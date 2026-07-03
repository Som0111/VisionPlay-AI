"""ONNX Runtime backend for custom models (e.g. YOLO detectors).

``load()`` builds an ``onnxruntime.InferenceSession`` over the ordered
execution-provider list; ``infer()`` runs ``session.run`` on the frame and
returns the outputs as a standardized
:class:`~visionplay.vision.inference.results.TensorOutput` — named NumPy
arrays, no ONNX Runtime types exposed to plugins.

**Generic-model scope (v1)**: this backend feeds ``frame.image`` to a
*single-input* model, casting to the input's declared dtype. It does not
resize, normalize, or transpose to NCHW — that preprocessing is model
specific and belongs to the consuming plugin/model, not this generic runner.
Multi-input models are rejected explicitly rather than guessed at.

**Ordered execution-provider contract** (``docs/architecture.md`` §5): the
backend requests execution providers as an *ordered preference list* —
``onnxruntime``'s native mechanism — rather than picking one implicitly.
The runtime tries each provider in order and falls through to the next if
one is unavailable. v1 populates the list with ``["CPUExecutionProvider"]``
only; enabling GPU later means prepending ``"DmlExecutionProvider"`` /
``"CUDAExecutionProvider"`` ahead of the CPU entry (a config change), with
CPU remaining as the guaranteed fallback — no code-path change in this
module or the surrounding pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from visionplay.vision.inference.backend_base import InferenceBackend, InferenceError
from visionplay.vision.inference.device import DeviceConfig, DeviceType
from visionplay.vision.inference.model_registry import ModelFormat, ModelSpec
from visionplay.vision.inference.results import TensorOutput
from visionplay.vision.pipeline.frame_types import Frame

__all__ = ["ONNXBackend"]

#: DeviceType → default ordered execution-provider list. GPU support later
#: adds an entry whose list *starts* with the GPU provider and *ends* with
#: CPU as the fallback — an additive change only.
_DEFAULT_PROVIDERS: dict[DeviceType, tuple[str, ...]] = {
    DeviceType.CPU: ("CPUExecutionProvider",),
}

#: ONNX Runtime input element-type strings → NumPy dtype. Used to cast
#: ``frame.image`` to whatever the model's input expects before running.
_ONNX_ELEMENT_TYPE_TO_NUMPY: dict[str, type[np.generic]] = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(uint8)": np.uint8,
    "tensor(int8)": np.int8,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
}


class ONNXBackend(InferenceBackend):
    """Runs one ONNX model over frames via ``onnxruntime.InferenceSession``.

    The model artifact itself is materialized by the
    :class:`~visionplay.vision.inference.model_registry.ModelRegistry`;
    the pipeline resolves the spec to a local path via ``registry.ensure()``
    and hands both to this constructor. The backend never downloads.
    """

    def __init__(
        self,
        spec: ModelSpec,
        model_path: Path,
        device: DeviceConfig | None = None,
        providers: Sequence[str] | None = None,
    ) -> None:
        """Configure the backend; nothing is loaded until :meth:`load`.

        Args:
            spec: Registry spec of the model this backend runs. Must be
                :attr:`ModelFormat.ONNX` — format mismatches fail here, at
                construction, not at first frame.
            model_path: Verified local artifact path (from the registry).
            device: Target compute device; ``None`` means CPU (v1 default).
            providers: Ordered execution-provider preference list (see
                module docstring). ``None`` derives the default from
                ``device`` — ``("CPUExecutionProvider",)`` for CPU.

        Raises:
            InferenceError: If the spec is not an ONNX model or the
                provider list is empty.
        """
        super().__init__(device)
        if spec.format is not ModelFormat.ONNX:
            raise InferenceError(
                f"Model {spec.model_id!r} has format {spec.format.value!r}; "
                f"ONNXBackend requires {ModelFormat.ONNX.value!r}"
            )
        if providers is not None:
            resolved = tuple(providers)
        else:
            resolved = _DEFAULT_PROVIDERS[self.device.type]
        if not resolved:
            raise InferenceError(
                f"ONNXBackend for {spec.model_id!r} needs at least one execution provider"
            )
        self._spec = spec
        self._model_path = model_path
        self._providers = resolved
        # onnxruntime.InferenceSession once loaded; None while unloaded. Typed
        # Any because onnxruntime ships no stubs (see pyproject mypy overrides).
        self._session: Any = None

    @property
    def name(self) -> str:
        """Result key / manifest string, e.g. ``"onnx.yolo_nano"``."""
        return f"onnx.{self._spec.model_id}"

    @property
    def model_path(self) -> Path:
        """Local path of the model artifact this backend loads."""
        return self._model_path

    @property
    def providers(self) -> tuple[str, ...]:
        """Ordered execution-provider preference list, first is most preferred."""
        return self._providers

    def load(self) -> None:
        """Build the ``onnxruntime.InferenceSession`` and mark the backend ready.

        ``onnxruntime`` is imported here, not at module import, so constructing
        a backend and probing availability stay cheap and dependency-free.
        Idempotent: a second call with a session already built is a no-op.

        Raises:
            InferenceError: If the model file is missing, ``onnxruntime`` is
                not importable, or the session cannot be created — all
                user-presentable, never silent.
        """
        if self._session is not None:
            return
        if not self._model_path.is_file():
            raise InferenceError(f"Model file for {self.name!r} not found at {self._model_path}")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise InferenceError(
                f"ONNX Runtime is not available; cannot load backend {self.name!r}"
            ) from exc
        try:
            self._session = ort.InferenceSession(
                str(self._model_path), providers=list(self._providers)
            )
        except Exception as exc:
            raise InferenceError(f"Failed to load ONNX model for {self.name!r}: {exc}") from exc

    def infer(self, frame: Frame) -> TensorOutput:
        """Run the model on ``frame.image`` and return its named outputs.

        Feeds the frame image to the model's single input (cast to the input's
        declared dtype) and wraps every output tensor in a
        :class:`~visionplay.vision.inference.results.TensorOutput`. No resize
        or normalization is applied — see the module docstring.

        Args:
            frame: The captured frame to run inference on.

        Returns:
            The model's outputs as standardized named NumPy arrays.

        Raises:
            InferenceError: If the backend is not loaded, the model is not
                single-input, or the runtime fails mid-inference.
        """
        if self._session is None:
            raise InferenceError(f"Backend {self.name!r} is not loaded; call load() first")

        inputs = self._session.get_inputs()
        if len(inputs) != 1:
            raise InferenceError(
                f"Backend {self.name!r} supports single-input models only, "
                f"but the model declares {len(inputs)} inputs"
            )
        input_meta = inputs[0]
        tensor = self._prepare_input(frame, input_meta.type)
        try:
            raw_outputs = self._session.run(None, {input_meta.name: tensor})
            names = [output.name for output in self._session.get_outputs()]
        except Exception as exc:
            raise InferenceError(f"ONNX inference failed for {self.name!r}: {exc}") from exc
        return TensorOutput(
            {name: np.asarray(array) for name, array in zip(names, raw_outputs, strict=True)}
        )

    @staticmethod
    def _prepare_input(frame: Frame, element_type: str) -> np.ndarray[Any, np.dtype[Any]]:
        """Cast the frame image to the dtype the model input expects.

        Args:
            frame: Frame whose ``image`` feeds the model.
            element_type: ONNX Runtime input type string, e.g. ``tensor(float)``.

        Returns:
            The image array as the model's input dtype (unchanged shape).
        """
        dtype = _ONNX_ELEMENT_TYPE_TO_NUMPY.get(element_type)
        if dtype is None:
            # Unknown/unmapped type: feed the image as-is and let the runtime
            # complain with its own (still user-presentable) error.
            return frame.image
        return frame.image.astype(dtype)

    def unload(self) -> None:
        """Release the session. Idempotent; safe to call if never loaded."""
        self._session = None

    def is_loaded(self) -> bool:
        """Return ``True`` while the session exists and can run inference."""
        return self._session is not None
