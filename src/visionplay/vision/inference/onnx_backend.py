"""ONNX Runtime backend stub for custom models (e.g. YOLO detectors).

Phase 0 ships lifecycle and configuration only — no ``onnxruntime`` import,
no session, no inference. Phase 2 replaces the body of ``load()`` with
``onnxruntime.InferenceSession(model_path, providers=self.providers)`` and
``infer()`` with real ``session.run(...)`` calls; the constructor signature
and provider contract defined here are already final.

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

from visionplay.vision.inference.backend_base import InferenceBackend, InferenceError
from visionplay.vision.inference.device import DeviceConfig, DeviceType
from visionplay.vision.inference.model_registry import ModelFormat, ModelSpec
from visionplay.vision.pipeline.frame_types import Frame

__all__ = ["ONNXBackend"]

#: DeviceType → default ordered execution-provider list. GPU support later
#: adds an entry whose list *starts* with the GPU provider and *ends* with
#: CPU as the fallback — an additive change only.
_DEFAULT_PROVIDERS: dict[DeviceType, tuple[str, ...]] = {
    DeviceType.CPU: ("CPUExecutionProvider",),
}


class ONNXBackend(InferenceBackend):
    """Runs one ONNX model over frames (stub — Phase 2 wires the session).

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
        self._loaded = False

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
        """Validate the artifact and mark the backend ready.

        Phase 2: replace the readiness flag with
        ``onnxruntime.InferenceSession(self.model_path, providers=self.providers)``.

        Raises:
            InferenceError: If the model file does not exist — a
                user-presentable failure, not a silent one.
        """
        if not self._model_path.is_file():
            raise InferenceError(f"Model file for {self.name!r} not found at {self._model_path}")
        self._loaded = True

    def infer(self, frame: Frame) -> Any:
        """Stub — validates lifecycle only; real ``session.run`` is Phase 2.

        Raises:
            InferenceError: If the backend is not loaded.
            NotImplementedError: Always, once loaded — no inference in Phase 0.
        """
        if not self._loaded:
            raise InferenceError(f"Backend {self.name!r} is not loaded; call load() first")
        raise NotImplementedError(
            f"ONNX inference for {self.name!r} is not implemented until Phase 2"
        )

    def unload(self) -> None:
        """Release the (future) session. Idempotent; safe if never loaded."""
        self._loaded = False

    def is_loaded(self) -> bool:
        """Return ``True`` while the backend is loaded and able to infer."""
        return self._loaded
