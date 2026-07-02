"""Abstract inference backend: the seam every model runtime implements.

Phase 0 defines only the interface and lifecycle; real inference arrives in
Phase 2. The contract mirrors :class:`~visionplay.vision.camera.camera_source.CameraSource`
on purpose — ``load()`` → ``infer()`` (repeatedly) → ``unload()``, plus a
context manager — so the pipeline drives cameras and backends with the same
idioms.

Ownership (``docs/architecture.md`` §4): the frame pipeline owns backend
instances and their lifecycles. Plugins declare needs via the manifest's
``required_backends`` and never construct backends; the pipeline runs
``infer()`` on its worker thread and stores each backend's output in
``frame.results`` under the backend's :attr:`InferenceBackend.name`.

Device handling (§5): every backend takes a
:class:`~visionplay.vision.inference.device.DeviceConfig` at construction
instead of hardcoding CPU. Call sites pass through whatever configuration
resolves to; they never branch on device type.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType
from typing import Any

from visionplay.vision.inference.device import DeviceConfig
from visionplay.vision.pipeline.frame_types import Frame

__all__ = ["InferenceBackend", "InferenceError"]


class InferenceError(Exception):
    """A backend failed to load, run, or stay usable.

    Messages must be user-presentable (capability negotiation and the UI
    surface them): say which backend/model failed and, where known, why.
    """


class InferenceBackend(ABC):
    """Abstract runner of one model/task over :class:`Frame` objects.

    Lifecycle: ``load()`` → ``infer()`` (repeatedly) → ``unload()``.
    Implementations must make ``unload()`` idempotent and safe on a backend
    that never loaded, so teardown paths need no state tracking. Backends
    are single-consumer and not required to be thread-safe — the pipeline
    drives each backend from exactly one worker thread.

    The class is also a context manager: ``with backend:`` loads on entry
    and unloads on exit, even when ``infer()`` raises.
    """

    def __init__(self, device: DeviceConfig | None = None) -> None:
        """Store the device this backend will run on.

        Args:
            device: Target compute device. ``None`` means CPU — the v1
                default and only option.
        """
        self._device = device if device is not None else DeviceConfig.cpu()

    @property
    def device(self) -> DeviceConfig:
        """The device configuration this backend was constructed with."""
        return self._device

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier for this backend's output.

        This is the key under which the pipeline stores results in
        ``frame.results`` and the string app manifests list in
        ``required_backends`` (e.g. ``"mediapipe.pose"``, ``"onnx.yolo"``).
        """

    @abstractmethod
    def load(self) -> None:
        """Acquire the model/runtime and prepare it for inference.

        Idempotent on an already-loaded backend. Loading is separated from
        construction so the pipeline can construct backends cheaply during
        capability negotiation and load only what the active app needs.

        Raises:
            InferenceError: If the backend cannot become ready (model file
                missing, runtime unavailable, ...). Must not fail silently.
        """

    @abstractmethod
    def infer(self, frame: Frame) -> Any:
        """Run the model on one frame and return its results.

        The return value is whatever this backend produces (landmarks,
        detections, ...); the pipeline stores it in ``frame.results`` under
        :attr:`name`. Per-frame cost must stay bounded — the pipeline drops
        frames rather than queueing behind a slow backend.

        Raises:
            InferenceError: If called before :meth:`load` or after
                :meth:`unload`, or on a runtime failure mid-stream.
            NotImplementedError: Phase 0 stubs raise this; real inference
                is Phase 2.
        """

    @abstractmethod
    def unload(self) -> None:
        """Release the model/runtime.

        Idempotent: safe to call multiple times and on a backend that was
        never loaded. Must not raise for those cases.
        """

    @abstractmethod
    def is_loaded(self) -> bool:
        """Return ``True`` while the backend is loaded and able to infer."""

    def __enter__(self) -> InferenceBackend:
        """Load the backend and return it (``with backend as b:``)."""
        self.load()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Unload the backend on scope exit, exception or not."""
        self.unload()
