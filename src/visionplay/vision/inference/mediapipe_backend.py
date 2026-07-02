"""MediaPipe landmark backend stub (hand/pose/face).

Phase 0 ships lifecycle and configuration only — no MediaPipe import, no
graph construction, no inference. Phase 2 replaces the body of ``load()``
with MediaPipe Tasks graph creation and ``infer()`` with real landmark
extraction; the constructor signature and :attr:`MediaPipeBackend.name`
contract defined here are already final.

Device handling (``docs/architecture.md`` §5): MediaPipe's Tasks API takes
a ``delegate`` (``CPU``/``GPU``) per task. The :attr:`MediaPipeBackend.delegate`
property derives it from the backend's :class:`DeviceConfig` via a mapping —
v1 maps CPU only, and GPU support later adds one mapping entry without
touching the surrounding pipeline.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from visionplay.vision.inference.backend_base import InferenceBackend, InferenceError
from visionplay.vision.inference.device import DeviceConfig, DeviceType
from visionplay.vision.pipeline.frame_types import Frame

__all__ = ["MediaPipeBackend", "MediaPipeTask"]

#: DeviceType → MediaPipe Tasks delegate name. GPU support later appends an
#: entry here (e.g. ``DeviceType.GPU: "GPU"``) — an additive change only.
_DELEGATES: dict[DeviceType, str] = {
    DeviceType.CPU: "CPU",
}


class MediaPipeTask(Enum):
    """The MediaPipe landmark task a backend instance runs.

    Values are the suffix of the backend :attr:`~MediaPipeBackend.name`
    (``"mediapipe.<value>"``) — the strings app manifests reference in
    ``required_backends``.
    """

    HAND_LANDMARKS = "hands"
    POSE_LANDMARKS = "pose"
    FACE_LANDMARKS = "face"


class MediaPipeBackend(InferenceBackend):
    """Runs one MediaPipe landmark task over frames (stub — Phase 2 wires it).

    One instance handles exactly one task; an app needing both hands and
    pose declares two ``required_backends`` and the pipeline owns two
    instances.
    """

    def __init__(self, task: MediaPipeTask, device: DeviceConfig | None = None) -> None:
        """Configure the backend; nothing is loaded until :meth:`load`.

        Args:
            task: Which landmark task this instance runs.
            device: Target compute device; ``None`` means CPU (v1 default).
        """
        super().__init__(device)
        self._task = task
        self._loaded = False

    @property
    def name(self) -> str:
        """Result key / manifest string, e.g. ``"mediapipe.pose"``."""
        return f"mediapipe.{self._task.value}"

    @property
    def task(self) -> MediaPipeTask:
        """The landmark task this instance is configured for."""
        return self._task

    @property
    def delegate(self) -> str:
        """MediaPipe Tasks delegate derived from the device config (v1: ``"CPU"``)."""
        return _DELEGATES[self.device.type]

    def load(self) -> None:
        """Mark the backend ready. Phase 2: build the MediaPipe Tasks graph here."""
        self._loaded = True

    def infer(self, frame: Frame) -> Any:
        """Stub — validates lifecycle only; real landmark extraction is Phase 2.

        Raises:
            InferenceError: If the backend is not loaded.
            NotImplementedError: Always, once loaded — no inference in Phase 0.
        """
        if not self._loaded:
            raise InferenceError(f"Backend {self.name!r} is not loaded; call load() first")
        raise NotImplementedError(
            f"MediaPipe inference for {self.name!r} is not implemented until Phase 2"
        )

    def unload(self) -> None:
        """Release the (future) graph. Idempotent; safe if never loaded."""
        self._loaded = False

    def is_loaded(self) -> bool:
        """Return ``True`` while the backend is loaded and able to infer."""
        return self._loaded
