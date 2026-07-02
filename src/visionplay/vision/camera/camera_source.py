"""Abstract frame-producing source: the seam every capture backend implements.

M0.4 defines only the interface; the concrete ``cv2.VideoCapture`` webcam
source (with MSMF/DirectShow fallback) arrives with the camera manager.
The abstraction is backend-agnostic on purpose: webcams, video files, IP
streams, screen capture, and the synthetic sources used in CI tests all
implement the same four-method contract, so the pipeline above never knows
which one it is driving.

Error philosophy (``docs/checklists/phase-0-checklist.md`` M0.4): failures raise
:class:`CameraError` with a user-presentable message — no silent
``None``-returns that the UI can't explain. The one legitimate ``None``
from :meth:`CameraSource.read` is *end of stream* (e.g. a video file ran
out), which is an expected condition, not a failure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType

from visionplay.vision.pipeline.frame_types import Frame

__all__ = ["CameraError", "CameraSource"]


class CameraError(Exception):
    """A capture source failed to open, read, or stay connected.

    Messages must be user-presentable (the UI surfaces them verbatim):
    say which device/source failed and, where known, why.
    """


class CameraSource(ABC):
    """Abstract producer of :class:`~visionplay.vision.pipeline.frame_types.Frame` objects.

    Lifecycle: ``open()`` → ``read()`` (repeatedly) → ``release()``.
    Implementations must make ``release()`` idempotent and safe to call on
    a source that never opened, so teardown paths need no state tracking.
    Sources are single-consumer and not required to be thread-safe — the
    pipeline (M0.5) drives a source from exactly one worker thread.

    The class is also a context manager: ``with source:`` opens on entry
    and releases on exit, even when a read raises.
    """

    @abstractmethod
    def open(self) -> None:
        """Acquire the underlying device/stream and prepare it for reading.

        Raises:
            CameraError: If the source cannot be opened (device missing,
                busy, unsupported mode, ...). Must not fail silently.
        """

    @abstractmethod
    def read(self) -> Frame | None:
        """Capture and return the next frame.

        Returns:
            The next :class:`Frame`, with ``frame_id`` increasing
            monotonically across the life of the source — or ``None`` on
            normal end of stream (finite sources such as video files).
            Live sources never return ``None``; they block for the next
            frame or raise.

        Raises:
            CameraError: On a read *failure* (device disconnected, source
                not open). Failure is never signalled by ``None``.
        """

    @abstractmethod
    def release(self) -> None:
        """Release the underlying device/stream.

        Idempotent: safe to call multiple times and on a source that was
        never opened. Must not raise for those cases.
        """

    @abstractmethod
    def is_open(self) -> bool:
        """Return ``True`` while the source is open and able to serve reads."""

    def __enter__(self) -> CameraSource:
        """Open the source and return it (``with source as s:``)."""
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Release the source on scope exit, exception or not."""
        self.release()
