"""Concrete webcam capture source over ``cv2.VideoCapture``.

Implements the :class:`~visionplay.vision.camera.camera_source.CameraSource`
contract for a live, locally attached camera. Backend selection is explicit
(``docs/phase-0-checklist.md`` M0.4): on Windows, MSMF is tried first and
DirectShow second — MSMF is the modern API but some UVC devices/drivers only
behave under DirectShow. Other platforms use OpenCV's auto-selection, which
keeps this module portable even though only Windows is packaged in v1.

Live-source semantics: :meth:`WebcamSource.read` never returns ``None`` —
a webcam has no end of stream, so a failed read is always a
:class:`~visionplay.vision.camera.camera_source.CameraError` (device
unplugged, grabbed by another app, driver stall). The pipeline's
consecutive-failure limit decides when to give up.

Threading: per the ``CameraSource`` contract this class is single-consumer
and not thread-safe; the frame pipeline drives it from exactly one worker
thread.
"""

from __future__ import annotations

import logging
import sys
import time

import cv2
import numpy as np

from visionplay.vision.camera.camera_source import CameraError, CameraSource
from visionplay.vision.pipeline.frame_types import ColorFormat, Frame

__all__ = ["WebcamSource"]

logger = logging.getLogger(__name__)

#: Human-readable names for the capture backends we may try (error messages).
_BACKEND_NAMES: dict[int, str] = {
    cv2.CAP_MSMF: "MSMF",
    cv2.CAP_DSHOW: "DirectShow",
    cv2.CAP_ANY: "auto",
}


def _capture_backends() -> tuple[int, ...]:
    """Return the ``cv2.VideoCapture`` backends to try, in preference order."""
    if sys.platform == "win32":
        return (cv2.CAP_MSMF, cv2.CAP_DSHOW)
    return (cv2.CAP_ANY,)


class WebcamSource(CameraSource):
    """A live camera driven through ``cv2.VideoCapture``.

    Frame size is a *request*: drivers snap unsupported resolutions to the
    nearest mode they offer, so consumers must trust ``Frame.width/height``
    (derived from the delivered array), never the requested values.
    """

    def __init__(
        self,
        device_index: int = 0,
        *,
        frame_width: int | None = None,
        frame_height: int | None = None,
    ) -> None:
        """Configure the source; the device is not touched until :meth:`open`.

        Args:
            device_index: OS camera index (0 is the default/built-in camera).
            frame_width: Requested capture width in pixels; ``None`` keeps
                the driver default.
            frame_height: Requested capture height in pixels; ``None`` keeps
                the driver default.

        Raises:
            ValueError: If ``device_index`` is negative or a requested
                dimension is not positive.
        """
        if device_index < 0:
            raise ValueError(f"device_index must be >= 0, got {device_index}")
        for name, value in (("frame_width", frame_width), ("frame_height", frame_height)):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        self._device_index = device_index
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._capture: cv2.VideoCapture | None = None
        self._next_frame_id = 0

    def open(self) -> None:
        """Open the camera, trying each platform backend in order.

        A no-op if the source is already open.

        Raises:
            CameraError: If no backend can open the device — the message
                names the device index and every backend tried.
        """
        if self._capture is not None:
            return
        tried: list[str] = []
        for backend in _capture_backends():
            backend_name = _BACKEND_NAMES.get(backend, str(backend))
            capture = cv2.VideoCapture(self._device_index, backend)
            if capture.isOpened():
                self._request_frame_size(capture)
                self._capture = capture
                logger.info("Camera %d opened via %s backend", self._device_index, backend_name)
                return
            capture.release()
            tried.append(backend_name)
        raise CameraError(
            f"Camera {self._device_index} could not be opened "
            f"(backends tried: {', '.join(tried)}). "
            "Check that a camera is connected and not in use by another application."
        )

    def read(self) -> Frame | None:
        """Capture the next frame from the live camera.

        Returns:
            The next BGR :class:`Frame`; never ``None`` — a live camera has
            no end of stream.

        Raises:
            CameraError: If the source is not open or the device stopped
                delivering frames (disconnected, driver stall).
        """
        capture = self._capture
        if capture is None:
            raise CameraError(f"Camera {self._device_index} is not open")
        ok, image = capture.read()
        if not ok or image is None:
            raise CameraError(
                f"Camera {self._device_index} stopped delivering frames "
                "(device disconnected or claimed by another application)"
            )
        frame = Frame.from_image(
            frame_id=self._next_frame_id,
            timestamp=time.time(),
            image=np.ascontiguousarray(image, dtype=np.uint8),
            color_format=ColorFormat.BGR,
        )
        self._next_frame_id += 1
        return frame

    def release(self) -> None:
        """Release the camera. Idempotent; safe on a never-opened source."""
        capture = self._capture
        self._capture = None
        if capture is not None:
            capture.release()
            logger.info("Camera %d released", self._device_index)

    def is_open(self) -> bool:
        """Return ``True`` while the device is open and able to serve reads."""
        return self._capture is not None and bool(self._capture.isOpened())

    def _request_frame_size(self, capture: cv2.VideoCapture) -> None:
        """Ask the driver for the configured resolution (best effort)."""
        if self._frame_width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._frame_width))
        if self._frame_height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._frame_height))
