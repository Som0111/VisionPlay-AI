"""Live camera view widget: renders frames with an FPS overlay.

Receives frames exclusively through its :meth:`CameraView.show_frame` slot,
connected to :class:`~visionplay.ui.widgets.frame_bridge.FrameBridge`'s
``frame_ready`` signal — Qt's queued delivery guarantees the slot runs on
the widget's own thread, so painting here is always main-thread-safe.

The FPS counter measures *displayed* frames (slot invocations), not camera
capture rate: with the frame bus's latest-wins policy those can legitimately
differ, and what the user cares about on screen is how fluid the view is.
"""

from __future__ import annotations

from collections import deque
from time import monotonic

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPaintEvent, QPixmap
from PySide6.QtWidgets import QWidget

from visionplay.vision.pipeline.frame_types import ColorFormat, Frame

__all__ = ["CameraView", "frame_to_qimage"]

#: Rolling window (in frames) over which the displayed FPS is averaged.
_FPS_WINDOW: int = 30

#: Status shown before the first frame arrives.
_WAITING_STATUS: str = "Waiting for camera..."


def frame_to_qimage(frame: Frame) -> QImage:
    """Convert a pipeline frame to a ``QImage``, honoring its color format.

    The returned image owns its pixels (deep copy), so it stays valid after
    the frame's ndarray is garbage-collected or reused.

    Args:
        frame: A BGR, RGB, or grayscale frame.

    Returns:
        A ``QImage`` of the same dimensions and channel layout.
    """
    image = np.ascontiguousarray(frame.image)
    if frame.color_format is ColorFormat.GRAY:
        qt_format = QImage.Format.Format_Grayscale8
        bytes_per_line = frame.width
    elif frame.color_format is ColorFormat.RGB:
        qt_format = QImage.Format.Format_RGB888
        bytes_per_line = frame.width * 3
    else:  # BGR — OpenCV capture native order
        qt_format = QImage.Format.Format_BGR888
        bytes_per_line = frame.width * 3
    return QImage(image.data, frame.width, frame.height, bytes_per_line, qt_format).copy()


class CameraView(QWidget):
    """Widget that displays the live camera feed with an FPS overlay.

    Until the first frame arrives (or after the stream ends) a status
    message is shown instead; :meth:`show_status` lets the application
    surface camera errors in place of the feed.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create an empty view showing the waiting status."""
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._status: str | None = _WAITING_STATUS
        self._frame_times: deque[float] = deque(maxlen=_FPS_WINDOW)
        self._frames_shown = 0
        self.setMinimumSize(320, 240)

    @property
    def frames_shown(self) -> int:
        """Total frames rendered since the widget was created."""
        return self._frames_shown

    @property
    def status(self) -> str | None:
        """The status message currently displayed, or ``None`` during live feed."""
        return self._status

    @property
    def fps(self) -> float:
        """Displayed frames per second, averaged over a rolling window."""
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        if span <= 0:
            return 0.0
        return (len(self._frame_times) - 1) / span

    def show_frame(self, frame: Frame) -> None:
        """Display a new frame (connected to ``FrameBridge.frame_ready``).

        Args:
            frame: The frame to render. Converted and copied immediately,
                so the caller may release the underlying array.
        """
        self._frame_times.append(monotonic())
        self._pixmap = QPixmap.fromImage(frame_to_qimage(frame))
        self._frames_shown += 1
        self._status = None
        self.update()

    def show_status(self, message: str) -> None:
        """Display a status/error message (connected to ``FrameBridge.stream_ended``).

        Args:
            message: User-presentable text, e.g. a camera error.
        """
        self._status = message
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 (Qt naming)
        """Paint the current frame scaled to fit, plus FPS/status overlays."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        if self._pixmap is not None:
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            offset_x = (self.width() - scaled.width()) // 2
            offset_y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(offset_x, offset_y, scaled)
            self._paint_fps(painter)
        if self._status is not None:
            self._paint_status(painter, self._status)
        painter.end()

    def _paint_fps(self, painter: QPainter) -> None:
        """Draw the FPS counter in the top-left corner."""
        text = f"{self.fps:5.1f} FPS"
        metrics = painter.fontMetrics()
        padding = 6
        box = metrics.boundingRect(text).adjusted(-padding, -padding, padding, padding)
        box.moveTopLeft(self.rect().topLeft())
        painter.fillRect(box, QColor(0, 0, 0, 160))
        painter.setPen(QColor(0, 255, 128))
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

    def _paint_status(self, painter: QPainter, message: str) -> None:
        """Draw a status message centered over the view."""
        painter.setPen(QColor(220, 220, 220))
        painter.drawText(
            self.rect(),
            Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
            message,
        )
