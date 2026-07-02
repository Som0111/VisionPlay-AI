"""Hand-tracking demo widget: passthrough camera render + placeholder overlay.

Phase 1 has no real MediaPipe backend running (M1.4 wires the pipeline
seam, not inference), so ``frame.results.get(RESULTS_KEY)`` is always
absent — this widget renders the live frame with a fixed "not yet
available" banner instead of landmarks, proving the render path works end
to end. Phase 2 replaces the banner with real landmark drawing.

Receives frames only through :meth:`HandTrackingWidget.on_frame_ready` —
never by calling into ``plugin.py``/``processor.py`` directly.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPaintEvent, QPixmap
from PySide6.QtWidgets import QWidget

from visionplay.ui.widgets.camera_view import frame_to_qimage
from visionplay.vision.pipeline.frame_types import Frame

from .processor import RESULTS_KEY

__all__ = ["NO_DATA_MESSAGE", "HandTrackingWidget"]

#: Shown whenever frame.results has no RESULTS_KEY entry yet (always, in Phase 1).
NO_DATA_MESSAGE: str = "Hand-tracking data not yet available (Phase 2)."


class HandTrackingWidget(QWidget):
    """Renders the passthrough camera frame with a hand-tracking status overlay."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create an empty view; nothing is drawn until the first frame arrives."""
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._overlay_message = NO_DATA_MESSAGE
        self.setMinimumSize(320, 240)

    @property
    def overlay_message(self) -> str:
        """The status text currently overlaid on the frame."""
        return self._overlay_message

    def on_frame_ready(self, frame: Frame) -> None:
        """Slot: connect to this app's per-frame Qt signal, never call directly.

        Args:
            frame: The frame most recently processed by ``plugin.py``.
        """
        self._pixmap = QPixmap.fromImage(frame_to_qimage(frame))
        hands = frame.results.get(RESULTS_KEY)
        self._overlay_message = NO_DATA_MESSAGE if hands is None else f"Hands: {hands!r}"
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 (Qt naming)
        """Paint the current frame scaled to fit, plus the status overlay."""
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
        painter.setPen(QColor(220, 220, 220))
        painter.drawText(
            self.rect(),
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter | Qt.TextFlag.TextWordWrap,
            self._overlay_message,
        )
        painter.end()
