"""Air Canvas widget: renders the composited canvas frame + a status bar.

The frame arriving here is already composited by ``processor.py`` (camera
image + toolbar + strokes + cursor drawn in place on the pipeline worker
thread), so this widget's job is presentation only: paint the frame scaled
to fit and overlay a one-line status/hint. Gesture selection happens on the
frame's toolbar strip, not through Qt controls — the processor owns all
interaction state, and the widget never reads results by calling into the
plugin/processor; it only receives them via :meth:`on_frame_ready`.

M3.3 adds one narrow, deliberate exception: keyboard shortcuts (undo/redo/
save) and the export file dialog are Qt-thread-only concerns with no gesture
equivalent, so they need *some* way to reach ``processor.py``. Rather than
mutating processor state directly from here, :meth:`keyPressEvent` only
calls the processor's ``request_*`` methods, which enqueue onto a
thread-safe queue that ``processor.process()`` drains on the pipeline
worker thread — state is still mutated from exactly one thread. This widget
still never reads results any way but the per-frame signal.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QKeyEvent, QKeySequence, QPainter, QPaintEvent, QPixmap
from PySide6.QtWidgets import QFileDialog, QWidget

from visionplay.ui.widgets.camera_view import frame_to_qimage
from visionplay.vision.pipeline.frame_types import Frame

from .processor import RESULTS_KEY, AirCanvasProcessor

__all__ = ["HINT_MESSAGE", "NO_DATA_MESSAGE", "AirCanvasWidget"]

#: Shown whenever frame.results has no RESULTS_KEY entry (backend unavailable).
NO_DATA_MESSAGE: str = "Hand-tracking backend unavailable — Air Canvas needs it to draw."

#: Shown while the backend is delivering results.
HINT_MESSAGE: str = "Pinch to draw · toolbar on top selects color/brush/eraser · open palm clears"

#: Default filename offered by the export dialog.
_DEFAULT_EXPORT_NAME: str = "air_canvas.png"


class AirCanvasWidget(QWidget):
    """Renders the composited Air Canvas frame with a status/hint overlay."""

    def __init__(
        self, processor: AirCanvasProcessor | None = None, parent: QWidget | None = None
    ) -> None:
        """Create an empty view; nothing is drawn until the first frame arrives.

        Args:
            processor: This app's processor, needed only to route keyboard
                shortcuts (undo/redo/save) through its thread-safe
                ``request_*`` methods. ``None`` disables shortcuts — the
                widget still renders frames normally (used by tests that
                don't exercise keyboard handling).
            parent: Optional Qt parent.
        """
        super().__init__(parent)
        self._processor = processor
        self._pixmap: QPixmap | None = None
        self._overlay_message = NO_DATA_MESSAGE
        self.setMinimumSize(320, 240)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    @property
    def overlay_message(self) -> str:
        """The status text currently overlaid on the frame."""
        return self._overlay_message

    def on_frame_ready(self, frame: Frame) -> None:
        """Slot: connect to this app's per-frame Qt signal, never call directly.

        Args:
            frame: The frame most recently composited by ``processor.py``.
        """
        self._pixmap = QPixmap.fromImage(frame_to_qimage(frame))
        available = frame.results.get(RESULTS_KEY) is not None
        self._overlay_message = HINT_MESSAGE if available else NO_DATA_MESSAGE
        self.update()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 (Qt naming)
        """Route undo/redo/save shortcuts to the processor; else default handling."""
        if self._processor is not None and self._handle_shortcut(event):
            event.accept()
            return
        super().keyPressEvent(event)

    def _handle_shortcut(self, event: QKeyEvent) -> bool:
        """Handle one shortcut key event. Returns ``True`` if it was consumed."""
        processor = self._processor
        assert processor is not None
        if event.matches(QKeySequence.StandardKey.Undo):
            processor.request_undo()
            return True
        if event.matches(QKeySequence.StandardKey.Redo):
            processor.request_redo()
            return True
        if event.matches(QKeySequence.StandardKey.Save):
            self._export()
            return True
        return False

    def _export(self) -> None:
        """Prompt for a destination PNG and queue the export request."""
        processor = self._processor
        assert processor is not None
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Export Air Canvas", _DEFAULT_EXPORT_NAME, "PNG Images (*.png)"
        )
        if not path_str:
            return
        path = Path(path_str)
        if path.suffix.lower() != ".png":
            path = path.with_suffix(".png")
        processor.request_export(path, include_background=processor.export_include_background)

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
