"""Fruit Ninja widget: renders the composited game frame + a status bar.

Same split as Air Canvas: ``processor.py`` composites pieces/halves/blade
trail/HUD/start-or-game-over overlay onto the frame in place on the
pipeline worker thread; this widget's job is presentation only. Gesture
play happens entirely through the fingertip swipe, not Qt controls — the
processor owns all interaction/game state, and the widget never reads it
except via :meth:`on_frame_ready`.

The one deliberate exception (mirroring Air Canvas's keyboard shortcuts):
starting/restarting a run is a Qt-thread-only concern with no gesture
equivalent yet, so :meth:`keyPressEvent` forwards Space/Enter to the
processor's thread-safe
:meth:`~visionplay.apps.fruit_ninja.processor.FruitNinjaProcessor.request_start`
rather than mutating processor state directly.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QKeyEvent, QPainter, QPaintEvent, QPixmap
from PySide6.QtWidgets import QWidget

from visionplay.ui.widgets.camera_view import frame_to_qimage
from visionplay.vision.pipeline.frame_types import Frame

from .processor import RESULTS_KEY, FruitNinjaProcessor

__all__ = ["HINT_MESSAGE", "NO_DATA_MESSAGE", "FruitNinjaWidget"]

#: Shown whenever frame.results has no RESULTS_KEY entry (backend unavailable).
NO_DATA_MESSAGE: str = "Hand-tracking backend unavailable — Fruit Ninja needs it to play."

#: Shown while the backend is delivering results.
HINT_MESSAGE: str = "Fast swipe to slice fruit · avoid bombs · Space to start/restart"

#: Keys that (re)start a run — Space or Enter, either accepted so a
#: first-time player doesn't need to guess which one this app expects.
_START_KEYS: frozenset[Qt.Key] = frozenset(
    {Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter}
)


class FruitNinjaWidget(QWidget):
    """Renders the composited Fruit Ninja frame with a status/hint overlay."""

    def __init__(
        self, processor: FruitNinjaProcessor | None = None, parent: QWidget | None = None
    ) -> None:
        """Create an empty view; nothing is drawn until the first frame arrives.

        Args:
            processor: This app's processor, needed only to route the
                start/restart keyboard shortcut through its thread-safe
                ``request_start`` method. ``None`` disables the shortcut —
                the widget still renders frames normally (used by tests
                that don't exercise keyboard handling).
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
        """Route the start/restart shortcut to the processor; else default handling."""
        if self._processor is not None and event.key() in _START_KEYS:
            self._processor.request_start()
            event.accept()
            return
        super().keyPressEvent(event)

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
