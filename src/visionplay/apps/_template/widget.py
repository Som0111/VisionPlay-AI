"""Template widget — copy and fill in your app's Qt rendering/controls.

The only place in an app where Qt-specific code belongs
(``docs/plugin-development.md``). This widget receives per-frame data only
through its :meth:`TemplateWidget.on_frame_ready` slot — the pipeline/UI
wiring connects that slot to a thread-safe Qt signal (M1.6). It must never
call into ``plugin.py`` or ``processor.py`` directly; that reintroduces a
threading hazard the signal connection exists to avoid.
"""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from visionplay.vision.pipeline.frame_types import Frame

__all__ = ["TemplateWidget"]


class TemplateWidget(QWidget):
    """Template per-app widget — replace with real rendering/controls."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the widget's chrome. No frame has been received yet."""
        super().__init__(parent)
        self._status_label = QLabel("Template app — no rendering implemented yet.")
        layout = QVBoxLayout(self)
        layout.addWidget(self._status_label)

    def on_frame_ready(self, frame: Frame) -> None:
        """Slot: connect to the app's per-frame Qt signal, never call directly.

        TODO: replace with real rendering. Runs on the Qt main thread —
        safe to touch widgets here, unlike ``plugin.py``'s ``on_frame``.

        Args:
            frame: The frame most recently processed by ``plugin.py``.
        """
        self._status_label.setText(f"Frame {frame.frame_id} received.")
