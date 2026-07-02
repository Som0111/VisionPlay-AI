"""Main application window: launcher placeholder + live camera view (M0.6).

Phase 0 scope only — the launcher panel is an empty placeholder that the
real app launcher/dashboard replaces in Phase 1. The window knows nothing
about the vision pipeline; the application bootstrap (``app.py``) connects
the frame bridge's signals to :attr:`MainWindow.camera_view` and listens to
:attr:`MainWindow.closing` to tear the pipeline down.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

from visionplay import __version__
from visionplay.ui.widgets.camera_view import CameraView

__all__ = ["MainWindow"]

#: Object name of the launcher placeholder panel (findable in tests/QSS).
LAUNCHER_PLACEHOLDER_NAME: str = "launcherPlaceholder"


class MainWindow(QMainWindow):
    """Top-level window: launcher placeholder on the left, camera view right."""

    #: Emitted from ``closeEvent`` before the window closes, so the
    #: application can stop the frame bridge and pipeline while the camera
    #: view still exists.
    closing = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the window chrome and child widgets (nothing starts running)."""
        super().__init__(parent)
        self.setWindowTitle(f"VisionPlay AI v{__version__}")
        self.resize(1024, 640)

        self._camera_view = CameraView()

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.addWidget(self._build_launcher_placeholder(), stretch=1)
        layout.addWidget(self._camera_view, stretch=3)
        self.setCentralWidget(central)

    @property
    def camera_view(self) -> CameraView:
        """The live camera view the frame bridge renders into."""
        return self._camera_view

    def _build_launcher_placeholder(self) -> QWidget:
        """Build the empty launcher panel (real launcher lands in Phase 1)."""
        panel = QFrame()
        panel.setObjectName(LAUNCHER_PLACEHOLDER_NAME)
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(panel)
        title = QLabel("Apps")
        hint = QLabel("The app launcher arrives in Phase 1.")
        hint.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(hint)
        layout.addStretch(1)
        return panel

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 (Qt naming)
        """Announce the close so the pipeline can be stopped, then close."""
        self.closing.emit()
        super().closeEvent(event)
