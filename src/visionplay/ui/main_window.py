"""Main application window: launcher + live camera view (M0.6, M1.6).

The window knows nothing about the plugin registry or vision pipeline —
the application bootstrap (``app.py``) populates :attr:`MainWindow.launcher`
via ``set_apps`` and connects its ``app_launch_requested`` signal, connects
the frame bridge's signals to :attr:`MainWindow.camera_view`, and listens
to :attr:`MainWindow.closing` to tear the pipeline down.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QHBoxLayout, QMainWindow, QWidget

from visionplay import __version__
from visionplay.ui.launcher.launcher_widget import LauncherWidget
from visionplay.ui.widgets.camera_view import CameraView

__all__ = ["MainWindow"]


class MainWindow(QMainWindow):
    """Top-level window: launcher on the left, camera view on the right."""

    #: Emitted from ``closeEvent`` before the window closes, so the
    #: application can stop the active app and frame pipeline while the
    #: camera view still exists.
    closing = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the window chrome and child widgets (nothing starts running)."""
        super().__init__(parent)
        self.setWindowTitle(f"VisionPlay AI v{__version__}")
        self.resize(1024, 640)

        self._launcher = LauncherWidget()
        self._camera_view = CameraView()

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.addWidget(self._launcher, stretch=1)
        layout.addWidget(self._camera_view, stretch=3)
        self.setCentralWidget(central)

    @property
    def launcher(self) -> LauncherWidget:
        """The launcher widget listing discovered apps and signaling launch intent."""
        return self._launcher

    @property
    def camera_view(self) -> CameraView:
        """The live camera view the frame bridge renders into."""
        return self._camera_view

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 (Qt naming)
        """Announce the close so the pipeline can be stopped, then close."""
        self.closing.emit()
        super().closeEvent(event)
