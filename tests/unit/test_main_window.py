"""Unit tests for visionplay.ui.main_window (offscreen Qt)."""

from __future__ import annotations

from PySide6.QtWidgets import QApplication, QFrame

import visionplay
from visionplay.ui.main_window import LAUNCHER_PLACEHOLDER_NAME, MainWindow
from visionplay.ui.widgets.camera_view import CameraView


class TestConstruction:
    def test_window_title_carries_version(self, qapp: QApplication) -> None:
        window = MainWindow()
        assert visionplay.__version__ in window.windowTitle()

    def test_camera_view_is_embedded(self, qapp: QApplication) -> None:
        window = MainWindow()
        assert isinstance(window.camera_view, CameraView)
        assert window.centralWidget() is not None
        assert window.camera_view.parent() is window.centralWidget()

    def test_launcher_placeholder_is_present(self, qapp: QApplication) -> None:
        window = MainWindow()
        placeholder = window.findChild(QFrame, LAUNCHER_PLACEHOLDER_NAME)
        assert placeholder is not None


class TestClose:
    def test_close_emits_closing_signal(self, qapp: QApplication) -> None:
        window = MainWindow()
        emitted: list[bool] = []
        window.closing.connect(lambda: emitted.append(True))
        window.show()
        assert window.close()
        assert emitted == [True]
