"""Unit tests for visionplay.ui.widgets.camera_view (offscreen Qt)."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from visionplay.ui.widgets import camera_view as camera_view_module
from visionplay.ui.widgets.camera_view import CameraView, frame_to_qimage
from visionplay.vision.pipeline.frame_types import ColorFormat, Frame


def make_frame(
    frame_id: int = 0,
    color_format: ColorFormat = ColorFormat.BGR,
    fill: int = 0,
) -> Frame:
    channels = () if color_format is ColorFormat.GRAY else (3,)
    return Frame.from_image(
        frame_id=frame_id,
        timestamp=float(frame_id),
        image=np.full((4, 6, *channels), fill, dtype=np.uint8),
        color_format=color_format,
    )


class TestFrameToQImage:
    def test_bgr_frame_converts_with_correct_size(self) -> None:
        qimage = frame_to_qimage(make_frame())
        assert (qimage.width(), qimage.height()) == (6, 4)

    def test_bgr_channel_order_is_honored(self) -> None:
        # Pure blue in BGR is (255, 0, 0) — must come out blue, not red.
        image = np.zeros((4, 6, 3), dtype=np.uint8)
        image[:, :, 0] = 255
        frame = Frame.from_image(frame_id=0, timestamp=0.0, image=image)
        color = frame_to_qimage(frame).pixelColor(0, 0)
        assert (color.red(), color.green(), color.blue()) == (0, 0, 255)

    def test_rgb_frame_converts(self) -> None:
        image = np.zeros((4, 6, 3), dtype=np.uint8)
        image[:, :, 0] = 255  # red in RGB
        frame = Frame.from_image(
            frame_id=0, timestamp=0.0, image=image, color_format=ColorFormat.RGB
        )
        color = frame_to_qimage(frame).pixelColor(0, 0)
        assert (color.red(), color.green(), color.blue()) == (255, 0, 0)

    def test_gray_frame_converts(self) -> None:
        qimage = frame_to_qimage(make_frame(color_format=ColorFormat.GRAY, fill=200))
        assert qimage.pixelColor(0, 0).red() == 200

    def test_result_owns_its_pixels(self) -> None:
        frame = make_frame(fill=255)
        qimage = frame_to_qimage(frame)
        frame.image[:] = 0  # mutate the source array after conversion
        assert qimage.pixelColor(0, 0).blue() == 255


class TestCameraView:
    def test_initial_state_waits_for_camera(self, qapp: QApplication) -> None:
        view = CameraView()
        assert view.frames_shown == 0
        assert view.fps == 0.0
        assert view.status is not None and "Waiting" in view.status

    def test_show_frame_updates_counters_and_clears_status(self, qapp: QApplication) -> None:
        view = CameraView()
        view.show_frame(make_frame(0))
        view.show_frame(make_frame(1))
        assert view.frames_shown == 2
        assert view.status is None

    def test_show_status_displays_message(self, qapp: QApplication) -> None:
        view = CameraView()
        view.show_frame(make_frame())
        view.show_status("Camera unplugged")
        assert view.status == "Camera unplugged"

    def test_fps_reflects_frame_arrival_times(
        self, qapp: QApplication, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ticks: Iterator[float] = iter(i * 0.1 for i in range(100))
        monkeypatch.setattr(camera_view_module, "monotonic", lambda: next(ticks))
        view = CameraView()
        for frame_id in range(5):
            view.show_frame(make_frame(frame_id))
        assert view.fps == pytest.approx(10.0)  # one frame every 0.1s

    def test_fps_window_is_rolling(
        self, qapp: QApplication, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 100 frames at 10 FPS: window must cap the sample count, not grow.
        ticks = iter(i * 0.1 for i in range(200))
        monkeypatch.setattr(camera_view_module, "monotonic", lambda: next(ticks))
        view = CameraView()
        for frame_id in range(100):
            view.show_frame(make_frame(frame_id))
        assert view.fps == pytest.approx(10.0)

    def test_paints_waiting_status_without_frames(self, qapp: QApplication) -> None:
        view = CameraView()
        view.resize(320, 240)
        assert not view.grab().isNull()  # exercises paintEvent's status path

    def test_paints_frame_with_fps_overlay(self, qapp: QApplication) -> None:
        view = CameraView()
        view.resize(320, 240)
        view.show_frame(make_frame(fill=128))
        rendered = view.grab().toImage()
        assert not rendered.isNull()
        # The frame is mid-gray; the widget background is black. Some pixel
        # in the centre must show the frame content, proving it was drawn.
        centre = rendered.pixelColor(rendered.width() // 2, rendered.height() // 2)
        assert centre.red() > 0

    def test_key_press_emits_key_pressed_with_the_key_code(self, qapp: QApplication) -> None:
        view = CameraView()
        received: list[int] = []
        view.key_pressed.connect(received.append)
        event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier)
        view.keyPressEvent(event)
        assert received == [Qt.Key.Key_Space]

    def test_frame_fills_the_widget_with_no_letterbox_bars(self, qapp: QApplication) -> None:
        # A 16:9 frame in a tall, narrow widget: KeepAspectRatio would leave
        # black bars top and bottom; the fix must fill every corner instead.
        view = CameraView()
        view.resize(200, 400)
        view.show_frame(make_frame(fill=200))
        rendered = view.grab().toImage()
        corners = [
            rendered.pixelColor(0, 0),
            rendered.pixelColor(rendered.width() - 1, 0),
            rendered.pixelColor(0, rendered.height() - 1),
            rendered.pixelColor(rendered.width() - 1, rendered.height() - 1),
        ]
        assert all(color.red() > 0 for color in corners)
