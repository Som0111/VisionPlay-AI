"""Unit tests for visionplay.app: bootstrap wiring, startup, clean shutdown.

Runs fully headless: paths are rooted in a temp directory and the camera is
a synthetic source, so no real webcam, user profile, or display is touched.
The frame-delivery test proves the whole M0.6 render path — capture worker
→ frame bus → bridge thread → queued Qt signal → widget — end to end.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from visionplay.app import VisionPlayApp
from visionplay.core.paths import AppPaths
from visionplay.vision.camera.camera_source import CameraError, CameraSource
from visionplay.vision.pipeline.frame_types import Frame


def wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
    """Poll ``predicate`` until true or ``timeout`` elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def process_events_until(
    qapp: QApplication, predicate: Callable[[], bool], timeout: float = 5.0
) -> bool:
    """Pump the Qt event loop until ``predicate`` is true or ``timeout`` elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class FakeSource(CameraSource):
    """Endless synthetic camera: never runs out of frames."""

    def __init__(self) -> None:
        self._opened = False
        self._next_id = 0
        self.release_calls = 0

    def open(self) -> None:
        self._opened = True

    def read(self) -> Frame | None:
        if not self._opened:
            raise CameraError("read() on a source that is not open")
        frame = Frame.from_image(
            frame_id=self._next_id,
            timestamp=float(self._next_id),
            image=np.zeros((4, 4, 3), dtype=np.uint8),
        )
        self._next_id += 1
        return frame

    def release(self) -> None:
        self.release_calls += 1
        self._opened = False

    def is_open(self) -> bool:
        return self._opened


@pytest.fixture(autouse=True)
def no_leaked_threads() -> Iterator[None]:
    """Every test must end with no pipeline/bridge threads alive."""
    yield
    leaked = [t.name for t in threading.enumerate() if t.name.startswith("visionplay-frame")]
    assert leaked == []


@pytest.fixture
def app(tmp_path: Path) -> Iterator[VisionPlayApp]:
    """A VisionPlayApp on temp paths and a synthetic camera; always shut down."""
    instance = VisionPlayApp(AppPaths.for_root(tmp_path), source=FakeSource())
    yield instance
    instance.shutdown()


class TestBootstrap:
    def test_first_run_creates_config_and_log_files(self, tmp_path: Path) -> None:
        app = VisionPlayApp(AppPaths.for_root(tmp_path), source=FakeSource())
        assert app.paths.config_file.exists()
        assert app.paths.log_file.exists()
        app.shutdown()

    def test_wires_platform_objects_without_starting_capture(self, app: VisionPlayApp) -> None:
        assert app.config.get("camera", "device_index") == 0
        assert app.event_bus is not None
        assert not app.pipeline.is_running()
        assert app.window is None  # UI not built until start()


class TestStartup:
    def test_start_builds_window_and_begins_capture(
        self, qapp: QApplication, app: VisionPlayApp
    ) -> None:
        window = app.start()
        assert app.window is window
        assert app.pipeline.is_running()

    def test_start_twice_raises(self, qapp: QApplication, app: VisionPlayApp) -> None:
        app.start()
        with pytest.raises(RuntimeError, match="already started"):
            app.start()

    def test_frames_reach_the_camera_view(self, qapp: QApplication, app: VisionPlayApp) -> None:
        window = app.start()
        # Capture worker -> bus -> bridge -> queued signal -> widget render.
        assert process_events_until(qapp, lambda: window.camera_view.frames_shown > 0)
        assert window.camera_view.status is None  # waiting message cleared


class TestShutdown:
    def test_closing_the_window_stops_capture_and_releases_camera(
        self, qapp: QApplication, tmp_path: Path
    ) -> None:
        source = FakeSource()
        app = VisionPlayApp(AppPaths.for_root(tmp_path), source=source)
        window = app.start()
        window.show()
        assert process_events_until(qapp, lambda: window.camera_view.frames_shown > 0)
        assert window.close()  # triggers MainWindow.closing -> app.shutdown
        assert not app.pipeline.is_running()
        assert source.release_calls == 1
        assert not source.is_open()

    def test_shutdown_is_idempotent(self, qapp: QApplication, app: VisionPlayApp) -> None:
        app.start()
        app.shutdown()
        app.shutdown()
        assert not app.pipeline.is_running()

    def test_shutdown_before_start_is_safe(self, app: VisionPlayApp) -> None:
        app.shutdown()
        assert not app.pipeline.is_running()

    def test_camera_error_message_reaches_the_view(
        self, qapp: QApplication, tmp_path: Path
    ) -> None:
        class FailingSource(FakeSource):
            def open(self) -> None:
                raise CameraError("Camera 0 could not be opened")

        app = VisionPlayApp(AppPaths.for_root(tmp_path), source=FailingSource())
        window = app.start()
        status_shows_error = lambda: (  # noqa: E731
            window.camera_view.status is not None and "Camera 0" in window.camera_view.status
        )
        assert process_events_until(qapp, status_shows_error)
        app.shutdown()
