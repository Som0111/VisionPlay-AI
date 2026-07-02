"""Unit tests for visionplay.ui.widgets.frame_bridge."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from visionplay.ui.widgets.frame_bridge import FrameBridge
from visionplay.vision.camera.camera_source import CameraError, CameraSource
from visionplay.vision.pipeline.frame_bus import FrameBus, FramePipeline
from visionplay.vision.pipeline.frame_types import Frame


def make_frame(frame_id: int = 0) -> Frame:
    return Frame.from_image(
        frame_id=frame_id,
        timestamp=float(frame_id),
        image=np.zeros((4, 4, 3), dtype=np.uint8),
    )


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
    """Pump the Qt event loop until ``predicate`` is true or ``timeout`` elapses.

    Bridge signals are emitted from the consumer thread and delivered as
    *queued* connections — receivers only run once the main thread's event
    loop processes them, exactly as in the real application.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class FakeSource(CameraSource):
    """Synthetic source: N frames then EOF; can fail on open."""

    def __init__(self, total_frames: int | None = 3, fail_open: bool = False) -> None:
        self._total = total_frames
        self._fail_open = fail_open
        self._opened = False
        self._next_id = 0

    def open(self) -> None:
        if self._fail_open:
            raise CameraError("Fake device 0 could not be opened")
        self._opened = True

    def read(self) -> Frame | None:
        if not self._opened:
            raise CameraError("read() on a source that is not open")
        if self._total is not None and self._next_id >= self._total:
            return None  # end of stream
        frame = make_frame(self._next_id)
        self._next_id += 1
        return frame

    def release(self) -> None:
        self._opened = False

    def is_open(self) -> bool:
        return self._opened


@pytest.fixture(autouse=True)
def no_leaked_threads() -> None:
    """Every test must end with no pipeline/bridge threads alive."""
    yield
    leaked = [t.name for t in threading.enumerate() if t.name.startswith("visionplay-frame")]
    assert leaked == []


class TestLifecycle:
    def test_start_and_stop_cleanly(self, qapp: QApplication) -> None:
        pipeline = FramePipeline(FakeSource(total_frames=None))
        bridge = FrameBridge(pipeline)
        pipeline.start()
        bridge.start()
        assert bridge.is_running()
        bridge.stop()
        assert not bridge.is_running()
        pipeline.stop()

    def test_start_while_running_raises(self, qapp: QApplication) -> None:
        pipeline = FramePipeline(FakeSource(total_frames=None))
        bridge = FrameBridge(pipeline)
        pipeline.start()
        bridge.start()
        try:
            with pytest.raises(RuntimeError, match="already running"):
                bridge.start()
        finally:
            bridge.stop()
            pipeline.stop()

    def test_stop_before_start_is_safe(self, qapp: QApplication) -> None:
        FrameBridge(FramePipeline(FakeSource())).stop()

    def test_stop_is_idempotent(self, qapp: QApplication) -> None:
        pipeline = FramePipeline(FakeSource(total_frames=None))
        bridge = FrameBridge(pipeline)
        pipeline.start()
        bridge.start()
        bridge.stop()
        bridge.stop()
        pipeline.stop()


class TestDelivery:
    def test_frames_are_reemitted_in_order(self, qapp: QApplication) -> None:
        pipeline = FramePipeline(FakeSource(total_frames=3), FrameBus(capacity=3))
        bridge = FrameBridge(pipeline)
        seen: list[int] = []
        bridge.frame_ready.connect(lambda frame: seen.append(frame.frame_id))
        pipeline.start()
        bridge.start()
        assert process_events_until(qapp, lambda: len(seen) == 3)
        assert seen == [0, 1, 2]
        bridge.stop()
        pipeline.stop()

    def test_stream_end_emits_user_message(self, qapp: QApplication) -> None:
        pipeline = FramePipeline(FakeSource(total_frames=1), FrameBus(capacity=1))
        bridge = FrameBridge(pipeline)
        messages: list[str] = []
        bridge.stream_ended.connect(messages.append)
        pipeline.start()
        bridge.start()
        assert process_events_until(qapp, lambda: messages == ["Camera stream ended."])
        assert not bridge.is_running()  # consumer exits on stream end
        bridge.stop()
        pipeline.stop()

    def test_pipeline_error_reaches_stream_ended_message(self, qapp: QApplication) -> None:
        pipeline = FramePipeline(FakeSource(fail_open=True))
        bridge = FrameBridge(pipeline)
        messages: list[str] = []
        bridge.stream_ended.connect(messages.append)
        pipeline.start()
        bridge.start()
        assert process_events_until(qapp, lambda: bool(messages) and "Fake device 0" in messages[0])
        assert not bridge.is_running()
        bridge.stop()
        pipeline.stop()

    def test_stop_does_not_emit_stream_ended(self, qapp: QApplication) -> None:
        pipeline = FramePipeline(FakeSource(total_frames=None))
        bridge = FrameBridge(pipeline)
        ended: list[str] = []
        bridge.stream_ended.connect(ended.append)
        pipeline.start()
        bridge.start()
        bridge.stop()
        pipeline.stop()
        qapp.processEvents()  # flush any queued deliveries before asserting
        assert ended == []
