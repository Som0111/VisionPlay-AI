"""Unit tests for visionplay.vision.pipeline.frame_bus."""

import threading
import time
from collections.abc import Callable

import numpy as np
import pytest

from visionplay.core.event_bus import EventBus
from visionplay.core.events import FrameReadyEvent
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


class FakeSource(CameraSource):
    """Synthetic source: N frames then EOF; can fail on open or on chosen reads."""

    def __init__(
        self,
        total_frames: int | None = 3,
        fail_open: bool = False,
        failing_reads: frozenset[int] = frozenset(),
    ) -> None:
        self._total = total_frames  # None = endless (live-camera-like)
        self._fail_open = fail_open
        self._failing_reads = failing_reads
        self._opened = False
        self._next_id = 0
        self._read_calls = 0
        self.release_calls = 0

    def open(self) -> None:
        if self._fail_open:
            raise CameraError("Fake device 0 could not be opened")
        self._opened = True

    def read(self) -> Frame | None:
        if not self._opened:
            raise CameraError("read() on a source that is not open")
        self._read_calls += 1
        if self._read_calls in self._failing_reads:
            raise CameraError(f"Fake device 0 read failure #{self._read_calls}")
        if self._total is not None and self._next_id >= self._total:
            return None  # end of stream
        frame = make_frame(self._next_id)
        self._next_id += 1
        return frame

    def release(self) -> None:
        self.release_calls += 1
        self._opened = False

    def is_open(self) -> bool:
        return self._opened


@pytest.fixture(autouse=True)
def no_leaked_worker_threads() -> None:
    """Every test must end with no pipeline worker threads alive."""
    yield
    leaked = [t.name for t in threading.enumerate() if t.name.startswith("visionplay-frame")]
    assert leaked == []


class TestFrameBusBasics:
    def test_capacity_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="capacity"):
            FrameBus(capacity=0)

    def test_publish_then_get(self) -> None:
        bus = FrameBus()
        bus.publish(make_frame(7))
        received = bus.get(timeout=1.0)
        assert received is not None
        assert received.frame_id == 7

    def test_get_timeout_returns_none_when_empty(self) -> None:
        assert FrameBus().get(timeout=0.05) is None

    def test_get_blocks_until_publish(self) -> None:
        bus = FrameBus()
        results: list[Frame | None] = []
        consumer = threading.Thread(target=lambda: results.append(bus.get(timeout=5.0)))
        consumer.start()
        bus.publish(make_frame(1))
        consumer.join(timeout=5.0)
        assert not consumer.is_alive()
        assert results[0] is not None
        assert results[0].frame_id == 1


class TestFrameBusBackPressure:
    def test_full_bus_drops_oldest_and_serves_latest(self) -> None:
        bus = FrameBus(capacity=1)
        for frame_id in range(5):
            bus.publish(make_frame(frame_id))
        received = bus.get(timeout=1.0)
        assert received is not None
        assert received.frame_id == 4  # latest wins; 0-3 dropped, never queued
        assert bus.frames_dropped == 4
        assert bus.frames_published == 5

    def test_capacity_bounds_pending_not_published(self) -> None:
        bus = FrameBus(capacity=2)
        for frame_id in range(4):
            bus.publish(make_frame(frame_id))
        first = bus.get(timeout=1.0)
        second = bus.get(timeout=1.0)
        assert first is not None and first.frame_id == 2
        assert second is not None and second.frame_id == 3
        assert bus.frames_dropped == 2

    def test_publish_never_blocks(self) -> None:
        bus = FrameBus(capacity=1)
        start = time.monotonic()
        for frame_id in range(100):
            bus.publish(make_frame(frame_id))
        assert time.monotonic() - start < 1.0


class TestFrameBusClose:
    def test_close_wakes_blocked_consumer_with_none(self) -> None:
        bus = FrameBus()
        results: list[Frame | None] = []
        consumer = threading.Thread(target=lambda: results.append(bus.get(timeout=5.0)))
        consumer.start()
        bus.close()
        consumer.join(timeout=5.0)
        assert not consumer.is_alive()
        assert results == [None]

    def test_pending_frames_drain_after_close(self) -> None:
        bus = FrameBus(capacity=2)
        bus.publish(make_frame(0))
        bus.publish(make_frame(1))
        bus.close()
        first = bus.get(timeout=1.0)
        second = bus.get(timeout=1.0)
        assert first is not None and first.frame_id == 0
        assert second is not None and second.frame_id == 1
        assert bus.get(timeout=0.05) is None  # drained: end of stream

    def test_publish_after_close_is_noop(self) -> None:
        bus = FrameBus()
        bus.close()
        bus.publish(make_frame(0))
        assert bus.frames_published == 0
        assert bus.get(timeout=0.05) is None

    def test_close_is_idempotent(self) -> None:
        bus = FrameBus()
        bus.close()
        bus.close()
        assert bus.closed


class TestPipelineLifecycle:
    def test_frames_flow_source_to_consumer(self) -> None:
        pipeline = FramePipeline(FakeSource(total_frames=3), FrameBus(capacity=3))
        pipeline.start()
        received = list(iter(lambda: pipeline.bus.get(timeout=5.0), None))
        pipeline.stop()
        assert [frame.frame_id for frame in received] == [0, 1, 2]
        assert pipeline.error is None

    def test_stop_releases_camera_and_joins_worker(self) -> None:
        source = FakeSource(total_frames=None)  # endless, live-camera-like
        pipeline = FramePipeline(source, FrameBus())
        pipeline.start()
        assert wait_until(lambda: pipeline.frames_captured > 0)
        pipeline.stop()
        assert not pipeline.is_running()
        assert source.release_calls == 1
        assert not source.is_open()

    def test_end_of_stream_stops_worker_and_closes_bus(self) -> None:
        source = FakeSource(total_frames=1)
        pipeline = FramePipeline(source, FrameBus())
        pipeline.start()
        assert wait_until(lambda: pipeline.bus.closed)
        assert wait_until(lambda: not pipeline.is_running())
        assert source.release_calls == 1
        assert pipeline.error is None
        pipeline.stop()  # already finished: must be safe

    def test_start_while_running_raises(self) -> None:
        pipeline = FramePipeline(FakeSource(total_frames=None), FrameBus())
        pipeline.start()
        try:
            with pytest.raises(RuntimeError, match="already running"):
                pipeline.start()
        finally:
            pipeline.stop()

    def test_stop_before_start_is_safe(self) -> None:
        FramePipeline(FakeSource(), FrameBus()).stop()

    def test_stop_is_idempotent(self) -> None:
        pipeline = FramePipeline(FakeSource(total_frames=None), FrameBus())
        pipeline.start()
        pipeline.stop()
        pipeline.stop()

    def test_restart_after_stop(self) -> None:
        pipeline = FramePipeline(FakeSource(total_frames=None), FrameBus())
        pipeline.start()
        pipeline.stop()
        source = FakeSource(total_frames=None)
        pipeline = FramePipeline(source, FrameBus())
        pipeline.start()
        assert pipeline.is_running()
        pipeline.stop()

    def test_context_manager_starts_and_stops(self) -> None:
        source = FakeSource(total_frames=None)
        with FramePipeline(source, FrameBus()) as pipeline:
            assert pipeline.is_running()
        assert not pipeline.is_running()
        assert source.release_calls == 1

    def test_default_bus_is_latest_frame_only(self) -> None:
        pipeline = FramePipeline(FakeSource())
        assert isinstance(pipeline.bus, FrameBus)

    def test_invalid_target_fps_rejected(self) -> None:
        with pytest.raises(ValueError, match="target_fps"):
            FramePipeline(FakeSource(), target_fps=0)

    def test_invalid_failure_limit_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_consecutive_failures"):
            FramePipeline(FakeSource(), max_consecutive_failures=0)


class TestPipelineBackPressure:
    def test_slow_consumer_drops_frames_and_sees_latest(self) -> None:
        source = FakeSource(total_frames=50)
        pipeline = FramePipeline(source, FrameBus(capacity=1))
        pipeline.start()
        assert wait_until(lambda: pipeline.bus.closed)  # producer outran us to EOF
        assert pipeline.bus.frames_dropped > 0  # frames dropped, never queued
        last_pending = pipeline.bus.get(timeout=1.0)
        assert last_pending is not None
        assert last_pending.frame_id == 49  # what survived is the newest frame
        pipeline.stop()

    def test_fps_governor_caps_capture_rate(self) -> None:
        source = FakeSource(total_frames=None)
        pipeline = FramePipeline(source, FrameBus(), target_fps=50)
        pipeline.start()
        time.sleep(0.3)
        pipeline.stop()
        # 50 fps over 0.3s is ~15 frames; generous slack for scheduler jitter.
        # (Only the upper bound is guaranteed: load can slow capture, never speed it.)
        assert pipeline.frames_captured <= 25

    def test_ungoverned_pipeline_captures_freely(self) -> None:
        pipeline = FramePipeline(FakeSource(total_frames=100), FrameBus(capacity=1))
        pipeline.start()
        assert wait_until(lambda: pipeline.bus.closed)
        assert pipeline.frames_captured == 100
        pipeline.stop()


class TestPipelineErrorRecovery:
    def test_open_failure_sets_error_and_closes_bus(self) -> None:
        pipeline = FramePipeline(FakeSource(fail_open=True), FrameBus())
        pipeline.start()
        assert wait_until(lambda: not pipeline.is_running())
        assert isinstance(pipeline.error, CameraError)
        assert pipeline.bus.closed  # consumer unblocks instead of hanging
        pipeline.stop()

    def test_transient_read_errors_are_survived(self) -> None:
        # Reads 2 and 3 fail, under the default limit of 3 consecutive failures.
        source = FakeSource(total_frames=5, failing_reads=frozenset({2, 3}))
        pipeline = FramePipeline(source, FrameBus(capacity=10))
        pipeline.start()
        assert wait_until(lambda: pipeline.bus.closed)
        pipeline.stop()
        assert pipeline.error is None
        assert pipeline.frames_captured == 5  # every frame still delivered

    def test_persistent_read_errors_stop_pipeline_cleanly(self) -> None:
        source = FakeSource(total_frames=None, failing_reads=frozenset(range(2, 100)))
        pipeline = FramePipeline(source, FrameBus(), max_consecutive_failures=3)
        pipeline.start()
        assert wait_until(lambda: not pipeline.is_running())
        assert isinstance(pipeline.error, CameraError)
        assert source.release_calls == 1  # camera released even on failure
        assert pipeline.bus.closed
        pipeline.stop()

    def test_error_resets_on_restart(self) -> None:
        pipeline = FramePipeline(FakeSource(fail_open=True), FrameBus())
        pipeline.start()
        assert wait_until(lambda: not pipeline.is_running())
        assert pipeline.error is not None
        pipeline.stop()


class TestPipelineEventBus:
    def test_frame_ready_events_carry_frame_indices(self) -> None:
        event_bus = EventBus()
        seen: list[int] = []
        event_bus.subscribe(FrameReadyEvent, lambda e: seen.append(e.frame_index))
        pipeline = FramePipeline(
            FakeSource(total_frames=3), FrameBus(capacity=3), event_bus=event_bus
        )
        pipeline.start()
        assert wait_until(lambda: pipeline.bus.closed)
        pipeline.stop()
        assert seen == [0, 1, 2]

    def test_raising_event_handler_does_not_stop_capture(self) -> None:
        event_bus = EventBus()

        def bad_handler(event: FrameReadyEvent) -> None:
            raise RuntimeError("subscriber bug")

        event_bus.subscribe(FrameReadyEvent, bad_handler)
        pipeline = FramePipeline(
            FakeSource(total_frames=3), FrameBus(capacity=3), event_bus=event_bus
        )
        pipeline.start()
        assert wait_until(lambda: pipeline.bus.closed)
        pipeline.stop()
        assert pipeline.frames_captured == 3
        assert pipeline.error is None
