"""Unit tests for the FramePipeline frame-processor seam (M1.4).

Headless: synthetic camera source, real ``PluginRegistry`` with the fixture
apps package as the processor — proving capture → ``on_frame`` → publish on
the worker thread with no camera, Qt, or inference involved.
"""

import threading
import time
from collections.abc import Callable
from dataclasses import replace

import numpy as np
from tests.fixtures.plugin_apps_fixture._support import RecordingPlugin

from visionplay.core.event_bus import EventBus
from visionplay.core.events import GameStopEvent
from visionplay.core.plugin_registry import PluginRegistry
from visionplay.vision.camera.camera_source import CameraSource
from visionplay.vision.pipeline.frame_bus import FrameBus, FramePipeline
from visionplay.vision.pipeline.frame_types import Frame

FIXTURE_APPS_PACKAGE = "tests.fixtures.plugin_apps_fixture"


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
    """Synthetic source: N frames then end of stream."""

    def __init__(self, total_frames: int = 3) -> None:
        self._total = total_frames
        self._next_id = 0
        self._opened = False

    def open(self) -> None:
        self._opened = True

    def read(self) -> Frame | None:
        if self._next_id >= self._total:
            return None
        frame = make_frame(self._next_id)
        self._next_id += 1
        return frame

    def release(self) -> None:
        self._opened = False

    def is_open(self) -> bool:
        return self._opened


def drain(bus: FrameBus, timeout: float = 5.0) -> list[Frame]:
    """Collect frames until the bus closes and drains."""
    frames: list[Frame] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        frame = bus.get(timeout=0.25)
        if frame is not None:
            frames.append(frame)
        elif bus.closed:
            return frames
    raise TimeoutError("bus did not close in time")


def make_registry() -> tuple[PluginRegistry, EventBus]:
    bus = EventBus()
    registry = PluginRegistry(event_bus=bus, apps_package=FIXTURE_APPS_PACKAGE)
    registry.discover()
    return registry, bus


def active_plugin(registry: PluginRegistry) -> RecordingPlugin:
    app_id = registry.active_app_id
    assert app_id is not None
    plugin = registry._apps[app_id].plugin
    assert isinstance(plugin, RecordingPlugin)
    return plugin


class TestProcessorInvocation:
    def test_on_frame_invoked_once_per_captured_frame(self) -> None:
        registry, _ = make_registry()
        registry.start("valid_app")
        pipeline = FramePipeline(
            FakeSource(total_frames=4),
            FrameBus(capacity=16),
            frame_processor=registry.process_frame,
        )
        with pipeline:
            frames = drain(pipeline.bus)
        assert len(frames) == 4
        assert active_plugin(registry).calls.count("on_frame") == 4

    def test_on_frame_runs_on_worker_thread(self) -> None:
        registry, _ = make_registry()
        registry.start("valid_app")
        seen_threads: list[str] = []

        def spy(frame: Frame) -> Frame:
            seen_threads.append(threading.current_thread().name)
            return registry.process_frame(frame)

        pipeline = FramePipeline(
            FakeSource(total_frames=3), FrameBus(capacity=16), frame_processor=spy
        )
        with pipeline:
            drain(pipeline.bus)
        assert seen_threads  # ran at least once
        assert all(name == "visionplay-frame-pipeline" for name in seen_threads)
        assert threading.current_thread().name not in seen_threads

    def test_processed_frame_is_what_gets_published(self) -> None:
        def annotate(frame: Frame) -> Frame:
            annotated = replace(frame, results={"annotated": True})
            return annotated

        pipeline = FramePipeline(
            FakeSource(total_frames=2), FrameBus(capacity=16), frame_processor=annotate
        )
        with pipeline:
            frames = drain(pipeline.bus)
        assert len(frames) == 2
        assert all(frame.results == {"annotated": True} for frame in frames)


class TestPassthroughWithoutActivePlugin:
    def test_no_processor_set_passes_frames_through_unchanged(self) -> None:
        pipeline = FramePipeline(FakeSource(total_frames=3), FrameBus(capacity=16))
        with pipeline:
            frames = drain(pipeline.bus)
        assert [f.frame_id for f in frames] == [0, 1, 2]
        assert all(f.results == {} for f in frames)

    def test_registry_with_no_active_app_passes_frames_through_unchanged(self) -> None:
        registry, _ = make_registry()  # discovered, but nothing started
        pipeline = FramePipeline(
            FakeSource(total_frames=3),
            FrameBus(capacity=16),
            frame_processor=registry.process_frame,
        )
        with pipeline:
            frames = drain(pipeline.bus)
        assert [f.frame_id for f in frames] == [0, 1, 2]
        assert all(f.results == {} for f in frames)

    def test_clearing_processor_restores_passthrough(self) -> None:
        registry, _ = make_registry()
        registry.start("valid_app")
        pipeline = FramePipeline(
            FakeSource(total_frames=3),
            FrameBus(capacity=16),
            frame_processor=registry.process_frame,
        )
        pipeline.set_frame_processor(None)  # cleared before start: pure passthrough
        with pipeline:
            frames = drain(pipeline.bus)
        assert [f.frame_id for f in frames] == [0, 1, 2]
        assert active_plugin(registry).calls.count("on_frame") == 0


class TestFailureContainmentStaysInRegistry:
    def test_raising_plugin_stops_the_app_not_the_pipeline(self) -> None:
        registry, event_bus = make_registry()
        stops: list[GameStopEvent] = []
        event_bus.subscribe(GameStopEvent, stops.append)
        registry.start("failing_frame_app")

        total = 10  # more than DEFAULT_MAX_CONSECUTIVE_FRAME_FAILURES
        pipeline = FramePipeline(
            FakeSource(total_frames=total),
            FrameBus(capacity=16),
            frame_processor=registry.process_frame,
        )
        with pipeline:
            frames = drain(pipeline.bus)

        # The pipeline survived the raising plugin and delivered every frame...
        assert pipeline.error is None
        assert len(frames) == total
        # ...while the registry's guard stopped the app at its threshold.
        assert registry.active_app_id is None
        assert stops[-1].app_id == "failing_frame_app"
        assert stops[-1].reason == "error"

    def test_worker_thread_not_leaked_after_failure_stop(self) -> None:
        registry, _ = make_registry()
        registry.start("failing_frame_app")
        pipeline = FramePipeline(
            FakeSource(total_frames=10),
            FrameBus(capacity=16),
            frame_processor=registry.process_frame,
        )
        pipeline.start()
        drain(pipeline.bus)
        pipeline.stop()
        leaked = [t.name for t in threading.enumerate() if t.name.startswith("visionplay-frame")]
        assert leaked == []
