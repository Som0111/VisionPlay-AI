"""Unit tests for apps/hand_tracking_demo (M1.7).

Covers: the manifest is well-formed and discoverable, the processor
handles both an absent and a populated ``mediapipe.hands`` result without
raising, the plugin delegates every lifecycle stage, the widget renders a
placeholder overlay until real data exists, and — the concrete pipeline-
wiring proof this milestone exists for — the real registry discovers this
app, the launcher can list it, it starts/stops correctly, and its
``on_frame`` runs through the real ``FramePipeline`` with no real
inference involved.
"""

import time

import numpy as np
from PySide6.QtWidgets import QApplication

from visionplay.apps.hand_tracking_demo.manifest import MANIFEST
from visionplay.apps.hand_tracking_demo.plugin import Plugin
from visionplay.apps.hand_tracking_demo.processor import RESULTS_KEY, HandTrackingProcessor
from visionplay.apps.hand_tracking_demo.widget import NO_DATA_MESSAGE, HandTrackingWidget
from visionplay.core.event_bus import EventBus
from visionplay.core.plugin_base import AppPlugin
from visionplay.core.plugin_registry import PluginRegistry
from visionplay.ui.launcher.launcher_widget import LauncherWidget
from visionplay.vision.camera.camera_source import CameraSource
from visionplay.vision.pipeline.frame_bus import FrameBus, FramePipeline
from visionplay.vision.pipeline.frame_types import Frame


def make_frame(frame_id: int = 0, results: dict[str, object] | None = None) -> Frame:
    frame = Frame.from_image(
        frame_id=frame_id,
        timestamp=float(frame_id),
        image=np.zeros((4, 4, 3), dtype=np.uint8),
    )
    if results:
        frame.results.update(results)
    return frame


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


class TestManifest:
    def test_manifest_is_well_formed(self) -> None:
        assert MANIFEST.id == "hand_tracking_demo"
        assert MANIFEST.category == "ai_demos"

    def test_declares_mediapipe_hands_backend(self) -> None:
        assert MANIFEST.required_backends == ("mediapipe.hands",)


class TestProcessor:
    def test_handles_absent_results_without_raising(self) -> None:
        processor = HandTrackingProcessor()
        frame = make_frame(1)
        assert RESULTS_KEY not in frame.results
        result = processor.process(frame)
        assert result is frame

    def test_handles_populated_results_without_raising(self) -> None:
        processor = HandTrackingProcessor()
        frame = make_frame(2, results={RESULTS_KEY: [{"landmarks": []}]})
        result = processor.process(frame)
        assert result is frame
        assert result.results[RESULTS_KEY] == [{"landmarks": []}]

    def test_start_and_stop_do_not_raise(self) -> None:
        processor = HandTrackingProcessor()
        processor.start()
        processor.stop()


class TestPlugin:
    def test_is_an_app_plugin(self) -> None:
        assert isinstance(Plugin(), AppPlugin)

    def test_on_frame_delegates_to_processor_and_returns_frame(self) -> None:
        plugin = Plugin()
        frame = make_frame(7)
        assert plugin.on_frame(frame) is frame

    def test_full_lifecycle_does_not_raise(self) -> None:
        plugin = Plugin()
        plugin.on_load()
        plugin.on_start()
        plugin.on_frame(make_frame())
        plugin.on_stop()
        plugin.on_unload()


class TestWidget:
    def test_shows_placeholder_message_with_no_hand_data(self, qapp: QApplication) -> None:
        widget = HandTrackingWidget()
        widget.on_frame_ready(make_frame(1))
        assert widget.overlay_message == NO_DATA_MESSAGE

    def test_shows_different_message_once_hand_data_exists(self, qapp: QApplication) -> None:
        widget = HandTrackingWidget()
        widget.on_frame_ready(make_frame(2, results={RESULTS_KEY: [{"landmarks": []}]}))
        assert widget.overlay_message != NO_DATA_MESSAGE
        assert "landmarks" in widget.overlay_message


class TestPipelineWiring:
    """The concrete M1.7 proof: discovery, launcher, start/stop, on_frame — no inference."""

    def test_real_registry_discovers_the_app(self) -> None:
        registry = PluginRegistry(event_bus=EventBus())  # default apps_package="visionplay.apps"
        registry.discover()
        assert "hand_tracking_demo" in registry.manifests

    def test_appears_in_the_launcher(self, qapp: QApplication) -> None:
        registry = PluginRegistry(event_bus=EventBus())
        registry.discover()
        launcher = LauncherWidget(registry.manifests)
        assert "hand_tracking_demo" in launcher.manifests
        assert launcher.manifests["hand_tracking_demo"].name == "Hand Tracking Demo"

    def test_starts_and_stops_correctly(self) -> None:
        registry = PluginRegistry(event_bus=EventBus())
        registry.discover()
        registry.start("hand_tracking_demo")
        assert registry.active_app_id == "hand_tracking_demo"
        registry.stop_active()
        assert registry.active_app_id is None

    def test_on_frame_runs_through_the_real_pipeline_without_inference(self) -> None:
        registry = PluginRegistry(event_bus=EventBus())
        registry.discover()
        registry.start("hand_tracking_demo")

        pipeline = FramePipeline(
            FakeSource(total_frames=4),
            FrameBus(capacity=16),
            frame_processor=registry.process_frame,
        )
        with pipeline:
            frames = drain(pipeline.bus)

        assert pipeline.error is None
        assert len(frames) == 4
        # No backend populated mediapipe.hands — proves no real inference ran.
        assert all(RESULTS_KEY not in frame.results for frame in frames)
        registry.stop_active()
