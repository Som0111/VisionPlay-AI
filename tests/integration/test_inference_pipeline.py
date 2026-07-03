"""Integration tests: the full capture → backend → on_frame → publish path (M2.4).

Drives synthetic frames through the real ``FramePipeline`` with a **real**
``MediaPipeBackend`` instance (no fixture mocks): a real ``BackendManager``
resolves the hand-tracking demo app's ``required_backends``, the
``FrameInferenceRunner`` runs inference on the worker thread before the
plugin's ``on_frame``, and results reach the published frames on the bus —
the composition ``app.py`` wires, exercised headless (no Qt, no camera).

Uses the session-downloaded hand-landmarker model (see ``tests/conftest.py``);
skips when the model can't be fetched, so offline runs still pass.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from visionplay.core.event_bus import EventBus
from visionplay.core.plugin_registry import PluginRegistry
from visionplay.vision.camera.camera_source import CameraSource
from visionplay.vision.inference.backend_defaults import register_mediapipe_hands_backend
from visionplay.vision.inference.backend_manager import BackendManager
from visionplay.vision.inference.inference_runner import FrameInferenceRunner
from visionplay.vision.inference.model_registry import ModelRegistry
from visionplay.vision.inference.results import HandLandmarkResult
from visionplay.vision.pipeline.frame_bus import FrameBus, FramePipeline
from visionplay.vision.pipeline.frame_types import Frame

APP_ID = "hand_tracking_demo"
BACKEND_NAME = "mediapipe.hands"


class FakeSource(CameraSource):
    """Synthetic source: N blank BGR frames then end of stream."""

    def __init__(self, total_frames: int = 3) -> None:
        self._total = total_frames
        self._next_id = 0
        self._opened = False

    def open(self) -> None:
        self._opened = True

    def read(self) -> Frame | None:
        if self._next_id >= self._total:
            return None
        image = np.zeros((90, 120, 3), dtype=np.uint8)
        frame = Frame.from_image(
            frame_id=self._next_id, timestamp=float(self._next_id), image=image
        )
        self._next_id += 1
        return frame

    def release(self) -> None:
        self._opened = False

    def is_open(self) -> bool:
        return self._opened


def drain(bus: FrameBus, timeout: float = 30.0) -> list[Frame]:
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


@pytest.fixture
def platform(hand_landmarker_model: Path):
    """Registry + manager + runner wired the way ``app.py`` composes them."""
    event_bus = EventBus()
    registry = PluginRegistry(event_bus=event_bus)  # real visionplay.apps tree
    registry.discover()
    manager = BackendManager()
    # Cache-only model registry over the session-downloaded model: the real
    # backend loads real bytes, but the test never touches the network itself.
    register_mediapipe_hands_backend(manager, ModelRegistry(hand_landmarker_model.parent))
    runner = FrameInferenceRunner(
        manager,
        backends_for=lambda app_id: tuple(registry.manifests[app_id].required_backends),
        event_bus=event_bus,
    )
    yield registry, manager, runner
    registry.stop_active()
    runner.shutdown()


@pytest.mark.integration
class TestFullInferencePath:
    def test_results_populated_before_on_frame_and_published(self, platform) -> None:
        registry, manager, runner = platform
        registry.start(APP_ID)

        # Snapshot frame.results between the backend stage and on_frame — the
        # same seam app.py composes — to prove the §4 ordering with a real backend.
        results_at_dispatch: list[bool] = []

        def process(frame: Frame) -> Frame:
            frame = runner.run(frame)
            results_at_dispatch.append(BACKEND_NAME in frame.results)
            return registry.process_frame(frame)

        pipeline = FramePipeline(FakeSource(total_frames=3), FrameBus(capacity=16))
        pipeline.set_frame_processor(process)
        with pipeline:
            frames = drain(pipeline.bus)

        assert pipeline.error is None
        assert len(frames) == 3
        assert results_at_dispatch == [True, True, True]
        for frame in frames:
            result = frame.results[BACKEND_NAME]
            assert isinstance(result, HandLandmarkResult)
            assert result.is_empty  # blank synthetic frames: no hand, no error

    def test_backend_stays_warm_across_frames_and_app_restart(self, platform) -> None:
        registry, manager, runner = platform
        registry.start(APP_ID)
        backend = manager.acquire(BACKEND_NAME)
        assert backend.is_loaded()

        pipeline = FramePipeline(FakeSource(total_frames=4), FrameBus(capacity=16))
        pipeline.set_frame_processor(lambda frame: registry.process_frame(runner.run(frame)))
        with pipeline:
            drain(pipeline.bus)

        # No per-frame reload: still the same warm instance after N frames...
        assert manager.acquire(BACKEND_NAME) is backend
        assert backend.is_loaded()

        # ...and across a stop/restart of the app that needs it.
        registry.stop_active()
        assert runner.active_backends == ()
        assert backend.is_loaded()  # kept warm by the manager
        registry.start(APP_ID)
        assert manager.acquire(BACKEND_NAME) is backend

    def test_mid_stream_infer_failure_is_contained(self, platform) -> None:
        registry, manager, runner = platform
        registry.start(APP_ID)
        # Sabotage the real backend after load: every infer() now raises a
        # real InferenceError ("not loaded"), the M2.4 mid-stream failure case.
        manager.acquire(BACKEND_NAME).unload()

        pipeline = FramePipeline(FakeSource(total_frames=3), FrameBus(capacity=16))
        pipeline.set_frame_processor(lambda frame: registry.process_frame(runner.run(frame)))
        with pipeline:
            frames = drain(pipeline.bus)

        # The pipeline survives and keeps publishing; the failing backend's
        # result is simply absent — the case plugins already read defensively.
        assert pipeline.error is None
        assert len(frames) == 3
        assert all(BACKEND_NAME not in frame.results for frame in frames)
        assert registry.active_app_id == APP_ID
