"""Unit tests for visionplay.vision.inference.inference_runner."""

from __future__ import annotations

from typing import Any

import numpy as np

from visionplay.core.event_bus import EventBus
from visionplay.core.events import GameStartEvent, GameStopEvent
from visionplay.vision.inference.backend_base import InferenceBackend, InferenceError
from visionplay.vision.inference.backend_manager import BackendManager, BackendRegistration
from visionplay.vision.inference.device import DeviceConfig
from visionplay.vision.inference.inference_runner import FrameInferenceRunner
from visionplay.vision.pipeline.frame_types import Frame


class FakeBackend(InferenceBackend):
    """Backend that returns a fixed result, or raises on infer if configured."""

    def __init__(self, name: str, result: Any = "RESULT", *, fail_infer: bool = False) -> None:
        super().__init__()
        self._name = name
        self._result = result
        self._fail_infer = fail_infer
        self._loaded = False

    @property
    def name(self) -> str:
        return self._name

    def load(self) -> None:
        self._loaded = True

    def infer(self, frame: Frame) -> Any:
        if self._fail_infer:
            raise InferenceError(f"{self._name} boom")
        return self._result

    def unload(self) -> None:
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded


def make_frame() -> Frame:
    return Frame.from_image(frame_id=0, timestamp=0.0, image=np.zeros((2, 2, 3), dtype=np.uint8))


def register(manager: BackendManager, backend: FakeBackend) -> None:
    manager.register(
        BackendRegistration(
            name=backend.name,
            factory=lambda device: backend,
            probe=lambda: True,
        )
    )


def build(
    backends_by_app: dict[str, tuple[str, ...]],
    *backends: FakeBackend,
) -> tuple[FrameInferenceRunner, BackendManager, EventBus]:
    manager = BackendManager(DeviceConfig.cpu())
    for backend in backends:
        register(manager, backend)
    event_bus = EventBus()
    runner = FrameInferenceRunner(
        manager,
        backends_for=lambda app_id: backends_by_app.get(app_id, ()),
        event_bus=event_bus,
    )
    return runner, manager, event_bus


class TestActivation:
    def test_no_active_app_is_passthrough(self) -> None:
        runner, _, _ = build({})
        frame = make_frame()
        assert runner.run(frame) is frame
        assert frame.results == {}
        assert runner.active_backends == ()

    def test_game_start_loads_and_activates_declared_backends(self) -> None:
        backend = FakeBackend("mediapipe.hands")
        runner, manager, bus = build({"app": ("mediapipe.hands",)}, backend)
        bus.publish(GameStartEvent(app_id="app"))
        assert runner.active_backends == ("mediapipe.hands",)
        assert manager.is_loaded("mediapipe.hands")

    def test_run_populates_frame_results(self) -> None:
        backend = FakeBackend("mediapipe.hands", result="HANDS")
        runner, _, bus = build({"app": ("mediapipe.hands",)}, backend)
        bus.publish(GameStartEvent(app_id="app"))
        frame = make_frame()
        runner.run(frame)
        assert frame.results == {"mediapipe.hands": "HANDS"}

    def test_multiple_backends_all_populate(self) -> None:
        a = FakeBackend("mediapipe.hands", result="H")
        b = FakeBackend("onnx.yolo", result="Y")
        runner, _, bus = build({"app": ("mediapipe.hands", "onnx.yolo")}, a, b)
        bus.publish(GameStartEvent(app_id="app"))
        frame = make_frame()
        runner.run(frame)
        assert frame.results == {"mediapipe.hands": "H", "onnx.yolo": "Y"}

    def test_game_stop_clears_active_backends(self) -> None:
        backend = FakeBackend("mediapipe.hands")
        runner, _, bus = build({"app": ("mediapipe.hands",)}, backend)
        bus.publish(GameStartEvent(app_id="app"))
        bus.publish(GameStopEvent(app_id="app"))
        assert runner.active_backends == ()
        frame = make_frame()
        runner.run(frame)
        assert frame.results == {}

    def test_switching_apps_swaps_active_backends(self) -> None:
        a = FakeBackend("mediapipe.hands", result="H")
        b = FakeBackend("onnx.yolo", result="Y")
        runner, _, bus = build({"app_a": ("mediapipe.hands",), "app_b": ("onnx.yolo",)}, a, b)
        bus.publish(GameStartEvent(app_id="app_a"))
        bus.publish(GameStopEvent(app_id="app_a"))
        bus.publish(GameStartEvent(app_id="app_b"))
        assert runner.active_backends == ("onnx.yolo",)
        frame = make_frame()
        runner.run(frame)
        assert frame.results == {"onnx.yolo": "Y"}


class TestWarmCache:
    def test_backend_loaded_once_across_frames(self) -> None:
        backend = FakeBackend("mediapipe.hands")
        runner, manager, bus = build({"app": ("mediapipe.hands",)}, backend)
        bus.publish(GameStartEvent(app_id="app"))
        runner.run(make_frame())
        runner.run(make_frame())
        # acquire returns the warm instance; only the single start-time load ran.
        assert manager.acquire("mediapipe.hands") is backend
        assert backend.is_loaded()


class TestErrorContainment:
    def test_missing_backend_is_skipped_not_fatal(self) -> None:
        # App declares a backend that was never registered.
        runner, _, bus = build({"app": ("does.not.exist",)})
        bus.publish(GameStartEvent(app_id="app"))
        assert runner.active_backends == ()  # nothing loaded, but no raise
        frame = make_frame()
        assert runner.run(frame) is frame
        assert frame.results == {}

    def test_infer_failure_is_contained(self) -> None:
        good = FakeBackend("good", result="OK")
        bad = FakeBackend("bad", fail_infer=True)
        runner, _, bus = build({"app": ("bad", "good")}, bad, good)
        bus.publish(GameStartEvent(app_id="app"))
        frame = make_frame()
        runner.run(frame)  # must not raise
        assert frame.results == {"good": "OK"}  # failed backend absent, good present

    def test_bad_backends_lookup_does_not_break_start(self) -> None:
        manager = BackendManager()
        bus = EventBus()

        def boom(app_id: str) -> tuple[str, ...]:
            raise KeyError(app_id)

        runner = FrameInferenceRunner(manager, backends_for=boom, event_bus=bus)
        bus.publish(GameStartEvent(app_id="app"))  # must not raise
        assert runner.active_backends == ()


class TestShutdown:
    def test_shutdown_releases_backends_and_unsubscribes(self) -> None:
        backend = FakeBackend("mediapipe.hands")
        runner, manager, bus = build({"app": ("mediapipe.hands",)}, backend)
        bus.publish(GameStartEvent(app_id="app"))
        assert manager.is_loaded("mediapipe.hands")

        runner.shutdown()

        assert not manager.is_loaded("mediapipe.hands")
        assert not backend.is_loaded()
        assert bus.subscriber_count(GameStartEvent) == 0
        assert bus.subscriber_count(GameStopEvent) == 0

    def test_shutdown_is_idempotent(self) -> None:
        backend = FakeBackend("mediapipe.hands")
        runner, _, bus = build({"app": ("mediapipe.hands",)}, backend)
        bus.publish(GameStartEvent(app_id="app"))
        runner.shutdown()
        runner.shutdown()
        assert runner.active_backends == ()
