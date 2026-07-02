"""Unit tests for visionplay.core.plugin_registry."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pytest
from tests.fixtures.plugin_apps_fixture._support import RecordingPlugin

from visionplay.core.event_bus import EventBus
from visionplay.core.events import GameStartEvent, GameStopEvent
from visionplay.core.plugin_registry import (
    DEFAULT_MAX_CONSECUTIVE_FRAME_FAILURES,
    PluginRegistry,
)
from visionplay.vision.pipeline.frame_types import Frame

FIXTURE_APPS_PACKAGE = "tests.fixtures.plugin_apps_fixture"


def make_frame(frame_id: int = 0) -> Frame:
    return Frame.from_image(
        frame_id=frame_id,
        timestamp=float(frame_id),
        image=np.zeros((4, 4, 3), dtype=np.uint8),
    )


@contextmanager
def _capture_visionplay_logs(caplog: pytest.LogCaptureFixture) -> Iterator[None]:
    """Capture ERROR+ records from the ``visionplay`` logger tree.

    ``visionplay.core.logging_setup.setup_logging`` sets ``propagate = False``
    on the ``"visionplay"`` logger once configured (elsewhere in the test
    session), which would otherwise stop caplog's root-attached handler from
    ever seeing these records. Attaching caplog's handler directly to the
    ``"visionplay"`` logger sidesteps that: propagation is only blocked
    *above* it, not to its own handlers.
    """
    logger = logging.getLogger("visionplay")
    previous_level = logger.level
    logger.addHandler(caplog.handler)
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.removeHandler(caplog.handler)
        logger.setLevel(previous_level)


def make_registry(**kwargs: object) -> tuple[PluginRegistry, EventBus]:
    bus = EventBus()
    registry = PluginRegistry(event_bus=bus, apps_package=FIXTURE_APPS_PACKAGE, **kwargs)  # type: ignore[arg-type]
    return registry, bus


class TestDiscovery:
    def test_discovers_valid_apps(self) -> None:
        registry, _ = make_registry()
        registry.discover()
        assert "valid_app" in registry.manifests
        assert "failing_load_app" not in registry.manifests  # on_load raises, excluded
        assert "failing_frame_app" in registry.manifests

    def test_skips_underscore_prefixed_package(self) -> None:
        registry, _ = make_registry()
        registry.discover()
        assert "skipped_app" not in registry.manifests
        assert all(not app_id.startswith("_") for app_id in registry.manifests)

    def test_on_load_called_exactly_once_per_loaded_app(self) -> None:
        registry, _ = make_registry()
        registry.discover()
        plugin = registry._apps["valid_app"].plugin
        assert isinstance(plugin, RecordingPlugin)
        assert plugin.calls.count("on_load") == 1

    def test_discover_is_idempotent(self) -> None:
        registry, _ = make_registry()
        registry.discover()
        registry.discover()  # re-running discovery must not raise or duplicate entries
        assert list(registry.manifests).count("valid_app") == 1


class TestUnsupportedApiVersion:
    def test_unsupported_api_version_is_excluded(self) -> None:
        registry, _ = make_registry()
        registry.discover()
        assert "unsupported_api_app" not in registry.manifests

    def test_unsupported_api_version_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        registry, _ = make_registry()
        with _capture_visionplay_logs(caplog):
            registry.discover()
        assert any("unsupported_api_app" in record.getMessage() for record in caplog.records)

    def test_unsupported_api_version_does_not_prevent_other_apps(self) -> None:
        registry, _ = make_registry()
        registry.discover()
        assert "valid_app" in registry.manifests


class TestOnLoadFailureContainment:
    def test_failing_on_load_excludes_app_but_not_others(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        registry, _ = make_registry()
        with _capture_visionplay_logs(caplog):
            registry.discover()
        assert "failing_load_app" not in registry.manifests
        assert "valid_app" in registry.manifests
        assert any("failing_load_app" in record.getMessage() for record in caplog.records)

    def test_discover_does_not_raise(self) -> None:
        registry, _ = make_registry()
        registry.discover()  # must not raise despite failing_load_app/unsupported_api_app


class TestSingleActiveApp:
    def test_start_activates_app(self) -> None:
        registry, _ = make_registry()
        registry.discover()
        registry.start("valid_app")
        assert registry.active_app_id == "valid_app"

    def test_starting_second_app_stops_first(self) -> None:
        registry, _ = make_registry()
        registry.discover()
        registry.start("valid_app")
        registry.start("failing_frame_app")
        assert registry.active_app_id == "failing_frame_app"

    def test_stopped_app_on_stop_was_called(self) -> None:
        registry, _ = make_registry()
        registry.discover()
        registry.start("valid_app")
        plugin = registry._apps["valid_app"].plugin
        assert isinstance(plugin, RecordingPlugin)
        registry.start("failing_frame_app")
        assert "on_stop" in plugin.calls

    def test_start_unknown_app_raises_key_error(self) -> None:
        registry, _ = make_registry()
        registry.discover()
        with pytest.raises(KeyError):
            registry.start("does_not_exist")

    def test_stop_active_with_no_active_app_is_a_no_op(self) -> None:
        registry, _ = make_registry()
        registry.discover()
        registry.stop_active()  # must not raise
        assert registry.active_app_id is None


class TestStartStopEventOrdering:
    def test_start_publishes_game_start_event(self) -> None:
        registry, bus = make_registry()
        registry.discover()
        received: list[GameStartEvent] = []
        bus.subscribe(GameStartEvent, received.append)
        registry.start("valid_app")
        assert [e.app_id for e in received] == ["valid_app"]

    def test_switching_apps_publishes_stop_then_start(self) -> None:
        registry, bus = make_registry()
        registry.discover()
        events: list[str] = []
        bus.subscribe(GameStartEvent, lambda e: events.append(f"start:{e.app_id}"))
        bus.subscribe(GameStopEvent, lambda e: events.append(f"stop:{e.app_id}:{e.reason}"))

        registry.start("valid_app")
        registry.start("failing_frame_app")

        assert events == [
            "start:valid_app",
            "stop:valid_app:user",
            "start:failing_frame_app",
        ]

    def test_stop_active_publishes_game_stop_event_with_reason(self) -> None:
        registry, bus = make_registry()
        registry.discover()
        received: list[GameStopEvent] = []
        bus.subscribe(GameStopEvent, received.append)
        registry.start("valid_app")
        registry.stop_active(reason="user")
        assert received[0].app_id == "valid_app"
        assert received[0].reason == "user"


class TestOnFrameFailureThreshold:
    def test_on_frame_exceptions_are_contained(self) -> None:
        registry, _ = make_registry()
        registry.discover()
        registry.start("failing_frame_app")
        for _ in range(DEFAULT_MAX_CONSECUTIVE_FRAME_FAILURES - 1):
            result = registry.process_frame(make_frame())
            assert isinstance(result, Frame)  # never raises out of process_frame
        assert registry.active_app_id == "failing_frame_app"  # threshold not yet hit

    def test_threshold_stops_app_and_publishes_error_stop_event(self) -> None:
        registry, bus = make_registry()
        registry.discover()
        received: list[GameStopEvent] = []
        bus.subscribe(GameStopEvent, received.append)

        registry.start("failing_frame_app")
        for _ in range(DEFAULT_MAX_CONSECUTIVE_FRAME_FAILURES):
            registry.process_frame(make_frame())

        assert registry.active_app_id is None
        assert received[-1].app_id == "failing_frame_app"
        assert received[-1].reason == "error"

    def test_custom_threshold_is_honored(self) -> None:
        registry, _ = make_registry(max_consecutive_frame_failures=2)
        registry.discover()
        registry.start("failing_frame_app")
        registry.process_frame(make_frame())
        assert registry.active_app_id == "failing_frame_app"
        registry.process_frame(make_frame())
        assert registry.active_app_id is None

    def test_success_resets_failure_counter(self) -> None:
        registry, _ = make_registry(max_consecutive_frame_failures=2)
        registry.discover()
        registry.start("valid_app")
        plugin = registry._apps["valid_app"].plugin
        assert isinstance(plugin, RecordingPlugin)
        plugin.fail_on_frame = True
        registry.process_frame(make_frame())
        plugin.fail_on_frame = False
        registry.process_frame(make_frame())  # success resets the streak
        plugin.fail_on_frame = True
        registry.process_frame(make_frame())
        assert registry.active_app_id == "valid_app"  # only 1 consecutive failure again

    def test_no_active_app_process_frame_is_passthrough(self) -> None:
        registry, _ = make_registry()
        registry.discover()
        frame = make_frame(5)
        result = registry.process_frame(frame)
        assert result is frame


class TestUnloadAll:
    def test_unload_all_stops_active_app_and_calls_on_unload_on_all(self) -> None:
        registry, _ = make_registry()
        registry.discover()
        registry.start("valid_app")
        valid_plugin = registry._apps["valid_app"].plugin
        frame_plugin = registry._apps["failing_frame_app"].plugin
        assert isinstance(valid_plugin, RecordingPlugin)
        assert isinstance(frame_plugin, RecordingPlugin)

        registry.unload_all()

        assert registry.active_app_id is None
        assert "on_stop" in valid_plugin.calls
        assert "on_unload" in valid_plugin.calls
        assert "on_unload" in frame_plugin.calls

    def test_unload_all_does_not_raise_when_no_app_active(self) -> None:
        registry, _ = make_registry()
        registry.discover()
        registry.unload_all()
