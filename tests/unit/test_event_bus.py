"""Unit tests for visionplay.core.event_bus and visionplay.core.events."""

import pytest

from visionplay.core.event_bus import EventBus
from visionplay.core.events import (
    Event,
    FrameReadyEvent,
    GameStartEvent,
    GameStopEvent,
    ShutdownEvent,
)


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


class TestSingleSubscriber:
    def test_handler_receives_published_event(self, bus: EventBus) -> None:
        received: list[GameStartEvent] = []
        bus.subscribe(GameStartEvent, received.append)
        event = GameStartEvent(app_id="air_hockey")
        bus.publish(event)
        assert received == [event]

    def test_handler_not_called_for_other_event_types(self, bus: EventBus) -> None:
        received: list[GameStartEvent] = []
        bus.subscribe(GameStartEvent, received.append)
        bus.publish(GameStopEvent(app_id="air_hockey"))
        bus.publish(ShutdownEvent())
        assert received == []

    def test_duplicate_subscribe_delivers_once(self, bus: EventBus) -> None:
        received: list[ShutdownEvent] = []
        bus.subscribe(ShutdownEvent, received.append)
        bus.subscribe(ShutdownEvent, received.append)
        bus.publish(ShutdownEvent())
        assert len(received) == 1


class TestMultipleSubscribers:
    def test_all_subscribers_receive_event(self, bus: EventBus) -> None:
        first: list[FrameReadyEvent] = []
        second: list[FrameReadyEvent] = []
        bus.subscribe(FrameReadyEvent, first.append)
        bus.subscribe(FrameReadyEvent, second.append)
        event = FrameReadyEvent(frame_index=7)
        bus.publish(event)
        assert first == [event]
        assert second == [event]

    def test_delivery_in_subscription_order(self, bus: EventBus) -> None:
        order: list[str] = []
        bus.subscribe(ShutdownEvent, lambda e: order.append("first"))
        bus.subscribe(ShutdownEvent, lambda e: order.append("second"))
        bus.publish(ShutdownEvent())
        assert order == ["first", "second"]

    def test_independent_event_types(self, bus: EventBus) -> None:
        starts: list[GameStartEvent] = []
        stops: list[GameStopEvent] = []
        bus.subscribe(GameStartEvent, starts.append)
        bus.subscribe(GameStopEvent, stops.append)
        bus.publish(GameStartEvent(app_id="a"))
        bus.publish(GameStopEvent(app_id="a", reason="error"))
        assert len(starts) == 1
        assert len(stops) == 1
        assert stops[0].reason == "error"


class TestUnsubscribe:
    def test_unsubscribed_handler_no_longer_called(self, bus: EventBus) -> None:
        received: list[ShutdownEvent] = []
        bus.subscribe(ShutdownEvent, received.append)
        assert bus.unsubscribe(ShutdownEvent, received.append) is True
        bus.publish(ShutdownEvent())
        assert received == []

    def test_unsubscribe_unknown_handler_is_safe(self, bus: EventBus) -> None:
        assert bus.unsubscribe(ShutdownEvent, lambda e: None) is False

    def test_unsubscribe_twice_is_safe(self, bus: EventBus) -> None:
        handler: list[ShutdownEvent] = []
        bus.subscribe(ShutdownEvent, handler.append)
        assert bus.unsubscribe(ShutdownEvent, handler.append) is True
        assert bus.unsubscribe(ShutdownEvent, handler.append) is False

    def test_unsubscribe_leaves_other_handlers(self, bus: EventBus) -> None:
        kept: list[ShutdownEvent] = []
        removed: list[ShutdownEvent] = []
        bus.subscribe(ShutdownEvent, kept.append)
        bus.subscribe(ShutdownEvent, removed.append)
        bus.unsubscribe(ShutdownEvent, removed.append)
        bus.publish(ShutdownEvent())
        assert len(kept) == 1
        assert removed == []

    def test_unsubscribe_during_publish_affects_next_publish_only(self, bus: EventBus) -> None:
        calls: list[str] = []

        def self_removing(event: ShutdownEvent) -> None:
            calls.append("called")
            bus.unsubscribe(ShutdownEvent, self_removing)

        bus.subscribe(ShutdownEvent, self_removing)
        bus.publish(ShutdownEvent())
        bus.publish(ShutdownEvent())
        assert calls == ["called"]


class TestNoSubscribers:
    def test_publish_with_no_subscribers_is_noop(self, bus: EventBus) -> None:
        bus.publish(ShutdownEvent())  # must not raise

    def test_publish_after_all_unsubscribed(self, bus: EventBus) -> None:
        handler: list[ShutdownEvent] = []
        bus.subscribe(ShutdownEvent, handler.append)
        bus.unsubscribe(ShutdownEvent, handler.append)
        bus.publish(ShutdownEvent())  # must not raise


class TestClear:
    def test_clear_removes_all_subscriptions(self, bus: EventBus) -> None:
        starts: list[GameStartEvent] = []
        stops: list[GameStopEvent] = []
        bus.subscribe(GameStartEvent, starts.append)
        bus.subscribe(GameStopEvent, stops.append)
        bus.clear()
        bus.publish(GameStartEvent(app_id="a"))
        bus.publish(GameStopEvent(app_id="a"))
        assert starts == []
        assert stops == []

    def test_subscriber_count(self, bus: EventBus) -> None:
        assert bus.subscriber_count(ShutdownEvent) == 0
        bus.subscribe(ShutdownEvent, lambda e: None)
        assert bus.subscriber_count(ShutdownEvent) == 1
        bus.clear()
        assert bus.subscriber_count(ShutdownEvent) == 0


class TestEventTypes:
    def test_events_are_immutable(self) -> None:
        event = GameStartEvent(app_id="a")
        with pytest.raises(AttributeError):
            event.app_id = "b"  # type: ignore[misc]

    def test_timestamp_autofilled(self) -> None:
        assert FrameReadyEvent(frame_index=0).timestamp > 0

    def test_all_events_derive_from_event(self) -> None:
        for cls in (FrameReadyEvent, GameStartEvent, GameStopEvent, ShutdownEvent):
            assert issubclass(cls, Event)

    def test_game_stop_default_reason(self) -> None:
        assert GameStopEvent(app_id="a").reason == "user"
