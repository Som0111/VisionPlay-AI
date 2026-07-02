"""Minimal synchronous in-process pub/sub for platform events.

Exists only because ``core/`` and ``vision/`` are Qt-free and therefore
cannot use signals/slots (``docs/architecture.md`` §4). Scope is
deliberately narrow: platform-level events between the Qt-free layers.
UI-to-UI communication uses native Qt signals, never this bus, and it must
not grow into a general message broker.

Semantics:

- **Synchronous** — :meth:`EventBus.publish` calls every handler inline,
  in subscription order, on the caller's thread, and returns when the last
  one has. There is no queueing, no threading, no async (per M0.3 scope);
  thread-safety, if ever needed, is a caller concern.
- **Typed** — handlers subscribe to a concrete event class and receive
  only instances of exactly that class (no base-class dispatch).
- **Tolerant** — publishing an event nobody subscribed to is a no-op, and
  unsubscribing a handler that isn't registered is safe (returns ``False``
  rather than raising), so teardown code never has to track registration
  state defensively.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar, cast

from visionplay.core.events import Event

__all__ = ["EventBus"]

E = TypeVar("E", bound=Event)

#: Internal storage type: handlers erased to the base ``Event`` signature.
_Handler = Callable[[Event], None]


class EventBus:
    """Synchronous publish/subscribe hub keyed by concrete event class.

    Not a singleton: the application composes one bus at startup and passes
    it to the components that need it, same as ``Config`` and ``AppPaths``.

    Example::

        bus = EventBus()
        bus.subscribe(GameStartEvent, lambda e: log.info("started %s", e.app_id))
        bus.publish(GameStartEvent(app_id="air_hockey"))
    """

    def __init__(self) -> None:
        """Create an empty bus with no subscriptions."""
        self._subscribers: dict[type[Event], list[_Handler]] = {}

    def subscribe(self, event_type: type[E], handler: Callable[[E], None]) -> None:
        """Register ``handler`` to be called for every published ``event_type``.

        The same handler may be registered once per event type; duplicate
        registrations for the same type are ignored (a handler is never
        called twice for one event).

        Args:
            event_type: Concrete :class:`~visionplay.core.events.Event`
                subclass to listen for.
            handler: Callable invoked with the event instance. Runs on the
                publisher's thread; must be fast and must not raise —
                exceptions propagate to the publisher.
        """
        handlers = self._subscribers.setdefault(event_type, [])
        erased = cast(_Handler, handler)
        if erased not in handlers:
            handlers.append(erased)

    def unsubscribe(self, event_type: type[E], handler: Callable[[E], None]) -> bool:
        """Remove a previously registered handler.

        Safe to call with a handler that was never registered (or already
        removed) — that is reported via the return value, not an exception.

        Args:
            event_type: Event class the handler was subscribed to.
            handler: The exact callable passed to :meth:`subscribe`.

        Returns:
            ``True`` if the handler was found and removed, ``False`` otherwise.
        """
        handlers = self._subscribers.get(event_type)
        if handlers is None:
            return False
        try:
            handlers.remove(cast(_Handler, handler))
        except ValueError:
            return False
        if not handlers:
            del self._subscribers[event_type]
        return True

    def publish(self, event: Event) -> None:
        """Deliver ``event`` synchronously to all subscribers of its exact type.

        Handlers run in subscription order on the calling thread. An event
        with no subscribers is silently ignored. Handlers are snapshotted
        before dispatch, so a handler that subscribes/unsubscribes during
        delivery affects future publishes, not the current one.

        Args:
            event: The event instance to deliver.
        """
        for handler in tuple(self._subscribers.get(type(event), ())):
            handler(event)

    def clear(self) -> None:
        """Drop every subscription (all event types). Used at shutdown/in tests."""
        self._subscribers.clear()

    def subscriber_count(self, event_type: type[Event]) -> int:
        """Return how many handlers are registered for ``event_type``."""
        return len(self._subscribers.get(event_type, ()))
