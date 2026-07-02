"""Platform event types published over the :class:`~visionplay.core.event_bus.EventBus`.

Events are small, immutable dataclasses. They carry identifiers and
metadata only — never heavyweight payloads like frame arrays: frames flow
through the vision pipeline's ``frame_bus`` (M0.5), not the event bus, and
``core/`` must not depend on ``vision/`` types (see ``docs/architecture.md``
§1 and §4). ``FrameReadyEvent`` therefore references a frame by sequence
number and timestamp, not by the ndarray itself.

All events derive from :class:`Event`, which is what the bus's subscribe
signature is typed against. Subscription is by concrete event class —
subscribing to a base class does not deliver subclass events (no hierarchy
dispatch; keep it simple until a real need appears).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time

__all__ = [
    "Event",
    "FrameReadyEvent",
    "GameStartEvent",
    "GameStopEvent",
    "ShutdownEvent",
]


@dataclass(frozen=True, slots=True)
class Event:
    """Base class for all platform events.

    Attributes:
        timestamp: Unix time the event was created (seconds). Auto-filled;
            useful for logging/latency diagnostics.
    """

    timestamp: float = field(default_factory=time, kw_only=True)


@dataclass(frozen=True, slots=True)
class FrameReadyEvent(Event):
    """A new camera frame has been captured by the pipeline.

    Metadata only — the frame data itself travels through the vision
    pipeline's frame bus, never through the event bus.

    Attributes:
        frame_index: Monotonic sequence number assigned by the capture source.
    """

    frame_index: int


@dataclass(frozen=True, slots=True)
class GameStartEvent(Event):
    """An app/game plugin has started (its ``on_start`` completed).

    Attributes:
        app_id: The plugin's manifest ``id``.
    """

    app_id: str


@dataclass(frozen=True, slots=True)
class GameStopEvent(Event):
    """An app/game plugin has stopped.

    Attributes:
        app_id: The plugin's manifest ``id``.
        reason: Why it stopped — e.g. ``"user"`` (normal exit) or
            ``"error"`` (registry stopped it after repeated failures).
    """

    app_id: str
    reason: str = "user"


@dataclass(frozen=True, slots=True)
class ShutdownEvent(Event):
    """The application is shutting down; subscribers should release resources."""
