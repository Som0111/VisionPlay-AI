"""Runs the active app's inference backends over each frame (M2.3A).

This is the ``backends`` stage of the per-frame data flow
(``docs/architecture.md`` §4): ``capture → backends → on_frame → publish``.
It sits between the camera and the plugin's ``on_frame`` — the pipeline calls
:meth:`FrameInferenceRunner.run` on the worker thread just before dispatching
to the active plugin, so results are already attached to ``frame.results``
when ``on_frame`` reads them.

Which backends run is driven entirely by the active app: the runner listens
for :class:`~visionplay.core.events.GameStartEvent` /
:class:`~visionplay.core.events.GameStopEvent` on the event bus (the Qt-free
platform-event channel, per §4) and, on start, resolves the app's
``required_backends`` and acquires them from the
:class:`~visionplay.vision.inference.backend_manager.BackendManager` (which
loads and warm-caches them). Plugins never touch the manager themselves — the
pipeline owns backend lifecycles and delivers results on the frame (§3/§4).

Lifecycle (v1): backends are loaded on app start and kept **warm** by the
manager across app switches and every ``on_frame`` — they are released only at
:meth:`shutdown`. Unloading a backend the moment no active app needs it is a
later optimization; for v1's small backend set, keeping them warm is simpler
and matches the "shared/warm across app switches" intent of §4.

Error containment: a backend that fails to load (at start) or to infer (per
frame) never crashes the pipeline. A load failure drops that backend for the
session (the plugin sees its result absent — the same defensive case it
already handles); an infer failure is logged once per backend and skipped for
that frame. :meth:`run` therefore satisfies the ``FrameProcessor`` no-raise
contract, and the event handlers never raise back into the publisher.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from visionplay.core.event_bus import EventBus
from visionplay.core.events import GameStartEvent, GameStopEvent
from visionplay.vision.inference.backend_manager import BackendManager
from visionplay.vision.pipeline.frame_types import Frame

__all__ = ["BackendsForApp", "FrameInferenceRunner"]

logger = logging.getLogger(__name__)

#: Resolves an app id to the backend names it declares in its manifest's
#: ``required_backends``. Injected so the runner (in ``vision/``) never needs
#: the plugin registry (in ``core/``) — the app bootstrap supplies the lookup.
BackendsForApp = Callable[[str], Sequence[str]]


class FrameInferenceRunner:
    """Runs the active app's declared backends over each captured frame.

    Construct one with the process-wide :class:`BackendManager`, a lookup from
    app id to its ``required_backends``, and the platform event bus. It
    subscribes to game start/stop immediately; call :meth:`shutdown` to
    unsubscribe and release backends.
    """

    def __init__(
        self,
        manager: BackendManager,
        backends_for: BackendsForApp,
        event_bus: EventBus,
    ) -> None:
        """Wire the runner and subscribe to app start/stop events.

        Args:
            manager: Owns and warm-caches the backend instances.
            backends_for: Maps an app id to its declared backend names.
            event_bus: Platform bus carrying ``GameStartEvent``/
                ``GameStopEvent`` (published by the plugin registry).
        """
        self._manager = manager
        self._backends_for = backends_for
        self._event_bus = event_bus
        #: Backends to run each frame — the loaded subset of the active app's
        #: needs. A tuple so the worker thread reads a consistent snapshot.
        self._active: tuple[str, ...] = ()
        #: Backends whose per-frame inference has already been logged as
        #: failing this session, to avoid a warning every frame.
        self._logged_failures: set[str] = set()
        event_bus.subscribe(GameStartEvent, self._on_game_start)
        event_bus.subscribe(GameStopEvent, self._on_game_stop)

    @property
    def active_backends(self) -> tuple[str, ...]:
        """Backend names currently run per frame (the loaded active subset)."""
        return self._active

    def run(self, frame: Frame) -> Frame:
        """Infer every active backend and attach results to ``frame.results``.

        The pipeline's per-frame hook (a ``FrameProcessor``): runs on the
        worker thread, before the active plugin's ``on_frame``. Never raises —
        a backend that fails is logged (once per session) and skipped, leaving
        its result absent for this frame.

        Args:
            frame: The captured frame; mutated in place via ``frame.results``.

        Returns:
            The same frame, with each successful backend's output stored under
            the backend's name.
        """
        for name in self._active:
            try:
                frame.results[name] = self._manager.acquire(name).infer(frame)
            except Exception as exc:  # containment: a backend must never crash the pipeline
                if name not in self._logged_failures:
                    logger.warning("Backend %r inference failed; skipping: %s", name, exc)
                    self._logged_failures.add(name)
        return frame

    def shutdown(self) -> None:
        """Unsubscribe from the bus and release all backends. Idempotent."""
        self._event_bus.unsubscribe(GameStartEvent, self._on_game_start)
        self._event_bus.unsubscribe(GameStopEvent, self._on_game_stop)
        self._active = ()
        self._manager.release_all()

    # -- Event handlers (run on the publisher's thread; must not raise) -------

    def _on_game_start(self, event: GameStartEvent) -> None:
        """Load the started app's backends and make them the active set."""
        try:
            names = tuple(self._backends_for(event.app_id))
        except Exception:  # a bad lookup must not break app start
            logger.exception("Could not resolve backends for app %r", event.app_id)
            names = ()

        loaded: list[str] = []
        for name in names:
            try:
                self._manager.acquire(name)  # constructs + loads + warm-caches
                loaded.append(name)
            except Exception as exc:  # degrade, don't fail the launch
                logger.warning(
                    "Backend %r unavailable for app %r; it will run without it: %s",
                    name,
                    event.app_id,
                    exc,
                )
        self._logged_failures = set()
        self._active = tuple(loaded)

    def _on_game_stop(self, event: GameStopEvent) -> None:
        """Stop running backends; the manager keeps them warm for reuse."""
        self._active = ()
        self._logged_failures = set()
