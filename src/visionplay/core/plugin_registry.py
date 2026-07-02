"""Plugin discovery, lifecycle guarding, and single-active-app enforcement (M1.2).

:class:`PluginRegistry` is the only thing in ``core/`` that turns the
:class:`~visionplay.core.plugin_base.AppPlugin` *contract* (M1.1) into
running apps. It owns three responsibilities, all internal-only in v1
(``docs/architecture.md`` §3):

- **Discovery** — ``pkgutil.iter_modules`` over an apps package, skipping
  underscore-prefixed packages (``_template``) exactly like the real
  ``apps/`` tree. Each discovered app package must expose a
  ``manifest.py`` module with a module-level ``MANIFEST: AppManifest`` and
  a ``plugin.py`` module with a module-level ``Plugin`` class (an
  :class:`~visionplay.core.plugin_base.AppPlugin` subclass) — this is the
  minimal discovery contract M1.2 needs; it does not prescribe
  ``processor.py``/``widget.py`` shape, which is a per-app concern.
- **Failure containment** — every call into a plugin is wrapped so a
  misbehaving app can log and stop itself, never crash the shell
  (``docs/architecture.md`` §3, §7). ``on_frame`` failures are counted
  consecutively per app; hitting the threshold stops that app rather than
  letting it keep failing silently forever.
- **Single active app** — v1 runs exactly one app at a time
  (``docs/architecture.md`` §4). Starting a new app stops whichever one is
  currently active first.

No Qt, no camera access, and no inference backends are constructed here —
the registry only calls the lifecycle methods a plugin already implements;
running backends and delivering frames is the frame pipeline's job (M1.4).
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from visionplay.core.event_bus import EventBus
from visionplay.core.events import GameStartEvent, GameStopEvent
from visionplay.core.plugin_base import CURRENT_API_VERSION, AppManifest, AppPlugin
from visionplay.vision.pipeline.frame_types import Frame

__all__ = ["DEFAULT_MAX_CONSECUTIVE_FRAME_FAILURES", "PluginRegistry"]

logger = logging.getLogger(__name__)

#: Default app package the registry scans in production. Tests point the
#: registry at a fixture package instead of importing this one.
_DEFAULT_APPS_PACKAGE = "visionplay.apps"

#: Consecutive on_frame failures an app is allowed before the registry
#: stops it. Not user-configurable via config.yaml (yet) — a constructor
#: argument is enough for M1.2's scope.
DEFAULT_MAX_CONSECUTIVE_FRAME_FAILURES: int = 5


@dataclass
class _LoadedApp:
    """Internal bookkeeping for one discovered, loaded app."""

    manifest: AppManifest
    plugin: AppPlugin
    consecutive_frame_failures: int = field(default=0)


class PluginRegistry:
    """Discovers apps, runs their lifecycle safely, and enforces one active app.

    Not a singleton: the application bootstrap owns one instance and wires
    it to the frame pipeline (M1.4) and launcher (M1.5), the same way it
    owns the :class:`~visionplay.core.event_bus.EventBus` and ``Config``.
    """

    def __init__(
        self,
        event_bus: EventBus,
        apps_package: str = _DEFAULT_APPS_PACKAGE,
        max_consecutive_frame_failures: int = DEFAULT_MAX_CONSECUTIVE_FRAME_FAILURES,
    ) -> None:
        """Create an empty registry; call :meth:`discover` to populate it.

        Args:
            event_bus: Bus that ``GameStartEvent``/``GameStopEvent`` are
                published on.
            apps_package: Dotted, importable package to scan for apps.
                Defaults to the real ``visionplay.apps`` tree; tests pass a
                fixture package instead.
            max_consecutive_frame_failures: How many consecutive
                ``on_frame`` exceptions an active app tolerates before the
                registry stops it.
        """
        self._event_bus = event_bus
        self._apps_package = apps_package
        self._max_consecutive_frame_failures = max_consecutive_frame_failures
        self._apps: dict[str, _LoadedApp] = {}
        self._active_app_id: str | None = None

    @property
    def manifests(self) -> Mapping[str, AppManifest]:
        """Manifests of every successfully loaded app, keyed by app id."""
        return {app_id: loaded.manifest for app_id, loaded in self._apps.items()}

    @property
    def active_app_id(self) -> str | None:
        """The currently active app's id, or ``None`` if no app is active."""
        return self._active_app_id

    # -- Discovery ---------------------------------------------------------

    def discover(self) -> None:
        """Import each non-underscore app under ``apps_package`` and load it.

        Safe to call once at startup. A single app that fails to import,
        fails validation, or raises in ``on_load`` is logged and excluded —
        it never prevents other apps from loading.
        """
        try:
            package = importlib.import_module(self._apps_package)
        except ImportError:
            logger.exception("Could not import apps package %r", self._apps_package)
            return

        package_path = getattr(package, "__path__", None)
        if package_path is None:
            logger.error("Apps package %r is not a package (no __path__)", self._apps_package)
            return

        for module_info in pkgutil.iter_modules(package_path):
            if module_info.name.startswith("_"):
                continue
            self._load_app(module_info.name)

    def _load_app(self, app_module_name: str) -> None:
        """Import, validate, and register one discovered app package.

        Logs and returns (never raises) on any failure — an unimportable
        module, a missing/invalid ``MANIFEST``, an unsupported
        ``api_version``, a missing/invalid ``Plugin`` class, a plugin that
        fails to construct, or a plugin whose ``on_load`` raises.
        """
        qualified_name = f"{self._apps_package}.{app_module_name}"

        manifest = self._import_manifest(qualified_name)
        if manifest is None:
            return

        if manifest.api_version != CURRENT_API_VERSION:
            logger.error(
                "App %r targets api_version %d, registry supports %d — skipping",
                manifest.id,
                manifest.api_version,
                CURRENT_API_VERSION,
            )
            return

        plugin = self._construct_plugin(qualified_name, manifest.id)
        if plugin is None:
            return

        if not self._guard(manifest.id, "on_load", plugin.on_load):
            return

        self._apps[manifest.id] = _LoadedApp(manifest=manifest, plugin=plugin)

    def _import_manifest(self, qualified_name: str) -> AppManifest | None:
        try:
            manifest_module = importlib.import_module(f"{qualified_name}.manifest")
        except Exception:
            logger.exception("Failed to import manifest for app module %r", qualified_name)
            return None
        manifest = getattr(manifest_module, "MANIFEST", None)
        if not isinstance(manifest, AppManifest):
            logger.error(
                "App module %r has no module-level MANIFEST: AppManifest in manifest.py",
                qualified_name,
            )
            return None
        return manifest

    def _construct_plugin(self, qualified_name: str, app_id: str) -> AppPlugin | None:
        try:
            plugin_module = importlib.import_module(f"{qualified_name}.plugin")
        except Exception:
            logger.exception("Failed to import plugin module for app %r", app_id)
            return None
        plugin_cls = getattr(plugin_module, "Plugin", None)
        if not (isinstance(plugin_cls, type) and issubclass(plugin_cls, AppPlugin)):
            logger.error("App %r plugin.py has no Plugin class implementing AppPlugin", app_id)
            return None
        try:
            return plugin_cls()
        except Exception:
            logger.exception("Failed to construct Plugin for app %r", app_id)
            return None

    # -- Lifecycle guard -----------------------------------------------------

    def _guard(self, app_id: str, method_name: str, call: Callable[[], object]) -> bool:
        """Call a plugin lifecycle method, catching and logging any exception.

        Returns:
            ``True`` if ``call`` completed without raising, ``False`` otherwise.
        """
        try:
            call()
        except Exception:
            logger.exception("App %r raised from %s()", app_id, method_name)
            return False
        return True

    # -- Start/stop (single active app) --------------------------------------

    def start(self, app_id: str) -> None:
        """Make ``app_id`` the active app, stopping any currently active app first.

        Args:
            app_id: Id of a previously discovered app (see :attr:`manifests`).

        Raises:
            KeyError: If ``app_id`` was not discovered/loaded. This is a
                caller/programming error, not a plugin failure, so it is not
                swallowed by the lifecycle guard.
        """
        if app_id not in self._apps:
            raise KeyError(f"Unknown app id {app_id!r}")

        if self._active_app_id is not None and self._active_app_id != app_id:
            self.stop_active(reason="user")

        loaded = self._apps[app_id]
        loaded.consecutive_frame_failures = 0
        if self._guard(app_id, "on_start", loaded.plugin.on_start):
            self._active_app_id = app_id
            self._event_bus.publish(GameStartEvent(app_id=app_id))

    def stop_active(self, reason: str = "user") -> None:
        """Stop the active app, if any.

        Args:
            reason: Recorded on the published ``GameStopEvent`` — ``"user"``
                for a normal stop/switch, ``"error"`` when the failure
                threshold in :meth:`process_frame` triggers the stop.
        """
        if self._active_app_id is None:
            return
        app_id = self._active_app_id
        loaded = self._apps[app_id]
        self._active_app_id = None
        self._guard(app_id, "on_stop", loaded.plugin.on_stop)
        self._event_bus.publish(GameStopEvent(app_id=app_id, reason=reason))

    # -- Per-frame dispatch ---------------------------------------------------

    def process_frame(self, frame: Frame) -> Frame:
        """Run the active app's ``on_frame``, guarded, on the calling thread.

        Called by the frame pipeline (M1.4) on its worker thread, once per
        captured frame. With no active app this is a no-op passthrough.
        Consecutive failures are tracked per app; hitting
        ``max_consecutive_frame_failures`` stops the app (``reason="error"``)
        instead of calling it again next frame.

        Args:
            frame: The captured frame for this tick.

        Returns:
            The plugin's returned frame, or the input frame unchanged if no
            app is active or the app's ``on_frame`` raised.
        """
        if self._active_app_id is None:
            return frame

        app_id = self._active_app_id
        loaded = self._apps[app_id]
        result: Frame = frame

        try:
            result = loaded.plugin.on_frame(frame)
        except Exception:
            logger.exception("App %r raised from on_frame()", app_id)
            loaded.consecutive_frame_failures += 1
            if loaded.consecutive_frame_failures >= self._max_consecutive_frame_failures:
                logger.error(
                    "App %r exceeded %d consecutive on_frame failures — stopping",
                    app_id,
                    self._max_consecutive_frame_failures,
                )
                self.stop_active(reason="error")
            return frame

        loaded.consecutive_frame_failures = 0
        return result

    # -- Shutdown -------------------------------------------------------------

    def unload_all(self) -> None:
        """Stop the active app (if any) and call ``on_unload`` on every loaded app.

        Guarded the same way as every other lifecycle call — one app's
        ``on_unload`` raising does not stop the others from being unloaded.
        """
        self.stop_active(reason="user")
        for app_id, loaded in self._apps.items():
            self._guard(app_id, "on_unload", loaded.plugin.on_unload)
