"""The plugin contract: `AppManifest` + `AppPlugin` (M1.1).

Every game, utility, or demo in VisionPlay AI is a plugin implementing
:class:`AppPlugin`, described by one :class:`AppManifest` instance
(``docs/plugin-development.md``). This module defines the contract only —
no discovery, no lifecycle guarding, no pipeline wiring. Those are the
registry's job (Phase 1, M1.2) and the frame pipeline's job (M1.4).

**Stability**: treat this module as a public API, not an implementation
detail. External, pip-installable plugins are an explicit later phase
(``docs/architecture.md`` §3, §7 "Plugin interface freeze"); when that
phase lands, only the *discovery* mechanism changes
(``pkgutil.iter_modules`` → ``importlib.metadata.entry_points``) — the
``AppPlugin``/``AppManifest`` contract defined here must not need to
change. :attr:`AppManifest.api_version` exists from day one so that future
loader can reject a plugin built against an incompatible contract version,
even though nothing checks it yet (that arrives with the registry in
M1.2).

**Layering note**: this module lives in ``core/`` per
``docs/architecture.md`` §2, but ``on_frame`` is typed against
:class:`~visionplay.vision.pipeline.frame_types.Frame`. ``Frame`` is a
dependency-free value object (only ``numpy``, no ``core/`` imports), so
referencing it here does not create a cycle — it is the one deliberate
exception to "``core/`` never imports downward," made because the
lifecycle contract is meaningless without the type it operates on.

No Qt, registry, launcher, or inference logic belongs in this module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

from visionplay.vision.pipeline.frame_types import Frame

__all__ = ["APP_CATEGORIES", "CURRENT_API_VERSION", "AppManifest", "AppPlugin"]

#: The only app categories the launcher groups by (``docs/plugin-development.md``).
#: An app's category lives here, in the manifest, only — never in its folder path
#: (``docs/architecture.md`` §2), so this set is the single source of truth for
#: what a valid ``category`` value is.
APP_CATEGORIES: frozenset[str] = frozenset({"gesture_games", "fitness", "face_ar", "ai_demos"})

#: The `AppPlugin`/`AppManifest` contract version this codebase implements.
#: `AppManifest.api_version` is checked against this by the registry (M1.2) to
#: reject plugins built against an incompatible contract instead of crashing at
#: runtime. Bump this only when the lifecycle contract itself changes.
CURRENT_API_VERSION: int = 1


@dataclass(frozen=True, slots=True)
class AppManifest:
    """Immutable declaration of one app's identity, category, and needs.

    See ``docs/plugin-development.md`` § "`AppManifest` fields" for the
    authoritative field-by-field description.

    Attributes:
        id: Unique, stable identifier. Never reused across apps, even if an
            old app is removed — the registry and any persisted settings
            key off this string.
        name: Display name shown in the launcher.
        category: One of :data:`APP_CATEGORIES`. Validated at construction
            time so an app can't silently end up ungrouped in the launcher.
        version: The app's own version, independent of the platform
            version (``visionplay.__version__``).
        api_version: Which version of the `AppPlugin` contract this app
            targets. Compared against :data:`CURRENT_API_VERSION` by the
            registry (M1.2), not by this class — construction never fails
            on a stale ``api_version`` by itself, since a manifest may be
            inspected before a compatibility decision is made.
        required_backends: Inference backends/capabilities this app needs
            (e.g. ``("mediapipe.hands",)``), used for launcher capability
            negotiation. Accepts any sequence at construction and is
            normalized to a ``tuple`` so the manifest stays hashable and
            immutable.
        icon: Path to the app's launcher icon, relative to its own
            ``assets/`` folder.

    Raises:
        ValueError: If ``category`` is not one of :data:`APP_CATEGORIES`.
    """

    id: str
    name: str
    category: str
    version: str
    api_version: int
    required_backends: Sequence[str] = field(default_factory=tuple)
    icon: str = ""

    def __post_init__(self) -> None:
        if self.category not in APP_CATEGORIES:
            supported = ", ".join(sorted(APP_CATEGORIES))
            raise ValueError(f"Unknown app category {self.category!r} (supported: {supported})")
        if not isinstance(self.required_backends, tuple):
            object.__setattr__(self, "required_backends", tuple(self.required_backends))


class AppPlugin(ABC):
    """Framework-agnostic lifecycle every app plugin implements.

    Deliberately free of Qt: keeping the interface itself UI-agnostic is
    what leaves room for a future sandboxed/external plugin loader without
    a breaking interface change (``docs/architecture.md`` §3). Concrete
    apps put their Qt-specific rendering/controls in their own
    ``widget.py``, never here.

    Execution model (enforced by the registry/pipeline in later
    milestones, not by this class): ``on_load`` and ``on_unload`` run on
    the registry's discovery path; ``on_start``/``on_stop`` run when the
    app becomes/stops being the active app; ``on_frame`` runs once per
    captured frame, **on the pipeline worker thread, never the Qt main
    thread** — implementations must not touch Qt objects from it and must
    keep per-frame cost bounded, since a slow ``on_frame`` causes the
    pipeline to drop frames for this app rather than queue them
    (``docs/architecture.md`` §4).
    """

    @abstractmethod
    def on_load(self) -> None:
        """Called once when the registry discovers and instantiates this plugin.

        Cheap setup only — no camera or model access yet. Model/camera
        resources are acquired in :meth:`on_start`.
        """

    @abstractmethod
    def on_start(self) -> None:
        """Called when the user opens this app from the launcher.

        Acquire any camera/model resources this app needs here. Backends
        are never constructed directly — the pipeline owns backend
        lifecycles and delivers results on the frame passed to
        :meth:`on_frame` (``docs/architecture.md`` §4).
        """

    @abstractmethod
    def on_frame(self, frame: Frame) -> Frame:
        """Process one captured frame; called on the pipeline worker thread.

        The frame arrives with any declared backends' results already
        attached under ``frame.results``. Delegate logic to the app's own
        ``processor.py``; never touch Qt objects here. Results reach the
        app's ``widget.py`` only via the thread-safe Qt signal the
        pipeline provides — never by the widget calling back into this
        method or the plugin directly.

        Args:
            frame: The captured frame for this tick, results already
                attached by the pipeline.

        Returns:
            The frame to publish downstream (typically the same frame,
            optionally annotated).
        """

    @abstractmethod
    def on_stop(self) -> None:
        """Called when this app stops being the active app.

        Release camera/model resources acquired in :meth:`on_start`.
        """

    @abstractmethod
    def on_unload(self) -> None:
        """Called when this app is removed from the registry.

        Not expected to happen in v1's static, internal-only discovery,
        but must be implemented for forward compatibility with a future
        plugin loader that can unload apps at runtime.
        """
