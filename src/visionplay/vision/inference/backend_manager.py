"""Ownership and lifecycle of inference backends, shared across app switches.

The :class:`BackendManager` is the layer the pipeline (M2.3) talks to when an
app declares ``required_backends``: it constructs each named
:class:`~visionplay.vision.inference.backend_base.InferenceBackend` on first
use, loads it, and keeps it **warm** so switching between two apps that need
the same backend does not reload it (``docs/architecture.md`` §4 — "Backend
ownership: pipeline owns backends, plugins declare needs; keeps backends
shared/warm across app switches").

The manager stays free of any concrete backend knowledge — it never contains
an ``if name == "mediapipe.hands"`` chain that grows with every new backend.
Instead callers register a :class:`BackendRegistration` per backend name at
startup (see
:mod:`~visionplay.vision.inference.backend_defaults` for the built-ins). Each
registration pairs a *factory* (turns a
:class:`~visionplay.vision.inference.device.DeviceConfig` into a backend) with
a *probe* (answers "is this backend runnable right now?" without loading it),
which is the primitive the launcher's capability negotiation (M2.3) calls.

Plugins never touch this class: only the pipeline constructs and queries a
:class:`BackendManager` (``docs/architecture.md`` §3/§4 — plugins never
instantiate backends).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from visionplay.vision.inference.backend_base import InferenceBackend, InferenceError
from visionplay.vision.inference.device import DeviceConfig

__all__ = ["BackendManager", "BackendRegistration"]

#: A factory turns the resolved device into a ready-to-load backend instance.
#: The manager supplies the device (from configuration) — factories never pick
#: it themselves, mirroring the "call sites never branch on device" rule
#: (``docs/architecture.md`` §5).
BackendFactory = Callable[[DeviceConfig], InferenceBackend]

#: A probe reports whether a backend can run *right now* (runtime dependency
#: importable, model registered, ...) without constructing or loading it. It
#: must not raise; the manager treats a raising probe as "unavailable".
AvailabilityProbe = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class BackendRegistration:
    """How to build and availability-check one backend, keyed by name.

    Attributes:
        name: Stable backend identifier — the string apps list in
            ``required_backends`` and the key results land under in
            ``frame.results`` (e.g. ``"mediapipe.hands"``, ``"onnx.yolo_nano"``).
            Must equal the :attr:`InferenceBackend.name` the factory produces;
            :meth:`BackendManager.acquire` enforces this.
        factory: Builds an unloaded backend from a device config. Called at
            most once per name by the manager, on first :meth:`acquire`.
        probe: Returns ``True`` if the backend is runnable now. Cheap and
            side-effect free — it must not load the backend or touch the
            camera. Used for capability negotiation before an app launches.
    """

    name: str
    factory: BackendFactory
    probe: AvailabilityProbe


class BackendManager:
    """Registry of backend factories plus a warm cache of loaded instances.

    Construct one per process with the resolved device and pass it to the
    pipeline. Registration happens once at startup (via
    :mod:`~visionplay.vision.inference.backend_defaults`); the pipeline then
    calls :meth:`is_available` during capability negotiation and
    :meth:`acquire` when an app starts.

    Instances are cached by name: the first :meth:`acquire` constructs and
    loads the backend, every later one returns that same warm instance, so an
    app switch between two apps needing the same backend never reloads it.
    """

    def __init__(self, device: DeviceConfig | None = None) -> None:
        """Create an empty manager bound to one device.

        Args:
            device: Device every backend is constructed for. ``None`` means
                CPU — the v1 default and only option (``docs/architecture.md``
                §5). Resolve it from configuration with
                :func:`~visionplay.vision.inference.backend_defaults.device_from_config`.
        """
        self._device = device if device is not None else DeviceConfig.cpu()
        self._registrations: dict[str, BackendRegistration] = {}
        self._instances: dict[str, InferenceBackend] = {}

    @property
    def device(self) -> DeviceConfig:
        """The device every backend this manager builds runs on."""
        return self._device

    def register(self, registration: BackendRegistration) -> None:
        """Add a backend's factory and probe under its name.

        Re-registering the identical :class:`BackendRegistration` is a no-op
        (startup wiring may run more than once); registering a *different*
        registration under a name already taken is an error — a silently
        shadowed backend would be a hard-to-trace bug.

        Args:
            registration: The name/factory/probe triple to add.

        Raises:
            InferenceError: If a different registration already holds the name.
        """
        existing = self._registrations.get(registration.name)
        if existing is not None and existing != registration:
            raise InferenceError(
                f"Backend name {registration.name!r} is already registered "
                f"with a different factory"
            )
        self._registrations[registration.name] = registration

    def registered_names(self) -> tuple[str, ...]:
        """Return every registered backend name, in registration order."""
        return tuple(self._registrations)

    def is_registered(self, name: str) -> bool:
        """Return ``True`` if a factory is registered under ``name``."""
        return name in self._registrations

    def is_available(self, name: str) -> bool:
        """Return ``True`` if the named backend can run right now.

        An unregistered name is unavailable rather than an error — capability
        negotiation asks about backends that may not exist. A probe that
        raises is also treated as unavailable, so one misbehaving probe can
        never break the launcher's negotiation pass.

        Args:
            name: Backend identifier to check.
        """
        registration = self._registrations.get(name)
        if registration is None:
            return False
        try:
            return bool(registration.probe())
        except Exception:
            # A probe must never crash capability negotiation; unknown == no.
            return False

    def acquire(self, name: str) -> InferenceBackend:
        """Return the loaded, warm backend for ``name``, building it if needed.

        First call constructs the backend from its factory and calls
        :meth:`~InferenceBackend.load`; later calls return the cached instance
        without reloading. A load failure is propagated and nothing is cached,
        so a retry re-attempts cleanly.

        Args:
            name: Backend identifier previously passed to :meth:`register`.

        Returns:
            The single warm :class:`InferenceBackend` instance for ``name``.

        Raises:
            InferenceError: If no factory is registered under ``name``, or if
                the factory yields a backend whose :attr:`InferenceBackend.name`
                does not match ``name``.
        """
        existing = self._instances.get(name)
        if existing is not None:
            return existing

        registration = self._registrations.get(name)
        if registration is None:
            available = ", ".join(self._registrations) or "<none>"
            raise InferenceError(
                f"No inference backend registered under name {name!r} " f"(registered: {available})"
            )

        backend = registration.factory(self._device)
        if backend.name != name:
            raise InferenceError(
                f"Backend factory for {name!r} produced a backend named "
                f"{backend.name!r}; names must match"
            )
        backend.load()
        self._instances[name] = backend
        return backend

    def loaded_names(self) -> tuple[str, ...]:
        """Return the names of backends currently constructed and warm."""
        return tuple(self._instances)

    def is_loaded(self, name: str) -> bool:
        """Return ``True`` if ``name`` has a warm instance in the cache."""
        return name in self._instances

    def release(self, name: str) -> None:
        """Unload and drop the warm instance for ``name`` if one exists.

        Idempotent and safe for a name that was never acquired. The pipeline
        (M2.3) calls this when no active app still needs the backend.

        Args:
            name: Backend identifier to release.
        """
        backend = self._instances.pop(name, None)
        if backend is not None:
            backend.unload()

    def release_all(self) -> None:
        """Unload and drop every warm instance — the process-shutdown path."""
        for name in tuple(self._instances):
            self.release(name)
