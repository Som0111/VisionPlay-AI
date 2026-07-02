"""Built-in backend registrations, dependency probes, and config resolvers.

This module keeps concrete backend knowledge out of
:class:`~visionplay.vision.inference.backend_manager.BackendManager`: it knows
that ``"mediapipe.*"`` needs the ``mediapipe`` package and that an
``"onnx.*"`` backend needs both ``onnxruntime`` and a registered model, and it
expresses each as a :class:`~visionplay.vision.inference.backend_manager.BackendRegistration`
the generic manager stores by name.

Availability probes deliberately check only that a backend *could* run — the
runtime dependency imports, the model is registered — without importing the
heavy runtime, constructing the backend, or downloading anything. That keeps
the launcher's capability-negotiation pass (M2.3) cheap and side-effect free.

The config resolvers turn the ``inference`` config namespace into the two
values the pipeline needs when it builds a manager: the target
:class:`~visionplay.vision.inference.device.DeviceConfig` and the model cache
directory. They take a plain mapping (``config.section("inference")``) rather
than a :class:`~visionplay.core.config.Config`, so this module stays decoupled
from the config type.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from visionplay.vision.inference.backend_manager import (
    BackendFactory,
    BackendManager,
    BackendRegistration,
)
from visionplay.vision.inference.device import DeviceConfig
from visionplay.vision.inference.mediapipe_backend import MediaPipeBackend, MediaPipeTask
from visionplay.vision.inference.model_registry import (
    ModelRegistry,
    ModelRegistryError,
    ModelSpec,
)
from visionplay.vision.inference.onnx_backend import ONNXBackend

__all__ = [
    "device_from_config",
    "models_dir_from_config",
    "probe_mediapipe",
    "probe_onnxruntime",
    "register_builtin_mediapipe_backends",
    "register_onnx_backend",
]

#: Import name of the MediaPipe runtime the landmark backends need.
_MEDIAPIPE_MODULE = "mediapipe"

#: Import name of the ONNX Runtime the ONNX backends need.
_ONNXRUNTIME_MODULE = "onnxruntime"


def _module_importable(module_name: str) -> bool:
    """Return ``True`` if ``module_name`` can be imported, without importing it.

    Uses :func:`importlib.util.find_spec`, which only inspects the import
    system — the heavy runtime (MediaPipe/ONNX) is never actually loaded. A
    missing parent package makes ``find_spec`` raise; that too means "not
    importable", so it is swallowed into ``False``.
    """
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def probe_mediapipe() -> bool:
    """Return ``True`` if the MediaPipe runtime is importable."""
    return _module_importable(_MEDIAPIPE_MODULE)


def probe_onnxruntime() -> bool:
    """Return ``True`` if the ONNX Runtime is importable."""
    return _module_importable(_ONNXRUNTIME_MODULE)


def _make_mediapipe_factory(task: MediaPipeTask) -> BackendFactory:
    """Build a factory that constructs a :class:`MediaPipeBackend` for ``task``.

    A helper (rather than an inline lambda) so ``task`` is bound per call and
    not captured by reference in a loop.
    """

    def factory(device: DeviceConfig) -> MediaPipeBackend:
        return MediaPipeBackend(task, device)

    return factory


#: One factory per landmark task, built once at import so that repeated
#: :func:`register_builtin_mediapipe_backends` calls produce *equal*
#: registrations — which the manager treats as an idempotent no-op rather than
#: a conflict (per-call closures would have distinct identities and clash).
_MEDIAPIPE_FACTORIES: dict[MediaPipeTask, BackendFactory] = {
    task: _make_mediapipe_factory(task) for task in MediaPipeTask
}


def register_builtin_mediapipe_backends(manager: BackendManager) -> None:
    """Register the standard MediaPipe landmark backends on ``manager``.

    Registers one backend per :class:`MediaPipeTask` (hands/pose/face) under
    its ``"mediapipe.<task>"`` name, each probed on the MediaPipe runtime
    being importable. Registration is cheap and constructs nothing — the
    backends are built lazily on first
    :meth:`~visionplay.vision.inference.backend_manager.BackendManager.acquire`.
    Idempotent: calling it twice on the same manager is a no-op.

    Args:
        manager: The manager to populate.
    """
    for task in MediaPipeTask:
        manager.register(
            BackendRegistration(
                name=f"mediapipe.{task.value}",
                factory=_MEDIAPIPE_FACTORIES[task],
                probe=probe_mediapipe,
            )
        )


def register_onnx_backend(
    manager: BackendManager,
    model_registry: ModelRegistry,
    spec: ModelSpec,
) -> None:
    """Register one ONNX backend for ``spec`` on ``manager``.

    The spec is registered in ``model_registry`` (so the model is known to the
    catalog), and the backend is exposed under ``"onnx.<model_id>"``. The
    factory resolves the model to a verified local path via
    :meth:`~visionplay.vision.inference.model_registry.ModelRegistry.ensure`
    at acquire time — downloading on first use, never at registration. The
    probe requires both the ONNX runtime and the model being registered.

    Args:
        manager: The manager to populate.
        model_registry: Catalog/cache the model is resolved through.
        spec: The model this backend runs.
    """
    model_registry.register(spec)

    def factory(device: DeviceConfig) -> ONNXBackend:
        model_path = model_registry.ensure(spec)
        return ONNXBackend(spec, model_path, device)

    def probe() -> bool:
        if not probe_onnxruntime():
            return False
        try:
            model_registry.get(spec.model_id)
        except ModelRegistryError:
            return False
        return True

    manager.register(
        BackendRegistration(name=f"onnx.{spec.model_id}", factory=factory, probe=probe)
    )


def device_from_config(inference_section: Mapping[str, Any]) -> DeviceConfig:
    """Resolve the target device from the ``inference`` config section.

    Reads the ``device`` mapping (default ``{"type": "cpu"}``) and delegates
    to :meth:`DeviceConfig.from_mapping`, so an unknown device type surfaces
    the same actionable error everywhere.

    Args:
        inference_section: The ``inference`` namespace, e.g.
            ``config.section("inference")``.

    Returns:
        The resolved :class:`DeviceConfig`.

    Raises:
        ValueError: If ``device`` is present but not a mapping, or names an
            unknown device type.
    """
    raw = inference_section.get("device", {})
    if not isinstance(raw, Mapping):
        raise ValueError(f"inference.device must be a mapping like {{'type': 'cpu'}}, got {raw!r}")
    return DeviceConfig.from_mapping(raw)


def models_dir_from_config(inference_section: Mapping[str, Any], default: Path) -> Path:
    """Resolve the model cache directory from the ``inference`` config section.

    An absent or null ``model_cache_dir`` uses ``default`` (normally
    ``AppPaths.models_dir``); a string overrides it, letting a user point the
    cache elsewhere.

    Args:
        inference_section: The ``inference`` namespace mapping.
        default: Directory used when no override is configured.

    Returns:
        The resolved model cache directory.

    Raises:
        ValueError: If ``model_cache_dir`` is set to a non-string value.
    """
    override = inference_section.get("model_cache_dir")
    if override is None:
        return default
    if not isinstance(override, str):
        raise ValueError(
            f"inference.model_cache_dir must be a string path or null, got {override!r}"
        )
    return Path(override)
