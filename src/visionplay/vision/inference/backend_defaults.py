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
    BackendManager,
    BackendRegistration,
)
from visionplay.vision.inference.device import DeviceConfig
from visionplay.vision.inference.mediapipe_backend import MediaPipeBackend, MediaPipeTask
from visionplay.vision.inference.model_catalog import HAND_LANDMARKER
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
    "register_default_backends",
    "register_mediapipe_hands_backend",
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


def register_mediapipe_hands_backend(
    manager: BackendManager,
    model_registry: ModelRegistry,
    spec: ModelSpec = HAND_LANDMARKER,
) -> None:
    """Register the MediaPipe hand-landmark backend on ``manager``.

    The model spec is registered in ``model_registry``, and the backend is
    exposed under ``"mediapipe.hands"``. The factory resolves the model to a
    verified local path via
    :meth:`~visionplay.vision.inference.model_registry.ModelRegistry.ensure`
    at acquire time — downloading on first use, never at registration or per
    frame. The probe requires both the MediaPipe runtime and the model being
    registered, mirroring the ONNX backend so capability negotiation is
    uniform across runtimes.

    Args:
        manager: The manager to populate.
        model_registry: Catalog/cache the model is resolved through.
        spec: The hand-landmarker model to run. Defaults to the bundled
            :data:`~visionplay.vision.inference.model_catalog.HAND_LANDMARKER`.
    """
    model_registry.register(spec)

    def factory(device: DeviceConfig) -> MediaPipeBackend:
        model_path = model_registry.ensure(spec)
        return MediaPipeBackend(MediaPipeTask.HAND_LANDMARKS, model_path, device)

    def probe() -> bool:
        if not probe_mediapipe():
            return False
        try:
            model_registry.get(spec.model_id)
        except ModelRegistryError:
            return False
        return True

    manager.register(BackendRegistration(name="mediapipe.hands", factory=factory, probe=probe))


def register_default_backends(manager: BackendManager, model_registry: ModelRegistry) -> None:
    """Register every built-in backend VisionPlay ships with, on ``manager``.

    The single entry point the app bootstrap (M2.3) calls to wire the standard
    backend set. Currently the MediaPipe hand-landmark backend; new built-ins
    are added here as they gain real implementations.

    Args:
        manager: The manager to populate.
        model_registry: Catalog/cache built-in models are resolved through.
    """
    register_mediapipe_hands_backend(manager, model_registry)


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
