"""YAML configuration with per-namespace sections and validated defaults.

The on-disk format is a single ``config.yaml`` whose top level is a mapping
of *namespaces* (sections) to key/value mappings::

    app:
      log_level: INFO
    camera:
      device_index: 0
      frame_width: 1280
      frame_height: 720
      mirror: true
    inference:
      device:
        type: cpu
      model_cache_dir: null

Platform code owns the ``app`` and ``camera`` namespaces; app plugins get
their own namespace keyed by their manifest ``id`` (Phase 1+), which is why
the accessors are namespace-first rather than a fixed schema object.

Design points:

- :func:`load_config` never fails on a *missing* file — it returns defaults
  (first-run behavior). A file that exists but is malformed raises
  :class:`ConfigError` with a message naming the file and the problem;
  silently reverting a user's edited-but-broken config to defaults would
  destroy their settings on the next save.
- Defaults are deep-merged under whatever the file provides, so a config
  written by an older version keeps working when new keys are introduced.
- No global mutable state: :class:`Config` instances are constructed at
  startup and passed down, mirroring :class:`~visionplay.core.paths.AppPaths`.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

__all__ = ["Config", "ConfigError", "default_config", "load_config"]

#: Log level names accepted for ``app.log_level`` (standard ``logging`` levels).
VALID_LOG_LEVELS: frozenset[str] = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})

#: Factory-fresh configuration. Deep-copied on use — never mutated.
_DEFAULTS: dict[str, dict[str, Any]] = {
    "app": {
        "log_level": "INFO",
    },
    "camera": {
        "device_index": 0,
        "frame_width": 1280,
        "frame_height": 720,
        "target_fps": 30,
        "mirror": True,
    },
    "inference": {
        # Resolved into a DeviceConfig on the vision side via
        # DeviceConfig.from_mapping — core does not import vision, so only the
        # structural shape is validated here, not the device *type* semantics.
        "device": {"type": "cpu"},
        # None -> use AppPaths.models_dir; a string overrides the model cache
        # location. Resolved by backend_defaults.models_dir_from_config.
        "model_cache_dir": None,
    },
}


class ConfigError(Exception):
    """A configuration file is unreadable, malformed, or fails validation.

    The message always includes enough context (file path, offending
    section/key) for a user to fix the file by hand.
    """


def default_config() -> dict[str, dict[str, Any]]:
    """Return a fresh copy of the built-in default configuration.

    Returns:
        A deep copy — callers may mutate the result freely.
    """
    return copy.deepcopy(_DEFAULTS)


class Config:
    """In-memory configuration with namespaced sections, backed by YAML.

    Obtain instances via :func:`load_config` (reads/creates the file) or
    directly for tests (``Config(data, path)``). Mutations happen in memory
    via :meth:`set`; nothing touches disk until :meth:`save`.
    """

    def __init__(self, data: dict[str, dict[str, Any]], path: Path) -> None:
        """Initialize from already-validated data.

        Args:
            data: Mapping of namespace -> key/value mapping. Validated by
                :func:`_validate`; passing unvalidated data is a bug.
            path: File the config was loaded from and will be saved to.
        """
        self._data = data
        self._path = path

    @property
    def path(self) -> Path:
        """The YAML file this configuration reads from and writes to."""
        return self._path

    def namespaces(self) -> tuple[str, ...]:
        """Return the names of all sections currently present."""
        return tuple(self._data)

    def section(self, namespace: str) -> dict[str, Any]:
        """Return a *copy* of one section's key/value mapping.

        Args:
            namespace: Section name, e.g. ``"camera"`` or a plugin id.

        Returns:
            A shallow copy (mutating it does not affect the config); empty
            dict if the namespace does not exist yet.
        """
        return dict(self._data.get(namespace, {}))

    def get(self, namespace: str, key: str, default: Any = None) -> Any:
        """Return one value, or ``default`` if the namespace/key is absent.

        Args:
            namespace: Section name.
            key: Key within the section.
            default: Returned when the key is not set.
        """
        return self._data.get(namespace, {}).get(key, default)

    def set(self, namespace: str, key: str, value: Any) -> None:
        """Set one value in memory, creating the namespace if needed.

        Only YAML-representable values (str/int/float/bool/None and
        lists/dicts thereof) survive a save round-trip; :meth:`save`
        enforces this.

        Args:
            namespace: Section name.
            key: Key within the section.
            value: New value.
        """
        self._data.setdefault(namespace, {})[key] = value

    def save(self) -> None:
        """Write the current state to :attr:`path` as YAML.

        Creates parent directories if needed. The write is atomic-ish:
        content goes to a ``.tmp`` sibling first, then replaces the target,
        so a crash mid-write cannot leave a truncated config.

        Raises:
            ConfigError: If a stored value is not YAML-serializable.
            OSError: On filesystem errors.
        """
        try:
            text = yaml.safe_dump(self._data, sort_keys=True, default_flow_style=False)
        except yaml.YAMLError as exc:
            raise ConfigError(
                f"Configuration contains a value that cannot be written as YAML: {exc}"
            ) from exc
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(self._path)


def load_config(path: Path, *, create_if_missing: bool = True) -> Config:
    """Load ``config.yaml``, merging file contents over built-in defaults.

    First-run behavior: if the file does not exist, a :class:`Config` with
    pure defaults is returned and (when ``create_if_missing``) immediately
    written to ``path`` so the user has a file to edit.

    Args:
        path: Location of the YAML config file.
        create_if_missing: Write the defaults to disk when the file is
            absent. Set ``False`` for read-only probing.

    Returns:
        A validated :class:`Config`.

    Raises:
        ConfigError: If the file exists but is not valid YAML, is not a
            mapping of sections, or fails value validation — the message
            names the file and the exact problem.
        OSError: On unreadable/unwritable filesystem locations.
    """
    if not path.exists():
        config = Config(default_config(), path)
        if create_if_missing:
            config.save()
        return config

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: not valid YAML: {exc}") from exc

    if raw is None:  # empty file — treat as "no overrides"
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(
            f"{path}: top level must be a mapping of sections, got {type(raw).__name__}"
        )

    data = _merge_defaults(raw, path)
    _validate(data, path)
    return Config(data, path)


def _merge_defaults(raw: dict[Any, Any], path: Path) -> dict[str, dict[str, Any]]:
    """Overlay file-provided sections onto the defaults.

    Unknown namespaces/keys are preserved verbatim (forward compatibility
    and plugin namespaces); known keys from the file win over defaults.

    Raises:
        ConfigError: If a section value is not a mapping.
    """
    merged = default_config()
    for name, section in raw.items():
        if not isinstance(name, str):
            raise ConfigError(f"{path}: section names must be strings, got {name!r}")
        if not isinstance(section, dict):
            raise ConfigError(
                f"{path}: section '{name}' must be a mapping of keys to values, "
                f"got {type(section).__name__}"
            )
        merged.setdefault(name, {}).update(section)
    return merged


def _validate(data: dict[str, dict[str, Any]], path: Path) -> None:
    """Validate platform-owned values, with actionable error messages.

    Only the ``app`` and ``camera`` namespaces are checked — plugin
    namespaces are opaque to core and validated by their owners.

    Raises:
        ConfigError: On the first invalid value found.
    """
    level = data["app"].get("log_level")
    if not isinstance(level, str) or level.upper() not in VALID_LOG_LEVELS:
        valid = ", ".join(sorted(VALID_LOG_LEVELS))
        raise ConfigError(f"{path}: app.log_level must be one of {valid}, got {level!r}")

    camera = data["camera"]
    for key in ("device_index", "frame_width", "frame_height", "target_fps"):
        value = camera.get(key)
        # bool is an int subclass; reject it explicitly.
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ConfigError(f"{path}: camera.{key} must be a non-negative integer, got {value!r}")
    mirror = camera.get("mirror")
    if not isinstance(mirror, bool):
        raise ConfigError(f"{path}: camera.mirror must be a boolean, got {mirror!r}")

    # Only the structural shape of the inference namespace is checked here.
    # Device-type semantics (is "cpu" a known device?) are the vision layer's
    # job via DeviceConfig.from_mapping — core must not import from vision/.
    inference = data["inference"]
    device = inference.get("device")
    if not isinstance(device, dict):
        raise ConfigError(
            f"{path}: inference.device must be a mapping like {{type: cpu}}, got {device!r}"
        )
    cache_override = inference.get("model_cache_dir")
    if cache_override is not None and not isinstance(cache_override, str):
        raise ConfigError(
            f"{path}: inference.model_cache_dir must be a string path or null, "
            f"got {cache_override!r}"
        )
