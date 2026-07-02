"""Platform-appropriate filesystem locations for VisionPlay.

All persistent state (configuration, logs, cached models, general cache)
lives under directories resolved via :mod:`platformdirs`, never hardcoded
Windows paths — v1 ships Windows-only, but this module must stay portable
(see ``docs/architecture.md`` §6).

The canonical entry point is :meth:`AppPaths.default`::

    paths = AppPaths.default()
    paths.ensure()          # create the directories on first run
    cfg_file = paths.config_file

There is deliberately no module-level singleton: callers construct an
:class:`AppPaths` once at startup and pass it down, and tests construct one
rooted in a temporary directory via :meth:`AppPaths.for_root`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from platformdirs import PlatformDirs

#: Application name used for platform directory resolution
#: (e.g. ``%LOCALAPPDATA%/VisionPlay/visionplay`` on Windows).
APP_NAME: str = "visionplay"

#: Author/organization segment used by platformdirs on Windows.
APP_AUTHOR: str = "VisionPlay"

#: Filename of the main configuration file inside :attr:`AppPaths.config_dir`.
CONFIG_FILENAME: str = "config.yaml"

#: Filename of the main log file inside :attr:`AppPaths.log_dir`.
LOG_FILENAME: str = "visionplay.log"


@dataclass(frozen=True, slots=True)
class AppPaths:
    """Resolved set of application directories.

    Instances are immutable value objects; resolving and creating
    directories are separate steps (:meth:`default` / :meth:`for_root`
    resolve, :meth:`ensure` creates).

    Attributes:
        config_dir: Directory for user configuration (``config.yaml``).
        cache_dir: Directory for general-purpose cached data.
        models_dir: Directory for the downloaded/checksum-verified model
            cache managed by ``model_registry.py`` (Phase 2).
        log_dir: Directory for rotating log files.
    """

    config_dir: Path
    cache_dir: Path
    models_dir: Path
    log_dir: Path

    @classmethod
    def default(cls) -> AppPaths:
        """Resolve the standard per-user directories for this platform.

        Uses :class:`platformdirs.PlatformDirs`, so the result is correct
        on Windows, macOS, and Linux without OS-specific branches. Does not
        create anything on disk; call :meth:`ensure` for that.

        Returns:
            An :class:`AppPaths` pointing at the platform-native locations.
        """
        dirs = PlatformDirs(appname=APP_NAME, appauthor=APP_AUTHOR, roaming=False)
        cache = Path(dirs.user_cache_dir)
        return cls(
            config_dir=Path(dirs.user_config_dir),
            cache_dir=cache,
            models_dir=cache / "models",
            log_dir=Path(dirs.user_log_dir),
        )

    @classmethod
    def for_root(cls, root: Path) -> AppPaths:
        """Resolve all directories beneath a single root directory.

        Intended for tests and portable/self-contained setups where state
        must not touch the real user profile.

        Args:
            root: Directory under which ``config/``, ``cache/``,
                ``cache/models/``, and ``logs/`` are laid out.

        Returns:
            An :class:`AppPaths` rooted at ``root`` (nothing is created).
        """
        return cls(
            config_dir=root / "config",
            cache_dir=root / "cache",
            models_dir=root / "cache" / "models",
            log_dir=root / "logs",
        )

    @property
    def config_file(self) -> Path:
        """Full path of the main configuration file (``config.yaml``)."""
        return self.config_dir / CONFIG_FILENAME

    @property
    def log_file(self) -> Path:
        """Full path of the main rotating log file."""
        return self.log_dir / LOG_FILENAME

    def ensure(self) -> AppPaths:
        """Create every directory (parents included) if it does not exist.

        Idempotent. Returns ``self`` so first-run setup can be a one-liner:
        ``paths = AppPaths.default().ensure()``.

        Raises:
            OSError: If a directory cannot be created (e.g. permissions).
        """
        for directory in (self.config_dir, self.cache_dir, self.models_dir, self.log_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return self
