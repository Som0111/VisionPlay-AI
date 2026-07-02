"""Application logging: rotating file + console, level driven by config.

Called once at startup (from the app bootstrap, after config is loaded)::

    setup_logging(paths.log_file, level=config.get("app", "log_level", "INFO"))

Handlers are installed on the ``visionplay`` package logger, not the root
logger, so noisy third-party libraries (OpenCV, MediaPipe, ONNX Runtime)
don't flood our log file and our setup doesn't fight theirs. Modules obtain
loggers the standard way (``logging.getLogger(__name__)``) and inherit the
configuration through the logger hierarchy.

The only module state is the ``logging`` module's own logger registry —
this module holds no globals of its own, and :func:`setup_logging` is
idempotent (calling it again reconfigures rather than duplicating handlers),
which keeps repeated calls in tests safe.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

__all__ = ["setup_logging"]

#: Name of the package logger all handlers attach to.
LOGGER_NAME: str = "visionplay"

#: Log line format: timestamp, level, module path, message.
LOG_FORMAT: str = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"

#: Timestamp format (ISO-8601-like, second resolution).
DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

#: Rotation policy: 1 MiB per file, 5 backups (~6 MiB worst case on disk).
MAX_LOG_BYTES: int = 1 * 1024 * 1024
BACKUP_COUNT: int = 5


def setup_logging(
    log_file: Path,
    level: int | str = logging.INFO,
    *,
    console: bool = True,
) -> logging.Logger:
    """Configure the ``visionplay`` logger with file (and console) output.

    Idempotent: existing handlers previously installed by this function are
    closed and replaced, so calling it twice (e.g. after a settings change
    or across tests) never duplicates output.

    Args:
        log_file: Path of the rotating log file. Its parent directory is
            created if missing. Rotation keeps :data:`BACKUP_COUNT` backups
            of :data:`MAX_LOG_BYTES` each.
        level: Minimum level to emit — a ``logging`` constant or a level
            name such as ``"DEBUG"`` (case-insensitive, as validated by
            :mod:`visionplay.core.config`).
        console: Also emit to ``stderr``. Disable for headless test runs
            that assert on file output only.

    Returns:
        The configured ``visionplay`` package logger.

    Raises:
        ValueError: If ``level`` is a string that names no known level.
        OSError: If the log directory/file cannot be created or opened.
    """
    if isinstance(level, str):
        numeric = logging.getLevelNamesMapping().get(level.upper())
        if numeric is None:
            raise ValueError(f"Unknown log level name: {level!r}")
        level = numeric

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    # Don't propagate to the root logger: avoids double output when the
    # embedding environment (pytest, IDEs) configures root itself.
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=MAX_LOG_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler(stream=sys.stderr)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
