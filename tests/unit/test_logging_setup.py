"""Unit tests for visionplay.core.logging_setup."""

import logging
from collections.abc import Iterator
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

from visionplay.core.logging_setup import LOGGER_NAME, setup_logging


@pytest.fixture(autouse=True)
def clean_logger() -> Iterator[None]:
    """Detach and close any handlers our setup installed, before and after."""

    def _reset() -> None:
        logger = logging.getLogger(LOGGER_NAME)
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    _reset()
    yield
    _reset()


def test_returns_the_package_logger(tmp_path: Path) -> None:
    logger = setup_logging(tmp_path / "app.log", console=False)
    assert logger.name == LOGGER_NAME


def test_writes_to_file(tmp_path: Path) -> None:
    log_file = tmp_path / "app.log"
    logger = setup_logging(log_file, level="DEBUG", console=False)
    logger.debug("hello from the test")
    for handler in logger.handlers:
        handler.flush()
    assert "hello from the test" in log_file.read_text(encoding="utf-8")


def test_creates_missing_log_dir(tmp_path: Path) -> None:
    log_file = tmp_path / "deep" / "nested" / "app.log"
    setup_logging(log_file, console=False)
    assert log_file.parent.is_dir()


def test_level_accepts_string_names_case_insensitive(tmp_path: Path) -> None:
    logger = setup_logging(tmp_path / "a.log", level="warning", console=False)
    assert logger.level == logging.WARNING


def test_level_accepts_int(tmp_path: Path) -> None:
    logger = setup_logging(tmp_path / "a.log", level=logging.ERROR, console=False)
    assert logger.level == logging.ERROR


def test_unknown_level_name_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="LOUD"):
        setup_logging(tmp_path / "a.log", level="LOUD", console=False)


def test_idempotent_no_duplicate_handlers(tmp_path: Path) -> None:
    setup_logging(tmp_path / "a.log", console=True)
    logger = setup_logging(tmp_path / "a.log", console=True)
    assert len(logger.handlers) == 2  # exactly one file + one console


def test_console_flag_controls_stream_handler(tmp_path: Path) -> None:
    logger = setup_logging(tmp_path / "a.log", console=False)
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], RotatingFileHandler)


def test_rotation_configured(tmp_path: Path) -> None:
    logger = setup_logging(tmp_path / "a.log", console=False)
    handler = logger.handlers[0]
    assert isinstance(handler, RotatingFileHandler)
    assert handler.maxBytes > 0
    assert handler.backupCount > 0


def test_no_propagation_to_root(tmp_path: Path) -> None:
    logger = setup_logging(tmp_path / "a.log", console=False)
    assert logger.propagate is False


def test_child_loggers_inherit(tmp_path: Path) -> None:
    log_file = tmp_path / "a.log"
    setup_logging(log_file, level="INFO", console=False)
    child = logging.getLogger(f"{LOGGER_NAME}.core.config")
    child.info("child message")
    for handler in logging.getLogger(LOGGER_NAME).handlers:
        handler.flush()
    assert "child message" in log_file.read_text(encoding="utf-8")
