"""Shared fixtures for unit tests.

Qt widgets are exercised on the ``offscreen`` platform plugin so UI tests
run headless (CI runners have no interactive display session). The
environment variable must be set before the first ``QApplication`` is
created, hence at import time here.
"""

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

#: Directory holding committed test assets (tiny ONNX models, ...).
FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    """The one ``QApplication`` shared by all UI tests (Qt allows only one)."""
    app = QApplication.instance() or QApplication([])
    assert isinstance(app, QApplication)
    return app


@pytest.fixture(scope="session")
def tiny_identity_onnx() -> Path:
    """Path to the committed single-input Identity ONNX fixture model."""
    path = FIXTURES_DIR / "onnx" / "tiny_identity.onnx"
    if not path.is_file():
        pytest.skip(f"missing ONNX fixture: {path}")
    return path


@pytest.fixture(scope="session")
def tiny_two_input_onnx() -> Path:
    """Path to the committed two-input (Add) ONNX fixture model."""
    path = FIXTURES_DIR / "onnx" / "tiny_two_input.onnx"
    if not path.is_file():
        pytest.skip(f"missing ONNX fixture: {path}")
    return path


@pytest.fixture(scope="session")
def hand_landmarker_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Real MediaPipe hand-landmarker model, downloaded once per session.

    Skips (rather than fails) when the model cannot be fetched — e.g. a CI
    runner with no network — so offline environments still run the rest of
    the suite. Downloaded locally, this exercises the real backend end to end.
    """
    from visionplay.vision.inference.model_catalog import HAND_LANDMARKER
    from visionplay.vision.inference.model_registry import (
        HttpModelDownloader,
        ModelRegistry,
        ModelRegistryError,
    )

    cache = tmp_path_factory.mktemp("mp_models")
    registry = ModelRegistry(cache, HttpModelDownloader())
    registry.register(HAND_LANDMARKER)
    try:
        return registry.ensure(HAND_LANDMARKER)
    except ModelRegistryError as exc:
        pytest.skip(f"hand_landmarker model unavailable (no network?): {exc}")
