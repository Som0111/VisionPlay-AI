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


# The real MediaPipe hand-landmarker model fixture (``hand_landmarker_model``)
# lives in tests/conftest.py, shared with the integration suite.
