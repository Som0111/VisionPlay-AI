"""Shared fixtures for unit tests.

Qt widgets are exercised on the ``offscreen`` platform plugin so UI tests
run headless (CI runners have no interactive display session). The
environment variable must be set before the first ``QApplication`` is
created, hence at import time here.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    """The one ``QApplication`` shared by all UI tests (Qt allows only one)."""
    app = QApplication.instance() or QApplication([])
    assert isinstance(app, QApplication)
    return app
