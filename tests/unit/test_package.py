"""Package smoke tests: version and entry-point wiring.

M0.1 gave ``python -m visionplay`` a banner stub; M0.6 replaced it with the
Qt application bootstrap. Launching the real GUI belongs to the app tests
(``test_app.py``) — here we only assert the entry point is wired to it.
"""

import visionplay
from visionplay import app
from visionplay.__main__ import main


def test_version_is_defined() -> None:
    assert visionplay.__version__


def test_entry_point_is_the_qt_bootstrap() -> None:
    assert main is app.main
