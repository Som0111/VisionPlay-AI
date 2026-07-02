"""M0.1 smoke tests: the package imports and the entry point exits cleanly."""

import pytest

import visionplay
from visionplay.__main__ import main


def test_version_is_defined() -> None:
    assert visionplay.__version__


def test_main_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main() == 0
    assert visionplay.__version__ in capsys.readouterr().out
