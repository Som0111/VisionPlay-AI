"""Unit tests for visionplay.core.paths."""

from pathlib import Path

import pytest

from visionplay.core import paths as paths_mod
from visionplay.core.paths import CONFIG_FILENAME, LOG_FILENAME, AppPaths


class TestForRoot:
    def test_lays_out_all_dirs_under_root(self, tmp_path: Path) -> None:
        p = AppPaths.for_root(tmp_path)
        assert p.config_dir == tmp_path / "config"
        assert p.cache_dir == tmp_path / "cache"
        assert p.models_dir == tmp_path / "cache" / "models"
        assert p.log_dir == tmp_path / "logs"

    def test_does_not_create_anything(self, tmp_path: Path) -> None:
        p = AppPaths.for_root(tmp_path / "sub")
        assert not p.config_dir.exists()


class TestDefault:
    def test_uses_platformdirs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeDirs:
            def __init__(self, **kwargs: object) -> None:
                self.user_config_dir = str(tmp_path / "cfg")
                self.user_cache_dir = str(tmp_path / "cache")
                self.user_log_dir = str(tmp_path / "logs")

        monkeypatch.setattr(paths_mod, "PlatformDirs", FakeDirs)
        p = AppPaths.default()
        assert p.config_dir == tmp_path / "cfg"
        assert p.cache_dir == tmp_path / "cache"
        assert p.models_dir == tmp_path / "cache" / "models"
        assert p.log_dir == tmp_path / "logs"

    def test_resolves_absolute_paths_without_patching(self) -> None:
        p = AppPaths.default()
        for d in (p.config_dir, p.cache_dir, p.models_dir, p.log_dir):
            assert d.is_absolute()

    def test_models_dir_nested_in_cache(self) -> None:
        p = AppPaths.default()
        assert p.cache_dir in p.models_dir.parents


class TestEnsure:
    def test_creates_all_directories(self, tmp_path: Path) -> None:
        p = AppPaths.for_root(tmp_path).ensure()
        for d in (p.config_dir, p.cache_dir, p.models_dir, p.log_dir):
            assert d.is_dir()

    def test_idempotent(self, tmp_path: Path) -> None:
        p = AppPaths.for_root(tmp_path)
        assert p.ensure() is p
        assert p.ensure() is p  # second call must not raise

    def test_ensure_returns_self_for_chaining(self, tmp_path: Path) -> None:
        p = AppPaths.for_root(tmp_path)
        assert p.ensure() is p


class TestFileProperties:
    def test_config_file(self, tmp_path: Path) -> None:
        p = AppPaths.for_root(tmp_path)
        assert p.config_file == p.config_dir / CONFIG_FILENAME

    def test_log_file(self, tmp_path: Path) -> None:
        p = AppPaths.for_root(tmp_path)
        assert p.log_file == p.log_dir / LOG_FILENAME


def test_apppaths_is_immutable(tmp_path: Path) -> None:
    p = AppPaths.for_root(tmp_path)
    with pytest.raises(AttributeError):
        p.config_dir = tmp_path  # type: ignore[misc]
