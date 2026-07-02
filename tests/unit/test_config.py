"""Unit tests for visionplay.core.config."""

from pathlib import Path

import pytest
import yaml

from visionplay.core.config import Config, ConfigError, default_config, load_config


@pytest.fixture
def cfg_path(tmp_path: Path) -> Path:
    return tmp_path / "config.yaml"


class TestDefaultConfig:
    def test_contains_platform_namespaces(self) -> None:
        d = default_config()
        assert d["app"]["log_level"] == "INFO"
        assert d["camera"]["device_index"] == 0

    def test_returns_independent_copies(self) -> None:
        a = default_config()
        a["app"]["log_level"] = "DEBUG"
        assert default_config()["app"]["log_level"] == "INFO"


class TestFirstRun:
    def test_missing_file_returns_defaults(self, cfg_path: Path) -> None:
        config = load_config(cfg_path)
        assert config.get("app", "log_level") == "INFO"

    def test_missing_file_is_created(self, cfg_path: Path) -> None:
        load_config(cfg_path)
        assert cfg_path.exists()
        on_disk = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        assert on_disk == default_config()

    def test_create_if_missing_false_does_not_write(self, cfg_path: Path) -> None:
        load_config(cfg_path, create_if_missing=False)
        assert not cfg_path.exists()


class TestRoundTrip:
    def test_save_then_load_preserves_values(self, cfg_path: Path) -> None:
        config = load_config(cfg_path)
        config.set("app", "log_level", "DEBUG")
        config.set("my_plugin", "difficulty", "hard")
        config.save()

        reloaded = load_config(cfg_path)
        assert reloaded.get("app", "log_level") == "DEBUG"
        assert reloaded.get("my_plugin", "difficulty") == "hard"

    def test_file_values_win_over_defaults(self, cfg_path: Path) -> None:
        cfg_path.write_text("app:\n  log_level: WARNING\n", encoding="utf-8")
        config = load_config(cfg_path)
        assert config.get("app", "log_level") == "WARNING"
        # untouched defaults still filled in
        assert config.get("camera", "frame_width") == 1280

    def test_unknown_sections_preserved(self, cfg_path: Path) -> None:
        cfg_path.write_text("future_thing:\n  x: 1\n", encoding="utf-8")
        config = load_config(cfg_path)
        assert config.get("future_thing", "x") == 1

    def test_empty_file_treated_as_defaults(self, cfg_path: Path) -> None:
        cfg_path.write_text("", encoding="utf-8")
        assert load_config(cfg_path).get("app", "log_level") == "INFO"

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        deep = tmp_path / "a" / "b" / "config.yaml"
        config = Config(default_config(), deep)
        config.save()
        assert deep.exists()


class TestAccessors:
    def test_get_default_for_missing_key(self, cfg_path: Path) -> None:
        config = load_config(cfg_path, create_if_missing=False)
        assert config.get("app", "nope", 42) == 42
        assert config.get("no_such_ns", "k") is None

    def test_section_returns_copy(self, cfg_path: Path) -> None:
        config = load_config(cfg_path, create_if_missing=False)
        section = config.section("camera")
        section["device_index"] = 99
        assert config.get("camera", "device_index") == 0

    def test_section_missing_namespace_is_empty(self, cfg_path: Path) -> None:
        config = load_config(cfg_path, create_if_missing=False)
        assert config.section("ghost") == {}

    def test_namespaces_lists_sections(self, cfg_path: Path) -> None:
        config = load_config(cfg_path, create_if_missing=False)
        assert set(config.namespaces()) >= {"app", "camera"}

    def test_set_creates_namespace(self, cfg_path: Path) -> None:
        config = load_config(cfg_path, create_if_missing=False)
        config.set("new_ns", "k", True)
        assert config.get("new_ns", "k") is True

    def test_path_property(self, cfg_path: Path) -> None:
        assert load_config(cfg_path, create_if_missing=False).path == cfg_path


class TestValidationErrors:
    def test_invalid_yaml_raises_with_filename(self, cfg_path: Path) -> None:
        cfg_path.write_text("app: [unclosed", encoding="utf-8")
        with pytest.raises(ConfigError, match="not valid YAML"):
            load_config(cfg_path)

    def test_non_mapping_top_level(self, cfg_path: Path) -> None:
        cfg_path.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="top level must be a mapping"):
            load_config(cfg_path)

    def test_non_mapping_section(self, cfg_path: Path) -> None:
        cfg_path.write_text("app: 5\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="section 'app' must be a mapping"):
            load_config(cfg_path)

    def test_bad_log_level(self, cfg_path: Path) -> None:
        cfg_path.write_text("app:\n  log_level: LOUD\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="app.log_level"):
            load_config(cfg_path)

    @pytest.mark.parametrize("bad", ["-1", "'zero'", "true", "1.5"])
    def test_bad_camera_values(self, cfg_path: Path, bad: str) -> None:
        cfg_path.write_text(f"camera:\n  device_index: {bad}\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="camera.device_index"):
            load_config(cfg_path)

    def test_unserializable_value_raises_on_save(self, cfg_path: Path) -> None:
        config = load_config(cfg_path, create_if_missing=False)
        config.set("app", "bad", object())
        with pytest.raises(ConfigError, match="cannot be written as YAML"):
            config.save()
