"""Unit tests for visionplay.vision.inference.device."""

import dataclasses

import pytest

from visionplay.vision.inference.device import DeviceConfig, DeviceType


class TestDefaults:
    def test_default_is_cpu(self) -> None:
        assert DeviceConfig().type is DeviceType.CPU

    def test_cpu_factory(self) -> None:
        assert DeviceConfig.cpu() == DeviceConfig(type=DeviceType.CPU)


class TestFromMapping:
    def test_explicit_cpu(self) -> None:
        assert DeviceConfig.from_mapping({"type": "cpu"}) == DeviceConfig.cpu()

    def test_case_insensitive(self) -> None:
        assert DeviceConfig.from_mapping({"type": "CPU"}) == DeviceConfig.cpu()

    def test_missing_type_defaults_to_cpu(self) -> None:
        assert DeviceConfig.from_mapping({}) == DeviceConfig.cpu()

    def test_unknown_type_raises_with_supported_list(self) -> None:
        with pytest.raises(ValueError, match=r"'cuda'.*supported: cpu"):
            DeviceConfig.from_mapping({"type": "cuda"})


class TestValueObject:
    def test_immutable(self) -> None:
        config = DeviceConfig.cpu()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.type = DeviceType.CPU  # type: ignore[misc]

    def test_equality_and_hash(self) -> None:
        assert DeviceConfig.cpu() == DeviceConfig.cpu()
        assert hash(DeviceConfig.cpu()) == hash(DeviceConfig.cpu())
