"""Unit tests for visionplay.vision.inference.backend_defaults."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from visionplay.vision.inference.backend_defaults import (
    device_from_config,
    models_dir_from_config,
    register_builtin_mediapipe_backends,
    register_onnx_backend,
)
from visionplay.vision.inference.backend_manager import BackendManager
from visionplay.vision.inference.device import DeviceConfig, DeviceType
from visionplay.vision.inference.model_registry import (
    ModelDownloader,
    ModelFormat,
    ModelRegistry,
    ModelSpec,
)

MODEL_BYTES = b"fake onnx model bytes"
MODEL_SHA256 = hashlib.sha256(MODEL_BYTES).hexdigest()


def make_spec() -> ModelSpec:
    return ModelSpec(
        model_id="yolo_nano",
        format=ModelFormat.ONNX,
        url="https://example.invalid/yolo_nano.onnx",
        sha256=MODEL_SHA256,
        filename="yolo_nano.onnx",
    )


class FakeDownloader(ModelDownloader):
    def __init__(self) -> None:
        self.calls = 0

    def download(self, url: str, destination: Path) -> None:
        self.calls += 1
        destination.write_bytes(MODEL_BYTES)


class TestMediaPipeRegistration:
    def test_registers_all_three_landmark_backends(self) -> None:
        manager = BackendManager()
        register_builtin_mediapipe_backends(manager)
        assert set(manager.registered_names()) == {
            "mediapipe.hands",
            "mediapipe.pose",
            "mediapipe.face",
        }

    def test_registration_constructs_nothing(self) -> None:
        manager = BackendManager()
        register_builtin_mediapipe_backends(manager)
        assert manager.loaded_names() == ()

    def test_availability_tracks_the_mediapipe_probe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        manager = BackendManager()
        register_builtin_mediapipe_backends(manager)
        monkeypatch.setattr(
            "visionplay.vision.inference.backend_defaults._module_importable",
            lambda module_name: module_name == "mediapipe",
        )
        assert manager.is_available("mediapipe.hands")

    def test_unavailable_when_runtime_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        manager = BackendManager()
        register_builtin_mediapipe_backends(manager)
        monkeypatch.setattr(
            "visionplay.vision.inference.backend_defaults._module_importable",
            lambda module_name: False,
        )
        assert not manager.is_available("mediapipe.hands")

    def test_idempotent_registration(self) -> None:
        manager = BackendManager()
        register_builtin_mediapipe_backends(manager)
        register_builtin_mediapipe_backends(manager)  # same registrations -> no-op
        assert len(manager.registered_names()) == 3


class TestOnnxRegistration:
    def test_registers_under_model_id_name(self, tmp_path: Path) -> None:
        manager = BackendManager()
        registry = ModelRegistry(tmp_path, FakeDownloader())
        register_onnx_backend(manager, registry, make_spec())
        assert manager.is_registered("onnx.yolo_nano")

    def test_spec_is_registered_in_model_registry(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path, FakeDownloader())
        register_onnx_backend(BackendManager(), registry, make_spec())
        assert registry.get("yolo_nano") == make_spec()

    def test_available_when_runtime_present_and_model_registered(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = BackendManager()
        registry = ModelRegistry(tmp_path, FakeDownloader())
        register_onnx_backend(manager, registry, make_spec())
        monkeypatch.setattr(
            "visionplay.vision.inference.backend_defaults._module_importable",
            lambda module_name: module_name == "onnxruntime",
        )
        assert manager.is_available("onnx.yolo_nano")

    def test_unavailable_when_runtime_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = BackendManager()
        registry = ModelRegistry(tmp_path, FakeDownloader())
        register_onnx_backend(manager, registry, make_spec())
        monkeypatch.setattr(
            "visionplay.vision.inference.backend_defaults._module_importable",
            lambda module_name: False,
        )
        assert not manager.is_available("onnx.yolo_nano")

    def test_probe_does_not_download(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        manager = BackendManager()
        downloader = FakeDownloader()
        registry = ModelRegistry(tmp_path, downloader)
        register_onnx_backend(manager, registry, make_spec())
        monkeypatch.setattr(
            "visionplay.vision.inference.backend_defaults._module_importable",
            lambda module_name: True,
        )
        manager.is_available("onnx.yolo_nano")
        assert downloader.calls == 0  # capability check never fetches bytes


class TestDeviceFromConfig:
    def test_defaults_to_cpu_when_absent(self) -> None:
        assert device_from_config({}) == DeviceConfig.cpu()

    def test_reads_cpu_type(self) -> None:
        assert device_from_config({"device": {"type": "cpu"}}).type is DeviceType.CPU

    def test_non_mapping_device_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a mapping"):
            device_from_config({"device": "cpu"})

    def test_unknown_device_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown device type"):
            device_from_config({"device": {"type": "quantum"}})


class TestModelsDirFromConfig:
    def test_uses_default_when_absent(self, tmp_path: Path) -> None:
        assert models_dir_from_config({}, tmp_path) == tmp_path

    def test_uses_default_when_null(self, tmp_path: Path) -> None:
        assert models_dir_from_config({"model_cache_dir": None}, tmp_path) == tmp_path

    def test_override_string_is_used(self, tmp_path: Path) -> None:
        override = tmp_path / "elsewhere"
        assert models_dir_from_config({"model_cache_dir": str(override)}, tmp_path) == override

    def test_non_string_override_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="string path or null"):
            models_dir_from_config({"model_cache_dir": 5}, tmp_path)
