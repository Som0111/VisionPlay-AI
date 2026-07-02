"""Unit tests for visionplay.vision.inference.onnx_backend."""

from pathlib import Path

import numpy as np
import pytest

from visionplay.vision.inference.backend_base import InferenceBackend, InferenceError
from visionplay.vision.inference.device import DeviceConfig
from visionplay.vision.inference.model_registry import ModelFormat, ModelSpec
from visionplay.vision.inference.onnx_backend import ONNXBackend
from visionplay.vision.pipeline.frame_types import Frame

FAKE_SHA256 = "a" * 64


def make_spec(model_format: ModelFormat = ModelFormat.ONNX) -> ModelSpec:
    return ModelSpec(
        model_id="yolo_nano",
        format=model_format,
        url="https://example.invalid/yolo_nano.onnx",
        sha256=FAKE_SHA256,
        filename="yolo_nano.onnx",
    )


def make_frame() -> Frame:
    return Frame.from_image(frame_id=0, timestamp=0.0, image=np.zeros((4, 4, 3), dtype=np.uint8))


class TestConfiguration:
    def test_is_inference_backend(self, tmp_path: Path) -> None:
        backend = ONNXBackend(make_spec(), tmp_path / "yolo_nano.onnx")
        assert isinstance(backend, InferenceBackend)

    def test_name_derived_from_model_id(self, tmp_path: Path) -> None:
        assert ONNXBackend(make_spec(), tmp_path / "m.onnx").name == "onnx.yolo_nano"

    def test_default_device_is_cpu(self, tmp_path: Path) -> None:
        assert ONNXBackend(make_spec(), tmp_path / "m.onnx").device == DeviceConfig.cpu()

    def test_default_providers_are_cpu_only(self, tmp_path: Path) -> None:
        backend = ONNXBackend(make_spec(), tmp_path / "m.onnx")
        assert backend.providers == ("CPUExecutionProvider",)

    def test_explicit_provider_order_is_preserved(self, tmp_path: Path) -> None:
        # The GPU-later story: a preferred provider ahead of the CPU fallback.
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        backend = ONNXBackend(make_spec(), tmp_path / "m.onnx", providers=providers)
        assert backend.providers == ("DmlExecutionProvider", "CPUExecutionProvider")

    def test_empty_provider_list_raises(self, tmp_path: Path) -> None:
        with pytest.raises(InferenceError, match="at least one execution provider"):
            ONNXBackend(make_spec(), tmp_path / "m.onnx", providers=[])

    def test_non_onnx_spec_rejected_at_construction(self, tmp_path: Path) -> None:
        with pytest.raises(InferenceError, match="'tflite'"):
            ONNXBackend(make_spec(ModelFormat.TFLITE), tmp_path / "m.tflite")

    def test_model_path_property(self, tmp_path: Path) -> None:
        path = tmp_path / "yolo_nano.onnx"
        assert ONNXBackend(make_spec(), path).model_path == path


class TestLifecycle:
    def test_load_with_existing_model_file(self, tmp_path: Path) -> None:
        path = tmp_path / "yolo_nano.onnx"
        path.write_bytes(b"fake model bytes")
        backend = ONNXBackend(make_spec(), path)
        backend.load()
        assert backend.is_loaded()
        backend.unload()
        assert not backend.is_loaded()

    def test_load_with_missing_model_file_raises(self, tmp_path: Path) -> None:
        backend = ONNXBackend(make_spec(), tmp_path / "missing.onnx")
        with pytest.raises(InferenceError, match="not found"):
            backend.load()
        assert not backend.is_loaded()

    def test_unload_idempotent_and_safe_when_never_loaded(self, tmp_path: Path) -> None:
        backend = ONNXBackend(make_spec(), tmp_path / "m.onnx")
        backend.unload()
        backend.unload()
        assert not backend.is_loaded()

    def test_context_manager_loads_and_unloads(self, tmp_path: Path) -> None:
        path = tmp_path / "yolo_nano.onnx"
        path.write_bytes(b"fake model bytes")
        backend = ONNXBackend(make_spec(), path)
        with backend:
            assert backend.is_loaded()
        assert not backend.is_loaded()


class TestInferStub:
    def test_infer_before_load_raises_inference_error(self, tmp_path: Path) -> None:
        backend = ONNXBackend(make_spec(), tmp_path / "m.onnx")
        with pytest.raises(InferenceError, match="onnx.yolo_nano.*not loaded"):
            backend.infer(make_frame())

    def test_infer_after_load_is_not_implemented_until_phase_2(self, tmp_path: Path) -> None:
        path = tmp_path / "yolo_nano.onnx"
        path.write_bytes(b"fake model bytes")
        backend = ONNXBackend(make_spec(), path)
        backend.load()
        with pytest.raises(NotImplementedError, match="Phase 2"):
            backend.infer(make_frame())
