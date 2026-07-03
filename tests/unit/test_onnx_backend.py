"""Unit tests for visionplay.vision.inference.onnx_backend.

Real inference is exercised against the committed tiny ONNX fixtures (see
``tests/fixtures/onnx/``) run through onnxruntime — no network, no camera.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from visionplay.vision.inference.backend_base import InferenceBackend, InferenceError
from visionplay.vision.inference.device import DeviceConfig
from visionplay.vision.inference.model_registry import ModelFormat, ModelSpec
from visionplay.vision.inference.onnx_backend import ONNXBackend
from visionplay.vision.inference.results import TensorOutput
from visionplay.vision.pipeline.frame_types import Frame

FAKE_SHA256 = "a" * 64


def make_spec(
    model_format: ModelFormat = ModelFormat.ONNX, model_id: str = "yolo_nano"
) -> ModelSpec:
    return ModelSpec(
        model_id=model_id,
        format=model_format,
        url="https://example.invalid/model.onnx",
        sha256=FAKE_SHA256,
        filename="model.onnx",
    )


def make_frame(height: int = 4, width: int = 5) -> Frame:
    image = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)
    return Frame.from_image(frame_id=0, timestamp=0.0, image=image)


class TestConfiguration:
    def test_is_inference_backend(self, tmp_path: Path) -> None:
        backend = ONNXBackend(make_spec(), tmp_path / "model.onnx")
        assert isinstance(backend, InferenceBackend)

    def test_name_derived_from_model_id(self, tmp_path: Path) -> None:
        assert ONNXBackend(make_spec(), tmp_path / "m.onnx").name == "onnx.yolo_nano"

    def test_default_device_is_cpu(self, tmp_path: Path) -> None:
        assert ONNXBackend(make_spec(), tmp_path / "m.onnx").device == DeviceConfig.cpu()

    def test_default_providers_are_cpu_only(self, tmp_path: Path) -> None:
        backend = ONNXBackend(make_spec(), tmp_path / "m.onnx")
        assert backend.providers == ("CPUExecutionProvider",)

    def test_explicit_provider_order_is_preserved(self, tmp_path: Path) -> None:
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
    def test_load_real_model(self, tiny_identity_onnx: Path) -> None:
        backend = ONNXBackend(make_spec(), tiny_identity_onnx)
        assert not backend.is_loaded()
        backend.load()
        assert backend.is_loaded()
        backend.unload()
        assert not backend.is_loaded()

    def test_load_is_idempotent(self, tiny_identity_onnx: Path) -> None:
        backend = ONNXBackend(make_spec(), tiny_identity_onnx)
        backend.load()
        backend.load()  # no raise, still loaded
        assert backend.is_loaded()

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        backend = ONNXBackend(make_spec(), tmp_path / "missing.onnx")
        with pytest.raises(InferenceError, match="not found"):
            backend.load()
        assert not backend.is_loaded()

    def test_load_invalid_model_bytes_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "garbage.onnx"
        bad.write_bytes(b"not a real onnx model")
        backend = ONNXBackend(make_spec(), bad)
        with pytest.raises(InferenceError, match="Failed to load ONNX model"):
            backend.load()
        assert not backend.is_loaded()

    def test_unload_idempotent_and_safe_when_never_loaded(self, tmp_path: Path) -> None:
        backend = ONNXBackend(make_spec(), tmp_path / "m.onnx")
        backend.unload()
        backend.unload()
        assert not backend.is_loaded()

    def test_context_manager_loads_and_unloads(self, tiny_identity_onnx: Path) -> None:
        backend = ONNXBackend(make_spec(), tiny_identity_onnx)
        with backend:
            assert backend.is_loaded()
        assert not backend.is_loaded()


class TestInfer:
    def test_infer_returns_tensor_output(self, tiny_identity_onnx: Path) -> None:
        backend = ONNXBackend(make_spec(), tiny_identity_onnx)
        backend.load()
        frame = make_frame()
        output = backend.infer(frame)
        assert isinstance(output, TensorOutput)
        # The fixture is an Identity model: output == input, cast to float32.
        assert np.allclose(output.first(), frame.image.astype(np.float32))
        assert output.names() == ("output",)

    def test_infer_before_load_raises(self, tiny_identity_onnx: Path) -> None:
        backend = ONNXBackend(make_spec(), tiny_identity_onnx)
        with pytest.raises(InferenceError, match="onnx.yolo_nano.*not loaded"):
            backend.infer(make_frame())

    def test_infer_after_unload_raises(self, tiny_identity_onnx: Path) -> None:
        backend = ONNXBackend(make_spec(), tiny_identity_onnx)
        backend.load()
        backend.unload()
        with pytest.raises(InferenceError, match="not loaded"):
            backend.infer(make_frame())

    def test_multi_input_model_rejected(self, tiny_two_input_onnx: Path) -> None:
        backend = ONNXBackend(make_spec(model_id="two_input"), tiny_two_input_onnx)
        backend.load()
        with pytest.raises(InferenceError, match="single-input models only"):
            backend.infer(make_frame())
