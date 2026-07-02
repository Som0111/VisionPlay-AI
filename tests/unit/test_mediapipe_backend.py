"""Unit tests for visionplay.vision.inference.mediapipe_backend."""

import numpy as np
import pytest

from visionplay.vision.inference.backend_base import InferenceBackend, InferenceError
from visionplay.vision.inference.device import DeviceConfig
from visionplay.vision.inference.mediapipe_backend import MediaPipeBackend, MediaPipeTask
from visionplay.vision.pipeline.frame_types import Frame


def make_frame() -> Frame:
    return Frame.from_image(frame_id=0, timestamp=0.0, image=np.zeros((4, 4, 3), dtype=np.uint8))


class TestConfiguration:
    @pytest.mark.parametrize(
        ("task", "expected_name"),
        [
            (MediaPipeTask.HAND_LANDMARKS, "mediapipe.hands"),
            (MediaPipeTask.POSE_LANDMARKS, "mediapipe.pose"),
            (MediaPipeTask.FACE_LANDMARKS, "mediapipe.face"),
        ],
    )
    def test_name_per_task(self, task: MediaPipeTask, expected_name: str) -> None:
        assert MediaPipeBackend(task).name == expected_name

    def test_is_inference_backend(self) -> None:
        assert isinstance(MediaPipeBackend(MediaPipeTask.POSE_LANDMARKS), InferenceBackend)

    def test_default_device_is_cpu(self) -> None:
        assert MediaPipeBackend(MediaPipeTask.POSE_LANDMARKS).device == DeviceConfig.cpu()

    def test_cpu_device_maps_to_cpu_delegate(self) -> None:
        backend = MediaPipeBackend(MediaPipeTask.HAND_LANDMARKS, device=DeviceConfig.cpu())
        assert backend.delegate == "CPU"

    def test_task_property(self) -> None:
        assert MediaPipeBackend(MediaPipeTask.FACE_LANDMARKS).task is MediaPipeTask.FACE_LANDMARKS


class TestLifecycle:
    def test_load_and_unload(self) -> None:
        backend = MediaPipeBackend(MediaPipeTask.POSE_LANDMARKS)
        assert not backend.is_loaded()
        backend.load()
        assert backend.is_loaded()
        backend.unload()
        assert not backend.is_loaded()

    def test_unload_idempotent_and_safe_when_never_loaded(self) -> None:
        backend = MediaPipeBackend(MediaPipeTask.POSE_LANDMARKS)
        backend.unload()
        backend.unload()
        assert not backend.is_loaded()

    def test_context_manager_loads_and_unloads(self) -> None:
        backend = MediaPipeBackend(MediaPipeTask.HAND_LANDMARKS)
        with backend:
            assert backend.is_loaded()
        assert not backend.is_loaded()


class TestInferStub:
    def test_infer_before_load_raises_inference_error(self) -> None:
        backend = MediaPipeBackend(MediaPipeTask.POSE_LANDMARKS)
        with pytest.raises(InferenceError, match="mediapipe.pose.*not loaded"):
            backend.infer(make_frame())

    def test_infer_after_load_is_not_implemented_until_phase_2(self) -> None:
        backend = MediaPipeBackend(MediaPipeTask.POSE_LANDMARKS)
        backend.load()
        with pytest.raises(NotImplementedError, match="Phase 2"):
            backend.infer(make_frame())
