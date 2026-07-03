"""Unit tests for visionplay.vision.inference.mediapipe_backend.

The configuration/guard tests need no model and run everywhere. The tests that
actually build a MediaPipe graph and run inference are marked ``integration``
and use the ``hand_landmarker_model`` fixture, which downloads the real model
once per session (and skips if the network is unavailable).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from visionplay.vision.inference.backend_base import InferenceBackend, InferenceError
from visionplay.vision.inference.device import DeviceConfig
from visionplay.vision.inference.mediapipe_backend import MediaPipeBackend, MediaPipeTask
from visionplay.vision.inference.results import HandLandmarkResult
from visionplay.vision.pipeline.frame_types import ColorFormat, Frame

DUMMY_MODEL = Path("nonexistent") / "hand_landmarker.task"


def make_frame(color_format: ColorFormat = ColorFormat.BGR) -> Frame:
    image = np.zeros((90, 120, 3), dtype=np.uint8)
    return Frame.from_image(frame_id=0, timestamp=0.0, image=image, color_format=color_format)


class TestConfiguration:
    def test_is_inference_backend(self) -> None:
        assert isinstance(
            MediaPipeBackend(MediaPipeTask.HAND_LANDMARKS, DUMMY_MODEL), InferenceBackend
        )

    def test_name_per_task(self) -> None:
        backend = MediaPipeBackend(MediaPipeTask.HAND_LANDMARKS, DUMMY_MODEL)
        assert backend.name == "mediapipe.hands"

    def test_default_device_is_cpu(self) -> None:
        backend = MediaPipeBackend(MediaPipeTask.HAND_LANDMARKS, DUMMY_MODEL)
        assert backend.device == DeviceConfig.cpu()

    def test_cpu_device_maps_to_cpu_delegate(self) -> None:
        backend = MediaPipeBackend(
            MediaPipeTask.HAND_LANDMARKS, DUMMY_MODEL, device=DeviceConfig.cpu()
        )
        assert backend.delegate == "CPU"

    def test_task_property(self) -> None:
        backend = MediaPipeBackend(MediaPipeTask.HAND_LANDMARKS, DUMMY_MODEL)
        assert backend.task is MediaPipeTask.HAND_LANDMARKS

    def test_model_path_property(self) -> None:
        assert MediaPipeBackend(MediaPipeTask.HAND_LANDMARKS, DUMMY_MODEL).model_path == DUMMY_MODEL


class TestLoadGuards:
    def test_not_loaded_initially(self) -> None:
        assert not MediaPipeBackend(MediaPipeTask.HAND_LANDMARKS, DUMMY_MODEL).is_loaded()

    def test_infer_before_load_raises(self) -> None:
        backend = MediaPipeBackend(MediaPipeTask.HAND_LANDMARKS, DUMMY_MODEL)
        with pytest.raises(InferenceError, match="mediapipe.hands.*not loaded"):
            backend.infer(make_frame())

    @pytest.mark.parametrize("task", [MediaPipeTask.POSE_LANDMARKS, MediaPipeTask.FACE_LANDMARKS])
    def test_unimplemented_task_load_raises(self, task: MediaPipeTask) -> None:
        backend = MediaPipeBackend(task, DUMMY_MODEL)
        with pytest.raises(InferenceError, match="not implemented"):
            backend.load()

    def test_load_missing_model_file_raises(self, tmp_path: Path) -> None:
        backend = MediaPipeBackend(MediaPipeTask.HAND_LANDMARKS, tmp_path / "absent.task")
        with pytest.raises(InferenceError, match="not found"):
            backend.load()

    def test_unload_idempotent_when_never_loaded(self) -> None:
        backend = MediaPipeBackend(MediaPipeTask.HAND_LANDMARKS, DUMMY_MODEL)
        backend.unload()
        backend.unload()
        assert not backend.is_loaded()


@pytest.mark.integration
class TestRealInference:
    def test_load_and_unload(self, hand_landmarker_model: Path) -> None:
        backend = MediaPipeBackend(MediaPipeTask.HAND_LANDMARKS, hand_landmarker_model)
        backend.load()
        assert backend.is_loaded()
        backend.unload()
        assert not backend.is_loaded()

    def test_context_manager(self, hand_landmarker_model: Path) -> None:
        backend = MediaPipeBackend(MediaPipeTask.HAND_LANDMARKS, hand_landmarker_model)
        with backend:
            assert backend.is_loaded()
        assert not backend.is_loaded()

    def test_blank_frame_returns_empty_result(self, hand_landmarker_model: Path) -> None:
        backend = MediaPipeBackend(MediaPipeTask.HAND_LANDMARKS, hand_landmarker_model)
        backend.load()
        try:
            result = backend.infer(make_frame())
        finally:
            backend.unload()
        assert isinstance(result, HandLandmarkResult)
        assert result.is_empty

    @pytest.mark.parametrize("color_format", [ColorFormat.BGR, ColorFormat.RGB, ColorFormat.GRAY])
    def test_infer_handles_every_color_format(
        self, hand_landmarker_model: Path, color_format: ColorFormat
    ) -> None:
        # GRAY frames are single-channel; build one that matches the format.
        if color_format is ColorFormat.GRAY:
            image = np.zeros((90, 120), dtype=np.uint8)
            frame = Frame.from_image(0, 0.0, image, color_format=ColorFormat.GRAY)
        else:
            frame = make_frame(color_format)
        backend = MediaPipeBackend(MediaPipeTask.HAND_LANDMARKS, hand_landmarker_model)
        backend.load()
        try:
            result = backend.infer(frame)
        finally:
            backend.unload()
        assert isinstance(result, HandLandmarkResult)
