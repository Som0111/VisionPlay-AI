"""Unit tests for visionplay.vision.inference.mediapipe_backend.

The configuration/guard tests need no model and run everywhere. The tests that
actually build a MediaPipe graph and run inference are marked ``integration``
and use the ``*_landmarker_model`` fixtures, which download the real models
once per session (and skip if the network is unavailable).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from visionplay.vision.inference.backend_base import InferenceBackend, InferenceError
from visionplay.vision.inference.device import DeviceConfig
from visionplay.vision.inference.mediapipe_backend import MediaPipeBackend, MediaPipeTask
from visionplay.vision.inference.results import (
    FaceLandmarkResult,
    HandLandmarkResult,
    PoseLandmarkResult,
)
from visionplay.vision.pipeline.frame_types import ColorFormat, Frame

DUMMY_MODEL = Path("nonexistent") / "hand_landmarker.task"

ALL_TASKS = list(MediaPipeTask)


def make_frame(color_format: ColorFormat = ColorFormat.BGR) -> Frame:
    image = np.zeros((90, 120, 3), dtype=np.uint8)
    return Frame.from_image(frame_id=0, timestamp=0.0, image=image, color_format=color_format)


def make_frame_at(timestamp: float) -> Frame:
    image = np.zeros((90, 120, 3), dtype=np.uint8)
    return Frame.from_image(frame_id=0, timestamp=timestamp, image=image)


class TestConfiguration:
    def test_is_inference_backend(self) -> None:
        assert isinstance(
            MediaPipeBackend(MediaPipeTask.HAND_LANDMARKS, DUMMY_MODEL), InferenceBackend
        )

    @pytest.mark.parametrize(
        ("task", "expected"),
        [
            (MediaPipeTask.HAND_LANDMARKS, "mediapipe.hands"),
            (MediaPipeTask.POSE_LANDMARKS, "mediapipe.pose"),
            (MediaPipeTask.FACE_LANDMARKS, "mediapipe.face"),
        ],
    )
    def test_name_per_task(self, task: MediaPipeTask, expected: str) -> None:
        assert MediaPipeBackend(task, DUMMY_MODEL).name == expected

    def test_default_device_is_cpu(self) -> None:
        backend = MediaPipeBackend(MediaPipeTask.HAND_LANDMARKS, DUMMY_MODEL)
        assert backend.device == DeviceConfig.cpu()

    def test_cpu_device_maps_to_cpu_delegate(self) -> None:
        backend = MediaPipeBackend(
            MediaPipeTask.HAND_LANDMARKS, DUMMY_MODEL, device=DeviceConfig.cpu()
        )
        assert backend.delegate == "CPU"

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_task_property(self, task: MediaPipeTask) -> None:
        assert MediaPipeBackend(task, DUMMY_MODEL).task is task

    def test_model_path_property(self) -> None:
        assert MediaPipeBackend(MediaPipeTask.HAND_LANDMARKS, DUMMY_MODEL).model_path == DUMMY_MODEL


class TestLoadGuards:
    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_not_loaded_initially(self, task: MediaPipeTask) -> None:
        assert not MediaPipeBackend(task, DUMMY_MODEL).is_loaded()

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_infer_before_load_raises(self, task: MediaPipeTask) -> None:
        backend = MediaPipeBackend(task, DUMMY_MODEL)
        with pytest.raises(InferenceError, match=f"mediapipe.{task.value}.*not loaded"):
            backend.infer(make_frame())

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_load_missing_model_file_raises(self, task: MediaPipeTask, tmp_path: Path) -> None:
        backend = MediaPipeBackend(task, tmp_path / "absent.task")
        with pytest.raises(InferenceError, match="not found"):
            backend.load()

    def test_unload_idempotent_when_never_loaded(self) -> None:
        backend = MediaPipeBackend(MediaPipeTask.HAND_LANDMARKS, DUMMY_MODEL)
        backend.unload()
        backend.unload()
        assert not backend.is_loaded()


class TestVideoModeTimestamps:
    """``_next_timestamp_ms`` — the monotonic tick VIDEO mode requires.

    Pure arithmetic, no MediaPipe graph needed; the real graph's rejection
    of a non-increasing timestamp is covered by
    ``TestRealInference.test_infer_tolerates_non_increasing_capture_timestamps``.
    """

    def test_first_call_uses_capture_time_in_milliseconds(self) -> None:
        backend = MediaPipeBackend(MediaPipeTask.HAND_LANDMARKS, DUMMY_MODEL)
        assert backend._next_timestamp_ms(1.5) == 1500

    def test_increasing_capture_times_pass_through(self) -> None:
        backend = MediaPipeBackend(MediaPipeTask.HAND_LANDMARKS, DUMMY_MODEL)
        backend._next_timestamp_ms(0.0)
        assert backend._next_timestamp_ms(0.05) == 50

    def test_duplicate_timestamp_is_bumped_by_one_millisecond(self) -> None:
        backend = MediaPipeBackend(MediaPipeTask.HAND_LANDMARKS, DUMMY_MODEL)
        first = backend._next_timestamp_ms(2.0)
        second = backend._next_timestamp_ms(2.0)
        assert second == first + 1

    def test_out_of_order_timestamp_is_bumped_past_the_last_value(self) -> None:
        backend = MediaPipeBackend(MediaPipeTask.HAND_LANDMARKS, DUMMY_MODEL)
        first = backend._next_timestamp_ms(5.0)
        second = backend._next_timestamp_ms(1.0)  # earlier than the first call
        assert second == first + 1

    def test_load_resets_the_timestamp_sequence(self, tmp_path: Path) -> None:
        backend = MediaPipeBackend(MediaPipeTask.HAND_LANDMARKS, tmp_path / "absent.task")
        backend._next_timestamp_ms(10.0)
        with pytest.raises(InferenceError):
            backend.load()  # fails (missing model file) but still resets state
        assert backend._next_timestamp_ms(0.0) == 0


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

    def test_infer_tolerates_non_increasing_capture_timestamps(
        self, hand_landmarker_model: Path
    ) -> None:
        # A real VIDEO-mode graph rejects a non-increasing timestamp outright;
        # this proves the backend's own monotonic guard keeps every call
        # (including out-of-order ones) succeeding instead of raising.
        backend = MediaPipeBackend(MediaPipeTask.HAND_LANDMARKS, hand_landmarker_model)
        backend.load()
        try:
            for timestamp in (0.0, 0.03, 0.03, 0.01, 0.06):
                result = backend.infer(make_frame_at(timestamp))
                assert isinstance(result, HandLandmarkResult)
        finally:
            backend.unload()


@pytest.mark.integration
class TestRealPoseInference:
    def test_load_and_unload(self, pose_landmarker_model: Path) -> None:
        backend = MediaPipeBackend(MediaPipeTask.POSE_LANDMARKS, pose_landmarker_model)
        backend.load()
        assert backend.is_loaded()
        backend.unload()
        assert not backend.is_loaded()

    def test_blank_frame_returns_empty_result(self, pose_landmarker_model: Path) -> None:
        backend = MediaPipeBackend(MediaPipeTask.POSE_LANDMARKS, pose_landmarker_model)
        with backend:
            result = backend.infer(make_frame())
        assert isinstance(result, PoseLandmarkResult)
        assert result.is_empty


@pytest.mark.integration
class TestRealFaceInference:
    def test_load_and_unload(self, face_landmarker_model: Path) -> None:
        backend = MediaPipeBackend(MediaPipeTask.FACE_LANDMARKS, face_landmarker_model)
        backend.load()
        assert backend.is_loaded()
        backend.unload()
        assert not backend.is_loaded()

    def test_blank_frame_returns_empty_result(self, face_landmarker_model: Path) -> None:
        backend = MediaPipeBackend(MediaPipeTask.FACE_LANDMARKS, face_landmarker_model)
        with backend:
            result = backend.infer(make_frame())
        assert isinstance(result, FaceLandmarkResult)
        assert result.is_empty
