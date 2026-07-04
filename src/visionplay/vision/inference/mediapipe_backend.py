"""MediaPipe landmark backend (hand/pose/face).

``load()`` builds a real MediaPipe Tasks graph; ``infer()`` runs it on a frame
and returns standardized landmark output
(:class:`~visionplay.vision.inference.results.HandLandmarkResult`,
:class:`~visionplay.vision.inference.results.PoseLandmarkResult`, or
:class:`~visionplay.vision.inference.results.FaceLandmarkResult`) — no
MediaPipe types cross into plugin code. All three landmark tasks are
implemented; which one an instance runs is fixed by the
:class:`MediaPipeTask` it is constructed with.

The model asset (a ``.task`` bundle) is not bundled or downloaded by this
class — the pipeline resolves it through the
:class:`~visionplay.vision.inference.model_registry.ModelRegistry` and passes
a verified local ``model_path`` to the constructor, symmetric with
:class:`~visionplay.vision.inference.onnx_backend.ONNXBackend`.

Device handling (``docs/architecture.md`` §5): MediaPipe's Tasks API takes a
``delegate`` (``CPU``/``GPU``) per task. The :attr:`MediaPipeBackend.delegate`
property derives it from the backend's :class:`DeviceConfig` via a mapping —
v1 maps CPU only, and GPU support later adds one mapping entry without
touching the surrounding pipeline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from visionplay.vision.inference.backend_base import InferenceBackend, InferenceError
from visionplay.vision.inference.device import DeviceConfig, DeviceType
from visionplay.vision.inference.results import (
    FaceLandmarkResult,
    FaceLandmarks,
    HandLandmarkResult,
    HandLandmarks,
    LandmarkPoint,
    PoseLandmarkResult,
    PoseLandmarks,
)
from visionplay.vision.pipeline.frame_types import ColorFormat, Frame

__all__ = ["LandmarkResult", "MediaPipeBackend", "MediaPipeTask"]

#: The standardized output types a MediaPipe backend can produce, one per task.
LandmarkResult = HandLandmarkResult | PoseLandmarkResult | FaceLandmarkResult

#: DeviceType → MediaPipe Tasks delegate name. GPU support later appends an
#: entry here (e.g. ``DeviceType.GPU: "GPU"``) — an additive change only.
_DELEGATES: dict[DeviceType, str] = {
    DeviceType.CPU: "CPU",
}

#: Maximum number of hands the hand landmarker detects per frame.
_MAX_HANDS: int = 2

#: Maximum number of people the pose landmarker detects per frame. One keeps
#: per-frame CPU cost bounded; fitness-style apps track a single subject.
_MAX_POSES: int = 1

#: Maximum number of faces the face landmarker detects per frame. Two leaves
#: room for multi-face AR filters without unbounded per-frame cost.
_MAX_FACES: int = 2


class MediaPipeTask(Enum):
    """The MediaPipe landmark task a backend instance runs.

    Values are the suffix of the backend :attr:`~MediaPipeBackend.name`
    (``"mediapipe.<value>"``) — the strings app manifests reference in
    ``required_backends``.
    """

    HAND_LANDMARKS = "hands"
    POSE_LANDMARKS = "pose"
    FACE_LANDMARKS = "face"


@dataclass(frozen=True, slots=True)
class _TaskBinding:
    """How one :class:`MediaPipeTask` maps onto the MediaPipe Tasks API.

    Attributes:
        landmarker_name: Landmarker class name on ``mediapipe.tasks.python.vision``.
        options_name: Matching options class name on the same module.
        count_option: The options kwarg limiting detections per frame.
        max_count: Value passed for :attr:`count_option`.
        convert: Maps the task's native result to the standardized shape.
    """

    landmarker_name: str
    options_name: str
    count_option: str
    max_count: int
    convert: Callable[[Any], LandmarkResult]


class MediaPipeBackend(InferenceBackend):
    """Runs one MediaPipe landmark task over frames.

    One instance handles exactly one task; an app needing both hands and pose
    declares two ``required_backends`` and the pipeline owns two instances.
    """

    def __init__(
        self,
        task: MediaPipeTask,
        model_path: Path,
        device: DeviceConfig | None = None,
    ) -> None:
        """Configure the backend; nothing is loaded until :meth:`load`.

        Args:
            task: Which landmark task this instance runs.
            model_path: Verified local path to the task's ``.task`` model
                bundle (resolved by the pipeline via the model registry).
            device: Target compute device; ``None`` means CPU (v1 default).
        """
        super().__init__(device)
        self._task = task
        self._model_path = model_path
        # mediapipe landmarker once loaded; None while unloaded. Typed Any
        # because mediapipe ships no stubs (see pyproject mypy overrides).
        self._landmarker: Any = None
        # Last timestamp (ms) passed to detect_for_video; VIDEO mode requires
        # strictly increasing values per landmarker instance. Reset on load()
        # so each graph starts its own monotonic sequence.
        self._last_timestamp_ms: int | None = None

    @property
    def name(self) -> str:
        """Result key / manifest string, e.g. ``"mediapipe.pose"``."""
        return f"mediapipe.{self._task.value}"

    @property
    def task(self) -> MediaPipeTask:
        """The landmark task this instance is configured for."""
        return self._task

    @property
    def model_path(self) -> Path:
        """Local path of the ``.task`` model bundle this backend loads."""
        return self._model_path

    @property
    def delegate(self) -> str:
        """MediaPipe Tasks delegate derived from the device config (v1: ``"CPU"``)."""
        return _DELEGATES[self.device.type]

    def load(self) -> None:
        """Build the MediaPipe Tasks graph and mark the backend ready.

        ``mediapipe`` is imported here, not at module import, so constructing a
        backend and probing availability stay cheap. Idempotent: a second call
        with a graph already built is a no-op.

        Raises:
            InferenceError: If the model file is missing, ``mediapipe`` is not
                importable, or the graph cannot be created — all
                user-presentable.
        """
        if self._landmarker is not None:
            return
        self._last_timestamp_ms = None
        if not self._model_path.is_file():
            raise InferenceError(f"Model file for {self.name!r} not found at {self._model_path}")
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise InferenceError(
                f"MediaPipe is not available; cannot load backend {self.name!r}"
            ) from exc
        binding = _TASK_BINDINGS[self._task]
        try:
            delegate = getattr(mp_python.BaseOptions.Delegate, self.delegate)
            options_cls = getattr(vision, binding.options_name)
            landmarker_cls = getattr(vision, binding.landmarker_name)
            options = options_cls(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(self._model_path), delegate=delegate
                ),
                # VIDEO mode (not IMAGE) so MediaPipe tracks landmarks across
                # frames instead of re-detecting each one in isolation — the
                # single biggest lever against frame-to-frame landmark jitter
                # on a live camera stream. Requires detect_for_video() with a
                # strictly increasing timestamp per call (see infer()).
                running_mode=vision.RunningMode.VIDEO,
                **{binding.count_option: binding.max_count},
            )
            self._landmarker = landmarker_cls.create_from_options(options)
        except Exception as exc:
            raise InferenceError(
                f"Failed to load MediaPipe graph for {self.name!r}: {exc}"
            ) from exc

    def infer(self, frame: Frame) -> LandmarkResult:
        """Run landmark detection on ``frame`` and return standardized output.

        Args:
            frame: The captured frame to run inference on.

        Returns:
            The task's standardized result (:class:`HandLandmarkResult`,
            :class:`PoseLandmarkResult`, or :class:`FaceLandmarkResult`);
            empty when nothing is in view (the normal case, not an error).

        Raises:
            InferenceError: If the backend is not loaded or detection fails.
        """
        if self._landmarker is None:
            raise InferenceError(f"Backend {self.name!r} is not loaded; call load() first")
        try:
            import mediapipe as mp

            rgb = _to_rgb(frame)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = self._next_timestamp_ms(frame.timestamp)
            detection = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception as exc:
            raise InferenceError(f"MediaPipe inference failed for {self.name!r}: {exc}") from exc
        return _TASK_BINDINGS[self._task].convert(detection)

    def _next_timestamp_ms(self, capture_timestamp: float) -> int:
        """Convert a frame's capture time to a strictly increasing millisecond tick.

        VIDEO mode rejects a timestamp that does not exceed the previous one
        per landmarker instance. Real capture (``time.time()``-based) is
        already increasing; this only guards the boundary case of two frames
        landing on the same millisecond (fast capture) or out-of-order
        synthetic timestamps in tests, by bumping to one past the last value
        instead of raising.
        """
        candidate = int(capture_timestamp * 1000)
        if self._last_timestamp_ms is not None and candidate <= self._last_timestamp_ms:
            candidate = self._last_timestamp_ms + 1
        self._last_timestamp_ms = candidate
        return candidate

    def unload(self) -> None:
        """Release the MediaPipe graph. Idempotent; safe if never loaded."""
        landmarker = self._landmarker
        self._landmarker = None
        self._last_timestamp_ms = None
        if landmarker is not None:
            landmarker.close()

    def is_loaded(self) -> bool:
        """Return ``True`` while the graph exists and can run inference."""
        return self._landmarker is not None


def _to_rgb(frame: Frame) -> npt.NDArray[np.uint8]:
    """Return the frame image as a contiguous uint8 RGB array for MediaPipe.

    MediaPipe expects SRGB; capture is BGR. Conversion is done with NumPy so
    this module needs no OpenCV import.
    """
    image = frame.image
    if frame.color_format is ColorFormat.RGB:
        rgb = image
    elif frame.color_format is ColorFormat.GRAY:
        rgb = np.stack([image, image, image], axis=-1)
    else:  # BGR — OpenCV capture native order
        rgb = image[:, :, ::-1]
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def _to_points(landmarks: Any) -> tuple[LandmarkPoint, ...]:
    """Map one MediaPipe normalized-landmark list to :class:`LandmarkPoint`s."""
    return tuple(LandmarkPoint(x=lm.x, y=lm.y, z=lm.z) for lm in landmarks)


def _to_hand_result(detection: Any) -> HandLandmarkResult:
    """Map a MediaPipe ``HandLandmarkerResult`` to the standardized shape."""
    hands: list[HandLandmarks] = []
    hand_landmarks = detection.hand_landmarks
    handedness = detection.handedness
    for index, landmarks in enumerate(hand_landmarks):
        label = "Unknown"
        score = 0.0
        if index < len(handedness) and handedness[index]:
            category = handedness[index][0]
            label = category.category_name
            score = float(category.score)
        hands.append(HandLandmarks(points=_to_points(landmarks), handedness=label, score=score))
    return HandLandmarkResult(hands=tuple(hands))


def _to_pose_result(detection: Any) -> PoseLandmarkResult:
    """Map a MediaPipe ``PoseLandmarkerResult`` to the standardized shape."""
    poses = tuple(
        PoseLandmarks(points=_to_points(landmarks)) for landmarks in detection.pose_landmarks
    )
    return PoseLandmarkResult(poses=poses)


def _to_face_result(detection: Any) -> FaceLandmarkResult:
    """Map a MediaPipe ``FaceLandmarkerResult`` to the standardized shape."""
    faces = tuple(
        FaceLandmarks(points=_to_points(landmarks)) for landmarks in detection.face_landmarks
    )
    return FaceLandmarkResult(faces=faces)


#: MediaPipeTask → Tasks-API binding. Adding a landmark task is one entry here
#: plus its standardized result type — the backend class itself is untouched.
_TASK_BINDINGS: dict[MediaPipeTask, _TaskBinding] = {
    MediaPipeTask.HAND_LANDMARKS: _TaskBinding(
        landmarker_name="HandLandmarker",
        options_name="HandLandmarkerOptions",
        count_option="num_hands",
        max_count=_MAX_HANDS,
        convert=_to_hand_result,
    ),
    MediaPipeTask.POSE_LANDMARKS: _TaskBinding(
        landmarker_name="PoseLandmarker",
        options_name="PoseLandmarkerOptions",
        count_option="num_poses",
        max_count=_MAX_POSES,
        convert=_to_pose_result,
    ),
    MediaPipeTask.FACE_LANDMARKS: _TaskBinding(
        landmarker_name="FaceLandmarker",
        options_name="FaceLandmarkerOptions",
        count_option="num_faces",
        max_count=_MAX_FACES,
        convert=_to_face_result,
    ),
}
