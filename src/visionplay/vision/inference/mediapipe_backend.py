"""MediaPipe landmark backend (hand/pose/face).

``load()`` builds a real MediaPipe Tasks graph; ``infer()`` runs it on a frame
and returns standardized
:class:`~visionplay.vision.inference.results.HandLandmarkResult` output — no
MediaPipe types cross into plugin code. v1 implements the **hand** landmark
task (``MediaPipeTask.HAND_LANDMARKS``); pose and face are reserved for a
later phase and their :meth:`MediaPipeBackend.load` raises until then.

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

from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from visionplay.vision.inference.backend_base import InferenceBackend, InferenceError
from visionplay.vision.inference.device import DeviceConfig, DeviceType
from visionplay.vision.inference.results import HandLandmarkResult, HandLandmarks, LandmarkPoint
from visionplay.vision.pipeline.frame_types import ColorFormat, Frame

__all__ = ["MediaPipeBackend", "MediaPipeTask"]

#: DeviceType → MediaPipe Tasks delegate name. GPU support later appends an
#: entry here (e.g. ``DeviceType.GPU: "GPU"``) — an additive change only.
_DELEGATES: dict[DeviceType, str] = {
    DeviceType.CPU: "CPU",
}

#: Maximum number of hands the hand landmarker detects per frame.
_MAX_HANDS: int = 2


class MediaPipeTask(Enum):
    """The MediaPipe landmark task a backend instance runs.

    Values are the suffix of the backend :attr:`~MediaPipeBackend.name`
    (``"mediapipe.<value>"``) — the strings app manifests reference in
    ``required_backends``.
    """

    HAND_LANDMARKS = "hands"
    POSE_LANDMARKS = "pose"
    FACE_LANDMARKS = "face"


class MediaPipeBackend(InferenceBackend):
    """Runs one MediaPipe landmark task over frames.

    One instance handles exactly one task; an app needing both hands and pose
    declares two ``required_backends`` and the pipeline owns two instances.
    v1 implements only :attr:`MediaPipeTask.HAND_LANDMARKS`.
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
        # mediapipe HandLandmarker once loaded; None while unloaded. Typed Any
        # because mediapipe ships no stubs (see pyproject mypy overrides).
        self._landmarker: Any = None

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
            InferenceError: If the task is not implemented in v1, the model
                file is missing, ``mediapipe`` is not importable, or the graph
                cannot be created — all user-presentable.
        """
        if self._landmarker is not None:
            return
        if self._task is not MediaPipeTask.HAND_LANDMARKS:
            raise InferenceError(
                f"MediaPipe task {self._task.value!r} is not implemented in this version; "
                f"only {MediaPipeTask.HAND_LANDMARKS.value!r} is available"
            )
        if not self._model_path.is_file():
            raise InferenceError(f"Model file for {self.name!r} not found at {self._model_path}")
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise InferenceError(
                f"MediaPipe is not available; cannot load backend {self.name!r}"
            ) from exc
        try:
            delegate = getattr(mp_python.BaseOptions.Delegate, self.delegate)
            options = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(self._model_path), delegate=delegate
                ),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=_MAX_HANDS,
            )
            self._landmarker = vision.HandLandmarker.create_from_options(options)
        except Exception as exc:
            raise InferenceError(
                f"Failed to load MediaPipe graph for {self.name!r}: {exc}"
            ) from exc

    def infer(self, frame: Frame) -> HandLandmarkResult:
        """Run hand-landmark detection on ``frame`` and return standardized output.

        Args:
            frame: The captured frame to run inference on.

        Returns:
            The detected hands as a
            :class:`~visionplay.vision.inference.results.HandLandmarkResult`;
            empty when no hand is in view (the normal case, not an error).

        Raises:
            InferenceError: If the backend is not loaded or detection fails.
        """
        if self._landmarker is None:
            raise InferenceError(f"Backend {self.name!r} is not loaded; call load() first")
        try:
            import mediapipe as mp

            rgb = _to_rgb(frame)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            detection = self._landmarker.detect(mp_image)
        except Exception as exc:
            raise InferenceError(f"MediaPipe inference failed for {self.name!r}: {exc}") from exc
        return _to_hand_result(detection)

    def unload(self) -> None:
        """Release the MediaPipe graph. Idempotent; safe if never loaded."""
        landmarker = self._landmarker
        self._landmarker = None
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


def _to_hand_result(detection: Any) -> HandLandmarkResult:
    """Map a MediaPipe ``HandLandmarkerResult`` to the standardized shape."""
    hands: list[HandLandmarks] = []
    hand_landmarks = detection.hand_landmarks
    handedness = detection.handedness
    for index, landmarks in enumerate(hand_landmarks):
        points = tuple(LandmarkPoint(x=lm.x, y=lm.y, z=lm.z) for lm in landmarks)
        label = "Unknown"
        score = 0.0
        if index < len(handedness) and handedness[index]:
            category = handedness[index][0]
            label = category.category_name
            score = float(category.score)
        hands.append(HandLandmarks(points=points, handedness=label, score=score))
    return HandLandmarkResult(hands=tuple(hands))
