"""The catalog of real downloadable model specs VisionPlay ships with.

Kept apart from :mod:`~visionplay.vision.inference.model_registry` (which is
the *mechanism* — download, verify, cache) so the concrete list of models
(the *policy*) lives in one obvious place. Each entry is an immutable
:class:`ModelSpec` with a real source URL and the SHA-256 the registry pins
the cache to; the bytes are fetched on first use and never committed to the
repo (CLAUDE.md conventions).

Adding a model here is how a new backend gets its weights: register the spec
with the :class:`~visionplay.vision.inference.model_registry.ModelRegistry`
and the matching backend factory (see
:mod:`~visionplay.vision.inference.backend_defaults`).
"""

from __future__ import annotations

from visionplay.vision.inference.model_registry import ModelFormat, ModelSpec

__all__ = ["FACE_LANDMARKER", "HAND_LANDMARKER", "POSE_LANDMARKER"]

#: MediaPipe Hand Landmarker Tasks bundle (float16), from Google's official
#: model store. Consumed by the ``mediapipe.hands`` backend via the Tasks API.
#: The SHA-256 pins the exact artifact; if Google ever re-publishes different
#: bytes at this URL, the registry rejects the mismatch rather than silently
#: serving an unexpected model.
HAND_LANDMARKER: ModelSpec = ModelSpec(
    model_id="hand_landmarker",
    format=ModelFormat.TASK,
    url=(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    ),
    sha256="fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
    filename="hand_landmarker.task",
)

#: MediaPipe Pose Landmarker Tasks bundle (lite variant, float16), from
#: Google's official model store. Consumed by the ``mediapipe.pose`` backend.
#: The lite variant is chosen deliberately: v1 is CPU-only and per-frame cost
#: must stay bounded (CLAUDE.md vision pipeline rules).
POSE_LANDMARKER: ModelSpec = ModelSpec(
    model_id="pose_landmarker",
    format=ModelFormat.TASK,
    url=(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    ),
    sha256="59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",
    filename="pose_landmarker_lite.task",
)

#: MediaPipe Face Landmarker Tasks bundle (float16), from Google's official
#: model store. Consumed by the ``mediapipe.face`` backend via the Tasks API.
FACE_LANDMARKER: ModelSpec = ModelSpec(
    model_id="face_landmarker",
    format=ModelFormat.TASK,
    url=(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    ),
    sha256="64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    filename="face_landmarker.task",
)
