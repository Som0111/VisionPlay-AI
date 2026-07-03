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

__all__ = ["HAND_LANDMARKER"]

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
