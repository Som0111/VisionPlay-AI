"""Standardized, framework-agnostic inference result value objects.

Backends return these instead of raw MediaPipe / ONNX Runtime structures, so a
plugin's ``processor.py`` can read ``frame.results["mediapipe.hands"]`` or
``frame.results["onnx.<id>"]`` without ever importing ``mediapipe`` or
``onnxruntime`` (``docs/architecture.md`` §4 — results reach plugins on the
frame; the plugin stays decoupled from the runtime that produced them).

Two shapes cover v1:

- Landmark results (:class:`HandLandmarkResult`) — MediaPipe landmark tasks.
  Coordinates are **normalized** to ``[0, 1]`` over the image, matching
  MediaPipe's own convention, so a plugin scales them to pixels itself and
  never depends on the capture resolution baked into the backend.
- Tensor results (:class:`TensorOutput`) — a generic ONNX model's raw named
  output tensors as plain NumPy arrays. Model-specific post-processing (e.g.
  decoding detection boxes) is a plugin/model concern, not the backend's, so
  the standardized shape is "named arrays", not parsed detections.

Everything here is immutable and depends only on NumPy — no Qt, no ML runtime.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy.typing as npt

__all__ = [
    "HandLandmarkResult",
    "HandLandmarks",
    "LandmarkPoint",
    "TensorOutput",
]


@dataclass(frozen=True, slots=True)
class LandmarkPoint:
    """One landmark, normalized to the image.

    Attributes:
        x: Horizontal position in ``[0, 1]`` (0 = left edge, 1 = right edge).
        y: Vertical position in ``[0, 1]`` (0 = top edge, 1 = bottom edge).
        z: Relative depth in roughly the same scale as ``x``; smaller is
            closer to the camera. Origin is landmark-set dependent (MediaPipe
            uses the wrist for hands) — treat it as relative, not metric.
    """

    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class HandLandmarks:
    """The landmarks of a single detected hand.

    Attributes:
        points: The hand's landmarks in MediaPipe's canonical order (21 for
            the hand model), each normalized via :class:`LandmarkPoint`.
        handedness: ``"Left"`` or ``"Right"`` as classified by the model
            (from the camera's point of view, per MediaPipe).
        score: Handedness classification confidence in ``[0, 1]``.
    """

    points: tuple[LandmarkPoint, ...]
    handedness: str
    score: float


@dataclass(frozen=True, slots=True)
class HandLandmarkResult:
    """All hands detected in one frame — the ``mediapipe.hands`` result.

    An empty :attr:`hands` tuple is the normal "no hand in view" case, not an
    error; plugins must handle it as such.

    Attributes:
        hands: One :class:`HandLandmarks` per detected hand, in the model's
            detection order. Empty when nothing was detected.
    """

    hands: tuple[HandLandmarks, ...] = ()

    @property
    def is_empty(self) -> bool:
        """``True`` when no hand was detected in the frame."""
        return not self.hands

    def __len__(self) -> int:
        """Number of hands detected."""
        return len(self.hands)


@dataclass(frozen=True, slots=True)
class TensorOutput:
    """A generic ONNX model's named output tensors as plain NumPy arrays.

    The standardized shape for :class:`~visionplay.vision.inference.onnx_backend.ONNXBackend`:
    the raw ``session.run`` outputs, keyed by the model's output names, with no
    ONNX Runtime types exposed. Decoding these into something meaningful
    (boxes, classes, ...) is model-specific and belongs to the consuming
    plugin, not this value object.

    Attributes:
        tensors: Mapping of output name to its NumPy array. The binding is
            immutable; the arrays themselves are treated as read-only.
    """

    tensors: Mapping[str, npt.NDArray[Any]] = field(default_factory=dict)

    def names(self) -> tuple[str, ...]:
        """Return the output tensor names, in mapping order."""
        return tuple(self.tensors)

    def __contains__(self, name: str) -> bool:
        """``True`` if ``name`` is one of the model's outputs."""
        return name in self.tensors

    def __getitem__(self, name: str) -> npt.NDArray[Any]:
        """Return the named output tensor.

        Raises:
            KeyError: If the model has no output with that name.
        """
        return self.tensors[name]

    def first(self) -> npt.NDArray[Any]:
        """Return the first output tensor — the common single-output case.

        Raises:
            ValueError: If the model produced no outputs.
        """
        for array in self.tensors.values():
            return array
        raise ValueError("TensorOutput has no tensors")
