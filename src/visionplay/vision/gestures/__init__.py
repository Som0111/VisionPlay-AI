"""Shared gesture/landmark utilities for app processors.

The one home for gesture logic reused across apps (Air Canvas, Fruit Ninja,
Virtual Mouse, ...) — plugins import from here instead of borrowing from a
sibling app, per the no-app-to-app-imports rule (``docs/architecture.md`` §3).

Everything here is pure computation over the standardized result types in
:mod:`~visionplay.vision.inference.results` and NumPy arrays — no Qt, no ML
runtime, fully headless-testable:

- :mod:`~visionplay.vision.gestures.landmarks` — hand landmark indices and
  normalized-coordinate geometry helpers.
- :mod:`~visionplay.vision.gestures.detection` — pinch and finger-extension
  detection over a :class:`~visionplay.vision.inference.results.HandLandmarks`.
- :mod:`~visionplay.vision.gestures.smoothing` — One-Euro filter for
  jitter-free cursors/strokes.
- :mod:`~visionplay.vision.gestures.velocity` — windowed velocity/speed
  estimation for swipe-style gestures.
"""

from __future__ import annotations

from visionplay.vision.gestures.detection import (
    DEFAULT_PINCH_THRESHOLD,
    count_extended_fingers,
    extended_fingers,
    is_pinching,
    pinch_distance,
)
from visionplay.vision.gestures.landmarks import (
    FINGERTIP_INDICES,
    HandLandmarkIndex,
    distance,
    midpoint,
    to_pixels,
)
from visionplay.vision.gestures.smoothing import OneEuroFilter
from visionplay.vision.gestures.velocity import VelocityTracker

__all__ = [
    "DEFAULT_PINCH_THRESHOLD",
    "FINGERTIP_INDICES",
    "HandLandmarkIndex",
    "OneEuroFilter",
    "VelocityTracker",
    "count_extended_fingers",
    "distance",
    "extended_fingers",
    "is_pinching",
    "midpoint",
    "pinch_distance",
    "to_pixels",
]
