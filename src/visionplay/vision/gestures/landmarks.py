"""Hand landmark indices and normalized-coordinate geometry helpers.

Indices follow MediaPipe's canonical 21-point hand topology, matching the
order of :attr:`~visionplay.vision.inference.results.HandLandmarks.points`
as produced by the ``mediapipe.hands`` backend. Geometry helpers operate on
:class:`~visionplay.vision.inference.results.LandmarkPoint` in the same
normalized ``[0, 1]`` image space the results use; only :func:`to_pixels`
crosses into pixel coordinates.
"""

from __future__ import annotations

import math
from enum import IntEnum

from visionplay.vision.inference.results import LandmarkPoint

__all__ = [
    "FINGERTIP_INDICES",
    "HandLandmarkIndex",
    "distance",
    "midpoint",
    "to_pixels",
]


class HandLandmarkIndex(IntEnum):
    """Index of each landmark in MediaPipe's canonical 21-point hand order."""

    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


#: The five fingertip indices, thumb to pinky.
FINGERTIP_INDICES: tuple[HandLandmarkIndex, ...] = (
    HandLandmarkIndex.THUMB_TIP,
    HandLandmarkIndex.INDEX_FINGER_TIP,
    HandLandmarkIndex.MIDDLE_FINGER_TIP,
    HandLandmarkIndex.RING_FINGER_TIP,
    HandLandmarkIndex.PINKY_TIP,
)


def distance(a: LandmarkPoint, b: LandmarkPoint) -> float:
    """Euclidean distance between two landmarks in the image plane.

    Deliberately 2-D (``x``/``y`` only): landmark ``z`` is relative depth on
    a landmark-set-dependent scale, so mixing it into a threshold-compared
    distance would make gesture thresholds unstable across hand poses.
    """
    return math.hypot(a.x - b.x, a.y - b.y)


def midpoint(a: LandmarkPoint, b: LandmarkPoint) -> LandmarkPoint:
    """Midpoint of two landmarks (all three components averaged)."""
    return LandmarkPoint(x=(a.x + b.x) / 2, y=(a.y + b.y) / 2, z=(a.z + b.z) / 2)


def to_pixels(point: LandmarkPoint, width: int, height: int) -> tuple[int, int]:
    """Map a normalized landmark to integer pixel coordinates.

    Landmarks can fall slightly outside ``[0, 1]`` near the frame edge;
    coordinates are clamped into the image so callers can index or draw
    without bounds checks.

    Args:
        point: The normalized landmark.
        width: Image width in pixels; must be positive.
        height: Image height in pixels; must be positive.

    Returns:
        ``(x, y)`` pixel coordinates, each clamped to the image.

    Raises:
        ValueError: If ``width`` or ``height`` is not positive.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive, got {width}x{height}")
    x = min(max(int(point.x * width), 0), width - 1)
    y = min(max(int(point.y * height), 0), height - 1)
    return x, y
