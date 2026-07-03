"""Pinch and finger-extension detection over standardized hand landmarks.

Pure predicates on a single
:class:`~visionplay.vision.inference.results.HandLandmarks` — stateless and
per-frame. Temporal behavior (debouncing a pinch into a pen-down event,
dwell timing, ...) belongs to the consuming app's processor, on top of these.

All thresholds are in the normalized ``[0, 1]`` landmark space, so they are
resolution-independent but *do* scale with how large the hand appears in
frame — a hand very close to the camera has larger normalized distances.
The defaults work at typical desktop webcam distances.
"""

from __future__ import annotations

from visionplay.vision.gestures.landmarks import HandLandmarkIndex, distance
from visionplay.vision.inference.results import HandLandmarks

__all__ = [
    "DEFAULT_PINCH_THRESHOLD",
    "count_extended_fingers",
    "extended_fingers",
    "is_pinching",
    "pinch_distance",
]

#: Thumb-to-index distance (normalized) at or below which a pinch is detected.
DEFAULT_PINCH_THRESHOLD: float = 0.06

#: A finger counts as extended when its tip is this factor farther from the
#: wrist than its middle joint — the margin rejects half-curled fingers.
_EXTENDED_RATIO: float = 1.1

#: Per-finger (tip, middle-joint) landmark pairs for extension detection,
#: thumb to pinky. The thumb has no PIP; its IP joint plays the same role.
_FINGER_JOINTS: tuple[tuple[HandLandmarkIndex, HandLandmarkIndex], ...] = (
    (HandLandmarkIndex.THUMB_TIP, HandLandmarkIndex.THUMB_IP),
    (HandLandmarkIndex.INDEX_FINGER_TIP, HandLandmarkIndex.INDEX_FINGER_PIP),
    (HandLandmarkIndex.MIDDLE_FINGER_TIP, HandLandmarkIndex.MIDDLE_FINGER_PIP),
    (HandLandmarkIndex.RING_FINGER_TIP, HandLandmarkIndex.RING_FINGER_PIP),
    (HandLandmarkIndex.PINKY_TIP, HandLandmarkIndex.PINKY_PIP),
)


def pinch_distance(hand: HandLandmarks) -> float:
    """Normalized distance between the thumb tip and index fingertip."""
    points = hand.points
    return distance(points[HandLandmarkIndex.THUMB_TIP], points[HandLandmarkIndex.INDEX_FINGER_TIP])


def is_pinching(hand: HandLandmarks, threshold: float = DEFAULT_PINCH_THRESHOLD) -> bool:
    """Return ``True`` when the thumb and index fingertips are pinched together.

    Args:
        hand: The hand to test.
        threshold: Maximum thumb-index distance (normalized) that counts as a
            pinch; must be positive.

    Raises:
        ValueError: If ``threshold`` is not positive.
    """
    if threshold <= 0:
        raise ValueError(f"Pinch threshold must be positive, got {threshold}")
    return pinch_distance(hand) <= threshold


def extended_fingers(hand: HandLandmarks) -> tuple[bool, bool, bool, bool, bool]:
    """Return which fingers are extended, thumb to pinky.

    A finger is extended when its tip is clearly farther from the wrist than
    its middle joint — an orientation-invariant heuristic that works with the
    hand upside down or sideways, unlike comparing raw ``y`` coordinates.
    """
    points = hand.points
    wrist = points[HandLandmarkIndex.WRIST]
    flags = tuple(
        distance(wrist, points[tip]) > distance(wrist, points[joint]) * _EXTENDED_RATIO
        for tip, joint in _FINGER_JOINTS
    )
    return flags[0], flags[1], flags[2], flags[3], flags[4]


def count_extended_fingers(hand: HandLandmarks) -> int:
    """Number of extended fingers (0-5), per :func:`extended_fingers`."""
    return sum(extended_fingers(hand))
