"""Unit tests for visionplay.vision.gestures.detection."""

from __future__ import annotations

import pytest

from visionplay.vision.gestures.detection import (
    DEFAULT_PINCH_THRESHOLD,
    count_extended_fingers,
    extended_fingers,
    is_pinching,
    pinch_distance,
)
from visionplay.vision.gestures.landmarks import HandLandmarkIndex
from visionplay.vision.inference.results import HandLandmarks, LandmarkPoint

#: Where the synthetic hand's wrist sits (roughly bottom-center of frame).
WRIST = (0.5, 0.9)


def make_hand(**overrides: tuple[float, float]) -> HandLandmarks:
    """Build a 21-point hand with every landmark at the wrist by default.

    Landmarks collapsed onto the wrist read as "not extended" (zero tip and
    joint distances), so each test positions only the landmarks it is about,
    keyed by the lowercase :class:`HandLandmarkIndex` name.
    """
    positions = {index: WRIST for index in HandLandmarkIndex}
    for name, xy in overrides.items():
        positions[HandLandmarkIndex[name.upper()]] = xy
    points = tuple(
        LandmarkPoint(x=positions[index][0], y=positions[index][1], z=0.0)
        for index in sorted(positions)
    )
    return HandLandmarks(points=points, handedness="Right", score=1.0)


class TestPinch:
    def test_distance_between_thumb_and_index_tips(self) -> None:
        hand = make_hand(thumb_tip=(0.5, 0.5), index_finger_tip=(0.53, 0.54))
        assert pinch_distance(hand) == pytest.approx(0.05)

    def test_touching_tips_pinch(self) -> None:
        hand = make_hand(thumb_tip=(0.5, 0.5), index_finger_tip=(0.51, 0.5))
        assert is_pinching(hand)

    def test_separated_tips_do_not_pinch(self) -> None:
        hand = make_hand(thumb_tip=(0.3, 0.5), index_finger_tip=(0.7, 0.5))
        assert not is_pinching(hand)

    def test_just_inside_threshold_pinches(self) -> None:
        hand = make_hand(
            thumb_tip=(0.5, 0.5), index_finger_tip=(0.5 + DEFAULT_PINCH_THRESHOLD * 0.99, 0.5)
        )
        assert is_pinching(hand)

    def test_just_outside_threshold_does_not_pinch(self) -> None:
        hand = make_hand(
            thumb_tip=(0.5, 0.5), index_finger_tip=(0.5 + DEFAULT_PINCH_THRESHOLD * 1.01, 0.5)
        )
        assert not is_pinching(hand)

    def test_custom_threshold_is_honored(self) -> None:
        hand = make_hand(thumb_tip=(0.5, 0.5), index_finger_tip=(0.6, 0.5))
        assert not is_pinching(hand)
        assert is_pinching(hand, threshold=0.2)

    def test_non_positive_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            is_pinching(make_hand(), threshold=0.0)


class TestExtendedFingers:
    def test_fist_has_no_extended_fingers(self) -> None:
        assert extended_fingers(make_hand()) == (False, False, False, False, False)
        assert count_extended_fingers(make_hand()) == 0

    def test_pointing_index_only(self) -> None:
        hand = make_hand(index_finger_pip=(0.5, 0.6), index_finger_tip=(0.5, 0.3))
        assert extended_fingers(hand) == (False, True, False, False, False)
        assert count_extended_fingers(hand) == 1

    def test_curled_finger_with_tip_near_joint_not_extended(self) -> None:
        # Tip barely past the PIP joint: inside the extension margin.
        hand = make_hand(index_finger_pip=(0.5, 0.6), index_finger_tip=(0.5, 0.585))
        assert not extended_fingers(hand)[1]

    def test_folded_finger_with_tip_below_joint_not_extended(self) -> None:
        # Tip curled back toward the wrist, closer than the PIP joint.
        hand = make_hand(index_finger_pip=(0.5, 0.6), index_finger_tip=(0.5, 0.75))
        assert not extended_fingers(hand)[1]

    def test_open_palm_counts_five(self) -> None:
        hand = make_hand(
            thumb_ip=(0.4, 0.7),
            thumb_tip=(0.32, 0.6),
            index_finger_pip=(0.46, 0.6),
            index_finger_tip=(0.44, 0.3),
            middle_finger_pip=(0.5, 0.6),
            middle_finger_tip=(0.5, 0.28),
            ring_finger_pip=(0.54, 0.6),
            ring_finger_tip=(0.56, 0.3),
            pinky_pip=(0.58, 0.65),
            pinky_tip=(0.62, 0.4),
        )
        assert count_extended_fingers(hand) == 5

    def test_orientation_invariant(self) -> None:
        # Same pointing gesture rotated sideways (hand pointing left).
        hand = make_hand(index_finger_pip=(0.6, 0.9), index_finger_tip=(0.9, 0.9))
        assert extended_fingers(hand)[1]
