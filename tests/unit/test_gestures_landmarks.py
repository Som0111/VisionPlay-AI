"""Unit tests for visionplay.vision.gestures.landmarks."""

from __future__ import annotations

import pytest

from visionplay.vision.gestures.landmarks import (
    FINGERTIP_INDICES,
    HandLandmarkIndex,
    distance,
    midpoint,
    to_pixels,
)
from visionplay.vision.inference.results import LandmarkPoint


class TestHandLandmarkIndex:
    def test_covers_all_21_landmarks(self) -> None:
        assert len(HandLandmarkIndex) == 21
        assert sorted(HandLandmarkIndex) == list(range(21))

    def test_canonical_anchors(self) -> None:
        assert HandLandmarkIndex.WRIST == 0
        assert HandLandmarkIndex.THUMB_TIP == 4
        assert HandLandmarkIndex.INDEX_FINGER_TIP == 8
        assert HandLandmarkIndex.PINKY_TIP == 20

    def test_fingertips_are_the_five_tips(self) -> None:
        assert FINGERTIP_INDICES == (4, 8, 12, 16, 20)


class TestDistance:
    def test_pythagorean_triple(self) -> None:
        a = LandmarkPoint(0.0, 0.0, 0.0)
        b = LandmarkPoint(0.3, 0.4, 0.0)
        assert distance(a, b) == pytest.approx(0.5)

    def test_ignores_z(self) -> None:
        a = LandmarkPoint(0.0, 0.0, 0.0)
        b = LandmarkPoint(0.3, 0.4, 5.0)
        assert distance(a, b) == pytest.approx(0.5)

    def test_zero_for_same_point(self) -> None:
        p = LandmarkPoint(0.5, 0.5, 0.1)
        assert distance(p, p) == 0.0

    def test_symmetric(self) -> None:
        a = LandmarkPoint(0.1, 0.9, 0.0)
        b = LandmarkPoint(0.8, 0.2, 0.0)
        assert distance(a, b) == pytest.approx(distance(b, a))


class TestMidpoint:
    def test_averages_all_components(self) -> None:
        a = LandmarkPoint(0.0, 0.2, -0.1)
        b = LandmarkPoint(1.0, 0.6, 0.3)
        mid = midpoint(a, b)
        assert mid.x == pytest.approx(0.5)
        assert mid.y == pytest.approx(0.4)
        assert mid.z == pytest.approx(0.1)


class TestToPixels:
    def test_scales_to_image(self) -> None:
        assert to_pixels(LandmarkPoint(0.5, 0.5, 0.0), 640, 480) == (320, 240)

    def test_origin_maps_to_zero(self) -> None:
        assert to_pixels(LandmarkPoint(0.0, 0.0, 0.0), 640, 480) == (0, 0)

    def test_clamps_out_of_range_coordinates(self) -> None:
        assert to_pixels(LandmarkPoint(-0.2, 1.3, 0.0), 640, 480) == (0, 479)

    def test_full_extent_stays_in_bounds(self) -> None:
        assert to_pixels(LandmarkPoint(1.0, 1.0, 0.0), 640, 480) == (639, 479)

    def test_invalid_dimensions_raise(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            to_pixels(LandmarkPoint(0.5, 0.5, 0.0), 0, 480)
