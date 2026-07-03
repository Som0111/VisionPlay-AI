"""Unit tests for visionplay.vision.gestures.velocity."""

from __future__ import annotations

import numpy as np
import pytest

from visionplay.vision.gestures.velocity import VelocityTracker


class TestParameters:
    def test_non_positive_window_raises(self) -> None:
        with pytest.raises(ValueError, match="window"):
            VelocityTracker(window=0.0)


class TestVelocity:
    def test_none_before_any_sample(self) -> None:
        tracker = VelocityTracker()
        assert tracker.velocity is None
        assert tracker.speed == 0.0

    def test_none_with_single_sample(self) -> None:
        tracker = VelocityTracker()
        tracker.update((0.5, 0.5), timestamp=0.0)
        assert tracker.velocity is None
        assert tracker.speed == 0.0

    def test_constant_motion_measured_exactly(self) -> None:
        tracker = VelocityTracker(window=1.0)
        # Move +0.3/s in x and -0.1/s in y for a third of a second.
        for i in range(11):
            t = i / 30.0
            tracker.update((0.3 * t, 1.0 - 0.1 * t), timestamp=t)
        velocity = tracker.velocity
        assert velocity is not None
        assert velocity == pytest.approx([0.3, -0.1])

    def test_speed_is_velocity_magnitude(self) -> None:
        tracker = VelocityTracker(window=1.0)
        tracker.update((0.0, 0.0), timestamp=0.0)
        tracker.update((0.3, 0.4), timestamp=1.0)
        assert tracker.speed == pytest.approx(0.5)

    def test_stationary_point_has_zero_speed(self) -> None:
        tracker = VelocityTracker()
        for i in range(10):
            tracker.update((0.5, 0.5), timestamp=i / 30.0)
        assert tracker.speed == pytest.approx(0.0)

    def test_old_samples_fall_out_of_window(self) -> None:
        tracker = VelocityTracker(window=0.1)
        # A slow far past, then a fast recent burst: the estimate must
        # reflect only the recent window.
        tracker.update((0.0, 0.0), timestamp=0.0)
        tracker.update((0.01, 0.0), timestamp=1.0)
        tracker.update((0.11, 0.0), timestamp=1.05)
        velocity = tracker.velocity
        assert velocity is not None
        assert velocity[0] == pytest.approx(2.0)

    def test_non_advancing_timestamp_is_ignored(self) -> None:
        tracker = VelocityTracker()
        tracker.update((0.0, 0.0), timestamp=1.0)
        tracker.update((5.0, 5.0), timestamp=1.0)  # duplicate: dropped
        assert tracker.velocity is None

    def test_scalar_positions_supported(self) -> None:
        tracker = VelocityTracker(window=1.0)
        tracker.update(0.0, timestamp=0.0)
        tracker.update(0.5, timestamp=0.5)
        velocity = tracker.velocity
        assert velocity is not None
        assert float(np.asarray(velocity)) == pytest.approx(1.0)


class TestReset:
    def test_reset_clears_history(self) -> None:
        tracker = VelocityTracker()
        tracker.update((0.0, 0.0), timestamp=0.0)
        tracker.update((1.0, 1.0), timestamp=0.05)
        tracker.reset()
        assert tracker.velocity is None
        assert tracker.speed == 0.0
