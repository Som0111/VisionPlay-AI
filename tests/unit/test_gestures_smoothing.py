"""Unit tests for visionplay.vision.gestures.smoothing."""

from __future__ import annotations

import numpy as np
import pytest

from visionplay.vision.gestures.smoothing import OneEuroFilter


class TestParameters:
    def test_non_positive_min_cutoff_raises(self) -> None:
        with pytest.raises(ValueError, match="min_cutoff"):
            OneEuroFilter(min_cutoff=0.0)

    def test_negative_beta_raises(self) -> None:
        with pytest.raises(ValueError, match="beta"):
            OneEuroFilter(beta=-0.1)

    def test_non_positive_derivative_cutoff_raises(self) -> None:
        with pytest.raises(ValueError, match="derivative_cutoff"):
            OneEuroFilter(derivative_cutoff=0.0)


class TestFiltering:
    def test_first_sample_passes_through(self) -> None:
        filt = OneEuroFilter()
        assert filt.filter(0.7, timestamp=0.0) == pytest.approx(0.7)

    def test_constant_signal_stays_constant(self) -> None:
        filt = OneEuroFilter()
        for i in range(20):
            value = filt.filter(0.5, timestamp=i / 30.0)
        assert value == pytest.approx(0.5)

    def test_converges_toward_step_input(self) -> None:
        filt = OneEuroFilter(min_cutoff=1.0)
        filt.filter(0.0, timestamp=0.0)
        for i in range(1, 120):
            value = filt.filter(1.0, timestamp=i / 30.0)
        assert value == pytest.approx(1.0, abs=1e-3)

    def test_reduces_jitter(self) -> None:
        rng = np.random.default_rng(42)
        raw = 0.5 + rng.normal(0.0, 0.02, size=200)
        filt = OneEuroFilter(min_cutoff=1.0)
        filtered = np.array([float(filt.filter(v, timestamp=i / 30.0)) for i, v in enumerate(raw)])
        assert filtered[20:].std() < raw[20:].std() * 0.5

    def test_higher_beta_tracks_fast_motion_more_closely(self) -> None:
        # A fast ramp: high beta should lag the true signal less.
        times = np.arange(60) / 30.0
        truth = times * 2.0
        lags = []
        for beta in (0.0, 5.0):
            filt = OneEuroFilter(min_cutoff=0.5, beta=beta)
            pairs = zip(times, truth, strict=True)
            filtered = [float(filt.filter(v, timestamp=t)) for t, v in pairs]
            lags.append(abs(truth[-1] - filtered[-1]))
        assert lags[1] < lags[0]

    def test_vector_input_keeps_shape(self) -> None:
        filt = OneEuroFilter()
        out = filt.filter((0.2, 0.8), timestamp=0.0)
        assert out.shape == (2,)
        out = filt.filter((0.3, 0.7), timestamp=1 / 30.0)
        assert out.shape == (2,)

    def test_non_advancing_timestamp_returns_previous(self) -> None:
        filt = OneEuroFilter()
        filt.filter(0.5, timestamp=1.0)
        smoothed = filt.filter(0.6, timestamp=1 / 30.0 + 1.0)
        repeat = filt.filter(999.0, timestamp=1.0)  # stale timestamp: ignored
        assert repeat == pytest.approx(smoothed)


class TestReset:
    def test_reset_forgets_history(self) -> None:
        filt = OneEuroFilter()
        filt.filter(0.0, timestamp=0.0)
        filt.filter(0.1, timestamp=1 / 30.0)
        filt.reset()
        # After reset the next sample passes through as if it were the first.
        assert filt.filter(0.9, timestamp=2 / 30.0) == pytest.approx(0.9)
