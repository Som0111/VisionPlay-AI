"""One-Euro filter for smoothing noisy landmark/cursor signals.

The One-Euro filter (Casiez, Roussel & Vogel, CHI 2012) is the standard
choice for interactive pointing: an adaptive low-pass filter whose cutoff
rises with signal speed, so slow movement is heavily smoothed (no jitter
while holding a drawing pose) while fast movement passes through with little
lag (swipes stay responsive). Air Canvas, Virtual Mouse, and any cursor-like
gesture consume this instead of rolling their own smoothing.

Stateful and single-signal: one filter instance smooths one value stream
(scalar or fixed-shape vector, e.g. an ``(x, y)`` fingertip). Feed it every
observation with its timestamp; call :meth:`OneEuroFilter.reset` when
tracking is lost so a reacquired hand does not get smoothed against stale
history.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

__all__ = ["OneEuroFilter"]


def _smoothing_factor(
    cutoff: npt.NDArray[np.float64] | float, dt: float
) -> npt.NDArray[np.float64]:
    """Exponential smoothing factor ``alpha`` for a cutoff frequency (Hz)."""
    tau = 1.0 / (2.0 * math.pi * np.asarray(cutoff, dtype=np.float64))
    return np.asarray(1.0 / (1.0 + tau / dt), dtype=np.float64)


class OneEuroFilter:
    """Adaptive low-pass filter tuned for interactive input.

    Tuning (per the original paper): start with the defaults; decrease
    ``min_cutoff`` if slow movement still jitters, increase ``beta`` if fast
    movement lags.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        derivative_cutoff: float = 1.0,
    ) -> None:
        """Configure the filter; state is created on the first sample.

        Args:
            min_cutoff: Minimum cutoff frequency in Hz — smoothing strength
                at rest. Must be positive.
            beta: Speed coefficient — how much the cutoff (and thus
                responsiveness) grows with signal speed. Must be >= 0.
            derivative_cutoff: Cutoff frequency in Hz for the internal
                derivative estimate. Must be positive.

        Raises:
            ValueError: If any parameter is out of range.
        """
        if min_cutoff <= 0:
            raise ValueError(f"min_cutoff must be positive, got {min_cutoff}")
        if beta < 0:
            raise ValueError(f"beta must be >= 0, got {beta}")
        if derivative_cutoff <= 0:
            raise ValueError(f"derivative_cutoff must be positive, got {derivative_cutoff}")
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._derivative_cutoff = derivative_cutoff
        self._last_time: float | None = None
        self._last_value: npt.NDArray[np.float64] | None = None
        self._last_derivative: npt.NDArray[np.float64] | None = None

    def filter(self, value: npt.ArrayLike, timestamp: float) -> npt.NDArray[np.float64]:
        """Smooth one observation and return the filtered value.

        Args:
            value: The raw observation — a scalar or fixed-shape array
                (e.g. ``(x, y)``). The shape must stay constant between
                :meth:`reset` calls.
            timestamp: Observation time in seconds (monotonic). A timestamp
                not later than the previous one returns the last filtered
                value unchanged rather than dividing by a non-positive dt.

        Returns:
            The filtered value, always as a float64 array.
        """
        x = np.asarray(value, dtype=np.float64)
        if self._last_value is None or self._last_time is None:
            self._last_time = timestamp
            self._last_value = x.copy()
            self._last_derivative = np.zeros_like(x)
            return x.copy()
        dt = timestamp - self._last_time
        if dt <= 0:
            return self._last_value.copy()
        assert self._last_derivative is not None
        derivative = (x - self._last_value) / dt
        alpha_d = _smoothing_factor(self._derivative_cutoff, dt)
        derivative_hat: npt.NDArray[np.float64] = (
            alpha_d * derivative + (1.0 - alpha_d) * self._last_derivative
        )
        cutoff = self._min_cutoff + self._beta * np.abs(derivative_hat)
        alpha = _smoothing_factor(cutoff, dt)
        value_hat: npt.NDArray[np.float64] = alpha * x + (1.0 - alpha) * self._last_value
        self._last_time = timestamp
        self._last_value = value_hat
        self._last_derivative = derivative_hat
        return value_hat.copy()

    def reset(self) -> None:
        """Forget all history; the next sample passes through unfiltered.

        Call when tracking is lost (hand left the frame) so reacquisition
        does not interpolate from a stale position.
        """
        self._last_time = None
        self._last_value = None
        self._last_derivative = None
