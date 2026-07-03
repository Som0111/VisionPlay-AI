"""Windowed velocity estimation for swipe-style gestures.

Estimates how fast a tracked point (fingertip, wrist, ...) is moving from
its recent positions — the primitive behind "a fast swipe slices, a slow
drift doesn't" (Fruit Ninja) and flick gestures generally. Velocity is
computed over a short trailing time window rather than the last two samples,
so a single noisy frame does not spike the estimate.

Units follow the positions fed in: normalized landmark coordinates give
velocity in image-widths per second, which is resolution-independent —
threshold in that space.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import numpy.typing as npt

__all__ = ["VelocityTracker"]

#: Default trailing window over which velocity is estimated, in seconds.
_DEFAULT_WINDOW_SECONDS: float = 0.1


class VelocityTracker:
    """Tracks one point's positions over time and estimates its velocity.

    Stateful and single-point: one tracker per tracked landmark. Feed every
    observation via :meth:`update`; call :meth:`reset` when tracking is lost
    so a reacquired point does not register a huge instantaneous jump.
    """

    def __init__(self, window: float = _DEFAULT_WINDOW_SECONDS) -> None:
        """Configure the estimation window.

        Args:
            window: Trailing time window in seconds over which velocity is
                estimated. Must be positive.

        Raises:
            ValueError: If ``window`` is not positive.
        """
        if window <= 0:
            raise ValueError(f"window must be positive, got {window}")
        self._window = window
        self._samples: deque[tuple[float, npt.NDArray[np.float64]]] = deque()

    def update(self, position: npt.ArrayLike, timestamp: float) -> None:
        """Record one observation of the tracked point.

        Args:
            position: The point's position — a scalar or fixed-shape array
                (e.g. ``(x, y)``); the shape must stay constant between
                :meth:`reset` calls.
            timestamp: Observation time in seconds (monotonic). A sample not
                later than the previous one is ignored.
        """
        point = np.asarray(position, dtype=np.float64)
        if self._samples and timestamp <= self._samples[-1][0]:
            return
        self._samples.append((timestamp, point))
        # Evict history older than the window, always keeping two samples so
        # a frame-rate stall cannot empty the estimate.
        cutoff = timestamp - self._window
        while len(self._samples) > 2 and self._samples[0][0] < cutoff:
            self._samples.popleft()

    @property
    def velocity(self) -> npt.NDArray[np.float64] | None:
        """Velocity over the trailing window (units/second).

        ``None`` until two samples have been observed — callers treat that
        as "not moving yet", the normal state right after (re)acquisition.
        """
        if len(self._samples) < 2:
            return None
        first_time, first_pos = self._samples[0]
        last_time, last_pos = self._samples[-1]
        result: npt.NDArray[np.float64] = (last_pos - first_pos) / (last_time - first_time)
        return result

    @property
    def speed(self) -> float:
        """Magnitude of :attr:`velocity`; ``0.0`` until it is available."""
        velocity = self.velocity
        if velocity is None:
            return 0.0
        return float(np.linalg.norm(velocity))

    def reset(self) -> None:
        """Forget all history — call when tracking of the point is lost."""
        self._samples.clear()
