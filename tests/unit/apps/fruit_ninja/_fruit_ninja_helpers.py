"""Shared synthetic-hand fixtures for Fruit Ninja tests (M3.4)."""

from __future__ import annotations

import numpy as np

from visionplay.apps.fruit_ninja.processor import RESULTS_KEY, FruitNinjaProcessor
from visionplay.vision.gestures import HandLandmarkIndex
from visionplay.vision.inference.results import HandLandmarkResult, HandLandmarks, LandmarkPoint
from visionplay.vision.pipeline.frame_types import Frame

__all__ = ["WRIST", "FrameFeeder", "make_hand"]

#: Wrist anchor of every synthetic hand (bottom-center of the frame);
#: irrelevant to the game itself since only the index fingertip is read.
WRIST: tuple[float, float] = (0.5, 0.95)


def make_hand(tip: tuple[float, float]) -> HandLandmarks:
    """Build a synthetic 21-point hand with its index fingertip at ``tip``."""
    positions: dict[HandLandmarkIndex, tuple[float, float]] = {
        index: WRIST for index in HandLandmarkIndex
    }
    positions[HandLandmarkIndex.INDEX_FINGER_TIP] = tip
    points = tuple(
        LandmarkPoint(x=positions[index][0], y=positions[index][1], z=0.0)
        for index in sorted(positions)
    )
    return HandLandmarks(points=points, handedness="Right", score=1.0)


class FrameFeeder:
    """Feeds a processor consecutive frames at a fixed simulated frame rate."""

    def __init__(self, processor: FruitNinjaProcessor, *, fps: float = 30.0) -> None:
        self.processor = processor
        self.next_id = 0
        self._dt = 1.0 / fps

    def feed(self, tip: tuple[float, float] | None) -> Frame:
        """Process one 480x640 BGR frame carrying the fingertip at ``tip`` (or no hand)."""
        frame = Frame.from_image(
            frame_id=self.next_id,
            timestamp=self.next_id * self._dt,
            image=np.zeros((480, 640, 3), dtype=np.uint8),
        )
        self.next_id += 1
        if tip is not None:
            frame.results[RESULTS_KEY] = HandLandmarkResult(hands=(make_hand(tip),))
        return self.processor.process(frame)

    def feed_many(self, tip: tuple[float, float] | None, count: int) -> Frame:
        frame = self.feed(tip)
        for _ in range(count - 1):
            frame = self.feed(tip)
        return frame

    def swipe(
        self, start: tuple[float, float], end: tuple[float, float], steps: int
    ) -> Frame:
        """Feed a fast linear fingertip swipe from ``start`` to ``end`` over ``steps`` frames.

        The first frame plants the fingertip at ``start`` (establishing a
        cursor with no prior position, so it can't itself register a slice);
        each subsequent frame steps linearly toward ``end``.
        """
        frame = self.feed(start)
        for step in range(1, steps + 1):
            t = step / steps
            point = (start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t)
            frame = self.feed(point)
        return frame
