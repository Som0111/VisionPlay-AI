"""Shared synthetic-hand fixtures for Air Canvas tests (M3.2 + M3.3).

Extracted from the M3.2 test module so the M3.3 test module (undo/redo,
layers, shapes, export) can build the same synthetic hands and frame feed
without duplicating this logic.
"""

from __future__ import annotations

import numpy as np

from visionplay.apps.air_canvas.processor import (
    RESULTS_KEY,
    TOOLBAR,
    TOOLBAR_HEIGHT,
    AirCanvasProcessor,
)
from visionplay.vision.gestures import HandLandmarkIndex
from visionplay.vision.inference.results import HandLandmarkResult, HandLandmarks, LandmarkPoint
from visionplay.vision.pipeline.frame_types import Frame

__all__ = ["CANVAS_POINT", "WRIST", "FrameFeeder", "make_hand", "region_center"]

#: Wrist anchor of every synthetic hand (bottom-center of the frame).
WRIST: tuple[float, float] = (0.5, 0.95)

#: A drawing position safely below the toolbar strip.
CANVAS_POINT: tuple[float, float] = (0.5, 0.5)


def region_center(action: str, payload: int = 0) -> tuple[float, float]:
    """Normalized center of the toolbar region with the given action/payload."""
    region = next(r for r in TOOLBAR if r.action == action and r.payload == payload)
    return ((region.x_min + region.x_max) / 2, TOOLBAR_HEIGHT / 2)


def make_hand(
    tip: tuple[float, float] = CANVAS_POINT,
    *,
    pinch: bool = False,
    pinch_gap: float | None = None,
    open_palm: bool = False,
) -> HandLandmarks:
    """Build a synthetic 21-point hand.

    Every landmark defaults to the wrist position (reads as a fist); the
    index fingertip is placed at ``tip``, and the thumb tip at a controlled
    distance from it (``pinch_gap``, or a tight/wide default per ``pinch``).
    ``open_palm=True`` instead poses all five fingers extended.
    """
    positions: dict[HandLandmarkIndex, tuple[float, float]] = {
        index: WRIST for index in HandLandmarkIndex
    }
    positions[HandLandmarkIndex.INDEX_FINGER_TIP] = tip
    if open_palm:
        positions.update(
            {
                HandLandmarkIndex.THUMB_IP: (0.4, 0.7),
                HandLandmarkIndex.THUMB_TIP: (0.32, 0.6),
                HandLandmarkIndex.INDEX_FINGER_PIP: (0.46, 0.6),
                HandLandmarkIndex.INDEX_FINGER_TIP: tip,
                HandLandmarkIndex.MIDDLE_FINGER_PIP: (0.5, 0.6),
                HandLandmarkIndex.MIDDLE_FINGER_TIP: (0.5, 0.28),
                HandLandmarkIndex.RING_FINGER_PIP: (0.54, 0.6),
                HandLandmarkIndex.RING_FINGER_TIP: (0.56, 0.3),
                HandLandmarkIndex.PINKY_PIP: (0.58, 0.65),
                HandLandmarkIndex.PINKY_TIP: (0.62, 0.4),
            }
        )
    else:
        gap = pinch_gap if pinch_gap is not None else (0.01 if pinch else 0.3)
        positions[HandLandmarkIndex.THUMB_TIP] = (tip[0] + gap, tip[1])
    points = tuple(
        LandmarkPoint(x=positions[index][0], y=positions[index][1], z=0.0)
        for index in sorted(positions)
    )
    return HandLandmarks(points=points, handedness="Right", score=1.0)


class FrameFeeder:
    """Feeds a processor consecutive frames with monotonic ids/timestamps."""

    def __init__(self, processor: AirCanvasProcessor) -> None:
        self.processor = processor
        self.next_id = 0

    def feed(self, hand: HandLandmarks | None, *, blank_results: bool = False) -> Frame:
        """Process one 240x320 BGR frame carrying ``hand`` (or no/empty result)."""
        frame = Frame.from_image(
            frame_id=self.next_id,
            timestamp=self.next_id / 30.0,
            image=np.zeros((240, 320, 3), dtype=np.uint8),
        )
        self.next_id += 1
        if hand is not None:
            frame.results[RESULTS_KEY] = HandLandmarkResult(hands=(hand,))
        elif blank_results:
            frame.results[RESULTS_KEY] = HandLandmarkResult()
        return self.processor.process(frame)

    def feed_many(self, hand: HandLandmarks | None, count: int) -> None:
        for _ in range(count):
            self.feed(hand)

    def settle(self, tip: tuple[float, float], frames: int = 30) -> None:
        """Hover (unpinched) at ``tip`` until the smoothing filter converges.

        The One-Euro-filtered cursor lags a positional jump exponentially;
        interactions that depend on an exact cursor position settle first,
        just as a real hand takes frames to travel there. 30 frames (1s at
        30fps) converges to well within any toolbar/erase hit-test radius
        regardless of the filter's exact tuning.
        """
        self.feed_many(make_hand(tip), frames)
