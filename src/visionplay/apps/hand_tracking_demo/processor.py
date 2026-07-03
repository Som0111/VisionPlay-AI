"""Hand-tracking demo processor: reads ``mediapipe.hands`` results defensively.

Pure logic, no ``PySide6`` import — unit-testable headless
(``docs/plugin-development.md``). The pipeline populates
``frame.results[RESULTS_KEY]`` with a real
:class:`~visionplay.vision.inference.results.HandLandmarkResult` when the
``mediapipe.hands`` backend runs; the key is read defensively because it is
absent whenever the backend is unavailable or failed mid-stream — an
expected, normal case to handle, not an error condition.
"""

from __future__ import annotations

from visionplay.vision.pipeline.frame_types import Frame

__all__ = ["RESULTS_KEY", "HandTrackingProcessor"]

#: Key the pipeline stores this app's declared backend's output under
#: (matches ``MANIFEST.required_backends``).
RESULTS_KEY: str = "mediapipe.hands"


class HandTrackingProcessor:
    """Per-frame logic for the hand-tracking demo.

    A deliberately minimal demo: it reads the landmark result defensively
    and passes the frame through for the widget to render. Gesture logic
    built on these landmarks is game territory (Phase 3), not this demo's.
    """

    def start(self) -> None:
        """Called from ``Plugin.on_start`` — nothing to initialize yet."""

    def process(self, frame: Frame) -> Frame:
        """Read hand-landmark results defensively; pass the frame through.

        Args:
            frame: The captured frame. ``frame.results.get(RESULTS_KEY)``
                is a ``HandLandmarkResult`` when the backend ran, ``None``
                when it is unavailable.

        Returns:
            The same frame, unmodified — the widget renders the landmarks.
        """
        _hands = frame.results.get(RESULTS_KEY)  # defensive read; absent when backend is down
        return frame

    def stop(self) -> None:
        """Called from ``Plugin.on_stop`` — nothing to release yet."""
