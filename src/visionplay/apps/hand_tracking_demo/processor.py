"""Hand-tracking demo processor: reads ``mediapipe.hands`` results defensively.

Pure logic, no ``PySide6`` import — unit-testable headless
(``docs/plugin-development.md``). No real backend runs in Phase 1 (M1.4
wires the pipeline's ``on_frame`` seam, not inference), so
``frame.results.get(RESULTS_KEY)`` is always ``None`` here — that is the
expected, normal case to handle, not an error condition. Phase 2 wiring a
real MediaPipe hands backend populates that key without requiring any
change to this read.
"""

from __future__ import annotations

from visionplay.vision.pipeline.frame_types import Frame

__all__ = ["RESULTS_KEY", "HandTrackingProcessor"]

#: Key the pipeline stores this app's declared backend's output under, once
#: Phase 2 wires a real ``mediapipe.hands`` backend (matches
#: ``MANIFEST.required_backends``).
RESULTS_KEY: str = "mediapipe.hands"


class HandTrackingProcessor:
    """Per-frame logic for the hand-tracking demo.

    Phase 1 has nothing to compute — it only proves that reading an absent
    backend result is handled gracefully. Phase 2 replaces the body of
    :meth:`process` with real per-frame landmark handling.
    """

    def start(self) -> None:
        """Called from ``Plugin.on_start`` — nothing to initialize yet."""

    def process(self, frame: Frame) -> Frame:
        """Read hand-landmark results defensively; pass the frame through.

        Args:
            frame: The captured frame. In Phase 1,
                ``frame.results.get(RESULTS_KEY)`` is always ``None`` since
                no backend runs yet.

        Returns:
            The same frame, unmodified — Phase 2 may annotate/replace it
            once real landmark data exists.
        """
        _hands = frame.results.get(RESULTS_KEY)  # forward-compatible read; unused in Phase 1
        return frame

    def stop(self) -> None:
        """Called from ``Plugin.on_stop`` — nothing to release yet."""
