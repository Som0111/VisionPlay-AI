"""Hand-tracking demo ``AppPlugin`` — lifecycle glue only.

Delegates every stage to :class:`~visionplay.apps.hand_tracking_demo.
processor.HandTrackingProcessor`, never touches Qt, and never constructs an
inference backend directly (the pipeline owns backend lifecycles —
``docs/architecture.md`` §4).
"""

from __future__ import annotations

from visionplay.core.plugin_base import AppPlugin
from visionplay.vision.pipeline.frame_types import Frame

from .processor import HandTrackingProcessor

__all__ = ["Plugin"]


class Plugin(AppPlugin):
    """Hand-tracking demo lifecycle glue."""

    def __init__(self) -> None:
        self._processor = HandTrackingProcessor()

    def on_load(self) -> None:
        """Cheap setup only — no camera/model access yet."""

    def on_start(self) -> None:
        """Start the processor's per-run state.

        No camera/model resources are acquired directly here — the
        pipeline owns the camera and (in Phase 2) the declared backend.
        """
        self._processor.start()

    def on_frame(self, frame: Frame) -> Frame:
        """Delegate per-frame handling to the processor."""
        return self._processor.process(frame)

    def on_stop(self) -> None:
        """Release the processor's per-run state."""
        self._processor.stop()

    def on_unload(self) -> None:
        """Not expected in v1's static discovery; implemented for forward compatibility."""
