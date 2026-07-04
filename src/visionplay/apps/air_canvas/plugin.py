"""Air Canvas ``AppPlugin`` — lifecycle glue only.

Delegates every stage to
:class:`~visionplay.apps.air_canvas.processor.AirCanvasProcessor`, never
touches Qt, and never constructs an inference backend directly (the
pipeline owns backend lifecycles — ``docs/architecture.md`` §4).
"""

from __future__ import annotations

from visionplay.core.plugin_base import AppPlugin
from visionplay.vision.pipeline.frame_types import Frame

from .processor import AirCanvasProcessor

__all__ = ["Plugin"]


class Plugin(AppPlugin):
    """Air Canvas lifecycle glue."""

    def __init__(self) -> None:
        self._processor = AirCanvasProcessor()

    @property
    def processor(self) -> AirCanvasProcessor:
        """This app's processor — exposed only so ``widget.py`` can route
        keyboard shortcuts (undo/redo/save) through its thread-safe
        ``request_*`` methods; never used to read per-frame results."""
        return self._processor

    def on_load(self) -> None:
        """Cheap setup only — no camera/model access yet."""

    def on_start(self) -> None:
        """Reset the canvas and tracking state for a fresh run."""
        self._processor.start()

    def on_frame(self, frame: Frame) -> Frame:
        """Delegate per-frame drawing/gesture handling to the processor."""
        return self._processor.process(frame)

    def on_stop(self) -> None:
        """Release the processor's per-run state."""
        self._processor.stop()

    def on_unload(self) -> None:
        """Not expected in v1's static discovery; implemented for forward compatibility."""
