"""Fixture plugin that records the ``frame.results`` it sees in ``on_frame``.

Proves the M2.3A ordering: inference backends populate ``frame.results``
before the pipeline dispatches to ``on_frame``.
"""

from __future__ import annotations

from typing import Any

from visionplay.core.plugin_base import AppPlugin
from visionplay.vision.pipeline.frame_types import Frame


class Plugin(AppPlugin):
    """Records a snapshot of each frame's results for the test to inspect."""

    def __init__(self) -> None:
        self.seen_results: list[dict[str, Any]] = []

    def on_load(self) -> None:
        pass

    def on_start(self) -> None:
        pass

    def on_frame(self, frame: Frame) -> Frame:
        self.seen_results.append(dict(frame.results))
        return frame

    def on_stop(self) -> None:
        pass

    def on_unload(self) -> None:
        pass
