"""Fixture plugin that must never be started — its backend is unsatisfiable.

Records lifecycle calls so tests can assert the capability guard prevented
``on_start`` from ever running.
"""

from __future__ import annotations

from visionplay.core.plugin_base import AppPlugin
from visionplay.vision.pipeline.frame_types import Frame


class Plugin(AppPlugin):
    """Records lifecycle calls; the negotiation tests assert on_start is absent."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def on_load(self) -> None:
        self.calls.append("on_load")

    def on_start(self) -> None:
        self.calls.append("on_start")

    def on_frame(self, frame: Frame) -> Frame:
        self.calls.append("on_frame")
        return frame

    def on_stop(self) -> None:
        self.calls.append("on_stop")

    def on_unload(self) -> None:
        self.calls.append("on_unload")
