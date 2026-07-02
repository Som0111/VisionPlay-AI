"""Shared fixture plugin base for plugin_registry tests.

Underscore-prefixed so plugin discovery's ``pkgutil.iter_modules`` skips it
the same way it skips ``_template`` — this module is support code, not an
app.
"""

from __future__ import annotations

from visionplay.core.plugin_base import AppPlugin
from visionplay.vision.pipeline.frame_types import Frame

__all__ = ["RecordingPlugin"]


class RecordingPlugin(AppPlugin):
    """Records every lifecycle call; subclasses opt into failure behavior."""

    #: Set on subclasses to make on_load()/on_frame() raise.
    fail_on_load: bool = False
    fail_on_frame: bool = False

    def __init__(self) -> None:
        self.calls: list[str] = []

    def on_load(self) -> None:
        self.calls.append("on_load")
        if self.fail_on_load:
            raise RuntimeError("fixture on_load failure")

    def on_start(self) -> None:
        self.calls.append("on_start")

    def on_frame(self, frame: Frame) -> Frame:
        self.calls.append("on_frame")
        if self.fail_on_frame:
            raise RuntimeError("fixture on_frame failure")
        return frame

    def on_stop(self) -> None:
        self.calls.append("on_stop")

    def on_unload(self) -> None:
        self.calls.append("on_unload")
