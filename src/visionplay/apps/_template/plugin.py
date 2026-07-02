"""Template ``AppPlugin`` — copy and fill in the lifecycle glue.

Delegates every lifecycle stage to ``processor.py``, never touches Qt, and
never constructs an inference backend directly (the pipeline owns backend
lifecycles and delivers results on ``frame.results`` —
``docs/architecture.md`` §4). This is lifecycle glue only: real logic
belongs in ``processor.py``, real rendering in ``widget.py``.
"""

from __future__ import annotations

from visionplay.core.plugin_base import AppPlugin
from visionplay.vision.pipeline.frame_types import Frame

from .processor import TemplateProcessor

__all__ = ["Plugin"]


class Plugin(AppPlugin):
    """Template lifecycle glue — replace ``TemplateProcessor`` with your app's."""

    def __init__(self) -> None:
        self._processor = TemplateProcessor()

    def on_load(self) -> None:
        """Called once when the registry discovers this plugin.

        TODO: cheap setup only — no camera/model access yet.
        """

    def on_start(self) -> None:
        """Called when the user opens this app from the launcher.

        TODO: acquire any camera/model resources this app needs, then
        start the processor's per-run state.
        """
        self._processor.start()

    def on_frame(self, frame: Frame) -> Frame:
        """Called once per frame, on the pipeline worker thread.

        TODO: nothing to add here for most apps — keep delegating to
        ``processor.py`` and never touch Qt objects from this method.
        """
        return self._processor.process(frame)

    def on_stop(self) -> None:
        """Called when this app stops being the active app.

        TODO: release camera/model resources acquired in ``on_start``.
        """
        self._processor.stop()

    def on_unload(self) -> None:
        """Called when this app is removed from the registry.

        TODO: not expected in v1's static discovery, but must be
        implemented for forward compatibility.
        """
