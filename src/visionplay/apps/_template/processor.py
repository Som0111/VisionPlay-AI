"""Template processor — copy and fill in with your app's CV/game logic.

Pure logic only: no ``PySide6`` import anywhere in this module, so it can
be unit-tested by feeding it synthetic frames in a plain pytest test, with
no display or event loop running (``docs/plugin-development.md``). All
per-frame work in a real app belongs here, delegated to from
``plugin.py``'s ``on_frame`` — never in ``widget.py``.
"""

from __future__ import annotations

from visionplay.vision.pipeline.frame_types import Frame

__all__ = ["TemplateProcessor"]


class TemplateProcessor:
    """Template pure-logic class — replace with your app's real state/logic.

    ``plugin.py`` owns one instance of this class for the app's lifetime
    and delegates every lifecycle stage to it. This template does nothing:
    :meth:`process` returns the frame unchanged.
    """

    def start(self) -> None:
        """Called from ``Plugin.on_start`` — TODO: initialize per-run state."""

    def process(self, frame: Frame) -> Frame:
        """Called from ``Plugin.on_frame`` — TODO: implement per-frame logic.

        Args:
            frame: The captured frame, with any declared backends'
                results already attached under ``frame.results``.

        Returns:
            The frame to publish downstream (annotate/replace as needed).
        """
        return frame

    def stop(self) -> None:
        """Called from ``Plugin.on_stop`` — TODO: release per-run state."""
