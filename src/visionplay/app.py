"""Qt application bootstrap and composition root (M0.6).

Wires the platform in dependency order — paths → config → logging →
event bus → camera source → frame pipeline → window + frame bridge — and
owns the shutdown path: closing the main window stops the bridge consumer
and the capture worker and releases the camera before the process exits.

:class:`VisionPlayApp` is the composition root; :func:`main` (the
``python -m visionplay`` entry) merely creates the ``QApplication``, runs
the event loop, and guarantees shutdown on the way out. Construction is
deliberately injectable (``paths``, ``source``) so startup can be tested
headless with a synthetic camera and a temp directory.
"""

from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication

from visionplay import __version__
from visionplay.core.config import Config, load_config
from visionplay.core.event_bus import EventBus
from visionplay.core.logging_setup import setup_logging
from visionplay.core.paths import AppPaths
from visionplay.ui.main_window import MainWindow
from visionplay.ui.widgets.frame_bridge import FrameBridge
from visionplay.vision.camera.camera_source import CameraSource
from visionplay.vision.camera.webcam_source import WebcamSource
from visionplay.vision.pipeline.frame_bus import FramePipeline

__all__ = ["VisionPlayApp", "main"]

logger = logging.getLogger(__name__)


class VisionPlayApp:
    """Composition root: owns the wired platform objects and their lifecycle.

    Constructing the class performs the headless bootstrap (paths, config,
    logging, event bus, pipeline); :meth:`start` adds the Qt pieces (window,
    frame bridge) and begins capture — it requires a live ``QApplication``.
    """

    def __init__(
        self,
        paths: AppPaths | None = None,
        *,
        source: CameraSource | None = None,
    ) -> None:
        """Bootstrap the headless platform layers.

        Args:
            paths: Filesystem locations; ``None`` resolves the platform
                defaults. Directories are created if missing.
            source: Camera source override for tests; ``None`` builds a
                :class:`WebcamSource` from the ``camera`` config section.
        """
        self._paths = (paths if paths is not None else AppPaths.default()).ensure()
        self._config = load_config(self._paths.config_file)
        setup_logging(self._paths.log_file, level=self._config.get("app", "log_level", "INFO"))
        logger.info("VisionPlay AI v%s starting", __version__)
        self._event_bus = EventBus()
        self._source = source if source is not None else _webcam_from_config(self._config)
        target_fps = self._config.get("camera", "target_fps", 30)
        self._pipeline = FramePipeline(
            self._source,
            event_bus=self._event_bus,
            target_fps=float(target_fps) if target_fps > 0 else None,
        )
        self._bridge: FrameBridge | None = None
        self._window: MainWindow | None = None

    @property
    def paths(self) -> AppPaths:
        """The resolved application directories."""
        return self._paths

    @property
    def config(self) -> Config:
        """The loaded application configuration."""
        return self._config

    @property
    def event_bus(self) -> EventBus:
        """The platform event bus shared by Qt-free layers."""
        return self._event_bus

    @property
    def pipeline(self) -> FramePipeline:
        """The camera capture pipeline (worker thread lives here)."""
        return self._pipeline

    @property
    def window(self) -> MainWindow | None:
        """The main window, once :meth:`start` has created it."""
        return self._window

    def start(self) -> MainWindow:
        """Create the main window, wire the frame bridge, begin capture.

        Must be called after a ``QApplication`` exists. The returned window
        is not shown — the caller decides when to ``show()`` it.

        Returns:
            The wired main window.

        Raises:
            RuntimeError: If the application was already started.
        """
        if self._window is not None:
            raise RuntimeError("VisionPlayApp is already started")
        window = MainWindow()
        bridge = FrameBridge(self._pipeline)
        # Cross-thread by design: the bridge emits from its consumer thread,
        # Qt queues delivery onto the widget's (main) thread.
        bridge.frame_ready.connect(window.camera_view.show_frame)
        bridge.stream_ended.connect(window.camera_view.show_status)
        window.closing.connect(self.shutdown)
        self._window = window
        self._bridge = bridge
        self._pipeline.start()
        bridge.start()
        logger.info("Frame pipeline and bridge started")
        return window

    def shutdown(self) -> None:
        """Stop the bridge and pipeline, releasing the camera. Idempotent.

        Runs when the main window closes (via :attr:`MainWindow.closing`)
        and again as a belt-and-braces measure after the event loop exits.
        """
        bridge = self._bridge
        self._bridge = None
        if bridge is not None:
            bridge.stop()
        self._pipeline.stop()
        logger.info("Frame pipeline and bridge stopped; camera released")


def _webcam_from_config(config: Config) -> WebcamSource:
    """Build the default webcam source from the ``camera`` config section.

    A configured width/height of 0 means "driver default".
    """
    camera = config.section("camera")
    width = camera.get("frame_width", 0)
    height = camera.get("frame_height", 0)
    return WebcamSource(
        camera.get("device_index", 0),
        frame_width=width if width > 0 else None,
        frame_height=height if height > 0 else None,
    )


def main(argv: list[str] | None = None) -> int:
    """Application entry point: run the Qt event loop until the window closes.

    Args:
        argv: Command-line arguments for ``QApplication``; defaults to
            ``sys.argv``.

    Returns:
        The Qt event loop's exit code.
    """
    qt_app = QApplication(argv if argv is not None else sys.argv)
    qt_app.setApplicationName("VisionPlay AI")
    app = VisionPlayApp()
    window = app.start()
    window.show()
    try:
        return qt_app.exec()
    finally:
        app.shutdown()
