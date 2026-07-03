"""Qt application bootstrap and composition root (M0.6, M1.6).

Wires the platform in dependency order — paths → config → logging →
event bus → plugin registry → camera source → frame pipeline → window +
frame bridge — and owns the shutdown path: closing the main window stops
the active app, then the bridge consumer and the capture worker, and
releases the camera before the process exits.

:class:`VisionPlayApp` is the composition root; :func:`main` (the
``python -m visionplay`` entry) merely creates the ``QApplication``, runs
the event loop, and guarantees shutdown on the way out. Construction is
deliberately injectable (``paths``, ``source``, ``apps_package``) so
startup can be tested headless with a synthetic camera, a temp directory,
and a fixture apps package.

Launcher wiring (M1.6): the frame pipeline's per-frame processor is set
once, at construction, to :meth:`_process_frame` — that runs the active
app's inference backends and then :meth:`PluginRegistry.process_frame`,
which dispatches to whichever app the registry considers active. So "telling
the pipeline which plugin is active" is simply a matter of telling the
*registry* (:meth:`PluginRegistry.start`); no per-launch pipeline
reconfiguration is needed. With no app ever started, the registry has no
active app and ``process_frame`` is a passthrough, so the M0.5/M0.6
camera-only behavior is unchanged.

Inference wiring (M2.3A): a :class:`FrameInferenceRunner` owns the shared
:class:`BackendManager` and, driven by ``GameStartEvent``/``GameStopEvent``
on the event bus, runs the active app's ``required_backends`` each frame —
populating ``frame.results`` **before** ``on_frame`` (``docs/architecture.md``
§4). Backends are constructed here from config-resolved device/cache settings
and registered via :func:`register_default_backends`; the runner is released
on shutdown. Apps that declare no backends leave the results empty, preserving
the pre-M2.3 behavior.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable

from PySide6.QtWidgets import QApplication

from visionplay import __version__
from visionplay.core.config import Config, load_config
from visionplay.core.event_bus import EventBus
from visionplay.core.logging_setup import setup_logging
from visionplay.core.paths import AppPaths
from visionplay.core.plugin_registry import PluginRegistry
from visionplay.ui.main_window import MainWindow
from visionplay.ui.widgets.frame_bridge import FrameBridge
from visionplay.vision.camera.camera_source import CameraSource
from visionplay.vision.camera.webcam_source import WebcamSource
from visionplay.vision.inference.backend_defaults import (
    device_from_config,
    models_dir_from_config,
    register_default_backends,
)
from visionplay.vision.inference.backend_manager import BackendManager
from visionplay.vision.inference.inference_runner import FrameInferenceRunner
from visionplay.vision.inference.model_registry import HttpModelDownloader, ModelRegistry
from visionplay.vision.pipeline.frame_bus import FramePipeline
from visionplay.vision.pipeline.frame_types import Frame

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
        apps_package: str = "visionplay.apps",
        backend_manager: BackendManager | None = None,
    ) -> None:
        """Bootstrap the headless platform layers.

        Args:
            paths: Filesystem locations; ``None`` resolves the platform
                defaults. Directories are created if missing.
            source: Camera source override for tests; ``None`` builds a
                :class:`WebcamSource` from the ``camera`` config section.
            apps_package: Dotted package the plugin registry discovers apps
                under. Defaults to the real ``visionplay.apps`` tree; tests
                pass a fixture apps package instead.
            backend_manager: Inference backend manager override for tests;
                ``None`` builds the real one from config (device, model cache)
                with the default backends registered.
        """
        self._paths = (paths if paths is not None else AppPaths.default()).ensure()
        self._config = load_config(self._paths.config_file)
        setup_logging(self._paths.log_file, level=self._config.get("app", "log_level", "INFO"))
        logger.info("VisionPlay AI v%s starting", __version__)
        self._event_bus = EventBus()
        self._registry = PluginRegistry(event_bus=self._event_bus, apps_package=apps_package)
        self._registry.discover()
        self._backend_manager = (
            backend_manager if backend_manager is not None else self._build_backend_manager()
        )
        self._inference_runner = FrameInferenceRunner(
            self._backend_manager,
            backends_for=_backends_for(self._registry),
            event_bus=self._event_bus,
        )
        self._source = source if source is not None else _webcam_from_config(self._config)
        target_fps = self._config.get("camera", "target_fps", 30)
        self._pipeline = FramePipeline(
            self._source,
            event_bus=self._event_bus,
            # The processor captures only the registry and runner — never
            # ``self`` — so the capture worker thread does not pin the whole
            # app/window/Qt-widget graph alive (which otherwise crashes the
            # offscreen Qt platform as widgets accumulate across app cycles).
            frame_processor=_compose_frame_processor(self._inference_runner, self._registry),
            target_fps=float(target_fps) if target_fps > 0 else None,
        )
        self._bridge: FrameBridge | None = None
        self._window: MainWindow | None = None

    def _build_backend_manager(self) -> BackendManager:
        """Construct the real backend manager from the ``inference`` config.

        Device and model-cache location are resolved from config; the built-in
        backends (v1: ``mediapipe.hands``) are registered against a
        checksum-verifying HTTP model registry. Nothing is downloaded or
        loaded here — that happens lazily when an app that needs a backend
        starts.
        """
        inference = self._config.section("inference")
        device = device_from_config(inference)
        models_dir = models_dir_from_config(inference, self._paths.models_dir)
        registry = ModelRegistry(models_dir, HttpModelDownloader())
        manager = BackendManager(device)
        register_default_backends(manager, registry)
        return manager

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
    def registry(self) -> PluginRegistry:
        """The plugin registry (apps discovered at construction time)."""
        return self._registry

    @property
    def backend_manager(self) -> BackendManager:
        """The inference backend manager shared across apps."""
        return self._backend_manager

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
        # Capability negotiation (M2.3): the launcher greys out apps whose
        # required_backends the manager can't satisfy right now. Set before
        # set_apps so the first render is already negotiated.
        window.launcher.set_backend_availability(self._backend_manager.is_available)
        window.launcher.set_apps(self._registry.manifests)
        window.launcher.app_launch_requested.connect(self._on_app_launch_requested)
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

    def _on_app_launch_requested(self, app_id: str) -> None:
        """Start the requested app, stopping any previously active one first.

        Connected to :attr:`~visionplay.ui.launcher.launcher_widget.LauncherWidget.
        app_launch_requested`. The registry's own exclusivity rule (M1.2)
        handles stopping whichever app was previously active; the pipeline
        needs no separate notification since its frame processor already
        always defers to :meth:`PluginRegistry.process_frame`.

        Capability guard (M2.3): the launcher already refuses to emit for a
        greyed-out app, but availability is re-checked here too — it can
        change between render and launch (e.g. a model cache wiped mid-
        session), and refusing to start beats crashing at load time.

        Args:
            app_id: The manifest id of the app the user selected.
        """
        manifest = self._registry.manifests.get(app_id)
        if manifest is not None:
            missing = [
                name
                for name in manifest.required_backends
                if not self._backend_manager.is_available(name)
            ]
            if missing:
                logger.warning(
                    "Refusing to launch app %r: required backends unavailable: %s",
                    app_id,
                    ", ".join(missing),
                )
                return
        self._registry.start(app_id)

    def shutdown(self) -> None:
        """Stop the active app, then the bridge and pipeline. Idempotent.

        Runs when the main window closes (via :attr:`MainWindow.closing`)
        and again as a belt-and-braces measure after the event loop exits.
        """
        self._registry.stop_active()
        bridge = self._bridge
        self._bridge = None
        if bridge is not None:
            bridge.stop()
        self._pipeline.stop()
        # Release backends only after the worker has stopped, so no in-flight
        # run() can touch a backend mid-teardown. Idempotent across repeat calls.
        self._inference_runner.shutdown()
        logger.info("Frame pipeline and bridge stopped; camera released")


def _backends_for(registry: PluginRegistry) -> Callable[[str], tuple[str, ...]]:
    """Build the app-id → required-backends lookup the inference runner uses.

    Returns a closure over ``registry`` only (not the app), so nothing in the
    inference/pipeline graph references the Qt window graph.
    """

    def lookup(app_id: str) -> tuple[str, ...]:
        return tuple(registry.manifests[app_id].required_backends)

    return lookup


def _compose_frame_processor(
    runner: FrameInferenceRunner, registry: PluginRegistry
) -> Callable[[Frame], Frame]:
    """Compose the pipeline's per-frame hook: backends first, then ``on_frame``.

    Runs on the capture worker thread. Inference populates ``frame.results``
    before the active plugin's ``on_frame`` reads it (``docs/architecture.md``
    §4). Both stages contain their own exceptions, honoring the pipeline's
    no-raise ``FrameProcessor`` contract. Captures only ``runner`` and
    ``registry`` so the worker thread never pins the app/window graph.
    """

    def process(frame: Frame) -> Frame:
        return registry.process_frame(runner.run(frame))

    return process


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
