"""Thread-safe bridge from the frame pipeline into Qt signal/slot land.

The vision layer is Qt-free by design: :class:`~visionplay.vision.pipeline.
frame_bus.FrameBus` is a blocking, pull-based handoff. Qt widgets, however,
must only be touched from the main thread. :class:`FrameBridge` closes that
gap — it runs a small consumer thread that pulls frames off the bus and
re-emits them as Qt signals. Because the emitting thread differs from the
receiving widget's thread affinity, Qt auto-selects a **queued connection**,
which is exactly the thread-safe delivery the M0.6 checklist requires:

    capture worker → FrameBus → FrameBridge thread → queued Qt signal → widget

The bridge deliberately takes the whole
:class:`~visionplay.vision.pipeline.frame_bus.FramePipeline`, not just its
bus: when the stream ends it needs :attr:`FramePipeline.error` to tell the
UI *why* (camera unplugged vs. normal stop), and that message travels on the
:attr:`stream_ended` signal so no Qt object is ever touched off-thread.
"""

from __future__ import annotations

import logging
import threading

from PySide6.QtCore import QObject, Signal

from visionplay.vision.pipeline.frame_bus import FramePipeline

__all__ = ["FrameBridge"]

logger = logging.getLogger(__name__)

#: Thread name for the bridge consumer (shows up in debuggers/leak checks).
_BRIDGE_THREAD_NAME: str = "visionplay-frame-bridge"

#: How often the consumer loop re-checks the stop flag while the bus is idle.
_POLL_TIMEOUT_S: float = 0.1


class FrameBridge(QObject):
    """Consumes a pipeline's frame bus and re-emits frames as Qt signals.

    Lifecycle mirrors :class:`FramePipeline`: ``start()`` spawns the
    consumer thread, ``stop()`` signals it and joins. ``stop()`` is
    idempotent and safe before ``start()``.
    """

    #: A new frame is available. Payload is a
    #: :class:`~visionplay.vision.pipeline.frame_types.Frame`. Emitted from
    #: the bridge thread — connect widget slots normally and Qt queues the
    #: delivery onto the widget's thread.
    frame_ready = Signal(object)

    #: The pipeline's stream ended (camera failure or end of stream).
    #: Payload is a user-presentable message. Not emitted on ``stop()``.
    stream_ended = Signal(str)

    def __init__(self, pipeline: FramePipeline, parent: QObject | None = None) -> None:
        """Wire the bridge to a pipeline; nothing runs until :meth:`start`.

        Args:
            pipeline: The pipeline whose bus to consume. The bridge only
                reads (``bus.get`` / ``error``); it never controls the
                pipeline's lifecycle.
            parent: Optional Qt parent.
        """
        super().__init__(parent)
        self._pipeline = pipeline
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def is_running(self) -> bool:
        """Return ``True`` while the consumer thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Spawn the consumer thread.

        Raises:
            RuntimeError: If the bridge is already running.
        """
        if self.is_running():
            raise RuntimeError("FrameBridge is already running")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name=_BRIDGE_THREAD_NAME, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the consumer to exit and join it. Idempotent, safe pre-start.

        Args:
            timeout: Max seconds to wait for the thread to finish.

        Raises:
            RuntimeError: If the thread fails to exit within ``timeout`` —
                a leaked consumer thread must be loud, never silent.
        """
        thread = self._thread
        if thread is None:
            return
        self._stop_event.set()
        thread.join(timeout=timeout)
        if thread.is_alive():
            raise RuntimeError(f"Frame bridge consumer did not stop within {timeout}s")
        self._thread = None

    # --- consumer thread -------------------------------------------------

    def _run(self) -> None:
        """Consumer body: pull frames until stopped or the bus closes."""
        bus = self._pipeline.bus
        while not self._stop_event.is_set():
            frame = bus.get(timeout=_POLL_TIMEOUT_S)
            if frame is not None:
                self.frame_ready.emit(frame)
            elif bus.closed:
                error = self._pipeline.error
                message = str(error) if error is not None else "Camera stream ended."
                logger.info("Frame stream ended: %s", message)
                self.stream_ended.emit(message)
                return
