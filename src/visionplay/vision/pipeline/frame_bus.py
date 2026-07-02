"""Frame bus and capture worker: camera frames off the main thread (M0.5).

Two pieces, matching ``docs/architecture.md`` §4:

- :class:`FrameBus` — a bounded, latest-wins handoff between the capture
  worker (producer) and whoever renders/processes frames (consumer). Under
  back-pressure it **drops the oldest frame instead of queueing**: a slow
  consumer sees fewer frames, never staler ones, and the producer never
  blocks.
- :class:`FramePipeline` — owns a dedicated worker thread that drives a
  :class:`~visionplay.vision.camera.camera_source.CameraSource`
  (open → read loop → release, all on the worker), runs the optional
  frame processor (M1.4 — the active app's ``on_frame`` seam, see below),
  publishes each frame to the bus, and optionally announces
  :class:`~visionplay.core.events.FrameReadyEvent` metadata on the
  :class:`~visionplay.core.event_bus.EventBus`.

Frame processor seam (M1.4, ``docs/architecture.md`` §4 data flow): between
capture and publish, the worker passes each frame through a settable
``Callable[[Frame], Frame]`` — in production the plugin registry's
``process_frame``, which dispatches to the active app's ``on_frame`` under
the registry's own guard. The pipeline deliberately adds **no** try/except
of its own around this call: containment policy (log, count consecutive
failures, stop the app) lives in one place, the registry (M1.2), and the
processor contract here is simply "must not raise". With no processor set
(or ``None``), captured frames pass through unchanged — the M0.5/M0.6
behavior.

Threading contract: the camera source is touched **only** by the worker
thread. Consumers interact with the pipeline through :class:`FrameBus`
(thread-safe) and the start/stop lifecycle. Nothing here imports Qt — the
Qt-signal bridge that carries frames into widgets is M0.6's job, layered on
top of this bus.

Error philosophy: a capture failure must never crash the process. Transient
read errors are tolerated up to a consecutive-failure limit; a persistent
failure (or a failure to open) stops the pipeline cleanly — camera
released, bus closed so consumers unblock, and the error preserved on
:attr:`FramePipeline.error` for the UI to present.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from types import TracebackType

from visionplay.core.event_bus import EventBus
from visionplay.core.events import FrameReadyEvent
from visionplay.vision.camera.camera_source import CameraError, CameraSource
from visionplay.vision.pipeline.frame_types import Frame

__all__ = ["FrameBus", "FramePipeline", "FrameProcessor"]

logger = logging.getLogger(__name__)

#: Per-frame hook run on the worker thread between capture and publish.
#: Must return the frame to publish and must not raise — exception
#: containment is the caller's contract (the plugin registry's guard, M1.2),
#: not the pipeline's.
FrameProcessor = Callable[[Frame], Frame]

#: Thread name for the capture worker (shows up in debuggers/leak checks).
_WORKER_THREAD_NAME: str = "visionplay-frame-pipeline"


class FrameBus:
    """Bounded, latest-wins, thread-safe frame handoff.

    Capacity is small by design (default 1: pure latest-frame semantics).
    :meth:`publish` never blocks — when the bus is full the **oldest**
    pending frame is discarded and counted in :attr:`frames_dropped`. This
    is the frame-skipping policy from the Phase 0 checklist: slow consumers
    drop frames, they don't build a stale queue.

    A bus can be :meth:`close`\\ d exactly once, by the producer side, to
    signal end of stream; blocked consumers wake and drain whatever is
    still pending, then receive ``None``.
    """

    def __init__(self, capacity: int = 1) -> None:
        """Create an open, empty bus.

        Args:
            capacity: Maximum pending frames (>= 1). 1 means a consumer
                only ever sees the newest frame.

        Raises:
            ValueError: If ``capacity`` is less than 1.
        """
        if capacity < 1:
            raise ValueError(f"FrameBus capacity must be >= 1, got {capacity}")
        self._frames: deque[Frame] = deque(maxlen=capacity)
        self._condition = threading.Condition()
        self._closed = False
        self._published = 0
        self._dropped = 0

    @property
    def closed(self) -> bool:
        """``True`` once the producer has closed the bus (end of stream)."""
        with self._condition:
            return self._closed

    @property
    def frames_published(self) -> int:
        """Total frames accepted via :meth:`publish` (drops included)."""
        with self._condition:
            return self._published

    @property
    def frames_dropped(self) -> int:
        """Frames evicted unseen because a newer frame arrived first."""
        with self._condition:
            return self._dropped

    def publish(self, frame: Frame) -> None:
        """Make ``frame`` available to the consumer; never blocks.

        If the bus is full, the oldest pending frame is evicted (counted
        as dropped). Publishing on a closed bus is a silent no-op — the
        worker may race a shutdown, and losing a frame at teardown is the
        expected outcome, not an error.

        Args:
            frame: The frame to hand off.
        """
        with self._condition:
            if self._closed:
                return
            if len(self._frames) == self._frames.maxlen:
                self._dropped += 1  # deque(maxlen) evicts the oldest on append
            self._frames.append(frame)
            self._published += 1
            self._condition.notify_all()

    def get(self, timeout: float | None = None) -> Frame | None:
        """Wait for and return the oldest pending frame.

        Blocks until a frame is available, the bus closes, or ``timeout``
        expires. After close, pending frames are still drained in order;
        only an empty closed bus returns ``None`` immediately.

        Args:
            timeout: Max seconds to wait; ``None`` waits indefinitely
                (a close still wakes the call).

        Returns:
            The next frame, or ``None`` on timeout or on end of stream.
        """
        with self._condition:
            if not self._condition.wait_for(lambda: self._frames or self._closed, timeout=timeout):
                return None  # timed out
            if self._frames:
                return self._frames.popleft()
            return None  # closed and drained

    def close(self) -> None:
        """Mark end of stream and wake all blocked consumers. Idempotent."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class FramePipeline:
    """Dedicated capture worker: drives a camera source, feeds a :class:`FrameBus`.

    Lifecycle: ``start()`` spawns the worker (which opens the source),
    ``stop()`` signals it and joins — releasing the camera and closing the
    bus on the way out, whatever the exit reason (stop request, end of
    stream, persistent capture failure). The class is also a context
    manager: ``with pipeline:`` starts on entry and stops on exit.

    A stopped pipeline may be :meth:`start`\\ ed again with a fresh bus;
    error state resets on restart.
    """

    def __init__(
        self,
        source: CameraSource,
        bus: FrameBus | None = None,
        *,
        event_bus: EventBus | None = None,
        frame_processor: FrameProcessor | None = None,
        target_fps: float | None = None,
        max_consecutive_failures: int = 3,
    ) -> None:
        """Wire the pipeline; nothing runs until :meth:`start`.

        Args:
            source: Capture source, used exclusively by the worker thread.
            bus: Frame handoff to publish into. ``None`` creates a
                capacity-1 (latest-frame-only) bus, exposed via :attr:`bus`.
            event_bus: If given, a :class:`FrameReadyEvent` (metadata only,
                never pixels) is published per captured frame. Handlers run
                on the worker thread and must be fast.
            frame_processor: Per-frame hook run on the worker thread after
                capture, before publish — the active plugin's ``on_frame``
                seam (see :data:`FrameProcessor` for its contract). ``None``
                (the default) publishes captured frames unchanged. Can also
                be set/cleared later via :meth:`set_frame_processor`.
            target_fps: FPS governor — upper bound on capture rate. ``None``
                captures as fast as the source delivers.
            max_consecutive_failures: Consecutive ``CameraError`` reads
                tolerated before the pipeline gives up and stops.

        Raises:
            ValueError: If ``target_fps`` or ``max_consecutive_failures``
                is not positive.
        """
        if target_fps is not None and target_fps <= 0:
            raise ValueError(f"target_fps must be positive, got {target_fps}")
        if max_consecutive_failures < 1:
            raise ValueError(
                f"max_consecutive_failures must be >= 1, got {max_consecutive_failures}"
            )
        self._source = source
        self._bus = bus if bus is not None else FrameBus(capacity=1)
        self._event_bus = event_bus
        self._frame_processor = frame_processor
        self._interval = None if target_fps is None else 1.0 / target_fps
        self._max_failures = max_consecutive_failures
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._error: Exception | None = None
        self._frames_captured = 0

    @property
    def bus(self) -> FrameBus:
        """The frame handoff consumers read from."""
        return self._bus

    @property
    def error(self) -> Exception | None:
        """The failure that stopped the pipeline, if any (``None`` while healthy)."""
        return self._error

    @property
    def frames_captured(self) -> int:
        """Frames successfully read from the source since the last start."""
        return self._frames_captured

    def is_running(self) -> bool:
        """Return ``True`` while the worker thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def set_frame_processor(self, processor: FrameProcessor | None) -> None:
        """Install (or clear) the per-frame hook; safe while running.

        Callable from any thread: a single attribute rebind is atomic under
        the GIL, and the worker re-reads the attribute once per frame — the
        new processor simply takes effect from the next captured frame.

        Args:
            processor: The hook to run on each captured frame (see
                :data:`FrameProcessor` for its no-raise contract), or
                ``None`` to restore plain passthrough.
        """
        self._frame_processor = processor

    def start(self) -> None:
        """Spawn the worker thread; it opens the source and begins capturing.

        Returns immediately — an open failure surfaces asynchronously via
        :attr:`error` and a closed :attr:`bus`, not as a raise here.

        Raises:
            RuntimeError: If the pipeline is already running.
        """
        if self.is_running():
            raise RuntimeError("FramePipeline is already running")
        self._stop_event.clear()
        self._error = None
        self._frames_captured = 0
        self._thread = threading.Thread(target=self._run, name=_WORKER_THREAD_NAME, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the worker to exit and join it. Idempotent, safe pre-start.

        The worker releases the camera and closes the bus on its way out.

        Args:
            timeout: Max seconds to wait for the worker to finish.

        Raises:
            RuntimeError: If the worker fails to exit within ``timeout`` —
                a leaked capture thread must be loud, never silent.
        """
        thread = self._thread
        if thread is None:
            return
        self._stop_event.set()
        thread.join(timeout=timeout)
        if thread.is_alive():
            raise RuntimeError(f"Frame pipeline worker did not stop within {timeout}s")
        self._thread = None

    def __enter__(self) -> FramePipeline:
        """Start the pipeline and return it (``with pipeline as p:``)."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Stop the pipeline on scope exit, exception or not."""
        self.stop()

    # --- worker thread -------------------------------------------------

    def _run(self) -> None:
        """Worker body: open → governed read loop → guaranteed cleanup."""
        try:
            self._source.open()
        except CameraError as exc:
            logger.error("Camera source failed to open: %s", exc)
            self._error = exc
            self._bus.close()
            return
        try:
            self._capture_loop()
        except Exception as exc:  # worker must never die loudly
            logger.exception("Unexpected error in frame pipeline worker")
            self._error = exc
        finally:
            self._source.release()
            self._bus.close()

    def _capture_loop(self) -> None:
        """Read frames until stop, end of stream, or persistent failure."""
        failures = 0
        next_deadline = time.monotonic()
        while not self._stop_event.is_set():
            if self._interval is not None:
                # FPS governor: wait out the remainder of this frame's slot,
                # but wake immediately on a stop request.
                delay = next_deadline - time.monotonic()
                if delay > 0 and self._stop_event.wait(delay):
                    return
                next_deadline = max(next_deadline + self._interval, time.monotonic())
            try:
                frame = self._source.read()
            except CameraError as exc:
                failures += 1
                if failures >= self._max_failures:
                    logger.error(
                        "Camera read failed %d times in a row, stopping pipeline: %s",
                        failures,
                        exc,
                    )
                    self._error = exc
                    return
                logger.warning("Camera read failed (%d/%d): %s", failures, self._max_failures, exc)
                continue
            failures = 0
            if frame is None:
                logger.info("Camera source reached end of stream")
                return
            self._frames_captured += 1
            # M1.4 seam: capture → active plugin's on_frame → publish. Read
            # once per frame so a concurrent set_frame_processor is safe. No
            # try/except here by design — the registry's guard (M1.2) owns
            # exception containment, and duplicating it would split policy.
            processor = self._frame_processor
            if processor is not None:
                frame = processor(frame)
            self._bus.publish(frame)
            self._announce(frame)

    def _announce(self, frame: Frame) -> None:
        """Publish frame metadata on the event bus; never let a handler kill capture."""
        if self._event_bus is None:
            return
        try:
            self._event_bus.publish(FrameReadyEvent(frame_index=frame.frame_id))
        except Exception:  # subscriber bugs must not stop capture
            logger.exception("FrameReadyEvent handler raised; continuing capture")
