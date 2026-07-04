"""Air Canvas processor: pinch-to-draw canvas state machine + frame compositing.

Pure logic, no ``PySide6`` import — unit-testable headless by feeding
synthetic frames with a
:class:`~visionplay.vision.inference.results.HandLandmarkResult` attached
(``docs/plugin-development.md``). Gesture primitives come from the shared
:mod:`visionplay.vision.gestures` toolkit (pinch distance, finger counting,
One-Euro smoothing, pixel mapping) — nothing gesture-generic is reimplemented
here.

Interaction model (MVP, M3.2):

- **Pinch = pen down.** Pinch on/off uses two thresholds (hysteresis) so the
  pen does not flap at the boundary; the smoothed index fingertip is the pen.
- **Gesture toolbar** across the top of the frame: color swatches, brush
  sizes, eraser toggle, clear, layer controls, shape/background toggles,
  help. A region activates on a pinch that *starts* inside it, or by
  hovering (dwell). Drawing is suppressed while the fingertip is in the
  toolbar strip.
- **Eraser tool**: while pinched, removes any stroke passing near the
  fingertip.
- **Open palm held** (all five fingers extended) clears the active layer.
- **Instruction strip**: usage lines, visible by default so a first-time
  user doesn't need to already know a gesture to learn the gestures.
  Toggled off/on via the "Help" toolbar cell.

Flagship features (M3.3):

- **Undo/redo**: every stroke add, erase drag, and clear is one entry on a
  command stack (:class:`_StrokeSetCommand`), each a full before/after
  snapshot of the affected layer's stroke list — simple to reason about and
  immune to index/equality edge cases. A fresh action clears the redo stack.
  Keyboard-triggered (``widget.py``), so requests arrive from the Qt thread;
  :meth:`AirCanvasProcessor.request_undo`/:meth:`request_redo` only enqueue
  onto a thread-safe queue that :meth:`process` drains on the pipeline
  worker thread — state is still mutated from exactly one thread, matching
  the vision pipeline rules in ``CLAUDE.md``.
- **Layers**: an ordered list of :class:`Layer`, index 0 painted first (the
  bottom). Add/select/toggle-visibility/reorder are toolbar actions;
  clearing and drawing always target the active layer. Compositing
  (skipping hidden layers, respecting order) happens in :meth:`_render`,
  the same place the toolbar/cursor/help strip are already drawn — keeping
  one rendering path instead of splitting raster logic between this module
  and ``widget.py``, and keeping it headless-testable per the M3.3 done-when
  criteria.
- **Shape recognition**: toggleable via the "Shapes" toolbar cell. A
  completed pen stroke is classified by :func:`classify_stroke_shape`
  (line/circle/rectangle/None) and, on a match, its point list is replaced
  by an idealized shape (:func:`_snap_stroke_shape`) — rendering code needs
  no changes since a snapped stroke is still just a point list.
- **Export**: :meth:`render_export_image` composites all visible layers
  onto either the current camera frame (BGR) or a transparent canvas
  (BGRA), toggled by the "BG" toolbar cell. Requested via
  :meth:`request_export` (same thread-safe queue as undo/redo) and written
  with ``cv2.imwrite`` when the queued command drains.

The processor draws the toolbar, strokes, and cursor onto the published
frame's image in place — the "optionally annotated" return path of
``AppPlugin.on_frame`` — so the canvas is visible through any frame
renderer. OpenCV drawing on a bounded number of pixels per frame keeps
per-frame cost bounded (CLAUDE.md vision pipeline rules).

The instruction strip is different: it must never sit on top of the
drawing surface, so instead of compositing text onto the camera image it is
rendered onto a separate black strip and appended *below* the frame with
``numpy.vstack`` — the frame handed back grows taller (a new ``Frame``, since
``Frame.width``/``Frame.height`` are immutable) but the camera/canvas area
itself stays completely clean. Toggling help off removes the strip
entirely, returning the frame to its original size.
"""

from __future__ import annotations

import logging
import math
import queue
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from visionplay.vision.gestures import (
    HandLandmarkIndex,
    OneEuroFilter,
    count_extended_fingers,
    pinch_distance,
)
from visionplay.vision.inference.results import HandLandmarkResult, HandLandmarks
from visionplay.vision.pipeline.frame_types import ColorFormat, Frame

__all__ = [
    "BRUSH_SIZES",
    "CLEAR_HOLD_FRAMES",
    "DWELL_FRAMES",
    "ERASE_RADIUS",
    "HELP_LINES",
    "PALETTE",
    "PINCH_DOWN_THRESHOLD",
    "PINCH_UP_THRESHOLD",
    "RESULTS_KEY",
    "TOOLBAR",
    "TOOLBAR_HEIGHT",
    "AirCanvasProcessor",
    "BrushColor",
    "Layer",
    "Stroke",
    "Tool",
    "ToolbarRegion",
    "classify_stroke_shape",
]

logger = logging.getLogger(__name__)

#: Key the pipeline stores this app's declared backend's output under
#: (matches ``MANIFEST.required_backends``).
RESULTS_KEY: str = "mediapipe.hands"

#: Thumb-index distance (normalized) at or below which the pen goes down.
PINCH_DOWN_THRESHOLD: float = 0.05

#: Thumb-index distance (normalized) above which a held pinch releases.
#: Wider than the down threshold on purpose — the gap is the hysteresis that
#: keeps the pen from flapping when the distance hovers near one value.
PINCH_UP_THRESHOLD: float = 0.08

#: Fraction of the frame height occupied by the gesture toolbar strip.
TOOLBAR_HEIGHT: float = 0.14

#: Consecutive frames the fingertip must hover one toolbar region to select it.
DWELL_FRAMES: int = 15

#: Consecutive open-palm frames required to clear the active layer.
CLEAR_HOLD_FRAMES: int = 20

#: Normalized radius around the fingertip within which the eraser removes strokes.
ERASE_RADIUS: float = 0.04

#: Available brush thicknesses in pixels (drawn 1:1 on the captured frame).
BRUSH_SIZES: tuple[int, ...] = (3, 6, 12)

#: Lines of the on-frame instruction panel, in display order.
HELP_LINES: tuple[str, ...] = (
    "Pinch (thumb + index) to draw",
    "Pinch or hover a toolbar tile to pick color / brush / eraser",
    "Hold an open palm to clear the active layer",
    "Layer/New/Hide/Up/Down manage layers; Shapes snaps strokes; BG toggles export background",
    "Ctrl+Z/Ctrl+Y undo/redo, Ctrl+S exports the canvas to PNG",
    "Pinch 'Help' to hide this panel",
)

#: Minimum points a completed stroke needs before shape classification runs
#: — anything shorter is too ambiguous to snap confidently.
_MIN_SHAPE_POINTS: int = 8

#: Endpoint gap (as a fraction of the stroke's bounding span) below which a
#: stroke counts as "closed" — a candidate for circle/rectangle, not line.
_CLOSED_GAP_RATIO: float = 0.15

#: Minimum circularity (``4*pi*area/perimeter**2``, 1.0 for a perfect circle)
#: to classify a closed stroke as a circle.
_CIRCLE_MIN_CIRCULARITY: float = 0.8

#: Minimum ratio of contour area to its minimum-area bounding rectangle's
#: area to classify a closed, non-circular stroke as a rectangle.
_RECTANGLE_MIN_FILL_RATIO: float = 0.82

#: Maximum perpendicular deviation from the endpoint-to-endpoint line (as a
#: fraction of that line's length) to classify an open stroke as a line.
_LINE_MAX_DEVIATION_RATIO: float = 0.06

#: Points sampled around a snapped circle's circumference.
_CIRCLE_SAMPLE_POINTS: int = 48

#: Scale factor applied to normalized ``[0, 1]`` points before handing them to
#: OpenCV contour functions — purely numerical headroom, cancelled out again
#: wherever a result is mapped back to normalized space.
_SHAPE_SCALE: float = 1000.0


class Tool(Enum):
    """The active drawing tool."""

    PEN = "pen"
    ERASER = "eraser"


@dataclass(frozen=True, slots=True)
class BrushColor:
    """One selectable pen color.

    Attributes:
        name: Display label used on the toolbar swatch.
        bgr: The color in BGR channel order (the capture-native format).
    """

    name: str
    bgr: tuple[int, int, int]


#: The selectable pen colors, in toolbar order.
PALETTE: tuple[BrushColor, ...] = (
    BrushColor("White", (255, 255, 255)),
    BrushColor("Red", (60, 60, 230)),
    BrushColor("Green", (80, 200, 80)),
    BrushColor("Blue", (230, 130, 40)),
)


@dataclass(slots=True)
class Stroke:
    """One continuous pen stroke on the canvas.

    Attributes:
        points: Fingertip path in normalized ``[0, 1]`` coordinates, so the
            canvas is independent of capture resolution. Replaced wholesale
            by :func:`_snap_stroke_shape` when shape recognition matches.
        color: Stroke color in BGR order.
        thickness: Line thickness in pixels.
    """

    color: tuple[int, int, int]
    thickness: int
    points: list[tuple[float, float]] = field(default_factory=list)


@dataclass(slots=True)
class Layer:
    """One layer of the canvas: an independently toggleable stack of strokes.

    Attributes:
        name: Display name (shown nowhere yet but useful for introspection
            and future UI); assigned on creation, never renamed by the app.
        visible: Whether :meth:`AirCanvasProcessor._render` composites this
            layer's strokes. Hidden layers are still drawable/undoable —
            only rendering skips them.
        strokes: Completed strokes on this layer, in draw order.
    """

    name: str
    visible: bool = True
    strokes: list[Stroke] = field(default_factory=list)


@dataclass(slots=True)
class _StrokeSetCommand:
    """One undoable mutation of a single layer's stroke list.

    Covers stroke add, erase, and clear uniformly: each is a transition from
    one full stroke-list snapshot to another. Snapshots are shallow copies —
    completed ``Stroke`` objects are never mutated in place after commit, so
    sharing references across ``before``/``after`` and the undo/redo stacks
    is safe.
    """

    layer: Layer
    before: list[Stroke]
    after: list[Stroke]

    def undo(self) -> None:
        self.layer.strokes = list(self.before)

    def redo(self) -> None:
        self.layer.strokes = list(self.after)


@dataclass(frozen=True, slots=True)
class ToolbarRegion:
    """One selectable cell of the gesture toolbar.

    Attributes:
        action: What activating the region does (e.g. ``"color"``,
            ``"brush"``, ``"eraser"``, ``"clear"``, ``"layer"``,
            ``"layer_add"``, ``"layer_eye"``, ``"layer_up"``,
            ``"layer_down"``, ``"shapes"``, ``"bg"``, ``"help"``).
        payload: Index into :data:`PALETTE`/:data:`BRUSH_SIZES` for
            ``"color"``/``"brush"`` actions; unused otherwise.
        x_min: Left edge in normalized coordinates.
        x_max: Right edge in normalized coordinates.
        label: Short static text drawn on the region; blank for cells whose
            label is computed per-frame instead (``"layer"``, ``"layer_eye"``).
    """

    action: str
    payload: int
    x_min: float
    x_max: float
    label: str

    def contains(self, x: float, y: float) -> bool:
        """``True`` when the normalized point is inside this region."""
        return self.x_min <= x < self.x_max and 0.0 <= y <= TOOLBAR_HEIGHT


def _build_toolbar() -> tuple[ToolbarRegion, ...]:
    """Lay the toolbar cells out evenly across the top strip.

    Order groups related tools together: colors, then Eraser right next to
    them (both are "what am I drawing/erasing with" choices), then brush
    size, then canvas/layer/mode toggles, and Help last.
    """
    cells: list[tuple[str, int, str]] = [
        ("color", index, color.name) for index, color in enumerate(PALETTE)
    ]
    cells.append(("eraser", 0, "Erase"))
    # Brush-size cells carry no text label — a size-proportional dot preview
    # is drawn instead (see _render_toolbar), which reads at a glance without
    # requiring the user to know what "px" means.
    cells += [("brush", index, "") for index in range(len(BRUSH_SIZES))]
    cells.append(("clear", 0, "Clear"))
    cells += [
        ("layer", 0, ""),  # dynamic "Ln/N" label, see _render_toolbar
        ("layer_add", 0, "New"),
        ("layer_eye", 0, ""),  # dynamic "Hide"/"Show" label
        ("layer_up", 0, "Up"),
        ("layer_down", 0, "Down"),
        ("shapes", 0, "Shapes"),
        ("bg", 0, "BG"),
        ("help", 0, "Help"),
    ]
    width = 1.0 / len(cells)
    return tuple(
        ToolbarRegion(action, payload, index * width, (index + 1) * width, label)
        for index, (action, payload, label) in enumerate(cells)
    )


#: The toolbar layout — fixed for the app's lifetime.
TOOLBAR: tuple[ToolbarRegion, ...] = _build_toolbar()


class AirCanvasProcessor:
    """Per-frame drawing/gesture logic for Air Canvas.

    ``plugin.py`` owns one instance for the app's lifetime; :meth:`start`
    resets it to a blank single-layer canvas for each run. Gesture-driven
    state transitions happen in :meth:`process`; keyboard/dialog-driven
    requests (undo, redo, export) arrive via :meth:`request_undo`,
    :meth:`request_redo`, and :meth:`request_export`, which only enqueue —
    :meth:`process` is still the sole place state is mutated, so it stays
    safe to call those from the Qt thread while frames are flowing.
    """

    def __init__(self) -> None:
        self._layers: list[Layer] = [Layer(name="Layer 1")]
        self._active_layer_index = 0
        self._active_stroke: Stroke | None = None
        self._tool = Tool.PEN
        self._color_index = 0
        self._brush_index = 0
        self._pinched = False
        self._cursor: tuple[float, float] | None = None
        # min_cutoff/beta tuned for drawing on normalized [0, 1] coordinates
        # (not pixels — speeds are naturally small, ~0.1-1.0 units/sec for a
        # deliberate stroke). A low beta keeps the filter from overreacting
        # to MediaPipe's own frame-to-frame landmark jitter (which shows up
        # as noise in the speed estimate, not just the position): strong
        # smoothing at rest so held strokes don't wobble, still responsive
        # enough to keep up with a real stroke.
        self._filter = OneEuroFilter(min_cutoff=0.8, beta=0.4)
        self._dwell_region: ToolbarRegion | None = None
        self._dwell_count = 0
        self._palm_count = 0
        self._palm_armed = True
        # Visible by default so a first-time user sees how to draw without
        # having to already know the "Help" gesture to reveal it.
        self._show_help = True
        self._shapes_enabled = False
        self._export_include_background = True
        self._undo_stack: list[_StrokeSetCommand] = []
        self._redo_stack: list[_StrokeSetCommand] = []
        self._erase_before: list[Stroke] | None = None
        self._erase_changed = False
        self._commands: queue.SimpleQueue[tuple[str, object, object]] = queue.SimpleQueue()

    # ------------------------------------------------------------------ state

    @property
    def show_help(self) -> bool:
        """``True`` while the on-frame instruction panel is visible."""
        return self._show_help

    @property
    def shapes_enabled(self) -> bool:
        """``True`` while completed pen strokes are snapped to shapes."""
        return self._shapes_enabled

    @property
    def export_include_background(self) -> bool:
        """``True`` when the next export composites onto the camera frame."""
        return self._export_include_background

    @property
    def strokes(self) -> tuple[Stroke, ...]:
        """Every stroke on the canvas across all layers, including any in
        progress — regardless of layer visibility (an inventory view, not a
        render view; see :meth:`_render` for the visibility-respecting one).
        """
        completed = tuple(stroke for layer in self._layers for stroke in layer.strokes)
        if self._active_stroke is not None:
            return (*completed, self._active_stroke)
        return completed

    @property
    def layers(self) -> tuple[Layer, ...]:
        """All layers, bottom (index 0) to top, in composite order."""
        return tuple(self._layers)

    @property
    def active_layer_index(self) -> int:
        """Index into :attr:`layers` of the layer new strokes/clears target."""
        return self._active_layer_index

    @property
    def can_undo(self) -> bool:
        """``True`` while :meth:`undo` has a command to reverse."""
        return bool(self._undo_stack)

    @property
    def can_redo(self) -> bool:
        """``True`` while :meth:`redo` has a command to reapply."""
        return bool(self._redo_stack)

    @property
    def tool(self) -> Tool:
        """The active tool."""
        return self._tool

    @property
    def brush_color(self) -> BrushColor:
        """The selected pen color."""
        return PALETTE[self._color_index]

    @property
    def brush_thickness(self) -> int:
        """The selected brush thickness in pixels."""
        return BRUSH_SIZES[self._brush_index]

    @property
    def is_pen_down(self) -> bool:
        """``True`` while a pinch is held (pen or eraser engaged)."""
        return self._pinched

    @property
    def cursor(self) -> tuple[float, float] | None:
        """Smoothed fingertip position, ``None`` when no hand is tracked."""
        return self._cursor

    @property
    def _active_layer(self) -> Layer:
        return self._layers[self._active_layer_index]

    def clear(self) -> None:
        """Erase every stroke on the active layer (undoable).

        Any stroke in progress is discarded, not committed. A no-op (and no
        undo entry pushed) when the active layer is already empty.
        """
        layer = self._active_layer
        if not layer.strokes and self._active_stroke is None:
            return
        before = list(layer.strokes)
        layer.strokes = []
        self._active_stroke = None
        self._push_command(layer, before)

    # -------------------------------------------------------------- undo/redo

    def undo(self) -> None:
        """Reverse the most recent stroke add/erase/clear, if any."""
        if not self._undo_stack:
            return
        command = self._undo_stack.pop()
        command.undo()
        self._redo_stack.append(command)

    def redo(self) -> None:
        """Reapply the most recently undone action, if any."""
        if not self._redo_stack:
            return
        command = self._redo_stack.pop()
        command.redo()
        self._undo_stack.append(command)

    def _push_command(self, layer: Layer, before: list[Stroke]) -> None:
        """Record one layer mutation and invalidate the redo stack."""
        after = list(layer.strokes)
        self._undo_stack.append(_StrokeSetCommand(layer=layer, before=before, after=after))
        self._redo_stack.clear()

    # ------------------------------------------------------- thread-safe requests

    def request_undo(self) -> None:
        """Queue an undo; applied on the next :meth:`process` call.

        Thread-safe — call this from the Qt thread (e.g. a keyboard
        shortcut in ``widget.py``) while frames are flowing on the pipeline
        worker thread.
        """
        self._commands.put(("undo", None, None))

    def request_redo(self) -> None:
        """Queue a redo; applied on the next :meth:`process` call.

        Thread-safe, see :meth:`request_undo`.
        """
        self._commands.put(("redo", None, None))

    def request_export(self, path: Path, *, include_background: bool) -> None:
        """Queue a PNG export; written on the next :meth:`process` call.

        Thread-safe, see :meth:`request_undo`. Actual file I/O happens on
        the pipeline worker thread rather than the Qt thread that presumably
        just closed a file dialog to obtain ``path``.

        Args:
            path: Destination file path.
            include_background: Composite onto the current camera frame
                (BGR) when ``True``; onto a transparent canvas (BGRA)
                otherwise.
        """
        self._commands.put(("export", path, include_background))

    def _drain_commands(self, frame: Frame) -> None:
        """Apply every queued undo/redo/export request, oldest first."""
        while True:
            try:
                name, arg1, arg2 = self._commands.get_nowait()
            except queue.Empty:
                return
            if name == "undo":
                self.undo()
            elif name == "redo":
                self.redo()
            elif name == "export":
                assert isinstance(arg1, Path)
                assert isinstance(arg2, bool)
                self._export(frame, arg1, include_background=arg2)

    def _export(self, frame: Frame, path: Path, *, include_background: bool) -> None:
        """Render and write one export image; logs (never raises) on failure."""
        image = self.render_export_image(frame, include_background=include_background)
        if not cv2.imwrite(str(path), image):
            logger.error("Failed to write exported Air Canvas image to %s", path)

    def render_export_image(
        self, frame: Frame, *, include_background: bool
    ) -> npt.NDArray[np.uint8]:
        """Composite every visible layer's strokes for export.

        Args:
            frame: Supplies the background (when ``include_background``) and
                the output dimensions otherwise.
            include_background: Draw onto a copy of ``frame``'s image (BGR)
                when ``True``; onto a transparent canvas (BGRA, alpha 0
                except where a stroke was drawn) otherwise.

        Returns:
            The image to write to disk — never the toolbar/cursor/help
            overlay, only the layered strokes.
        """
        if include_background:
            canvas = _to_bgr(frame.image, frame.color_format)
        else:
            height, width = frame.image.shape[:2]
            canvas = np.zeros((height, width, 4), dtype=np.uint8)
        for layer in self._layers:
            if not layer.visible:
                continue
            for stroke in layer.strokes:
                _draw_stroke(canvas, stroke, rgb=False)
        return canvas

    # -------------------------------------------------------------- lifecycle

    def start(self) -> None:
        """Called from ``Plugin.on_start`` — reset to a blank single-layer canvas."""
        self._layers = [Layer(name="Layer 1")]
        self._active_layer_index = 0
        self._active_stroke = None
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._erase_before = None
        self._erase_changed = False
        self._tool = Tool.PEN
        self._color_index = 0
        self._brush_index = 0
        self._show_help = True
        self._shapes_enabled = False
        self._export_include_background = True
        self._reset_tracking()

    def stop(self) -> None:
        """Called from ``Plugin.on_stop`` — drop per-run tracking state."""
        self._reset_tracking()

    def _reset_tracking(self) -> None:
        """Forget everything derived from recent frames (not the canvas)."""
        self._end_interaction()
        self._pinched = False
        self._cursor = None
        self._filter.reset()
        self._dwell_region = None
        self._dwell_count = 0
        self._palm_count = 0
        self._palm_armed = True

    # -------------------------------------------------------------- per frame

    def process(self, frame: Frame) -> Frame:
        """Advance the canvas state from one frame and composite onto it.

        Args:
            frame: The captured frame. ``frame.results.get(RESULTS_KEY)`` is
                a ``HandLandmarkResult`` when the backend ran, ``None`` when
                it is unavailable — an expected case, handled as "no hand".
                Also the sole point where queued undo/redo/export requests
                (from :meth:`request_undo`, :meth:`request_redo`,
                :meth:`request_export`) are applied.

        Returns:
            The frame to publish: the toolbar/strokes/cursor are drawn onto
            the original image in place, and — only while help is visible —
            a separate instruction strip is appended below it, which grows
            the image and so requires returning a new ``Frame`` (dimensions
            are immutable on the original).
        """
        self._drain_commands(frame)
        hand = _first_hand(frame.results.get(RESULTS_KEY))
        if hand is None:
            # Tracking lost: lift the pen and forget smoothing history so a
            # reacquired hand doesn't get a stale interpolated jump.
            self._reset_tracking()
        else:
            self._advance(hand, frame.timestamp)
        return self._render(frame)

    def _advance(self, hand: HandLandmarks, timestamp: float) -> None:
        """Run one tick of the gesture state machine for a tracked hand."""
        tip = hand.points[HandLandmarkIndex.INDEX_FINGER_TIP]
        smoothed = self._filter.filter((tip.x, tip.y), timestamp)
        x, y = float(smoothed[0]), float(smoothed[1])
        self._cursor = (x, y)

        was_pinched = self._pinched
        self._pinched = self._update_pinch(hand)
        pinch_started = self._pinched and not was_pinched

        self._update_clear_gesture(hand)

        region = _region_at(x, y)
        if region is not None:
            # In the toolbar strip: never draw; select by fresh pinch or dwell.
            self._end_interaction()
            self._update_dwell(region)
            if pinch_started or self._dwell_count >= DWELL_FRAMES:
                self._activate(region)
                self._dwell_count = 0
            return

        self._dwell_region = None
        self._dwell_count = 0
        if not self._pinched:
            self._end_interaction()
        elif self._tool is Tool.ERASER:
            if self._erase_before is None:
                self._erase_before = list(self._active_layer.strokes)
            self._erase_at(x, y)
        else:
            self._draw_to(x, y)

    def _update_pinch(self, hand: HandLandmarks) -> bool:
        """Hysteresis pinch: down at the tight threshold, up at the loose one."""
        gap = pinch_distance(hand)
        if self._pinched:
            return gap <= PINCH_UP_THRESHOLD
        return gap <= PINCH_DOWN_THRESHOLD

    def _update_clear_gesture(self, hand: HandLandmarks) -> None:
        """Clear the active layer after an open palm is held; re-arm on release."""
        if count_extended_fingers(hand) == 5:
            self._palm_count += 1
            if self._palm_armed and self._palm_count >= CLEAR_HOLD_FRAMES:
                self.clear()
                self._palm_armed = False  # one clear per palm hold
        else:
            self._palm_count = 0
            self._palm_armed = True

    def _update_dwell(self, region: ToolbarRegion) -> None:
        """Count consecutive frames hovering the same toolbar region."""
        if region is self._dwell_region:
            self._dwell_count += 1
        else:
            self._dwell_region = region
            self._dwell_count = 1

    def _activate(self, region: ToolbarRegion) -> None:
        """Apply one toolbar region's action."""
        if region.action == "color":
            self._color_index = region.payload
            self._tool = Tool.PEN  # picking a color always returns to the pen
        elif region.action == "brush":
            self._brush_index = region.payload
        elif region.action == "eraser":
            self._tool = Tool.ERASER
        elif region.action == "clear":
            self.clear()
        elif region.action == "help":
            self._show_help = not self._show_help
        elif region.action == "shapes":
            self._shapes_enabled = not self._shapes_enabled
        elif region.action == "bg":
            self._export_include_background = not self._export_include_background
        elif region.action == "layer":
            self._active_layer_index = (self._active_layer_index + 1) % len(self._layers)
        elif region.action == "layer_add":
            self._layers.append(Layer(name=f"Layer {len(self._layers) + 1}"))
            self._active_layer_index = len(self._layers) - 1
        elif region.action == "layer_eye":
            self._active_layer.visible = not self._active_layer.visible
        elif region.action == "layer_up":
            self._move_active_layer(1)
        elif region.action == "layer_down":
            self._move_active_layer(-1)

    def _move_active_layer(self, direction: int) -> None:
        """Swap the active layer with its neighbor toward ``direction``.

        A no-op at either end of the stack — there's nothing to swap with.
        """
        index = self._active_layer_index
        target = index + direction
        if not (0 <= target < len(self._layers)):
            return
        self._layers[index], self._layers[target] = self._layers[target], self._layers[index]
        self._active_layer_index = target

    def _draw_to(self, x: float, y: float) -> None:
        """Extend the active stroke to the smoothed fingertip, starting one if needed."""
        if self._active_stroke is None:
            self._active_stroke = Stroke(color=self.brush_color.bgr, thickness=self.brush_thickness)
        self._active_stroke.points.append((x, y))

    def _end_interaction(self) -> None:
        """Finalize whichever of pen-stroke/eraser-drag is in progress, if any."""
        self._end_stroke()
        self._end_erase()

    def _end_stroke(self) -> None:
        """Commit the stroke in progress, if any, to the active layer.

        When shape recognition is enabled, the stroke's points are replaced
        by an idealized snap (:func:`_snap_stroke_shape`) before committing —
        a no-match leaves the freehand points untouched.
        """
        if self._active_stroke is None:
            return
        stroke = self._active_stroke
        self._active_stroke = None
        if self._shapes_enabled:
            snapped = _snap_stroke_shape(stroke.points)
            if snapped is not None:
                stroke.points = snapped
        layer = self._active_layer
        before = list(layer.strokes)
        layer.strokes.append(stroke)
        self._push_command(layer, before)

    def _end_erase(self) -> None:
        """Finalize the eraser drag in progress, if any, as one undo entry."""
        if self._erase_before is None:
            return
        layer = self._active_layer
        before = self._erase_before
        changed = self._erase_changed
        self._erase_before = None
        self._erase_changed = False
        if changed:
            self._push_command(layer, before)

    def _erase_at(self, x: float, y: float) -> None:
        """Remove every stroke that passes within :data:`ERASE_RADIUS` of the point."""
        layer = self._active_layer
        kept = [stroke for stroke in layer.strokes if not _stroke_hit(stroke, x, y, ERASE_RADIUS)]
        if len(kept) != len(layer.strokes):
            self._erase_changed = True
        layer.strokes = kept

    # -------------------------------------------------------------- rendering

    def _render(self, frame: Frame) -> Frame:
        """Composite toolbar/layers/cursor onto the frame; append help below it.

        Only color frames are annotated — a GRAY frame passes through
        untouched rather than guessing how to draw on one channel. The
        toolbar/strokes/cursor are drawn onto ``frame.image`` in place (same
        dimensions, same ``Frame``); the instruction strip, when visible, is
        concatenated below via ``numpy.vstack`` and requires a new ``Frame``
        since it changes the image's height.
        """
        if frame.color_format is ColorFormat.GRAY:
            return frame
        image = frame.image
        rgb = frame.color_format is ColorFormat.RGB
        self._render_toolbar(image, rgb)
        for layer in self._layers:
            if not layer.visible:
                continue
            for stroke in layer.strokes:
                _draw_stroke(image, stroke, rgb)
        if self._active_stroke is not None and self._active_layer.visible:
            _draw_stroke(image, self._active_stroke, rgb)
        self._render_cursor(image, rgb)
        if not self._show_help:
            return frame
        strip = _build_instructions_strip(image.shape[1], rgb)
        combined = np.vstack([image, strip])
        annotated = Frame.from_image(frame.frame_id, frame.timestamp, combined, frame.color_format)
        annotated.results.update(frame.results)
        return annotated

    def _render_toolbar(self, image: npt.NDArray[np.uint8], rgb: bool) -> None:
        """Draw the toolbar strip: swatches, labels, and selection highlights."""
        height, width = image.shape[:2]
        strip_bottom = int(TOOLBAR_HEIGHT * height)
        cv2.rectangle(image, (0, 0), (width, strip_bottom), _color((40, 40, 40), rgb), -1)
        for region in TOOLBAR:
            x0, x1 = int(region.x_min * width), int(region.x_max * width)
            if region.action == "color":
                swatch = PALETTE[region.payload].bgr
                cv2.rectangle(
                    image, (x0 + 2, 2), (x1 - 2, strip_bottom - 2), _color(swatch, rgb), -1
                )
            elif region.action == "brush":
                # A dot sized to the actual stroke thickness reads instantly
                # ("this tile draws lines this thick"), unlike a "12px" label.
                center = ((x0 + x1) // 2, strip_bottom // 2)
                radius = max(BRUSH_SIZES[region.payload] // 2, 2)
                cv2.circle(image, center, radius, _color((235, 235, 235), rgb), -1)
            elif region.action == "layer":
                text = f"L{self._active_layer_index + 1}/{len(self._layers)}"
                self._draw_toolbar_text(image, text, x0, strip_bottom, rgb)
            elif region.action == "layer_eye":
                text = "Hide" if self._active_layer.visible else "Show"
                self._draw_toolbar_text(image, text, x0, strip_bottom, rgb)
            if self._region_is_selected(region):
                cv2.rectangle(
                    image, (x0 + 2, 2), (x1 - 2, strip_bottom - 2), _color((255, 255, 255), rgb), 2
                )
            if region.label:
                self._draw_toolbar_text(image, region.label, x0, strip_bottom, rgb)

    def _draw_toolbar_text(
        self, image: npt.NDArray[np.uint8], text: str, x0: int, strip_bottom: int, rgb: bool
    ) -> None:
        """Draw one line of small toolbar text at a cell's bottom-left corner."""
        cv2.putText(
            image,
            text,
            (x0 + 4, strip_bottom - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            _color((220, 220, 220), rgb),
            1,
            cv2.LINE_AA,
        )

    def _region_is_selected(self, region: ToolbarRegion) -> bool:
        """Whether a toolbar region reflects the current selection state."""
        if region.action == "color":
            return self._tool is Tool.PEN and region.payload == self._color_index
        if region.action == "brush":
            return region.payload == self._brush_index
        if region.action == "eraser":
            return self._tool is Tool.ERASER
        if region.action == "help":
            return self._show_help
        if region.action == "shapes":
            return self._shapes_enabled
        if region.action == "bg":
            return self._export_include_background
        return False

    def _render_cursor(self, image: npt.NDArray[np.uint8], rgb: bool) -> None:
        """Draw the fingertip cursor: a ring, filled while the pen is down."""
        if self._cursor is None:
            return
        height, width = image.shape[:2]
        x = min(max(int(self._cursor[0] * width), 0), width - 1)
        y = min(max(int(self._cursor[1] * height), 0), height - 1)
        if self._tool is Tool.ERASER:
            radius = max(int(ERASE_RADIUS * width), 4)
            cv2.circle(image, (x, y), radius, _color((200, 200, 200), rgb), 2)
        else:
            fill = -1 if self._pinched else 2
            cv2.circle(image, (x, y), 6, _color(self.brush_color.bgr, rgb), fill)


# ------------------------------------------------------------------ helpers


def _first_hand(result: object) -> HandLandmarks | None:
    """The first detected hand, or ``None`` for absent/empty/foreign results."""
    if isinstance(result, HandLandmarkResult) and result.hands:
        return result.hands[0]
    return None


def _region_at(x: float, y: float) -> ToolbarRegion | None:
    """The toolbar region under a normalized point, if any."""
    for region in TOOLBAR:
        if region.contains(x, y):
            return region
    return None


def _stroke_hit(stroke: Stroke, x: float, y: float, radius: float) -> bool:
    """``True`` when any point of the stroke lies within ``radius`` of the point."""
    return any(math.hypot(px - x, py - y) <= radius for px, py in stroke.points)


def _color(bgr: tuple[int, int, int], rgb: bool) -> tuple[int, int, int]:
    """Return the color in the frame's channel order."""
    return (bgr[2], bgr[1], bgr[0]) if rgb else bgr


def _draw_stroke(image: npt.NDArray[np.uint8], stroke: Stroke, rgb: bool) -> None:
    """Draw one stroke as pixels: a dot for a single point, a polyline otherwise.

    Works on both a 3-channel (BGR/RGB) and a 4-channel (BGRA export) canvas
    — a 4-channel image gets an appended opaque alpha so drawn pixels are
    fully visible against a transparent background.
    """
    if not stroke.points:
        return
    height, width = image.shape[:2]
    color: tuple[int, ...] = _color(stroke.color, rgb)
    if image.shape[2] == 4:
        color = (*color, 255)
    pixels = np.array(
        [(int(px * width), int(py * height)) for px, py in stroke.points], dtype=np.int32
    )
    if len(pixels) == 1:
        center = (int(pixels[0][0]), int(pixels[0][1]))
        cv2.circle(image, center, max(stroke.thickness // 2, 1), color, -1)
    else:
        cv2.polylines(image, [pixels], False, color, stroke.thickness, cv2.LINE_AA)


def _to_bgr(image: npt.NDArray[np.uint8], color_format: ColorFormat) -> npt.NDArray[np.uint8]:
    """Return a true-BGR copy of a captured frame's image, any source format."""
    if color_format is ColorFormat.RGB:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR).astype(np.uint8)
    if color_format is ColorFormat.GRAY:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR).astype(np.uint8)
    return image.copy()


def _build_instructions_strip(width: int, rgb: bool) -> npt.NDArray[np.uint8]:
    """Build a standalone black strip of ``HELP_LINES``, centered, one per row.

    A separate image ``vstack``-ed below the frame, not drawn onto it — the
    camera/canvas area stays completely clean; this is purely additional
    real estate below it.
    """
    line_height = 22
    height = 14 + line_height * len(HELP_LINES)
    strip = np.zeros((height, width, 3), dtype=np.uint8)
    strip[:] = _color((20, 20, 20), rgb)
    for index, line in enumerate(HELP_LINES):
        (text_width, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        x = max((width - text_width) // 2, 8)
        baseline = 22 + index * line_height
        cv2.putText(
            strip,
            line,
            (x, baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            _color((235, 235, 235), rgb),
            1,
            cv2.LINE_AA,
        )
    return strip


# --------------------------------------------------------------- shape recognition


def classify_stroke_shape(points: Sequence[tuple[float, float]]) -> str | None:
    """Classify a completed stroke's points as a near-line/circle/rectangle.

    Pure geometry over normalized ``[0, 1]`` points — no OpenCV image
    involved, just its contour-measurement functions. A stroke whose
    endpoints are close together (relative to its own bounding span) is
    treated as "closed" and tested against circle/rectangle; otherwise it is
    tested against line.

    Args:
        points: The stroke's fingertip path, in draw order.

    Returns:
        ``"line"``, ``"circle"``, ``"rectangle"``, or ``None`` when nothing
        matches confidently enough to snap.
    """
    if len(points) < _MIN_SHAPE_POINTS:
        return None
    array = np.asarray(points, dtype=np.float64) * _SHAPE_SCALE
    span = float(max(np.ptp(array[:, 0]), np.ptp(array[:, 1])))
    if span <= 0:
        return None
    gap = float(math.hypot(*(array[-1] - array[0])))
    closed = gap <= _CLOSED_GAP_RATIO * span

    if closed:
        return _classify_closed(array)
    return _classify_open(array, span)


def _classify_closed(array: npt.NDArray[np.float64]) -> str | None:
    """Circle/rectangle test for a stroke whose ends meet."""
    contour = array.astype(np.float32).reshape(-1, 1, 2)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if area <= 0 or perimeter <= 0:
        return None
    circularity = 4.0 * math.pi * area / (perimeter * perimeter)
    if circularity >= _CIRCLE_MIN_CIRCULARITY:
        return "circle"
    rect = cv2.minAreaRect(contour)
    rect_area = rect[1][0] * rect[1][1]
    if rect_area > 0 and area / rect_area >= _RECTANGLE_MIN_FILL_RATIO:
        return "rectangle"
    return None


def _classify_open(array: npt.NDArray[np.float64], span: float) -> str | None:
    """Line test for a stroke whose ends don't meet."""
    direction = array[-1] - array[0]
    length = float(math.hypot(*direction))
    if length <= 0:
        return None
    unit = direction / length
    normal = np.array([-unit[1], unit[0]])
    deviations = np.abs((array - array[0]) @ normal)
    if float(deviations.max()) <= _LINE_MAX_DEVIATION_RATIO * span:
        return "line"
    return None


def _snap_stroke_shape(
    points: Sequence[tuple[float, float]],
) -> list[tuple[float, float]] | None:
    """Replace a stroke's points with an idealized shape, or ``None`` if unmatched.

    The snapped point list is deliberately still "just points" — a circle
    becomes a densely sampled polygon, a rectangle its four corners plus a
    closing point — so :func:`_draw_stroke` needs no shape-aware branch.
    """
    kind = classify_stroke_shape(points)
    if kind is None:
        return None
    array = np.asarray(points, dtype=np.float64)
    if kind == "line":
        start = (float(array[0][0]), float(array[0][1]))
        end = (float(array[-1][0]), float(array[-1][1]))
        return [start, end]
    if kind == "circle":
        center = array.mean(axis=0)
        radius = float(np.mean(np.linalg.norm(array - center, axis=1)))
        angles = np.linspace(0.0, 2.0 * math.pi, _CIRCLE_SAMPLE_POINTS, endpoint=True)
        return [
            (
                float(center[0] + radius * math.cos(angle)),
                float(center[1] + radius * math.sin(angle)),
            )
            for angle in angles
        ]
    # rectangle
    contour = (array * _SHAPE_SCALE).astype(np.float32).reshape(-1, 1, 2)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect) / _SHAPE_SCALE
    corners = [(float(px), float(py)) for px, py in box]
    return [*corners, corners[0]]
