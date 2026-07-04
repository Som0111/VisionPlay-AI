"""Fruit Ninja processor: gesture-driven slicing game simulation + frame compositing.

Pure logic, no ``PySide6`` import — unit-testable headless by feeding
synthetic frames with a
:class:`~visionplay.vision.inference.results.HandLandmarkResult` attached,
the same pattern as ``air_canvas/processor.py``. Gesture primitives come
from the shared :mod:`visionplay.vision.gestures` toolkit: the smoothed
index fingertip feeds a
:class:`~visionplay.vision.gestures.VelocityTracker`, whose ``speed`` gates
whether a frame-to-frame fingertip segment counts as a slice — a fast swipe
cuts, a slow drift just leaves a fading trail.

Game model:

- **Pieces** (fruit or bomb) spawn from the bottom of the frame on a
  randomized interval and follow simple parabolic motion (constant
  horizontal velocity, constant downward gravity) in normalized ``[0, 1]``
  frame coordinates, so the game is resolution-independent like Air
  Canvas's canvas.
- **Slicing**: each frame, if the fingertip's tracked speed clears
  :data:`SLICE_SPEED_THRESHOLD`, the segment from the previous smoothed
  fingertip position to the current one is tested against every live
  piece's circle via :func:`segment_hits_circle`. A sliced fruit becomes
  two drifting halves (a pure visual, no further collision) and awards
  score; a sliced bomb ends the run immediately.
- **Combo**: consecutive fruit slices land within :data:`COMBO_WINDOW`
  seconds of each other to keep the multiplier climbing; a longer gap
  resets it to 1.
- **Misses**: a live fruit that falls past the bottom edge unsliced costs
  one life; running out of lives ends the run the same way a sliced bomb
  does, via :meth:`_end_run`.
- **States**: ``READY`` (no simulation running) -> ``PLAYING`` (spawner +
  physics active) -> ``GAME_OVER`` (frozen, cause/score on screen).
  :meth:`request_start` is the single thread-safe entry point back into
  ``PLAYING`` from either state — a keyboard/click affordance in
  ``widget.py`` calls it while the frame loop runs on the pipeline worker
  thread, following the same enqueue-then-drain pattern as Air Canvas's
  undo/redo (``CLAUDE.md`` vision pipeline rules: state is mutated from
  exactly one thread).

All game state is drawn onto the published frame's image in place (pieces,
halves, blade trail, HUD, start/game-over overlay) — the same "optionally
annotated frame" return path ``AppPlugin.on_frame`` uses — so ``widget.py``
stays a dumb frame renderer, per ``docs/plugin-development.md``.
"""

from __future__ import annotations

import logging
import math
import queue
import random
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
import numpy.typing as npt

from visionplay.vision.gestures import HandLandmarkIndex, OneEuroFilter, VelocityTracker
from visionplay.vision.inference.results import HandLandmarkResult, HandLandmarks
from visionplay.vision.pipeline.frame_types import ColorFormat, Frame

__all__ = [
    "BASE_SCORE",
    "BOMB_PROBABILITY",
    "COMBO_WINDOW",
    "GRAVITY",
    "INITIAL_LIVES",
    "MISS_Y",
    "PIECE_RADIUS",
    "RESULTS_KEY",
    "SLICE_SPEED_THRESHOLD",
    "SPAWN_INTERVAL_RANGE",
    "FruitNinjaProcessor",
    "GameState",
    "Half",
    "Piece",
    "PieceKind",
    "advance_piece",
    "segment_hits_circle",
    "spawn_piece",
]

logger = logging.getLogger(__name__)

#: Key the pipeline stores this app's declared backend's output under
#: (matches ``MANIFEST.required_backends``).
RESULTS_KEY: str = "mediapipe.hands"

#: Downward acceleration applied to every piece, in normalized units/s^2 —
#: tuned so a piece spawned near the bottom arcs up through roughly the
#: frame's middle before falling back off, at a believable flight time.
GRAVITY: float = 1.7

#: Random interval between spawn attempts, in seconds.
SPAWN_INTERVAL_RANGE: tuple[float, float] = (0.6, 1.1)

#: Fraction of spawns that are a bomb instead of a fruit.
BOMB_PROBABILITY: float = 0.15

#: Piece radius, normalized to frame width/height.
PIECE_RADIUS: float = 0.055

#: Lives a fresh run starts with.
INITIAL_LIVES: int = 3

#: Fingertip speed (normalized units/second) a frame-to-frame segment must
#: clear to count as a slice — well above Air Canvas's deliberate ~0.1-1.0
#: drawing speeds, so a slow drift across a piece never slices it.
SLICE_SPEED_THRESHOLD: float = 1.6

#: Gap between consecutive fruit slices, in seconds, that still counts
#: toward the combo; a longer gap resets the multiplier to 1.
COMBO_WINDOW: float = 0.6

#: Score awarded per fruit slice, multiplied by the current combo count.
BASE_SCORE: float = 10.0

#: How long a sliced half keeps drifting/rendering before it's dropped.
HALF_LIFETIME: float = 0.6

#: How long a blade-trail point is kept for rendering (not for slicing).
TRAIL_LIFETIME: float = 0.15

#: Largest elapsed time treated as one physics step. A pipeline frame-skip
#: (a slow app's frames are dropped, not queued — ``CLAUDE.md``) can leave a
#: large real gap between two processed frames; without this clamp a piece
#: would teleport or its velocity would blow up for one step instead of
#: just advancing a little less far that tick.
_MAX_DT: float = 0.1

#: Below this normalized y, a piece has fallen off the bottom of the frame.
MISS_Y: float = 1.08

#: Piece colors (BGR), cycled through for spawned fruit.
_FRUIT_COLORS: tuple[tuple[int, int, int], ...] = (
    (60, 200, 240),  # orange
    (60, 60, 230),  # red
    (70, 200, 70),  # green
    (40, 210, 230),  # yellow
)

#: Bomb fill color (BGR) — dark, unmistakably not a fruit.
_BOMB_COLOR: tuple[int, int, int] = (40, 40, 40)

#: Keys that (re)start a run, forwarded by ``widget.py``.
_START_MESSAGE: str = "Press SPACE to start"


class GameState(Enum):
    """The run's current phase."""

    READY = "ready"
    PLAYING = "playing"
    GAME_OVER = "game_over"


class PieceKind(Enum):
    """What a spawned piece is."""

    FRUIT = "fruit"
    BOMB = "bomb"


@dataclass(slots=True)
class Piece:
    """One airborne fruit or bomb.

    Attributes:
        kind: Fruit or bomb.
        x: Horizontal position, normalized ``[0, 1]``.
        y: Vertical position, normalized (0 top, 1 bottom; can exceed 1
            briefly while falling off-frame before being dropped).
        vx: Horizontal velocity, normalized units/second (constant).
        vy: Vertical velocity, normalized units/second (grows via gravity;
            negative is upward).
        radius: Collision/render radius, normalized.
        color: BGR render color.
    """

    kind: PieceKind
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    color: tuple[int, int, int]


@dataclass(slots=True)
class Half:
    """One drifting half of a sliced fruit — purely a rendering artifact.

    Attributes:
        x: Horizontal position, normalized.
        y: Vertical position, normalized.
        vx: Horizontal velocity, normalized units/second.
        vy: Vertical velocity, normalized units/second.
        radius: Render radius, normalized (inherited from the sliced piece).
        color: BGR render color (inherited from the sliced piece).
        side: ``-1`` for the half thrown left, ``1`` for the half thrown
            right — a rendering hint for which way the arc faces.
        age: Seconds since the slice; dropped past :data:`HALF_LIFETIME`.
    """

    x: float
    y: float
    vx: float
    vy: float
    radius: float
    color: tuple[int, int, int]
    side: int
    age: float = 0.0


@dataclass(slots=True)
class TrailPoint:
    """One recent fingertip position, kept only long enough to render a fading trail."""

    x: float
    y: float
    age: float = 0.0


class FruitNinjaProcessor:
    """Per-frame game simulation + gesture tracking for Fruit Ninja.

    ``plugin.py`` owns one instance for the app's lifetime; :meth:`start`
    resets it to a fresh ``READY`` run. :meth:`process` both advances the
    simulation (while ``PLAYING``) and composites it onto the frame every
    tick, matching Air Canvas's "processor renders, widget just displays"
    split.
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        """Create the processor.

        Args:
            rng: Source of randomness for spawning; ``None`` uses a fresh,
                unseeded :class:`random.Random` (real runs). Tests inject a
                seeded instance for deterministic spawn sequences.
        """
        self._rng = rng if rng is not None else random.Random()
        self._state = GameState.READY
        self._score = 0
        self._lives = INITIAL_LIVES
        self._combo_count = 0
        self._last_slice_time: float | None = None
        self._game_over_reason = ""
        self._pieces: list[Piece] = []
        self._halves: list[Half] = []
        self._trail: list[TrailPoint] = []
        # min_cutoff/beta tuned higher than Air Canvas's drawing filter —
        # a slicing swipe is deliberately fast, so the filter needs to keep
        # up with real motion rather than smoothing it away, while still
        # damping MediaPipe's own frame-to-frame landmark jitter at rest.
        self._filter = OneEuroFilter(min_cutoff=1.2, beta=1.5)
        self._velocity = VelocityTracker(window=0.08)
        self._cursor: tuple[float, float] | None = None
        self._last_timestamp: float | None = None
        self._next_spawn_at: float | None = None
        self._commands: queue.SimpleQueue[str] = queue.SimpleQueue()

    # ------------------------------------------------------------------ state

    @property
    def state(self) -> GameState:
        """The run's current phase."""
        return self._state

    @property
    def score(self) -> int:
        """Total score accumulated this run."""
        return self._score

    @property
    def lives(self) -> int:
        """Lives remaining this run."""
        return self._lives

    @property
    def combo(self) -> int:
        """Current combo multiplier (1 with no active streak)."""
        return max(self._combo_count, 1)

    @property
    def game_over_reason(self) -> str:
        """Human-readable cause of the last game over (``""`` if none yet)."""
        return self._game_over_reason

    @property
    def pieces(self) -> tuple[Piece, ...]:
        """Every live (unsliced) fruit/bomb currently in flight."""
        return tuple(self._pieces)

    @property
    def halves(self) -> tuple[Half, ...]:
        """Every sliced-fruit half still drifting/rendering."""
        return tuple(self._halves)

    @property
    def cursor(self) -> tuple[float, float] | None:
        """Smoothed fingertip position, ``None`` when no hand is tracked."""
        return self._cursor

    # -------------------------------------------------------- thread-safe requests

    def request_start(self) -> None:
        """Queue a (re)start; applied on the next :meth:`process` call.

        Thread-safe — call from the Qt thread (e.g. a keyboard shortcut in
        ``widget.py``) while frames are flowing on the pipeline worker
        thread. Valid from ``READY`` or ``GAME_OVER``; a no-op while
        already ``PLAYING``.
        """
        self._commands.put("start")

    def _drain_commands(self) -> None:
        """Apply every queued start request, oldest first."""
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return
            if command == "start" and self._state is not GameState.PLAYING:
                self._start_run()

    def _start_run(self) -> None:
        """Reset all run state and enter ``PLAYING``."""
        self._state = GameState.PLAYING
        self._score = 0
        self._lives = INITIAL_LIVES
        self._combo_count = 0
        self._last_slice_time = None
        self._game_over_reason = ""
        self._pieces = []
        self._halves = []
        self._trail = []
        self._next_spawn_at = self._last_timestamp

    # -------------------------------------------------------------- lifecycle

    def start(self) -> None:
        """Called from ``Plugin.on_start`` — reset to a fresh ``READY`` run."""
        self._state = GameState.READY
        self._score = 0
        self._lives = INITIAL_LIVES
        self._combo_count = 0
        self._last_slice_time = None
        self._game_over_reason = ""
        self._pieces = []
        self._halves = []
        self._trail = []
        self._last_timestamp = None
        self._next_spawn_at = None
        self._reset_tracking()

    def stop(self) -> None:
        """Called from ``Plugin.on_stop`` — drop per-run tracking state."""
        self._reset_tracking()

    def _reset_tracking(self) -> None:
        """Forget everything derived from recent frames."""
        self._cursor = None
        self._filter.reset()
        self._velocity.reset()

    # -------------------------------------------------------------- per frame

    def process(self, frame: Frame) -> Frame:
        """Advance the game one tick and composite it onto the frame.

        Args:
            frame: The captured frame. ``frame.results.get(RESULTS_KEY)`` is
                a ``HandLandmarkResult`` when the hand backend ran, ``None``
                when it is unavailable — an expected case, handled as "no
                hand". Also the sole point where a queued
                :meth:`request_start` is applied.

        Returns:
            The same frame, with pieces/halves/blade trail/HUD drawn onto
            its image in place.
        """
        self._drain_commands()
        dt = self._advance_time(frame.timestamp)
        hand = _first_hand(frame.results.get(RESULTS_KEY))
        if hand is None:
            self._reset_tracking()
        else:
            self._advance_cursor(hand, frame.timestamp)
        if self._state is GameState.PLAYING:
            self._step(dt, frame.timestamp)
        self._age_trail(dt)
        return self._render(frame)

    def _advance_time(self, timestamp: float) -> float:
        """Return the clamped elapsed time since the previous processed frame."""
        last = self._last_timestamp
        self._last_timestamp = timestamp
        if last is None:
            return 0.0
        return min(max(timestamp - last, 0.0), _MAX_DT)

    def _advance_cursor(self, hand: HandLandmarks, timestamp: float) -> None:
        """Smooth the fingertip, track its velocity, and test for a slice."""
        tip = hand.points[HandLandmarkIndex.INDEX_FINGER_TIP]
        smoothed = self._filter.filter((tip.x, tip.y), timestamp)
        previous = self._cursor
        x, y = float(smoothed[0]), float(smoothed[1])
        self._cursor = (x, y)
        self._trail.append(TrailPoint(x=x, y=y))
        self._velocity.update((x, y), timestamp)
        if (
            self._state is GameState.PLAYING
            and previous is not None
            and self._velocity.speed >= SLICE_SPEED_THRESHOLD
        ):
            self._slice_segment(previous, (x, y), timestamp)

    def _slice_segment(
        self, start: tuple[float, float], end: tuple[float, float], timestamp: float
    ) -> None:
        """Test one fast blade segment against every live piece."""
        survivors: list[Piece] = []
        for piece in self._pieces:
            if segment_hits_circle(start, end, (piece.x, piece.y), piece.radius):
                self._on_piece_sliced(piece, timestamp)
                if self._state is not GameState.PLAYING:
                    return  # a bomb slice already cleared self._pieces via _end_run
            else:
                survivors.append(piece)
        self._pieces = survivors

    def _on_piece_sliced(self, piece: Piece, timestamp: float) -> None:
        """Award score/combo for a sliced fruit, or end the run for a bomb."""
        if piece.kind is PieceKind.BOMB:
            self._end_run("Sliced a bomb!")
            return
        if self._last_slice_time is not None and timestamp - self._last_slice_time <= COMBO_WINDOW:
            self._combo_count += 1
        else:
            self._combo_count = 1
        self._last_slice_time = timestamp
        self._score += int(BASE_SCORE * self._combo_count)
        self._halves.append(
            Half(
                x=piece.x,
                y=piece.y,
                vx=piece.vx - 0.35,
                vy=piece.vy - 0.2,
                radius=piece.radius,
                color=piece.color,
                side=-1,
            )
        )
        self._halves.append(
            Half(
                x=piece.x,
                y=piece.y,
                vx=piece.vx + 0.35,
                vy=piece.vy - 0.2,
                radius=piece.radius,
                color=piece.color,
                side=1,
            )
        )

    def _end_run(self, reason: str) -> None:
        """Freeze the simulation in ``GAME_OVER`` with a display reason."""
        self._state = GameState.GAME_OVER
        self._game_over_reason = reason
        self._pieces = []

    def _step(self, dt: float, timestamp: float) -> None:
        """Advance spawning, piece/half physics, and miss handling."""
        self._maybe_spawn(timestamp)
        survivors: list[Piece] = []
        for piece in self._pieces:
            advance_piece(piece, dt)
            if piece.y - piece.radius > MISS_Y:
                if piece.kind is PieceKind.FRUIT:
                    self._on_piece_missed()
                # A missed bomb is simply gone — no penalty for letting it fall.
                continue
            survivors.append(piece)
        self._pieces = survivors
        for half in self._halves:
            _advance_half(half, dt)
            half.age += dt
        self._halves = [half for half in self._halves if half.age < HALF_LIFETIME]

    def _on_piece_missed(self) -> None:
        """One live fruit reached the bottom unsliced — costs a life."""
        self._lives -= 1
        if self._lives <= 0:
            self._end_run("Out of lives!")

    def _maybe_spawn(self, timestamp: float) -> None:
        """Spawn one piece if the randomized spawn interval has elapsed."""
        if self._next_spawn_at is None:
            self._next_spawn_at = timestamp
        if timestamp < self._next_spawn_at:
            return
        self._pieces.append(spawn_piece(self._rng))
        self._next_spawn_at = timestamp + self._rng.uniform(*SPAWN_INTERVAL_RANGE)

    def _age_trail(self, dt: float) -> None:
        """Age and prune the rendered blade trail (independent of slicing)."""
        for point in self._trail:
            point.age += dt
        self._trail = [point for point in self._trail if point.age < TRAIL_LIFETIME]

    # -------------------------------------------------------------- rendering

    def _render(self, frame: Frame) -> Frame:
        """Composite halves/pieces/trail/HUD/overlay onto the frame in place.

        Only color frames are annotated — a GRAY frame passes through
        untouched rather than guessing how to draw on one channel.
        """
        if frame.color_format is ColorFormat.GRAY:
            return frame
        image = frame.image
        rgb = frame.color_format is ColorFormat.RGB
        for half in self._halves:
            _draw_half(image, half, rgb)
        for piece in self._pieces:
            _draw_piece(image, piece, rgb)
        _draw_trail(image, self._trail, rgb)
        self._render_hud(image, rgb)
        if self._state is not GameState.PLAYING:
            self._render_overlay(image, rgb)
        return frame

    def _render_hud(self, image: npt.NDArray[np.uint8], rgb: bool) -> None:
        """Draw the score/lives/combo line across the top of the frame."""
        text = f"Score: {self._score}   Lives: {self._lives}   Combo x{self.combo}"
        cv2.putText(
            image,
            text,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            _color((235, 235, 235), rgb),
            2,
            cv2.LINE_AA,
        )

    def _render_overlay(self, image: npt.NDArray[np.uint8], rgb: bool) -> None:
        """Draw the centered start/game-over message block."""
        height, width = image.shape[:2]
        if self._state is GameState.READY:
            lines = ["FRUIT NINJA", "Fast swipe to slice - avoid the bombs", _START_MESSAGE]
        else:
            lines = [
                "GAME OVER",
                self._game_over_reason,
                f"Final score: {self._score}",
                _START_MESSAGE.replace("start", "restart"),
            ]
        top = height // 2 - 20 * (len(lines) - 1)
        for index, line in enumerate(lines):
            (text_width, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            x = max((width - text_width) // 2, 8)
            y = top + index * 40
            cv2.putText(
                image,
                line,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                _color((235, 235, 235), rgb),
                2,
                cv2.LINE_AA,
            )


# ------------------------------------------------------------------ helpers


def _first_hand(result: object) -> HandLandmarks | None:
    """The first detected hand, or ``None`` for absent/empty/foreign results."""
    if isinstance(result, HandLandmarkResult) and result.hands:
        return result.hands[0]
    return None


def spawn_piece(rng: random.Random) -> Piece:
    """Build one new piece at the bottom edge with an upward launch velocity."""
    kind = PieceKind.BOMB if rng.random() < BOMB_PROBABILITY else PieceKind.FRUIT
    x = rng.uniform(0.15, 0.85)
    vx = rng.uniform(-0.25, 0.25)
    vy = rng.uniform(-2.0, -1.5)  # negative is upward
    color = _BOMB_COLOR if kind is PieceKind.BOMB else rng.choice(_FRUIT_COLORS)
    return Piece(kind=kind, x=x, y=MISS_Y, vx=vx, vy=vy, radius=PIECE_RADIUS, color=color)


def advance_piece(piece: Piece, dt: float) -> None:
    """Step one piece's parabolic motion forward by ``dt`` seconds, in place."""
    piece.vy += GRAVITY * dt
    piece.x += piece.vx * dt
    piece.y += piece.vy * dt


def _advance_half(half: Half, dt: float) -> None:
    """Step one drifting half's motion forward by ``dt`` seconds, in place."""
    half.vy += GRAVITY * dt
    half.x += half.vx * dt
    half.y += half.vy * dt


def segment_hits_circle(
    start: tuple[float, float],
    end: tuple[float, float],
    center: tuple[float, float],
    radius: float,
) -> bool:
    """``True`` when the segment ``start``-``end`` passes within ``radius`` of ``center``.

    Standard point-to-segment distance: project the center onto the
    (clamped) segment and compare against ``radius``. A zero-length segment
    (fingertip motionless between two samples) degrades to a point check.
    """
    sx, sy = start
    ex, ey = end
    cx, cy = center
    dx, dy = ex - sx, ey - sy
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        return math.hypot(cx - sx, cy - sy) <= radius
    t = max(0.0, min(1.0, ((cx - sx) * dx + (cy - sy) * dy) / length_sq))
    closest_x, closest_y = sx + t * dx, sy + t * dy
    return math.hypot(cx - closest_x, cy - closest_y) <= radius


def _color(bgr: tuple[int, int, int], rgb: bool) -> tuple[int, int, int]:
    """Return the color in the frame's channel order."""
    return (bgr[2], bgr[1], bgr[0]) if rgb else bgr


def _draw_piece(image: npt.NDArray[np.uint8], piece: Piece, rgb: bool) -> None:
    """Draw one live piece as a filled circle; bombs get a small fuse-dot accent."""
    height, width = image.shape[:2]
    center = (int(piece.x * width), int(piece.y * height))
    radius_px = max(int(piece.radius * width), 4)
    cv2.circle(image, center, radius_px, _color(piece.color, rgb), -1, cv2.LINE_AA)
    if piece.kind is PieceKind.BOMB:
        cv2.circle(image, center, max(radius_px // 3, 2), _color((80, 80, 220), rgb), -1)


def _draw_half(image: npt.NDArray[np.uint8], half: Half, rgb: bool) -> None:
    """Draw one drifting half as a filled semicircle facing its throw direction."""
    height, width = image.shape[:2]
    center = (int(half.x * width), int(half.y * height))
    radius_px = max(int(half.radius * width), 4)
    start_angle = 0 if half.side < 0 else 180
    cv2.ellipse(
        image,
        center,
        (radius_px, radius_px),
        0,
        start_angle,
        start_angle + 180,
        _color(half.color, rgb),
        -1,
        cv2.LINE_AA,
    )


def _draw_trail(image: npt.NDArray[np.uint8], trail: list[TrailPoint], rgb: bool) -> None:
    """Draw the recent fingertip path as a connected polyline."""
    if len(trail) < 2:
        return
    height, width = image.shape[:2]
    pixels = [(int(point.x * width), int(point.y * height)) for point in trail]
    for i in range(1, len(pixels)):
        cv2.line(image, pixels[i - 1], pixels[i], _color((255, 255, 255), rgb), 3, cv2.LINE_AA)
