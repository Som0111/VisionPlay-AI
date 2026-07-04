"""Unit tests for apps/fruit_ninja (M3.4).

All processor tests drive :class:`FruitNinjaProcessor` headless with
synthetic frames carrying hand-crafted ``HandLandmarkResult`` payloads (via
``_fruit_ninja_helpers.FrameFeeder``) or, for scoring/collision/miss logic,
by calling the processor's own private simulation hooks directly with
hand-built :class:`Piece` fixtures — no Qt, camera, or MediaPipe involved,
and no dependence on the real gesture pipeline's exact filter/velocity
tuning for things that are really just game-rule tests.

The helper module is named ``_fruit_ninja_helpers`` rather than the
``_helpers`` name ``air_canvas`` uses: pytest's default import mode caches
bare module names process-wide, so two same-named helper modules in
sibling app test directories would collide (whichever app's tests import
first "wins" for both) — a real failure once more than one app's tests run
in the same session, not a hypothetical one.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
from _fruit_ninja_helpers import FrameFeeder

from visionplay.apps.fruit_ninja.manifest import MANIFEST
from visionplay.apps.fruit_ninja.plugin import Plugin
from visionplay.apps.fruit_ninja.processor import (
    BOMB_PROBABILITY,
    COMBO_WINDOW,
    GRAVITY,
    INITIAL_LIVES,
    MISS_Y,
    PIECE_RADIUS,
    RESULTS_KEY,
    SPAWN_INTERVAL_RANGE,
    FruitNinjaProcessor,
    GameState,
    Piece,
    PieceKind,
    advance_piece,
    segment_hits_circle,
    spawn_piece,
)
from visionplay.core.plugin_base import AppPlugin
from visionplay.vision.pipeline.frame_types import ColorFormat, Frame


def _fruit(x: float = 0.5, y: float = 0.5, vx: float = 0.0, vy: float = 0.0) -> Piece:
    return Piece(
        kind=PieceKind.FRUIT, x=x, y=y, vx=vx, vy=vy, radius=PIECE_RADIUS, color=(0, 0, 255)
    )


def _bomb(x: float = 0.5, y: float = 0.5) -> Piece:
    return Piece(
        kind=PieceKind.BOMB, x=x, y=y, vx=0.0, vy=0.0, radius=PIECE_RADIUS, color=(40, 40, 40)
    )


@pytest.fixture
def feeder() -> FrameFeeder:
    processor = FruitNinjaProcessor(rng=random.Random(1234))
    processor.start()
    return FrameFeeder(processor)


class TestManifest:
    def test_manifest_is_well_formed(self) -> None:
        assert MANIFEST.id == "fruit_ninja"
        assert MANIFEST.category == "gesture_games"

    def test_declares_mediapipe_hands_backend(self) -> None:
        assert MANIFEST.required_backends == ("mediapipe.hands",)


class TestPlugin:
    def test_is_an_app_plugin(self) -> None:
        assert isinstance(Plugin(), AppPlugin)

    def test_full_lifecycle_does_not_raise(self) -> None:
        plugin = Plugin()
        plugin.on_load()
        plugin.on_start()
        frame = Frame.from_image(0, 0.0, np.zeros((60, 80, 3), dtype=np.uint8))
        result = plugin.on_frame(frame)
        assert isinstance(result, Frame)
        assert result.frame_id == frame.frame_id
        plugin.on_stop()
        plugin.on_unload()


class TestDefensiveInput:
    def test_absent_results_key_does_not_raise(self, feeder: FrameFeeder) -> None:
        frame = feeder.feed(None)
        assert RESULTS_KEY not in frame.results
        assert feeder.processor.cursor is None

    def test_foreign_object_under_key_is_no_hand(self, feeder: FrameFeeder) -> None:
        frame = Frame.from_image(0, 0.0, np.zeros((480, 640, 3), dtype=np.uint8))
        frame.results[RESULTS_KEY] = [{"landmarks": []}]  # not a HandLandmarkResult
        feeder.processor.process(frame)
        assert feeder.processor.cursor is None

    def test_gray_frame_passes_through_untouched(self, feeder: FrameFeeder) -> None:
        image = np.zeros((240, 320), dtype=np.uint8)
        frame = Frame.from_image(99, 3.3, image, color_format=ColorFormat.GRAY)
        result = feeder.processor.process(frame)
        assert result is frame
        assert not frame.image.any()


class TestGameStateMachine:
    def test_starts_ready(self, feeder: FrameFeeder) -> None:
        assert feeder.processor.state is GameState.READY

    def test_request_start_enters_playing(self, feeder: FrameFeeder) -> None:
        feeder.processor.request_start()
        feeder.feed(None)
        assert feeder.processor.state is GameState.PLAYING

    def test_no_spawning_before_start(self, feeder: FrameFeeder) -> None:
        feeder.feed_many(None, 5)
        assert feeder.processor.pieces == ()

    def test_bomb_slice_ends_run_immediately(self, feeder: FrameFeeder) -> None:
        feeder.processor.request_start()
        feeder.feed(None)
        feeder.processor._on_piece_sliced(_bomb(), timestamp=1.0)
        assert feeder.processor.state is GameState.GAME_OVER
        assert feeder.processor.game_over_reason
        assert feeder.processor.pieces == ()

    def test_out_of_lives_ends_run(self, feeder: FrameFeeder) -> None:
        feeder.processor.request_start()
        feeder.feed(None)
        for _ in range(INITIAL_LIVES):
            feeder.processor._on_piece_missed()
        assert feeder.processor.state is GameState.GAME_OVER
        assert feeder.processor.lives == 0

    def test_restart_after_game_over_resets_everything(self, feeder: FrameFeeder) -> None:
        feeder.processor.request_start()
        feeder.feed(None)
        feeder.processor._on_piece_sliced(_bomb(), timestamp=1.0)
        assert feeder.processor.state is GameState.GAME_OVER

        feeder.processor.request_start()
        feeder.feed(None)
        assert feeder.processor.state is GameState.PLAYING
        assert feeder.processor.score == 0
        assert feeder.processor.lives == INITIAL_LIVES
        assert feeder.processor.combo == 1

    def test_request_start_while_playing_is_a_no_op(self, feeder: FrameFeeder) -> None:
        feeder.processor.request_start()
        feeder.feed(None)
        feeder.processor._pieces.clear()
        feeder.processor._pieces.append(_fruit())
        feeder.processor.request_start()  # should not reset an active run
        feeder.feed(None)
        assert feeder.processor.state is GameState.PLAYING
        assert len(feeder.processor.pieces) == 1


class TestScoringAndCombo:
    def test_first_slice_awards_base_score(self, feeder: FrameFeeder) -> None:
        feeder.processor.request_start()
        feeder.feed(None)
        feeder.processor._on_piece_sliced(_fruit(), timestamp=1.0)
        assert feeder.processor.score == 10
        assert feeder.processor.combo == 1

    def test_slices_within_combo_window_increase_multiplier(self, feeder: FrameFeeder) -> None:
        feeder.processor.request_start()
        feeder.feed(None)
        feeder.processor._on_piece_sliced(_fruit(), timestamp=1.0)
        feeder.processor._on_piece_sliced(_fruit(), timestamp=1.0 + COMBO_WINDOW - 0.01)
        assert feeder.processor.combo == 2
        assert feeder.processor.score == 10 + 20

    def test_gap_past_combo_window_resets_multiplier(self, feeder: FrameFeeder) -> None:
        feeder.processor.request_start()
        feeder.feed(None)
        feeder.processor._on_piece_sliced(_fruit(), timestamp=1.0)
        feeder.processor._on_piece_sliced(_fruit(), timestamp=1.0 + COMBO_WINDOW + 0.01)
        assert feeder.processor.combo == 1
        assert feeder.processor.score == 10 + 10

    def test_sliced_fruit_becomes_two_halves(self, feeder: FrameFeeder) -> None:
        feeder.processor.request_start()
        feeder.feed(None)
        feeder.processor._on_piece_sliced(_fruit(), timestamp=1.0)
        assert len(feeder.processor.halves) == 2


class TestMissHandling:
    def test_fruit_past_bottom_costs_a_life(self, feeder: FrameFeeder) -> None:
        feeder.processor.request_start()
        feeder.feed(None)
        feeder.processor._pieces.clear()
        feeder.processor._pieces.append(_fruit(y=MISS_Y + PIECE_RADIUS + 0.01))
        feeder.feed(None)
        assert feeder.processor.lives == INITIAL_LIVES - 1
        assert feeder.processor.pieces == ()

    def test_bomb_past_bottom_costs_nothing(self, feeder: FrameFeeder) -> None:
        feeder.processor.request_start()
        feeder.feed(None)
        feeder.processor._pieces.clear()
        feeder.processor._pieces.append(_bomb(y=MISS_Y + PIECE_RADIUS + 0.01))
        feeder.feed(None)
        assert feeder.processor.lives == INITIAL_LIVES
        assert feeder.processor.pieces == ()
        assert feeder.processor.state is GameState.PLAYING


class TestPhysicsDeterminism:
    def testadvance_piece_applies_gravity_then_integrates_position(self) -> None:
        piece = _fruit(x=0.5, y=0.5, vx=0.2, vy=0.0)
        advance_piece(piece, dt=1.0)
        assert piece.vy == pytest.approx(GRAVITY)
        assert piece.y == pytest.approx(0.5 + GRAVITY)  # semi-implicit Euler: uses updated vy
        assert piece.x == pytest.approx(0.7)

    def test_repeated_small_steps_match_one_large_step_in_velocity(self) -> None:
        small = _fruit(vy=0.0)
        for _ in range(10):
            advance_piece(small, dt=0.1)
        large = _fruit(vy=0.0)
        advance_piece(large, dt=1.0)
        assert small.vy == pytest.approx(large.vy)


class TestSpawner:
    def test_spawned_piece_is_within_expected_bounds(self) -> None:
        piece = spawn_piece(random.Random(42))
        assert piece.kind in (PieceKind.FRUIT, PieceKind.BOMB)
        assert 0.15 <= piece.x <= 0.85
        assert piece.vy < 0  # launched upward
        assert piece.radius == PIECE_RADIUS
        assert piece.y == pytest.approx(MISS_Y)

    def test_bomb_probability_is_approximately_respected(self) -> None:
        rng = random.Random(7)
        pieces = [spawn_piece(rng) for _ in range(2000)]
        bomb_fraction = sum(p.kind is PieceKind.BOMB for p in pieces) / len(pieces)
        assert bomb_fraction == pytest.approx(BOMB_PROBABILITY, abs=0.03)

    def test_first_piece_spawns_on_the_first_playing_frame(self, feeder: FrameFeeder) -> None:
        feeder.processor.request_start()
        feeder.feed(None)
        assert len(feeder.processor.pieces) == 1

    def test_a_second_piece_spawns_within_the_max_interval(self, feeder: FrameFeeder) -> None:
        feeder.processor.request_start()
        feeder.feed(None)
        # Advance well past the longest possible spawn interval, but well
        # short of the shortest possible flight time, so a second piece must
        # have appeared and neither has fallen off yet.
        feeder.feed_many(None, int(SPAWN_INTERVAL_RANGE[1] * 30) + 5)
        assert len(feeder.processor.pieces) >= 2


class TestSegmentCircleCollision:
    def test_segment_through_center_hits(self) -> None:
        assert segment_hits_circle((0.0, 0.5), (1.0, 0.5), (0.5, 0.5), 0.05)

    def test_segment_passing_outside_radius_misses(self) -> None:
        assert not segment_hits_circle((0.0, 0.5), (1.0, 0.5), (0.5, 0.7), 0.05)

    def test_segment_endpoint_exactly_at_radius_hits(self) -> None:
        assert segment_hits_circle((0.0, 0.5), (0.45, 0.5), (0.5, 0.5), 0.05)

    def test_zero_length_segment_inside_radius_hits(self) -> None:
        assert segment_hits_circle((0.5, 0.5), (0.5, 0.5), (0.51, 0.5), 0.05)

    def test_zero_length_segment_outside_radius_misses(self) -> None:
        assert not segment_hits_circle((0.5, 0.5), (0.5, 0.5), (0.7, 0.5), 0.05)

    def test_diagonal_segment_hits(self) -> None:
        assert segment_hits_circle((0.0, 0.0), (1.0, 1.0), (0.5, 0.5), 0.05)


class TestSliceIntegration:
    def test_slow_drift_through_a_piece_does_not_slice_it(self, feeder: FrameFeeder) -> None:
        feeder.processor.request_start()
        feeder.feed(None)
        feeder.processor._pieces.clear()
        feeder.processor._pieces.append(_fruit(x=0.5, y=0.5))
        # A slow hover across the piece's position over a full second.
        feeder.swipe((0.3, 0.5), (0.7, 0.5), steps=30)
        assert feeder.processor.score == 0
        assert len(feeder.processor.pieces) == 1

    def test_fast_swipe_through_a_piece_slices_it(self, feeder: FrameFeeder) -> None:
        feeder.processor.request_start()
        feeder.feed(None)
        feeder.processor._pieces.clear()
        feeder.processor._pieces.append(_fruit(x=0.5, y=0.5, vx=0.0, vy=0.0))
        # A fast swipe: same distance as the slow-drift case, far fewer frames.
        feeder.swipe((0.1, 0.5), (0.9, 0.5), steps=4)
        assert feeder.processor.score > 0
        assert feeder.processor.pieces == ()
        assert len(feeder.processor.halves) == 2


class TestDiscovery:
    def test_real_registry_discovers_the_app(self) -> None:
        from visionplay.core.event_bus import EventBus
        from visionplay.core.plugin_registry import PluginRegistry

        registry = PluginRegistry(event_bus=EventBus())
        registry.discover()
        assert "fruit_ninja" in registry.manifests
        assert registry.manifests["fruit_ninja"].name == "Fruit Ninja"

    def test_starts_and_stops_through_the_registry(self) -> None:
        from visionplay.core.event_bus import EventBus
        from visionplay.core.plugin_registry import PluginRegistry

        registry = PluginRegistry(event_bus=EventBus())
        registry.discover()
        registry.start("fruit_ninja")
        assert registry.active_app_id == "fruit_ninja"
        registry.stop_active()
        assert registry.active_app_id is None


class TestWidget:
    def test_shows_unavailable_message_without_backend_results(self, qapp: object) -> None:
        from visionplay.apps.fruit_ninja.widget import NO_DATA_MESSAGE, FruitNinjaWidget

        widget = FruitNinjaWidget()
        frame = Frame.from_image(0, 0.0, np.zeros((60, 80, 3), dtype=np.uint8))
        widget.on_frame_ready(frame)
        assert widget.overlay_message == NO_DATA_MESSAGE

    def test_shows_hint_once_backend_results_exist(self, qapp: object) -> None:
        from visionplay.apps.fruit_ninja.widget import HINT_MESSAGE, FruitNinjaWidget
        from visionplay.vision.inference.results import HandLandmarkResult

        widget = FruitNinjaWidget()
        frame = Frame.from_image(0, 0.0, np.zeros((60, 80, 3), dtype=np.uint8))
        frame.results[RESULTS_KEY] = HandLandmarkResult()
        widget.on_frame_ready(frame)
        assert widget.overlay_message == HINT_MESSAGE
