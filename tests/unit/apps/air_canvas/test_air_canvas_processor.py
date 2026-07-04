"""Unit tests for apps/air_canvas (M3.2).

All processor tests drive :class:`AirCanvasProcessor` headless with
synthetic frames carrying hand-crafted ``HandLandmarkResult`` payloads — no
Qt, camera, or MediaPipe involved. Widget and discovery tests mirror the
hand-tracking demo's coverage.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from _helpers import CANVAS_POINT, FrameFeeder, make_hand, region_center

from visionplay.apps.air_canvas.manifest import MANIFEST
from visionplay.apps.air_canvas.plugin import Plugin
from visionplay.apps.air_canvas.processor import (
    BRUSH_SIZES,
    CLEAR_HOLD_FRAMES,
    DWELL_FRAMES,
    HELP_LINES,
    PALETTE,
    PINCH_DOWN_THRESHOLD,
    PINCH_UP_THRESHOLD,
    RESULTS_KEY,
    TOOLBAR,
    TOOLBAR_HEIGHT,
    AirCanvasProcessor,
    Tool,
    ToolbarRegion,
)
from visionplay.core.plugin_base import AppPlugin
from visionplay.vision.inference.results import HandLandmarkResult
from visionplay.vision.pipeline.frame_types import ColorFormat, Frame


@pytest.fixture
def feeder() -> FrameFeeder:
    processor = AirCanvasProcessor()
    processor.start()
    return FrameFeeder(processor)


class TestManifest:
    def test_manifest_is_well_formed(self) -> None:
        assert MANIFEST.id == "air_canvas"
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
    def test_absent_results_key_is_no_hand(self, feeder: FrameFeeder) -> None:
        frame = feeder.feed(None)
        assert RESULTS_KEY not in frame.results
        assert feeder.processor.cursor is None
        assert not feeder.processor.is_pen_down

    def test_empty_result_is_no_hand(self, feeder: FrameFeeder) -> None:
        feeder.feed(None, blank_results=True)
        assert feeder.processor.cursor is None

    def test_foreign_object_under_key_is_no_hand(self, feeder: FrameFeeder) -> None:
        frame = Frame.from_image(0, 0.0, np.zeros((240, 320, 3), dtype=np.uint8))
        frame.results[RESULTS_KEY] = [{"landmarks": []}]  # not a HandLandmarkResult
        feeder.processor.process(frame)
        assert feeder.processor.cursor is None


class TestStrokeBuilding:
    def test_pinch_draws_one_point_per_frame(self, feeder: FrameFeeder) -> None:
        for step in range(5):
            feeder.feed(make_hand((0.3 + 0.05 * step, 0.5), pinch=True))
        strokes = feeder.processor.strokes
        assert len(strokes) == 1
        assert len(strokes[0].points) == 5

    def test_release_commits_stroke_and_new_pinch_starts_another(self, feeder: FrameFeeder) -> None:
        feeder.feed_many(make_hand((0.3, 0.5), pinch=True), 3)
        feeder.feed(make_hand((0.35, 0.5), pinch=False))
        feeder.feed_many(make_hand((0.6, 0.6), pinch=True), 2)
        assert len(feeder.processor.strokes) == 2
        assert feeder.processor.strokes[0] is not feeder.processor.strokes[1]

    def test_no_pinch_never_draws(self, feeder: FrameFeeder) -> None:
        feeder.feed_many(make_hand((0.4, 0.5), pinch=False), 10)
        assert feeder.processor.strokes == ()

    def test_stroke_records_selected_color_and_thickness(self, feeder: FrameFeeder) -> None:
        feeder.feed(make_hand(region_center("color", 1), pinch=True))  # select Red
        feeder.feed(make_hand(CANVAS_POINT, pinch=False))
        feeder.feed_many(make_hand(CANVAS_POINT, pinch=True), 2)
        stroke = feeder.processor.strokes[0]
        assert stroke.color == PALETTE[1].bgr
        assert stroke.thickness == BRUSH_SIZES[0]

    def test_points_are_smoothed_within_bounds(self, feeder: FrameFeeder) -> None:
        # Jittery vertical input: smoothed points must stay inside the
        # jitter envelope and inside the canvas.
        for step in range(10):
            y = 0.5 if step % 2 == 0 else 0.54
            feeder.feed(make_hand((0.4, y), pinch=True))
        points = feeder.processor.strokes[0].points
        assert all(0.0 <= x <= 1.0 and 0.45 <= y <= 0.55 for x, y in points)
        # After the first sample, filtering keeps values strictly inside the
        # raw extremes (an unfiltered path would sit exactly on them).
        assert all(0.5 < y < 0.54 for _, y in points[1:])


class TestPinchStateMachine:
    def test_hysteresis_gap_does_not_engage_pen_from_up(self, feeder: FrameFeeder) -> None:
        gap = (PINCH_DOWN_THRESHOLD + PINCH_UP_THRESHOLD) / 2
        feeder.feed_many(make_hand(CANVAS_POINT, pinch_gap=gap), 5)
        assert not feeder.processor.is_pen_down

    def test_hysteresis_gap_keeps_pen_down_once_engaged(self, feeder: FrameFeeder) -> None:
        gap = (PINCH_DOWN_THRESHOLD + PINCH_UP_THRESHOLD) / 2
        feeder.feed(make_hand(CANVAS_POINT, pinch_gap=0.02))
        assert feeder.processor.is_pen_down
        feeder.feed_many(make_hand(CANVAS_POINT, pinch_gap=gap), 5)
        assert feeder.processor.is_pen_down

    def test_wide_gap_releases_pen(self, feeder: FrameFeeder) -> None:
        feeder.feed(make_hand(CANVAS_POINT, pinch_gap=0.02))
        feeder.feed(make_hand(CANVAS_POINT, pinch_gap=PINCH_UP_THRESHOLD * 1.5))
        assert not feeder.processor.is_pen_down

    def test_tracking_loss_commits_stroke_and_lifts_pen(self, feeder: FrameFeeder) -> None:
        feeder.feed_many(make_hand((0.4, 0.5), pinch=True), 3)
        feeder.feed(None)
        assert len(feeder.processor.strokes) == 1
        assert not feeder.processor.is_pen_down
        assert feeder.processor.cursor is None


class TestToolbar:
    def test_pinch_in_color_region_selects_color(self, feeder: FrameFeeder) -> None:
        feeder.feed(make_hand(region_center("color", 2), pinch=True))
        assert feeder.processor.brush_color == PALETTE[2]

    def test_toolbar_pinch_does_not_draw(self, feeder: FrameFeeder) -> None:
        feeder.feed_many(make_hand(region_center("color", 0), pinch=True), 3)
        assert feeder.processor.strokes == ()

    def test_held_pinch_entering_toolbar_does_not_select(self, feeder: FrameFeeder) -> None:
        feeder.feed(make_hand(CANVAS_POINT, pinch=True))  # pinch starts on canvas
        feeder.feed(make_hand(region_center("color", 3), pinch=True))  # drags into toolbar
        assert feeder.processor.brush_color == PALETTE[0]

    def test_dwell_selects_without_pinch(self, feeder: FrameFeeder) -> None:
        feeder.feed_many(make_hand(region_center("eraser")), DWELL_FRAMES)
        assert feeder.processor.tool is Tool.ERASER

    def test_dwell_resets_when_leaving_region(self, feeder: FrameFeeder) -> None:
        feeder.feed_many(make_hand(region_center("eraser")), DWELL_FRAMES - 1)
        feeder.feed(make_hand(CANVAS_POINT))
        feeder.feed_many(make_hand(region_center("eraser")), DWELL_FRAMES - 1)
        assert feeder.processor.tool is Tool.PEN

    def test_brush_region_changes_thickness(self, feeder: FrameFeeder) -> None:
        feeder.feed(make_hand(region_center("brush", 2), pinch=True))
        assert feeder.processor.brush_thickness == BRUSH_SIZES[2]

    def test_selecting_color_returns_to_pen(self, feeder: FrameFeeder) -> None:
        feeder.feed_many(make_hand(region_center("eraser")), DWELL_FRAMES)
        assert feeder.processor.tool is Tool.ERASER
        feeder.settle(CANVAS_POINT)  # leave the strip, release dwell
        feeder.settle(region_center("color", 0))
        feeder.feed(make_hand(region_center("color", 0), pinch=True))
        assert feeder.processor.tool is Tool.PEN

    def test_clear_region_wipes_canvas(self, feeder: FrameFeeder) -> None:
        feeder.feed_many(make_hand((0.4, 0.5), pinch=True), 3)
        feeder.feed(make_hand((0.4, 0.5), pinch=False))
        feeder.settle(region_center("clear"))
        feeder.feed(make_hand(region_center("clear"), pinch=True))
        assert feeder.processor.strokes == ()

    def test_eraser_sits_immediately_after_the_color_swatches(self) -> None:
        actions = [region.action for region in TOOLBAR]
        assert actions[: len(PALETTE)] == ["color"] * len(PALETTE)
        assert actions[len(PALETTE)] == "eraser"

    def test_brush_regions_carry_no_text_label(self) -> None:
        brush_regions = [r for r in TOOLBAR if r.action == "brush"]
        assert len(brush_regions) == len(BRUSH_SIZES)
        assert all(region.label == "" for region in brush_regions)

    def test_regions_cover_the_strip_without_overlap(self) -> None:
        edges = sorted((r.x_min, r.x_max) for r in TOOLBAR)
        assert edges[0][0] == pytest.approx(0.0)
        assert edges[-1][1] == pytest.approx(1.0)
        for (_, right), (next_left, _) in itertools.pairwise(edges):
            assert right == pytest.approx(next_left)

    def test_region_contains_respects_strip_height(self) -> None:
        region = ToolbarRegion("clear", 0, 0.0, 1.0, "Clear")
        assert region.contains(0.5, TOOLBAR_HEIGHT / 2)
        assert not region.contains(0.5, TOOLBAR_HEIGHT * 2)


class TestEraser:
    def _draw_two_strokes(self, feeder: FrameFeeder) -> None:
        feeder.settle((0.2, 0.5))
        feeder.feed_many(make_hand((0.2, 0.5), pinch=True), 3)
        feeder.feed(make_hand((0.2, 0.5), pinch=False))
        feeder.settle((0.8, 0.5))
        feeder.feed_many(make_hand((0.8, 0.5), pinch=True), 3)
        feeder.feed(make_hand((0.8, 0.5), pinch=False))

    def _select_eraser(self, feeder: FrameFeeder) -> None:
        # Settle onto the (now narrower, M3.3-toolbar) cell first so the
        # smoothed cursor isn't still drifting in from a neighboring cell
        # when the dwell count starts.
        feeder.settle(region_center("eraser"))
        feeder.feed_many(make_hand(region_center("eraser")), DWELL_FRAMES)
        assert feeder.processor.tool is Tool.ERASER

    def test_eraser_removes_only_touched_stroke(self, feeder: FrameFeeder) -> None:
        self._draw_two_strokes(feeder)
        self._select_eraser(feeder)
        feeder.settle((0.2, 0.5))
        feeder.feed(make_hand((0.2, 0.5), pinch=True))
        strokes = feeder.processor.strokes
        assert len(strokes) == 1
        assert strokes[0].points[0][0] == pytest.approx(0.8, abs=0.05)

    def test_eraser_misses_leave_canvas_untouched(self, feeder: FrameFeeder) -> None:
        self._draw_two_strokes(feeder)
        self._select_eraser(feeder)
        feeder.settle((0.5, 0.9))
        feeder.feed(make_hand((0.5, 0.9), pinch=True))  # far from both strokes
        assert len(feeder.processor.strokes) == 2

    def test_eraser_does_not_create_strokes(self, feeder: FrameFeeder) -> None:
        self._select_eraser(feeder)
        feeder.feed_many(make_hand((0.5, 0.6), pinch=True), 5)
        assert feeder.processor.strokes == ()


class TestClearGesture:
    def test_open_palm_held_clears_canvas(self, feeder: FrameFeeder) -> None:
        feeder.feed_many(make_hand((0.4, 0.5), pinch=True), 3)
        feeder.feed(make_hand((0.4, 0.5), pinch=False))
        feeder.feed_many(make_hand((0.44, 0.3), open_palm=True), CLEAR_HOLD_FRAMES)
        assert feeder.processor.strokes == ()

    def test_short_palm_does_not_clear(self, feeder: FrameFeeder) -> None:
        feeder.feed_many(make_hand((0.4, 0.5), pinch=True), 3)
        feeder.feed(make_hand((0.4, 0.5), pinch=False))
        feeder.feed_many(make_hand((0.44, 0.3), open_palm=True), CLEAR_HOLD_FRAMES - 1)
        feeder.feed(make_hand((0.44, 0.3), pinch=False))  # palm closes before the hold
        assert len(feeder.processor.strokes) == 1


class TestRendering:
    def test_toolbar_strip_is_drawn_on_every_frame(self, feeder: FrameFeeder) -> None:
        frame = feeder.feed(None)
        strip = frame.image[: int(TOOLBAR_HEIGHT * 240)]
        assert strip.any()

    def test_strokes_appear_on_the_frame(self, feeder: FrameFeeder) -> None:
        feeder.feed_many(make_hand((0.3, 0.5), pinch=True), 2)
        frame = feeder.feed(make_hand((0.7, 0.5), pinch=True))
        row = frame.image[120]  # y == 0.5 of a 240-row frame
        assert row.any()

    def test_gray_frame_passes_through_untouched(self, feeder: FrameFeeder) -> None:
        image = np.zeros((240, 320), dtype=np.uint8)
        frame = Frame.from_image(99, 3.3, image, color_format=ColorFormat.GRAY)
        result = feeder.processor.process(frame)
        assert result is frame
        assert not frame.image.any()


class TestHelpPanel:
    def test_visible_by_default(self, feeder: FrameFeeder) -> None:
        assert feeder.processor.show_help

    def test_visible_help_grows_the_frame_with_a_strip_below_it(self, feeder: FrameFeeder) -> None:
        frame = feeder.feed(None)
        assert frame.height > 240  # taller than the raw 240x320 camera frame
        assert frame.image.shape[0] == frame.height  # metadata matches the array

    def test_camera_canvas_area_is_left_completely_clean(self, feeder: FrameFeeder) -> None:
        # No text/overlay anywhere in the untouched canvas below the toolbar
        # — only the appended strip (beyond row 240) carries the instructions.
        frame = feeder.feed(None)
        canvas_below_toolbar = frame.image[int(TOOLBAR_HEIGHT * 240) + 4 : 240]
        assert not canvas_below_toolbar.any()

    def test_strip_spans_the_full_frame_width(self, feeder: FrameFeeder) -> None:
        frame = feeder.feed(None)
        strip_row = frame.image[240 + 4]  # a few rows into the appended strip
        assert strip_row[0].any()  # left edge tinted by the strip background
        assert strip_row[-1].any()  # right edge tinted too

    def test_panel_is_removed_once_hidden(self, feeder: FrameFeeder) -> None:
        feeder.settle(region_center("help"))
        feeder.feed(make_hand(region_center("help"), pinch=True))
        assert not feeder.processor.show_help
        frame = feeder.feed(None)
        assert frame.height == 240
        assert frame.image.shape[0] == 240

    def test_help_toolbar_pinch_toggles_visibility(self, feeder: FrameFeeder) -> None:
        feeder.settle(region_center("help"))
        feeder.feed(make_hand(region_center("help"), pinch=True))
        assert not feeder.processor.show_help
        feeder.feed(make_hand(region_center("help"), pinch=False))
        feeder.settle(region_center("help"))
        feeder.feed(make_hand(region_center("help"), pinch=True))
        assert feeder.processor.show_help

    def test_toggling_help_does_not_draw(self, feeder: FrameFeeder) -> None:
        feeder.settle(region_center("help"))
        feeder.feed_many(make_hand(region_center("help"), pinch=True), 3)
        assert feeder.processor.strokes == ()

    def test_help_lines_are_non_empty(self) -> None:
        assert len(HELP_LINES) > 0
        assert all(line.strip() for line in HELP_LINES)


class TestLifecycleHelpReset:
    def test_start_shows_help_again_even_if_previously_hidden(self, feeder: FrameFeeder) -> None:
        feeder.settle(region_center("help"))
        feeder.feed(make_hand(region_center("help"), pinch=True))
        assert not feeder.processor.show_help
        feeder.processor.start()
        assert feeder.processor.show_help


class TestLifecycle:
    def test_start_resets_canvas_and_selection(self, feeder: FrameFeeder) -> None:
        feeder.feed(make_hand(region_center("brush", 2), pinch=True))
        feeder.feed_many(make_hand((0.4, 0.5), pinch=True), 3)
        feeder.processor.start()
        assert feeder.processor.strokes == ()
        assert feeder.processor.tool is Tool.PEN
        assert feeder.processor.brush_thickness == BRUSH_SIZES[0]
        assert feeder.processor.brush_color == PALETTE[0]

    def test_stop_commits_active_stroke_and_keeps_canvas(self, feeder: FrameFeeder) -> None:
        feeder.feed_many(make_hand((0.4, 0.5), pinch=True), 3)
        feeder.processor.stop()
        assert len(feeder.processor.strokes) == 1
        assert not feeder.processor.is_pen_down


class TestWidget:
    def test_shows_unavailable_message_without_backend_results(self, qapp: object) -> None:
        from visionplay.apps.air_canvas.widget import NO_DATA_MESSAGE, AirCanvasWidget

        widget = AirCanvasWidget()
        frame = Frame.from_image(0, 0.0, np.zeros((60, 80, 3), dtype=np.uint8))
        widget.on_frame_ready(frame)
        assert widget.overlay_message == NO_DATA_MESSAGE

    def test_shows_hint_once_backend_results_exist(self, qapp: object) -> None:
        from visionplay.apps.air_canvas.widget import HINT_MESSAGE, AirCanvasWidget

        widget = AirCanvasWidget()
        frame = Frame.from_image(0, 0.0, np.zeros((60, 80, 3), dtype=np.uint8))
        frame.results[RESULTS_KEY] = HandLandmarkResult()
        widget.on_frame_ready(frame)
        assert widget.overlay_message == HINT_MESSAGE


class TestDiscovery:
    def test_real_registry_discovers_the_app(self) -> None:
        from visionplay.core.event_bus import EventBus
        from visionplay.core.plugin_registry import PluginRegistry

        registry = PluginRegistry(event_bus=EventBus())
        registry.discover()
        assert "air_canvas" in registry.manifests
        assert registry.manifests["air_canvas"].name == "Air Canvas"

    def test_starts_and_stops_through_the_registry(self) -> None:
        from visionplay.core.event_bus import EventBus
        from visionplay.core.plugin_registry import PluginRegistry

        registry = PluginRegistry(event_bus=EventBus())
        registry.discover()
        registry.start("air_canvas")
        assert registry.active_app_id == "air_canvas"
        registry.stop_active()
        assert registry.active_app_id is None
