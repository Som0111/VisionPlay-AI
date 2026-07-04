"""Unit tests for Air Canvas M3.3: undo/redo, layers, shape recognition, export.

Gesture-driven state (undo/redo/layers/shapes/export toggles reachable from
the toolbar) is exercised the same headless way as the M3.2 suite — via
:class:`FrameFeeder` feeding synthetic hands, no Qt/camera/MediaPipe
involved. Keyboard-only actions (undo/redo/save) additionally get a small
widget-level test using the shared ``qapp`` fixture.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest
from _helpers import FrameFeeder, make_hand, region_center

from visionplay.apps.air_canvas.processor import AirCanvasProcessor, classify_stroke_shape
from visionplay.vision.pipeline.frame_types import Frame


@pytest.fixture
def feeder() -> FrameFeeder:
    processor = AirCanvasProcessor()
    processor.start()
    return FrameFeeder(processor)


def _draw_stroke(feeder: FrameFeeder, start: tuple[float, float], end: tuple[float, float]) -> None:
    """Draw and commit one short pinch stroke from ``start`` to ``end``."""
    feeder.settle(start)
    feeder.feed_many(make_hand(start, pinch=True), 3)
    feeder.feed_many(make_hand(end, pinch=True), 3)
    feeder.feed(make_hand(end, pinch=False))


def _activate(feeder: FrameFeeder, action: str, payload: int = 0) -> None:
    """Trigger exactly one toolbar activation via a pinch, not dwell.

    Dwell-hovering a region for many frames (as a real user might) fires
    repeatedly every ``DWELL_FRAMES`` — fine for idempotent M3.2 actions
    (reselecting the same color twice is a no-op) but wrong for M3.3's
    non-idempotent/toggle actions (adding a layer twice, or toggling
    visibility twice back to unchanged). A single pinch frame activates
    exactly once via ``pinch_started``. Dropping tracking first
    (``feed(None)``) resets the smoothing filter so that frame's position is
    unfiltered and lands exactly on the region, no settling required.
    """
    point = region_center(action, payload)
    feeder.feed(None)
    feeder.feed(make_hand(point, pinch=True))
    feeder.feed(make_hand(point, pinch=False))


class TestUndoRedo:
    def test_undo_reverses_the_last_stroke(self, feeder: FrameFeeder) -> None:
        _draw_stroke(feeder, (0.2, 0.5), (0.3, 0.5))
        assert len(feeder.processor.strokes) == 1
        feeder.processor.undo()
        assert feeder.processor.strokes == ()

    def test_redo_reapplies_an_undone_stroke(self, feeder: FrameFeeder) -> None:
        _draw_stroke(feeder, (0.2, 0.5), (0.3, 0.5))
        feeder.processor.undo()
        feeder.processor.redo()
        assert len(feeder.processor.strokes) == 1

    def test_multi_step_undo_redo(self, feeder: FrameFeeder) -> None:
        _draw_stroke(feeder, (0.2, 0.5), (0.25, 0.5))
        _draw_stroke(feeder, (0.6, 0.5), (0.65, 0.5))
        _draw_stroke(feeder, (0.8, 0.5), (0.85, 0.5))
        assert len(feeder.processor.strokes) == 3
        feeder.processor.undo()
        feeder.processor.undo()
        assert len(feeder.processor.strokes) == 1
        feeder.processor.redo()
        assert len(feeder.processor.strokes) == 2
        feeder.processor.redo()
        assert len(feeder.processor.strokes) == 3

    def test_new_stroke_invalidates_the_redo_stack(self, feeder: FrameFeeder) -> None:
        _draw_stroke(feeder, (0.2, 0.5), (0.25, 0.5))
        feeder.processor.undo()
        assert feeder.processor.can_redo
        _draw_stroke(feeder, (0.6, 0.5), (0.65, 0.5))
        assert not feeder.processor.can_redo
        feeder.processor.redo()  # no-op: nothing to redo
        assert len(feeder.processor.strokes) == 1

    def test_undo_past_the_start_is_a_no_op(self, feeder: FrameFeeder) -> None:
        assert not feeder.processor.can_undo
        feeder.processor.undo()
        assert feeder.processor.strokes == ()

    def test_redo_past_the_end_is_a_no_op(self, feeder: FrameFeeder) -> None:
        assert not feeder.processor.can_redo
        feeder.processor.redo()
        assert feeder.processor.strokes == ()

    def test_undo_reverses_an_erase_drag_as_one_step(self, feeder: FrameFeeder) -> None:
        _draw_stroke(feeder, (0.2, 0.5), (0.25, 0.5))
        _draw_stroke(feeder, (0.6, 0.5), (0.65, 0.5))
        _activate(feeder, "eraser")
        feeder.settle((0.2, 0.5))
        feeder.feed_many(make_hand((0.2, 0.5), pinch=True), 3)
        feeder.feed(make_hand((0.2, 0.5), pinch=False))
        assert len(feeder.processor.strokes) == 1
        feeder.processor.undo()
        assert len(feeder.processor.strokes) == 2

    def test_undo_reverses_clear(self, feeder: FrameFeeder) -> None:
        _draw_stroke(feeder, (0.2, 0.5), (0.25, 0.5))
        _activate(feeder, "clear")
        assert feeder.processor.strokes == ()
        feeder.processor.undo()
        assert len(feeder.processor.strokes) == 1

    def test_clearing_an_empty_layer_pushes_no_undo_entry(self, feeder: FrameFeeder) -> None:
        assert not feeder.processor.can_undo
        feeder.processor.clear()
        assert not feeder.processor.can_undo

    def test_request_undo_is_applied_on_the_next_process_call(self, feeder: FrameFeeder) -> None:
        _draw_stroke(feeder, (0.2, 0.5), (0.25, 0.5))
        assert len(feeder.processor.strokes) == 1
        feeder.processor.request_undo()
        # No hand this tick — request_undo must still be drained.
        feeder.feed(None)
        assert feeder.processor.strokes == ()

    def test_request_redo_is_applied_on_the_next_process_call(self, feeder: FrameFeeder) -> None:
        _draw_stroke(feeder, (0.2, 0.5), (0.25, 0.5))
        feeder.processor.undo()
        feeder.processor.request_redo()
        feeder.feed(None)
        assert len(feeder.processor.strokes) == 1


class TestLayers:
    def test_starts_with_one_visible_layer(self, feeder: FrameFeeder) -> None:
        assert len(feeder.processor.layers) == 1
        assert feeder.processor.active_layer_index == 0
        assert feeder.processor.layers[0].visible

    def test_layer_add_creates_and_activates_a_new_layer(self, feeder: FrameFeeder) -> None:
        _activate(feeder, "layer_add")
        assert len(feeder.processor.layers) == 2
        assert feeder.processor.active_layer_index == 1

    def test_layer_cycles_through_layers_and_wraps(self, feeder: FrameFeeder) -> None:
        _activate(feeder, "layer_add")
        assert feeder.processor.active_layer_index == 1
        _activate(feeder, "layer")
        assert feeder.processor.active_layer_index == 0

    def test_strokes_go_to_the_active_layer(self, feeder: FrameFeeder) -> None:
        _activate(feeder, "layer_add")
        _draw_stroke(feeder, (0.2, 0.5), (0.25, 0.5))
        assert feeder.processor.layers[0].strokes == []
        assert len(feeder.processor.layers[1].strokes) == 1

    def test_layer_eye_toggles_active_layer_visibility(self, feeder: FrameFeeder) -> None:
        assert feeder.processor.layers[0].visible
        _activate(feeder, "layer_eye")
        assert not feeder.processor.layers[0].visible
        _activate(feeder, "layer_eye")
        assert feeder.processor.layers[0].visible

    def test_hidden_layer_strokes_are_not_rendered_but_still_counted(
        self, feeder: FrameFeeder
    ) -> None:
        _draw_stroke(feeder, (0.3, 0.5), (0.7, 0.5))
        _activate(feeder, "layer_eye")  # hide the only (active) layer
        frame = feeder.feed(None)
        row = frame.image[120]  # y == 0.5 of a 240-row frame
        assert not row.any()
        assert len(feeder.processor.strokes) == 1  # still in the inventory

    def test_layer_up_moves_active_layer_toward_the_top(self, feeder: FrameFeeder) -> None:
        _activate(feeder, "layer_add")  # layers: [0, 1(active)]
        _activate(feeder, "layer")  # cycle back to layer 0 (active)
        assert feeder.processor.active_layer_index == 0
        original_top = feeder.processor.layers[1]
        _activate(feeder, "layer_up")
        assert feeder.processor.active_layer_index == 1
        assert feeder.processor.layers[0] is original_top

    def test_layer_up_at_the_top_is_a_no_op(self, feeder: FrameFeeder) -> None:
        _activate(feeder, "layer_up")
        assert feeder.processor.active_layer_index == 0
        assert len(feeder.processor.layers) == 1

    def test_layer_down_at_the_bottom_is_a_no_op(self, feeder: FrameFeeder) -> None:
        _activate(feeder, "layer_down")
        assert feeder.processor.active_layer_index == 0

    def test_compositing_order_draws_later_layers_on_top(self, feeder: FrameFeeder) -> None:
        # Two overlapping strokes on two different layers: the later
        # (top) layer's color must win at the overlap pixel.
        _draw_stroke(feeder, (0.45, 0.5), (0.55, 0.5))  # White, layer 0
        _activate(feeder, "layer_add")
        _activate(feeder, "color", 1)  # Red
        _draw_stroke(feeder, (0.45, 0.5), (0.55, 0.5))  # Red, layer 1 (on top)
        frame = feeder.feed(None)
        pixel = frame.image[120, int(0.5 * 320)]
        blue, green, red = (int(c) for c in pixel)
        # Anti-aliased line edges blend slightly with the white stroke below,
        # so check "clearly red, not white" rather than an exact BGR match.
        assert red > 200
        assert blue < 150 and green < 150

    def test_start_resets_to_one_layer(self, feeder: FrameFeeder) -> None:
        _activate(feeder, "layer_add")
        _activate(feeder, "layer_eye")
        feeder.processor.start()
        assert len(feeder.processor.layers) == 1
        assert feeder.processor.layers[0].visible
        assert feeder.processor.active_layer_index == 0
        assert not feeder.processor.can_undo
        assert not feeder.processor.can_redo


class TestShapeRecognition:
    def test_disabled_by_default(self, feeder: FrameFeeder) -> None:
        assert not feeder.processor.shapes_enabled

    def test_toggle_flips_state(self, feeder: FrameFeeder) -> None:
        _activate(feeder, "shapes")
        assert feeder.processor.shapes_enabled
        _activate(feeder, "shapes")
        assert not feeder.processor.shapes_enabled

    def test_clean_line_classifies_as_line(self) -> None:
        points = [(0.2 + 0.01 * i, 0.5) for i in range(20)]
        assert classify_stroke_shape(points) == "line"

    def test_clean_circle_classifies_as_circle(self) -> None:
        center, radius = (0.5, 0.5), 0.2
        points = [
            (center[0] + radius * math.cos(a), center[1] + radius * math.sin(a))
            for a in (2 * math.pi * i / 48 for i in range(48))
        ]
        assert classify_stroke_shape(points) == "circle"

    def test_clean_rectangle_classifies_as_rectangle(self) -> None:
        points: list[tuple[float, float]] = []
        for t in np.linspace(0, 1, 10, endpoint=False):
            points.append((0.3 + 0.4 * t, 0.3))
        for t in np.linspace(0, 1, 10, endpoint=False):
            points.append((0.7, 0.3 + 0.4 * t))
        for t in np.linspace(0, 1, 10, endpoint=False):
            points.append((0.7 - 0.4 * t, 0.7))
        for t in np.linspace(0, 1, 10, endpoint=False):
            points.append((0.3, 0.7 - 0.4 * t))
        points.append((0.3, 0.3))
        assert classify_stroke_shape([(float(x), float(y)) for x, y in points]) == "rectangle"

    def test_jagged_scribble_classifies_as_none(self) -> None:
        rng = np.random.default_rng(0)
        points = [(float(x), float(y)) for x, y in rng.uniform(0.2, 0.8, size=(30, 2))]
        assert classify_stroke_shape(points) is None

    def test_too_short_a_stroke_classifies_as_none(self) -> None:
        assert classify_stroke_shape([(0.5, 0.5), (0.51, 0.5)]) is None

    def test_enabled_shapes_snaps_a_drawn_line_to_two_points(self, feeder: FrameFeeder) -> None:
        _activate(feeder, "shapes")
        feeder.settle((0.2, 0.5))
        for step in range(15):
            feeder.feed(make_hand((0.2 + 0.02 * step, 0.5), pinch=True))
        feeder.feed(make_hand((0.5, 0.5), pinch=False))
        stroke = feeder.processor.strokes[0]
        assert len(stroke.points) == 2

    def test_disabled_shapes_leaves_the_stroke_freehand(self, feeder: FrameFeeder) -> None:
        feeder.settle((0.2, 0.5))
        for step in range(15):
            feeder.feed(make_hand((0.2 + 0.02 * step, 0.5), pinch=True))
        feeder.feed(make_hand((0.5, 0.5), pinch=False))
        stroke = feeder.processor.strokes[0]
        assert len(stroke.points) == 15


class TestExport:
    def test_bg_toggle_defaults_to_true_and_flips(self, feeder: FrameFeeder) -> None:
        assert feeder.processor.export_include_background
        _activate(feeder, "bg")
        assert not feeder.processor.export_include_background

    def test_render_export_with_background_draws_onto_the_frame(self, feeder: FrameFeeder) -> None:
        _draw_stroke(feeder, (0.3, 0.5), (0.7, 0.5))
        frame = Frame.from_image(0, 0.0, np.full((240, 320, 3), 30, dtype=np.uint8))
        image = feeder.processor.render_export_image(frame, include_background=True)
        assert image.shape == (240, 320, 3)
        assert image[120, 160].tolist() != [30, 30, 30]  # a stroke pixel overwrote the background
        assert image[10, 10].tolist() == [30, 30, 30]  # untouched background survives

    def test_render_export_without_background_is_transparent_elsewhere(
        self, feeder: FrameFeeder
    ) -> None:
        _draw_stroke(feeder, (0.3, 0.5), (0.7, 0.5))
        frame = Frame.from_image(0, 0.0, np.full((240, 320, 3), 30, dtype=np.uint8))
        image = feeder.processor.render_export_image(frame, include_background=False)
        assert image.shape == (240, 320, 4)
        assert image[10, 10, 3] == 0  # transparent where nothing was drawn
        assert image[120, 160, 3] == 255  # opaque where a stroke was drawn

    def test_hidden_layers_are_excluded_from_export(self, feeder: FrameFeeder) -> None:
        _draw_stroke(feeder, (0.3, 0.5), (0.7, 0.5))
        _activate(feeder, "layer_eye")
        frame = Frame.from_image(0, 0.0, np.full((240, 320, 3), 30, dtype=np.uint8))
        image = feeder.processor.render_export_image(frame, include_background=True)
        assert image[120, 160].tolist() == [30, 30, 30]

    def test_request_export_writes_a_png_on_the_next_process_call(
        self, feeder: FrameFeeder, tmp_path: Path
    ) -> None:
        _draw_stroke(feeder, (0.3, 0.5), (0.7, 0.5))
        destination = tmp_path / "canvas.png"
        feeder.processor.request_export(destination, include_background=True)
        feeder.feed(None)
        assert destination.is_file()
        written = cv2.imread(str(destination))
        assert written is not None
        assert written.shape[:2] == (240, 320)


class TestWidgetShortcuts:
    def test_ctrl_z_requests_undo(self, qapp: object) -> None:
        from PySide6.QtCore import QEvent, Qt
        from PySide6.QtGui import QKeyEvent

        from visionplay.apps.air_canvas.widget import AirCanvasWidget

        processor = AirCanvasProcessor()
        processor.start()
        _draw_stroke(FrameFeeder(processor), (0.2, 0.5), (0.3, 0.5))
        assert len(processor.strokes) == 1

        widget = AirCanvasWidget(processor=processor)
        widget.keyPressEvent(
            QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
        )
        processor.process(Frame.from_image(0, 0.0, np.zeros((240, 320, 3), dtype=np.uint8)))
        assert processor.strokes == ()

    def test_ctrl_y_requests_redo(self, qapp: object) -> None:
        from PySide6.QtCore import QEvent, Qt
        from PySide6.QtGui import QKeyEvent

        from visionplay.apps.air_canvas.widget import AirCanvasWidget

        processor = AirCanvasProcessor()
        processor.start()
        _draw_stroke(FrameFeeder(processor), (0.2, 0.5), (0.3, 0.5))
        processor.undo()
        assert processor.strokes == ()

        widget = AirCanvasWidget(processor=processor)
        widget.keyPressEvent(
            QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Y, Qt.KeyboardModifier.ControlModifier)
        )
        processor.process(Frame.from_image(0, 0.0, np.zeros((240, 320, 3), dtype=np.uint8)))
        assert len(processor.strokes) == 1

    def test_ctrl_s_exports_via_file_dialog(
        self, qapp: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from PySide6.QtCore import QEvent, Qt
        from PySide6.QtGui import QKeyEvent

        from visionplay.apps.air_canvas import widget as widget_module

        processor = AirCanvasProcessor()
        processor.start()
        _draw_stroke(FrameFeeder(processor), (0.3, 0.5), (0.7, 0.5))

        destination = tmp_path / "out.png"
        monkeypatch.setattr(
            widget_module.QFileDialog,
            "getSaveFileName",
            staticmethod(lambda *args, **kwargs: (str(destination), "")),
        )
        widget = widget_module.AirCanvasWidget(processor=processor)
        widget.keyPressEvent(
            QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_S, Qt.KeyboardModifier.ControlModifier)
        )
        processor.process(Frame.from_image(0, 0.0, np.zeros((240, 320, 3), dtype=np.uint8)))
        assert destination.is_file()

    def test_cancelling_the_dialog_does_not_queue_an_export(
        self, qapp: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from PySide6.QtCore import QEvent, Qt
        from PySide6.QtGui import QKeyEvent

        from visionplay.apps.air_canvas import widget as widget_module

        processor = AirCanvasProcessor()
        processor.start()
        monkeypatch.setattr(
            widget_module.QFileDialog,
            "getSaveFileName",
            staticmethod(lambda *args, **kwargs: ("", "")),
        )
        widget = widget_module.AirCanvasWidget(processor=processor)
        widget.keyPressEvent(
            QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_S, Qt.KeyboardModifier.ControlModifier)
        )
        assert not (tmp_path / "out.png").exists()

    def test_without_a_processor_key_events_do_not_raise(self, qapp: object) -> None:
        from PySide6.QtCore import QEvent, Qt
        from PySide6.QtGui import QKeyEvent

        from visionplay.apps.air_canvas.widget import AirCanvasWidget

        widget = AirCanvasWidget()
        widget.keyPressEvent(
            QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
        )
