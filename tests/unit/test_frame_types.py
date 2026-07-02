"""Unit tests for visionplay.vision.pipeline.frame_types."""

import numpy as np
import pytest

from visionplay.vision.pipeline.frame_types import ColorFormat, Frame


def make_image(width: int = 64, height: int = 48, channels: int = 3) -> np.ndarray:
    if channels == 1:
        return np.zeros((height, width), dtype=np.uint8)
    return np.zeros((height, width, channels), dtype=np.uint8)


class TestFrameCreation:
    def test_explicit_construction(self) -> None:
        image = make_image()
        frame = Frame(frame_id=1, timestamp=123.5, image=image, width=64, height=48)
        assert frame.frame_id == 1
        assert frame.timestamp == 123.5
        assert frame.image is image

    def test_from_image_derives_dimensions(self) -> None:
        frame = Frame.from_image(frame_id=0, timestamp=0.0, image=make_image(320, 240))
        assert frame.width == 320
        assert frame.height == 240

    def test_from_image_grayscale(self) -> None:
        frame = Frame.from_image(
            frame_id=0,
            timestamp=0.0,
            image=make_image(64, 48, channels=1),
            color_format=ColorFormat.GRAY,
        )
        assert frame.size == (64, 48)
        assert frame.color_format is ColorFormat.GRAY

    def test_mismatched_metadata_rejected(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            Frame(frame_id=0, timestamp=0.0, image=make_image(64, 48), width=48, height=64)


class TestFrameMetadata:
    def test_default_color_format_is_bgr(self) -> None:
        frame = Frame.from_image(frame_id=0, timestamp=0.0, image=make_image())
        assert frame.color_format is ColorFormat.BGR

    def test_shape_property(self) -> None:
        frame = Frame.from_image(frame_id=0, timestamp=0.0, image=make_image(64, 48))
        assert frame.shape == (48, 64, 3)

    def test_size_property_is_width_height(self) -> None:
        frame = Frame.from_image(frame_id=0, timestamp=0.0, image=make_image(320, 240))
        assert frame.size == (320, 240)

    def test_results_starts_empty(self) -> None:
        frame = Frame.from_image(frame_id=0, timestamp=0.0, image=make_image())
        assert frame.results == {}

    def test_results_is_per_frame(self) -> None:
        a = Frame.from_image(frame_id=0, timestamp=0.0, image=make_image())
        b = Frame.from_image(frame_id=1, timestamp=0.0, image=make_image())
        a.results["mediapipe.pose"] = "landmarks"
        assert b.results == {}

    def test_metadata_fields_are_immutable(self) -> None:
        frame = Frame.from_image(frame_id=0, timestamp=0.0, image=make_image())
        with pytest.raises(AttributeError):
            frame.frame_id = 99  # type: ignore[misc]

    def test_results_dict_is_fillable_in_place(self) -> None:
        frame = Frame.from_image(frame_id=0, timestamp=0.0, image=make_image())
        frame.results["onnx"] = [1, 2, 3]
        assert frame.results["onnx"] == [1, 2, 3]
