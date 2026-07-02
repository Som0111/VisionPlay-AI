"""Unit tests for visionplay.core.plugin_base."""

import dataclasses

import numpy as np
import pytest

from visionplay.core.plugin_base import (
    APP_CATEGORIES,
    CURRENT_API_VERSION,
    AppManifest,
    AppPlugin,
)
from visionplay.vision.pipeline.frame_types import Frame


def make_frame(frame_id: int = 0) -> Frame:
    return Frame.from_image(
        frame_id=frame_id,
        timestamp=float(frame_id),
        image=np.zeros((4, 4, 3), dtype=np.uint8),
    )


def make_manifest(**overrides: object) -> AppManifest:
    defaults: dict[str, object] = {
        "id": "hand_tracking_demo",
        "name": "Hand Tracking Demo",
        "category": "ai_demos",
        "version": "0.1.0",
        "api_version": CURRENT_API_VERSION,
        "required_backends": ("mediapipe.hands",),
        "icon": "assets/icon.png",
    }
    defaults.update(overrides)
    return AppManifest(**defaults)  # type: ignore[arg-type]


class FakePlugin(AppPlugin):
    """Minimal in-memory implementation recording lifecycle calls."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def on_load(self) -> None:
        self.calls.append("on_load")

    def on_start(self) -> None:
        self.calls.append("on_start")

    def on_frame(self, frame: Frame) -> Frame:
        self.calls.append("on_frame")
        return frame

    def on_stop(self) -> None:
        self.calls.append("on_stop")

    def on_unload(self) -> None:
        self.calls.append("on_unload")


class TestAppManifestValid:
    def test_constructs_with_all_fields(self) -> None:
        manifest = make_manifest()
        assert manifest.id == "hand_tracking_demo"
        assert manifest.name == "Hand Tracking Demo"
        assert manifest.category == "ai_demos"
        assert manifest.version == "0.1.0"
        assert manifest.api_version == CURRENT_API_VERSION
        assert manifest.required_backends == ("mediapipe.hands",)
        assert manifest.icon == "assets/icon.png"

    @pytest.mark.parametrize("category", sorted(APP_CATEGORIES))
    def test_accepts_every_defined_category(self, category: str) -> None:
        assert make_manifest(category=category).category == category

    def test_required_backends_normalized_to_tuple(self) -> None:
        manifest = make_manifest(required_backends=["mediapipe.hands", "onnx.yolo"])
        assert manifest.required_backends == ("mediapipe.hands", "onnx.yolo")
        assert isinstance(manifest.required_backends, tuple)

    def test_required_backends_defaults_to_empty_tuple(self) -> None:
        manifest = make_manifest(required_backends=())
        assert manifest.required_backends == ()

    def test_icon_defaults_to_empty_string(self) -> None:
        manifest = AppManifest(
            id="x",
            name="X",
            category="ai_demos",
            version="0.1.0",
            api_version=CURRENT_API_VERSION,
        )
        assert manifest.icon == ""


class TestAppManifestInvalid:
    def test_unknown_category_raises(self) -> None:
        with pytest.raises(ValueError, match=r"'not_a_category'"):
            make_manifest(category="not_a_category")

    def test_error_message_lists_supported_categories(self) -> None:
        with pytest.raises(ValueError, match=r"ai_demos.*face_ar.*fitness.*gesture_games"):
            make_manifest(category="bogus")

    def test_empty_category_raises(self) -> None:
        with pytest.raises(ValueError):
            make_manifest(category="")


class TestAppManifestFrozen:
    def test_cannot_reassign_field(self) -> None:
        manifest = make_manifest()
        with pytest.raises(dataclasses.FrozenInstanceError):
            manifest.name = "New Name"  # type: ignore[misc]

    def test_equality_and_hash(self) -> None:
        assert make_manifest() == make_manifest()
        assert hash(make_manifest()) == hash(make_manifest())


class TestAppPluginAbstractEnforcement:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            AppPlugin()  # type: ignore[abstract]

    def test_incomplete_subclass_rejected(self) -> None:
        class MissingMethods(AppPlugin):
            def on_load(self) -> None: ...
            def on_start(self) -> None: ...

            # on_frame, on_stop, on_unload not implemented

        with pytest.raises(TypeError, match="abstract"):
            MissingMethods()  # type: ignore[abstract]

    def test_complete_subclass_instantiates(self) -> None:
        assert isinstance(FakePlugin(), AppPlugin)


class TestAppPluginLifecycle:
    def test_on_frame_returns_a_frame(self) -> None:
        plugin = FakePlugin()
        frame = make_frame(3)
        result = plugin.on_frame(frame)
        assert isinstance(result, Frame)
        assert result.frame_id == 3

    def test_lifecycle_methods_are_callable_in_order(self) -> None:
        plugin = FakePlugin()
        plugin.on_load()
        plugin.on_start()
        plugin.on_frame(make_frame())
        plugin.on_stop()
        plugin.on_unload()
        assert plugin.calls == ["on_load", "on_start", "on_frame", "on_stop", "on_unload"]
