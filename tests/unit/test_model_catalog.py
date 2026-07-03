"""Unit tests for visionplay.vision.inference.model_catalog."""

from __future__ import annotations

import pytest

from visionplay.vision.inference.model_catalog import (
    FACE_LANDMARKER,
    HAND_LANDMARKER,
    POSE_LANDMARKER,
)
from visionplay.vision.inference.model_registry import ModelFormat, ModelSpec

ALL_SPECS = [HAND_LANDMARKER, POSE_LANDMARKER, FACE_LANDMARKER]


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.model_id)
class TestEverySpec:
    def test_is_a_task_bundle(self, spec: ModelSpec) -> None:
        assert spec.format is ModelFormat.TASK

    def test_url_is_https(self, spec: ModelSpec) -> None:
        assert spec.url.startswith("https://")

    def test_checksum_is_64_hex(self, spec: ModelSpec) -> None:
        digest = spec.sha256
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest.lower())


class TestIdentityFields:
    def test_hand_landmarker(self) -> None:
        assert HAND_LANDMARKER.model_id == "hand_landmarker"
        assert HAND_LANDMARKER.filename == "hand_landmarker.task"

    def test_pose_landmarker(self) -> None:
        assert POSE_LANDMARKER.model_id == "pose_landmarker"
        assert POSE_LANDMARKER.filename == "pose_landmarker_lite.task"

    def test_face_landmarker(self) -> None:
        assert FACE_LANDMARKER.model_id == "face_landmarker"
        assert FACE_LANDMARKER.filename == "face_landmarker.task"

    def test_identities_are_unique(self) -> None:
        assert len({spec.model_id for spec in ALL_SPECS}) == len(ALL_SPECS)
        assert len({spec.filename for spec in ALL_SPECS}) == len(ALL_SPECS)
        assert len({spec.sha256 for spec in ALL_SPECS}) == len(ALL_SPECS)
