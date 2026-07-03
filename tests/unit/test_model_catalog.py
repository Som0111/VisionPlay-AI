"""Unit tests for visionplay.vision.inference.model_catalog."""

from __future__ import annotations

from visionplay.vision.inference.model_catalog import HAND_LANDMARKER
from visionplay.vision.inference.model_registry import ModelFormat


class TestHandLandmarkerSpec:
    def test_is_a_task_bundle(self) -> None:
        assert HAND_LANDMARKER.format is ModelFormat.TASK

    def test_identity_fields(self) -> None:
        assert HAND_LANDMARKER.model_id == "hand_landmarker"
        assert HAND_LANDMARKER.filename == "hand_landmarker.task"

    def test_url_is_https(self) -> None:
        assert HAND_LANDMARKER.url.startswith("https://")

    def test_checksum_is_64_hex(self) -> None:
        digest = HAND_LANDMARKER.sha256
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest.lower())
