"""Unit tests for visionplay.vision.inference.results."""

from __future__ import annotations

import numpy as np
import pytest

from visionplay.vision.inference.results import (
    FaceLandmarkResult,
    FaceLandmarks,
    HandLandmarkResult,
    HandLandmarks,
    LandmarkPoint,
    PoseLandmarkResult,
    PoseLandmarks,
    TensorOutput,
)


class TestHandLandmarkResult:
    def test_default_is_empty(self) -> None:
        result = HandLandmarkResult()
        assert result.is_empty
        assert len(result) == 0
        assert result.hands == ()

    def test_populated_reports_hands(self) -> None:
        hand = HandLandmarks(
            points=(LandmarkPoint(0.1, 0.2, 0.0),),
            handedness="Left",
            score=0.98,
        )
        result = HandLandmarkResult(hands=(hand,))
        assert not result.is_empty
        assert len(result) == 1
        assert result.hands[0].handedness == "Left"
        assert result.hands[0].points[0].x == pytest.approx(0.1)

    def test_is_frozen(self) -> None:
        result = HandLandmarkResult()
        with pytest.raises(AttributeError):
            result.hands = ()  # type: ignore[misc]


class TestPoseLandmarkResult:
    def test_default_is_empty(self) -> None:
        result = PoseLandmarkResult()
        assert result.is_empty
        assert len(result) == 0
        assert result.poses == ()

    def test_populated_reports_poses(self) -> None:
        pose = PoseLandmarks(points=(LandmarkPoint(0.4, 0.5, 0.0),))
        result = PoseLandmarkResult(poses=(pose,))
        assert not result.is_empty
        assert len(result) == 1
        assert result.poses[0].points[0].y == pytest.approx(0.5)

    def test_is_frozen(self) -> None:
        result = PoseLandmarkResult()
        with pytest.raises(AttributeError):
            result.poses = ()  # type: ignore[misc]


class TestFaceLandmarkResult:
    def test_default_is_empty(self) -> None:
        result = FaceLandmarkResult()
        assert result.is_empty
        assert len(result) == 0
        assert result.faces == ()

    def test_populated_reports_faces(self) -> None:
        face = FaceLandmarks(points=(LandmarkPoint(0.7, 0.3, 0.0),))
        result = FaceLandmarkResult(faces=(face,))
        assert not result.is_empty
        assert len(result) == 1
        assert result.faces[0].points[0].x == pytest.approx(0.7)

    def test_is_frozen(self) -> None:
        result = FaceLandmarkResult()
        with pytest.raises(AttributeError):
            result.faces = ()  # type: ignore[misc]


class TestTensorOutput:
    def test_names_and_getitem(self) -> None:
        a = np.zeros((2, 2), dtype=np.float32)
        b = np.ones((3,), dtype=np.float32)
        out = TensorOutput({"a": a, "b": b})
        assert out.names() == ("a", "b")
        assert np.array_equal(out["a"], a)
        assert "b" in out
        assert "missing" not in out

    def test_first_returns_leading_tensor(self) -> None:
        first = np.arange(4, dtype=np.float32)
        out = TensorOutput({"out0": first, "out1": np.zeros(1, dtype=np.float32)})
        assert np.array_equal(out.first(), first)

    def test_first_on_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="no tensors"):
            TensorOutput().first()

    def test_missing_key_raises(self) -> None:
        with pytest.raises(KeyError):
            TensorOutput({"a": np.zeros(1, dtype=np.float32)})["nope"]

    def test_default_is_empty(self) -> None:
        assert TensorOutput().names() == ()
