"""Unit tests for visionplay.vision.camera.webcam_source.

CI-safe tests monkeypatch ``cv2.VideoCapture`` with a fake; the one test
that touches real hardware is marked ``webcam`` and skipped in CI.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pytest

from visionplay.vision.camera import webcam_source
from visionplay.vision.camera.camera_source import CameraError
from visionplay.vision.camera.webcam_source import WebcamSource
from visionplay.vision.pipeline.frame_types import ColorFormat, Frame

#: The genuine cv2 class, captured before the autouse fixture patches it.
_REAL_VIDEOCAPTURE = webcam_source.cv2.VideoCapture


class FakeVideoCapture:
    """Stand-in for ``cv2.VideoCapture``: scripted open/read behavior."""

    #: Backend values that refuse to open, shared across constructions.
    failing_backends: frozenset[int] = frozenset()
    #: Every instance constructed, for asserting fallback order.
    instances: ClassVar[list[FakeVideoCapture]] = []

    def __init__(self, index: int, backend: int) -> None:
        self.index = index
        self.backend = backend
        self.props: dict[int, float] = {}
        self.released = False
        self.read_ok = True
        self._opened = backend not in self.failing_backends
        FakeVideoCapture.instances.append(self)

    def isOpened(self) -> bool:  # noqa: N802 (cv2 naming)
        return self._opened and not self.released

    def read(self) -> tuple[bool, Any]:
        if not self.read_ok:
            return False, None
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    def set(self, prop: int, value: float) -> bool:
        self.props[prop] = value
        return True

    def release(self) -> None:
        self.released = True


@pytest.fixture(autouse=True)
def fake_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace cv2.VideoCapture with the fake and reset its class state."""
    FakeVideoCapture.failing_backends = frozenset()
    FakeVideoCapture.instances = []
    monkeypatch.setattr(webcam_source.cv2, "VideoCapture", FakeVideoCapture)


class TestConstruction:
    def test_negative_device_index_rejected(self) -> None:
        with pytest.raises(ValueError, match="device_index"):
            WebcamSource(-1)

    def test_nonpositive_dimensions_rejected(self) -> None:
        with pytest.raises(ValueError, match="frame_width"):
            WebcamSource(0, frame_width=0)
        with pytest.raises(ValueError, match="frame_height"):
            WebcamSource(0, frame_height=-1)


class TestOpen:
    def test_open_uses_first_backend_that_works(self) -> None:
        source = WebcamSource(0)
        source.open()
        assert source.is_open()
        assert len(FakeVideoCapture.instances) == 1

    def test_open_falls_back_when_first_backend_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(webcam_source, "_capture_backends", lambda: (11, 22))
        FakeVideoCapture.failing_backends = frozenset({11})
        source = WebcamSource(0)
        source.open()
        assert source.is_open()
        assert [c.backend for c in FakeVideoCapture.instances] == [11, 22]
        assert FakeVideoCapture.instances[0].released  # failed handle cleaned up

    def test_open_failure_raises_camera_error_naming_device(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(webcam_source, "_capture_backends", lambda: (11, 22))
        FakeVideoCapture.failing_backends = frozenset({11, 22})
        with pytest.raises(CameraError, match="Camera 3"):
            WebcamSource(3).open()

    def test_open_is_idempotent(self) -> None:
        source = WebcamSource(0)
        source.open()
        source.open()
        assert len(FakeVideoCapture.instances) == 1

    def test_requested_frame_size_forwarded_to_driver(self) -> None:
        source = WebcamSource(0, frame_width=1280, frame_height=720)
        source.open()
        assert set(FakeVideoCapture.instances[0].props.values()) == {1280.0, 720.0}

    def test_driver_defaults_when_no_size_requested(self) -> None:
        WebcamSource(0).open()
        assert FakeVideoCapture.instances[0].props == {}


class TestRead:
    def test_read_yields_bgr_frames_with_monotonic_ids(self) -> None:
        source = WebcamSource(0)
        source.open()
        first = source.read()
        second = source.read()
        assert isinstance(first, Frame) and isinstance(second, Frame)
        assert (first.frame_id, second.frame_id) == (0, 1)
        assert first.color_format is ColorFormat.BGR
        assert first.size == (640, 480)

    def test_read_before_open_raises(self) -> None:
        with pytest.raises(CameraError, match="not open"):
            WebcamSource(0).read()

    def test_failed_read_raises_never_returns_none(self) -> None:
        source = WebcamSource(0)
        source.open()
        FakeVideoCapture.instances[0].read_ok = False
        with pytest.raises(CameraError, match="stopped delivering"):
            source.read()


class TestMirror:
    def test_mirrored_by_default(self) -> None:
        source = WebcamSource(0)
        source.open()
        FakeVideoCapture.instances[0].read = lambda: (  # type: ignore[method-assign]
            True,
            _distinct_columns_frame(),
        )
        frame = source.read()
        assert frame is not None
        expected = np.flip(_distinct_columns_frame(), axis=1)
        assert np.array_equal(frame.image, expected)

    def test_mirror_disabled_leaves_image_unflipped(self) -> None:
        source = WebcamSource(0, mirror=False)
        source.open()
        raw = _distinct_columns_frame()
        FakeVideoCapture.instances[0].read = lambda: (True, raw)  # type: ignore[method-assign]
        frame = source.read()
        assert frame is not None
        assert np.array_equal(frame.image, raw)


def _distinct_columns_frame() -> np.ndarray:
    """A 4x4 BGR image whose columns are all distinct, to detect a horizontal flip."""
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    for column in range(4):
        image[:, column] = (column, column, column)
    return image


class TestRelease:
    def test_release_closes_device(self) -> None:
        source = WebcamSource(0)
        source.open()
        source.release()
        assert not source.is_open()
        assert FakeVideoCapture.instances[0].released

    def test_release_is_idempotent_and_safe_unopened(self) -> None:
        source = WebcamSource(0)
        source.release()  # never opened: must not raise
        source.open()
        source.release()
        source.release()


@pytest.mark.webcam
def test_real_webcam_end_to_end() -> None:
    """Opens physical camera 0 and reads one real frame (skipped in CI)."""
    source = WebcamSource(0)
    # Bypass the autouse fake for this test only.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(webcam_source.cv2, "VideoCapture", _REAL_VIDEOCAPTURE)
        with source:
            frame = source.read()
    assert frame is not None
    assert frame.width > 0 and frame.height > 0
