"""Unit tests for visionplay.vision.camera.camera_source."""

import numpy as np
import pytest

from visionplay.vision.camera.camera_source import CameraError, CameraSource
from visionplay.vision.pipeline.frame_types import Frame


class FakeSource(CameraSource):
    """Minimal in-memory implementation: serves N synthetic frames then EOF."""

    def __init__(self, total_frames: int = 3, fail_open: bool = False) -> None:
        self._total = total_frames
        self._fail_open = fail_open
        self._opened = False
        self._next_id = 0
        self.release_calls = 0

    def open(self) -> None:
        if self._fail_open:
            raise CameraError("Fake device 0 could not be opened")
        self._opened = True

    def read(self) -> Frame | None:
        if not self._opened:
            raise CameraError("read() on a source that is not open")
        if self._next_id >= self._total:
            return None  # end of stream
        frame = Frame.from_image(
            frame_id=self._next_id,
            timestamp=float(self._next_id),
            image=np.zeros((4, 4, 3), dtype=np.uint8),
        )
        self._next_id += 1
        return frame

    def release(self) -> None:
        self.release_calls += 1
        self._opened = False

    def is_open(self) -> bool:
        return self._opened


class TestAbstractInterface:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            CameraSource()  # type: ignore[abstract]

    def test_incomplete_subclass_rejected(self) -> None:
        class MissingMethods(CameraSource):
            def open(self) -> None: ...
            def is_open(self) -> bool:
                return False

            # read() and release() not implemented

        with pytest.raises(TypeError, match="abstract"):
            MissingMethods()  # type: ignore[abstract]

    def test_complete_subclass_instantiates(self) -> None:
        assert isinstance(FakeSource(), CameraSource)


class TestLifecycle:
    def test_open_read_release(self) -> None:
        source = FakeSource(total_frames=2)
        source.open()
        assert source.is_open()
        first = source.read()
        assert isinstance(first, Frame)
        assert first.frame_id == 0
        source.release()
        assert not source.is_open()

    def test_frame_ids_monotonic(self) -> None:
        source = FakeSource(total_frames=3)
        source.open()
        ids = [frame.frame_id for frame in iter(source.read, None)]
        assert ids == [0, 1, 2]

    def test_end_of_stream_returns_none(self) -> None:
        source = FakeSource(total_frames=1)
        source.open()
        assert source.read() is not None
        assert source.read() is None

    def test_read_before_open_raises(self) -> None:
        with pytest.raises(CameraError, match="not open"):
            FakeSource().read()

    def test_failed_open_raises_camera_error(self) -> None:
        with pytest.raises(CameraError, match="device 0"):
            FakeSource(fail_open=True).open()


class TestContextManager:
    def test_with_block_opens_and_releases(self) -> None:
        source = FakeSource()
        with source as active:
            assert active is source
            assert source.is_open()
        assert not source.is_open()
        assert source.release_calls == 1

    def test_release_called_even_on_exception(self) -> None:
        source = FakeSource()
        with pytest.raises(RuntimeError), source:
            raise RuntimeError("boom")
        assert source.release_calls == 1


def test_camera_error_is_exception() -> None:
    assert issubclass(CameraError, Exception)
