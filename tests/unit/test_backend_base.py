"""Unit tests for visionplay.vision.inference.backend_base."""

from typing import Any

import numpy as np
import pytest

from visionplay.vision.inference.backend_base import InferenceBackend, InferenceError
from visionplay.vision.inference.device import DeviceConfig, DeviceType
from visionplay.vision.pipeline.frame_types import Frame


def make_frame(frame_id: int = 0) -> Frame:
    return Frame.from_image(
        frame_id=frame_id,
        timestamp=float(frame_id),
        image=np.zeros((4, 4, 3), dtype=np.uint8),
    )


class FakeBackend(InferenceBackend):
    """Minimal in-memory implementation: echoes frame ids as results."""

    def __init__(self, device: DeviceConfig | None = None, fail_load: bool = False) -> None:
        super().__init__(device)
        self._fail_load = fail_load
        self._loaded = False
        self.unload_calls = 0

    @property
    def name(self) -> str:
        return "fake.echo"

    def load(self) -> None:
        if self._fail_load:
            raise InferenceError("Fake model 'echo' could not be loaded")
        self._loaded = True

    def infer(self, frame: Frame) -> Any:
        if not self._loaded:
            raise InferenceError("infer() on a backend that is not loaded")
        return {"frame_id": frame.frame_id}

    def unload(self) -> None:
        self.unload_calls += 1
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded


class TestAbstractInterface:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            InferenceBackend()  # type: ignore[abstract]

    def test_incomplete_subclass_rejected(self) -> None:
        class MissingMethods(InferenceBackend):
            def load(self) -> None: ...
            def is_loaded(self) -> bool:
                return False

            # name, infer() and unload() not implemented

        with pytest.raises(TypeError, match="abstract"):
            MissingMethods()  # type: ignore[abstract]

    def test_complete_subclass_instantiates(self) -> None:
        assert isinstance(FakeBackend(), InferenceBackend)


class TestDevice:
    def test_default_device_is_cpu(self) -> None:
        assert FakeBackend().device == DeviceConfig.cpu()

    def test_explicit_device_is_kept(self) -> None:
        device = DeviceConfig(type=DeviceType.CPU)
        assert FakeBackend(device=device).device is device


class TestLifecycle:
    def test_load_infer_unload(self) -> None:
        backend = FakeBackend()
        backend.load()
        assert backend.is_loaded()
        assert backend.infer(make_frame(7)) == {"frame_id": 7}
        backend.unload()
        assert not backend.is_loaded()

    def test_infer_before_load_raises(self) -> None:
        with pytest.raises(InferenceError, match="not loaded"):
            FakeBackend().infer(make_frame())

    def test_failed_load_raises_inference_error(self) -> None:
        with pytest.raises(InferenceError, match="'echo'"):
            FakeBackend(fail_load=True).load()

    def test_unload_idempotent_and_safe_when_never_loaded(self) -> None:
        backend = FakeBackend()
        backend.unload()
        backend.unload()
        assert backend.unload_calls == 2


class TestContextManager:
    def test_with_block_loads_and_unloads(self) -> None:
        backend = FakeBackend()
        with backend as active:
            assert active is backend
            assert backend.is_loaded()
        assert not backend.is_loaded()
        assert backend.unload_calls == 1

    def test_unload_called_even_on_exception(self) -> None:
        backend = FakeBackend()
        with pytest.raises(RuntimeError), backend:
            raise RuntimeError("boom")
        assert backend.unload_calls == 1


def test_inference_error_is_exception() -> None:
    assert issubclass(InferenceError, Exception)
