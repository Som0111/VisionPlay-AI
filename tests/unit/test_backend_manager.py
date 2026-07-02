"""Unit tests for visionplay.vision.inference.backend_manager.

These exercise the manager with *fake* backends only — no real MediaPipe/ONNX
runtime, no camera, no Qt — per M2.1's "Done when" criteria.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from visionplay.vision.inference.backend_base import InferenceBackend, InferenceError
from visionplay.vision.inference.backend_manager import BackendManager, BackendRegistration
from visionplay.vision.inference.device import DeviceConfig
from visionplay.vision.pipeline.frame_types import Frame


class FakeBackend(InferenceBackend):
    """Minimal in-memory backend that counts loads and records its device."""

    def __init__(self, name: str, device: DeviceConfig | None = None) -> None:
        super().__init__(device)
        self._name = name
        self._loaded = False
        self.load_calls = 0
        self.unload_calls = 0

    @property
    def name(self) -> str:
        return self._name

    def load(self) -> None:
        self.load_calls += 1
        self._loaded = True

    def infer(self, frame: Frame) -> Any:
        return None

    def unload(self) -> None:
        self.unload_calls += 1
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded


def make_frame() -> Frame:
    return Frame.from_image(frame_id=0, timestamp=0.0, image=np.zeros((2, 2, 3), dtype=np.uint8))


def registration(
    name: str = "fake.one",
    *,
    available: bool = True,
    backend: FakeBackend | None = None,
) -> tuple[BackendRegistration, list[FakeBackend]]:
    """Build a registration whose factory records every backend it creates."""
    created: list[FakeBackend] = []

    def factory(device: DeviceConfig) -> InferenceBackend:
        instance = backend if backend is not None else FakeBackend(name, device)
        created.append(instance)
        return instance

    reg = BackendRegistration(name=name, factory=factory, probe=lambda: available)
    return reg, created


class TestRegistration:
    def test_registered_name_is_reported(self) -> None:
        manager = BackendManager()
        reg, _ = registration("fake.one")
        manager.register(reg)
        assert manager.is_registered("fake.one")
        assert manager.registered_names() == ("fake.one",)

    def test_unregistered_name_is_not_registered(self) -> None:
        assert not BackendManager().is_registered("nope")

    def test_reregistering_identical_registration_is_noop(self) -> None:
        manager = BackendManager()
        reg, _ = registration("fake.one")
        manager.register(reg)
        manager.register(reg)
        assert manager.registered_names() == ("fake.one",)

    def test_conflicting_registration_raises(self) -> None:
        manager = BackendManager()
        first, _ = registration("fake.one")
        second, _ = registration("fake.one")
        manager.register(first)
        with pytest.raises(InferenceError, match="already registered"):
            manager.register(second)


class TestAvailability:
    def test_available_backend_reports_true(self) -> None:
        manager = BackendManager()
        reg, _ = registration("fake.one", available=True)
        manager.register(reg)
        assert manager.is_available("fake.one")

    def test_unavailable_backend_reports_false(self) -> None:
        manager = BackendManager()
        reg, _ = registration("fake.one", available=False)
        manager.register(reg)
        assert not manager.is_available("fake.one")

    def test_unregistered_name_reports_false_without_raising(self) -> None:
        assert not BackendManager().is_available("ghost")

    def test_raising_probe_is_treated_as_unavailable(self) -> None:
        manager = BackendManager()

        def boom() -> bool:
            raise RuntimeError("probe blew up")

        reg, _ = registration("fake.one")
        manager.register(BackendRegistration(name="fake.one", factory=reg.factory, probe=boom))
        assert not manager.is_available("fake.one")

    def test_availability_does_not_construct_or_load(self) -> None:
        manager = BackendManager()
        reg, created = registration("fake.one", available=True)
        manager.register(reg)
        manager.is_available("fake.one")
        assert created == []  # probe alone builds nothing
        assert not manager.is_loaded("fake.one")


class TestAcquire:
    def test_acquire_constructs_and_loads(self) -> None:
        manager = BackendManager()
        reg, created = registration("fake.one")
        manager.register(reg)
        backend = manager.acquire("fake.one")
        assert backend.is_loaded()
        assert len(created) == 1
        assert manager.is_loaded("fake.one")
        assert manager.loaded_names() == ("fake.one",)

    def test_same_name_returns_same_warm_instance_without_reloading(self) -> None:
        manager = BackendManager()
        reg, created = registration("fake.one")
        manager.register(reg)
        first = manager.acquire("fake.one")
        second = manager.acquire("fake.one")
        assert first is second
        assert len(created) == 1
        assert isinstance(first, FakeBackend)
        assert first.load_calls == 1  # warm cache: no duplicate load

    def test_acquire_passes_managers_device_to_factory(self) -> None:
        manager = BackendManager(DeviceConfig.cpu())
        reg, _ = registration("fake.one")
        manager.register(reg)
        backend = manager.acquire("fake.one")
        assert backend.device == DeviceConfig.cpu()

    def test_acquire_unregistered_name_raises_inference_error(self) -> None:
        with pytest.raises(InferenceError, match="No inference backend registered"):
            BackendManager().acquire("does.not.exist")

    def test_factory_name_mismatch_raises(self) -> None:
        manager = BackendManager()
        # Factory advertised as "fake.one" but builds a backend named "other".
        _, _ = registration("fake.one")
        mismatched = FakeBackend("other")
        manager.register(
            BackendRegistration(
                name="fake.one",
                factory=lambda device: mismatched,
                probe=lambda: True,
            )
        )
        with pytest.raises(InferenceError, match="names must match"):
            manager.acquire("fake.one")

    def test_load_failure_is_not_cached(self) -> None:
        manager = BackendManager()

        class FailingBackend(FakeBackend):
            def load(self) -> None:
                raise InferenceError("cannot load")

        failing = FailingBackend("fake.one")
        manager.register(
            BackendRegistration(
                name="fake.one",
                factory=lambda device: failing,
                probe=lambda: True,
            )
        )
        with pytest.raises(InferenceError, match="cannot load"):
            manager.acquire("fake.one")
        assert not manager.is_loaded("fake.one")


class TestRelease:
    def test_release_unloads_and_drops_instance(self) -> None:
        manager = BackendManager()
        reg, _ = registration("fake.one")
        manager.register(reg)
        backend = manager.acquire("fake.one")
        assert isinstance(backend, FakeBackend)
        manager.release("fake.one")
        assert not backend.is_loaded()
        assert backend.unload_calls == 1
        assert not manager.is_loaded("fake.one")

    def test_release_is_safe_for_never_acquired_name(self) -> None:
        manager = BackendManager()
        reg, _ = registration("fake.one")
        manager.register(reg)
        manager.release("fake.one")  # no instance yet — must not raise

    def test_reacquire_after_release_reloads(self) -> None:
        manager = BackendManager()
        reg, created = registration("fake.one")
        manager.register(reg)
        manager.acquire("fake.one")
        manager.release("fake.one")
        manager.acquire("fake.one")
        assert len(created) == 2  # a fresh instance after release

    def test_release_all_unloads_every_instance(self) -> None:
        manager = BackendManager()
        reg_a, _ = registration("fake.a")
        reg_b, _ = registration("fake.b")
        manager.register(reg_a)
        manager.register(reg_b)
        a = manager.acquire("fake.a")
        b = manager.acquire("fake.b")
        assert isinstance(a, FakeBackend)
        assert isinstance(b, FakeBackend)
        manager.release_all()
        assert manager.loaded_names() == ()
        assert not a.is_loaded()
        assert not b.is_loaded()


class TestDevice:
    def test_default_device_is_cpu(self) -> None:
        assert BackendManager().device == DeviceConfig.cpu()

    def test_explicit_device_is_kept(self) -> None:
        device = DeviceConfig.cpu()
        assert BackendManager(device).device is device
