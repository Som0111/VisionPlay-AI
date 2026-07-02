"""Unit tests for visionplay.vision.inference.model_registry."""

import hashlib
from pathlib import Path

import pytest

from visionplay.vision.inference.model_registry import (
    ModelDownloader,
    ModelFormat,
    ModelRegistry,
    ModelRegistryError,
    ModelSpec,
)

MODEL_BYTES = b"fake onnx model bytes"
MODEL_SHA256 = hashlib.sha256(MODEL_BYTES).hexdigest()


def make_spec(
    model_id: str = "yolo_nano",
    sha256: str = MODEL_SHA256,
    filename: str = "yolo_nano.onnx",
) -> ModelSpec:
    return ModelSpec(
        model_id=model_id,
        format=ModelFormat.ONNX,
        url=f"https://example.invalid/{filename}",
        sha256=sha256,
        filename=filename,
    )


class FakeDownloader(ModelDownloader):
    """Writes fixed bytes to the destination and counts invocations."""

    def __init__(self, payload: bytes = MODEL_BYTES) -> None:
        self._payload = payload
        self.calls = 0

    def download(self, url: str, destination: Path) -> None:
        self.calls += 1
        destination.write_bytes(self._payload)


class TestModelSpec:
    def test_valid_spec_constructs(self) -> None:
        assert make_spec().format is ModelFormat.ONNX

    def test_empty_model_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="model_id"):
            make_spec(model_id="")

    def test_malformed_sha256_rejected(self) -> None:
        with pytest.raises(ValueError, match="64 hex"):
            make_spec(sha256="abc123")

    def test_non_hex_sha256_rejected(self) -> None:
        with pytest.raises(ValueError, match="64 hex"):
            make_spec(sha256="z" * 64)

    def test_filename_with_path_separator_rejected(self) -> None:
        with pytest.raises(ValueError, match="bare file name"):
            make_spec(filename="../escape.onnx")


class TestCatalog:
    def test_register_and_get(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        spec = make_spec()
        registry.register(spec)
        assert registry.get("yolo_nano") == spec

    def test_reregistering_identical_spec_is_noop(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        registry.register(make_spec())
        registry.register(make_spec())
        assert registry.get("yolo_nano") == make_spec()

    def test_conflicting_reregistration_raises(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        registry.register(make_spec())
        with pytest.raises(ModelRegistryError, match="already registered"):
            registry.register(make_spec(filename="different.onnx"))

    def test_unknown_id_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ModelRegistryError, match="'nope'"):
            ModelRegistry(tmp_path).get("nope")

    def test_path_for_is_inside_models_dir(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        assert registry.path_for(make_spec()) == tmp_path / "yolo_nano.onnx"


class TestEnsure:
    def test_downloads_verifies_and_caches_on_first_use(self, tmp_path: Path) -> None:
        downloader = FakeDownloader()
        registry = ModelRegistry(tmp_path / "models", downloader)
        path = registry.ensure(make_spec())
        assert path == tmp_path / "models" / "yolo_nano.onnx"
        assert path.read_bytes() == MODEL_BYTES
        assert downloader.calls == 1
        assert registry.is_cached(make_spec())

    def test_cached_model_is_not_redownloaded(self, tmp_path: Path) -> None:
        downloader = FakeDownloader()
        registry = ModelRegistry(tmp_path, downloader)
        registry.ensure(make_spec())
        registry.ensure(make_spec())
        assert downloader.calls == 1

    def test_checksum_mismatch_raises_and_caches_nothing(self, tmp_path: Path) -> None:
        downloader = FakeDownloader(payload=b"tampered bytes")
        registry = ModelRegistry(tmp_path, downloader)
        with pytest.raises(ModelRegistryError, match="Checksum mismatch"):
            registry.ensure(make_spec())
        assert list(tmp_path.iterdir()) == []  # no artifact, no .part left behind

    def test_corrupt_cached_file_is_refetched(self, tmp_path: Path) -> None:
        spec = make_spec()
        registry = ModelRegistry(tmp_path, FakeDownloader())
        (tmp_path / spec.filename).write_bytes(b"bit-rotted garbage")
        assert not registry.is_cached(spec)
        path = registry.ensure(spec)
        assert path.read_bytes() == MODEL_BYTES

    def test_cache_miss_without_downloader_raises(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        with pytest.raises(ModelRegistryError, match="no downloader"):
            registry.ensure(make_spec())

    def test_valid_preexisting_cache_needs_no_downloader(self, tmp_path: Path) -> None:
        spec = make_spec()
        (tmp_path / spec.filename).write_bytes(MODEL_BYTES)
        registry = ModelRegistry(tmp_path)
        assert registry.ensure(spec) == tmp_path / spec.filename

    def test_uppercase_spec_digest_still_verifies(self, tmp_path: Path) -> None:
        spec = make_spec(sha256=MODEL_SHA256.upper())
        registry = ModelRegistry(tmp_path, FakeDownloader())
        assert registry.ensure(spec).read_bytes() == MODEL_BYTES


class TestIsCached:
    def test_false_when_absent(self, tmp_path: Path) -> None:
        assert not ModelRegistry(tmp_path).is_cached(make_spec())

    def test_true_only_with_matching_checksum(self, tmp_path: Path) -> None:
        spec = make_spec()
        registry = ModelRegistry(tmp_path)
        (tmp_path / spec.filename).write_bytes(MODEL_BYTES)
        assert registry.is_cached(spec)


def test_model_registry_error_is_exception() -> None:
    assert issubclass(ModelRegistryError, Exception)
