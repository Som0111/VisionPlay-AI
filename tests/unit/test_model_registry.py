"""Unit tests for visionplay.vision.inference.model_registry."""

import hashlib
import http.server
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from visionplay.vision.inference.model_registry import (
    HttpModelDownloader,
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


class TestModelFormat:
    def test_task_format_exists(self) -> None:
        assert ModelFormat.TASK.value == "task"

    def test_task_spec_constructs(self) -> None:
        spec = ModelSpec(
            model_id="hand_landmarker",
            format=ModelFormat.TASK,
            url="https://example.invalid/hand_landmarker.task",
            sha256=MODEL_SHA256,
            filename="hand_landmarker.task",
        )
        assert spec.format is ModelFormat.TASK


@contextmanager
def local_http_server(payload: bytes) -> Iterator[str]:
    """Serve ``payload`` for any GET on a background HTTP server; yield its URL."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (http.server API name)
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args: object) -> None:
            pass  # keep test output quiet

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield f"http://{host}:{port}/model.bin"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


class TestHttpModelDownloader:
    def test_downloads_bytes_to_destination(self, tmp_path: Path) -> None:
        dest = tmp_path / "out.bin"
        with local_http_server(MODEL_BYTES) as url:
            HttpModelDownloader().download(url, dest)
        assert dest.read_bytes() == MODEL_BYTES

    def test_connection_failure_raises_registry_error(self, tmp_path: Path) -> None:
        # Nothing is listening on this port after the server closes.
        with local_http_server(MODEL_BYTES) as url:
            dead_url = url
        with pytest.raises(ModelRegistryError, match="Failed to download"):
            HttpModelDownloader(timeout=1.0).download(dead_url, tmp_path / "out.bin")

    def test_end_to_end_ensure_with_real_downloader(self, tmp_path: Path) -> None:
        spec = make_spec()
        with local_http_server(MODEL_BYTES) as url:
            spec = ModelSpec(
                model_id=spec.model_id,
                format=spec.format,
                url=url,
                sha256=MODEL_SHA256,
                filename=spec.filename,
            )
            registry = ModelRegistry(tmp_path / "models", HttpModelDownloader())
            path = registry.ensure(spec)
        assert path.read_bytes() == MODEL_BYTES
        assert registry.is_cached(spec)

    def test_checksum_mismatch_from_real_downloader_rejected(self, tmp_path: Path) -> None:
        with local_http_server(b"tampered payload") as url:
            spec = ModelSpec(
                model_id="yolo_nano",
                format=ModelFormat.ONNX,
                url=url,
                sha256=MODEL_SHA256,  # does not match the served bytes
                filename="yolo_nano.onnx",
            )
            registry = ModelRegistry(tmp_path, HttpModelDownloader())
            with pytest.raises(ModelRegistryError, match="Checksum mismatch"):
                registry.ensure(spec)
        assert list(tmp_path.iterdir()) == []  # nothing partial left behind
