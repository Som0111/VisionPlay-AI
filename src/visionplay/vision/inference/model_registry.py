"""Format-tagged model catalog with checksum-verified download-on-first-use.

Large model files are never committed to the repo (CLAUDE.md conventions).
Instead, each model is described by an immutable :class:`ModelSpec` — id,
format, source URL, SHA-256 — and :class:`ModelRegistry` materializes it
into the local cache (``AppPaths.models_dir``) on first use, verifying the
checksum before anything downstream ever sees the file.

Models are tagged by **format** (:class:`ModelFormat`: ``onnx``/``tflite``),
never by device — device selection is a runtime backend concern
(``docs/architecture.md`` §5), so this registry needs no changes when GPU
support lands.

The actual network transfer sits behind the :class:`ModelDownloader`
abstraction. Phase 0 ships no real downloader — tests inject fakes, and the
HTTP implementation arrives with the first real model in Phase 2.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

__all__ = [
    "ModelDownloader",
    "ModelFormat",
    "ModelRegistry",
    "ModelRegistryError",
    "ModelSpec",
]

#: Bytes per read when hashing a model file (models are large; never slurp).
_HASH_CHUNK_SIZE: int = 1 << 20


class ModelRegistryError(Exception):
    """A model could not be registered, resolved, or materialized.

    Messages must be user-presentable: say which model failed and why
    (unknown id, checksum mismatch, no downloader, ...).
    """


class ModelFormat(Enum):
    """On-disk format of a model artifact.

    The registry tags models by format only — which *device* runs a model
    is decided by the backend at runtime, never recorded here.
    """

    ONNX = "onnx"
    TFLITE = "tflite"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Immutable description of one downloadable model artifact.

    Attributes:
        model_id: Unique registry key (e.g. ``"yolo_nano"``).
        format: Artifact format — see :class:`ModelFormat`.
        url: Source URL the artifact is fetched from on first use.
        sha256: Expected SHA-256 of the artifact, 64 hex characters. The
            cache never serves a file that does not match this digest.
        filename: Bare file name inside the model cache directory. Must not
            contain path separators — specs cannot escape the cache dir.
    """

    model_id: str
    format: ModelFormat
    url: str
    sha256: str
    filename: str

    def __post_init__(self) -> None:
        """Reject specs that could never verify or could escape the cache.

        Raises:
            ValueError: On an empty id, a malformed digest, or a filename
                containing path components.
        """
        if not self.model_id:
            raise ValueError("ModelSpec.model_id must be non-empty")
        digest = self.sha256.lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(
                f"ModelSpec.sha256 for {self.model_id!r} must be 64 hex characters, "
                f"got {self.sha256!r}"
            )
        if not self.filename or Path(self.filename).name != self.filename:
            raise ValueError(
                f"ModelSpec.filename for {self.model_id!r} must be a bare file name, "
                f"got {self.filename!r}"
            )


class ModelDownloader(ABC):
    """Abstract transfer of one artifact from a URL to a local path.

    Deliberately minimal: the registry handles caching, verification, and
    atomic placement; implementations only move bytes. Phase 0 has no real
    (network) implementation — tests substitute in-memory fakes.
    """

    @abstractmethod
    def download(self, url: str, destination: Path) -> None:
        """Fetch ``url`` and write its bytes to ``destination``.

        Args:
            url: Source location of the artifact.
            destination: File to (over)write. The parent directory exists.

        Raises:
            ModelRegistryError: If the transfer fails. Must not leave a
                partial file masquerading as complete — the registry
                verifies checksums, but failures should still be loud.
        """


class ModelRegistry:
    """Catalog of :class:`ModelSpec` entries plus the local artifact cache.

    Construct one per process with the app's models directory
    (``AppPaths.models_dir``) and pass it down — no module-level singleton,
    mirroring :class:`~visionplay.core.paths.AppPaths`.

    Resolution flow: ``register()`` specs at startup, then ``ensure()``
    returns a verified local path, downloading on first use. A cached file
    whose checksum no longer matches its spec is treated as corrupt and
    re-fetched, never served.
    """

    def __init__(self, models_dir: Path, downloader: ModelDownloader | None = None) -> None:
        """Create a registry over a cache directory.

        Args:
            models_dir: Directory holding cached artifacts. Created on
                first :meth:`ensure`, not here.
            downloader: Transfer implementation for cache misses. ``None``
                is valid for cache-only operation (Phase 0 default);
                :meth:`ensure` then fails loudly on a miss.
        """
        self._models_dir = models_dir
        self._downloader = downloader
        self._specs: dict[str, ModelSpec] = {}

    @property
    def models_dir(self) -> Path:
        """The cache directory this registry materializes artifacts into."""
        return self._models_dir

    def register(self, spec: ModelSpec) -> None:
        """Add a model to the catalog.

        Args:
            spec: The model description to register.

        Raises:
            ModelRegistryError: If a different spec is already registered
                under the same id. Re-registering an identical spec is a
                no-op — startup code may run more than once.
        """
        existing = self._specs.get(spec.model_id)
        if existing is not None and existing != spec:
            raise ModelRegistryError(
                f"Model id {spec.model_id!r} is already registered with a different spec"
            )
        self._specs[spec.model_id] = spec

    def get(self, model_id: str) -> ModelSpec:
        """Look up a registered spec by id.

        Raises:
            ModelRegistryError: If no model with that id is registered.
        """
        try:
            return self._specs[model_id]
        except KeyError:
            raise ModelRegistryError(f"Unknown model id {model_id!r}") from None

    def path_for(self, spec: ModelSpec) -> Path:
        """Return the cache path for a spec (whether or not it exists yet)."""
        return self._models_dir / spec.filename

    def is_cached(self, spec: ModelSpec) -> bool:
        """Return ``True`` if a checksum-valid copy is already cached.

        Hashes the full file, so this costs one read of the artifact — call
        it at load time, not per frame.
        """
        path = self.path_for(spec)
        return path.is_file() and _sha256_of(path) == spec.sha256.lower()

    def ensure(self, spec: ModelSpec) -> Path:
        """Return a verified local path for ``spec``, downloading if needed.

        A cached file with a matching checksum is returned as-is. A cached
        file that fails verification is discarded and re-fetched. Downloads
        land in a temporary sibling and are moved into place only after the
        checksum verifies, so the cache never contains a file that has not
        passed verification under its final name.

        Args:
            spec: The model to materialize.

        Returns:
            Path to the verified artifact inside :attr:`models_dir`.

        Raises:
            ModelRegistryError: If the model is not cached and no
                downloader is configured, or if the downloaded artifact's
                checksum does not match the spec.
        """
        path = self.path_for(spec)
        if path.is_file():
            if _sha256_of(path) == spec.sha256.lower():
                return path
            path.unlink()  # corrupt or stale: never serve, re-fetch below

        if self._downloader is None:
            raise ModelRegistryError(
                f"Model {spec.model_id!r} is not cached and no downloader is configured"
            )

        self._models_dir.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".part")
        try:
            self._downloader.download(spec.url, partial)
            actual = _sha256_of(partial)
            if actual != spec.sha256.lower():
                raise ModelRegistryError(
                    f"Checksum mismatch for model {spec.model_id!r}: "
                    f"expected sha256 {spec.sha256}, got {actual}"
                )
        except Exception:
            partial.unlink(missing_ok=True)
            raise
        partial.replace(path)
        return path


def _sha256_of(path: Path) -> str:
    """Return the lowercase hex SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()
