"""Fixtures shared by the unit and integration suites."""

from pathlib import Path

import pytest

#: Directory holding committed test assets (tiny ONNX models, fixture apps, ...).
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _ensure_real_model(spec_name: str, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Download one catalog model into a session cache, skipping when offline.

    Skips (rather than fails) when the model cannot be fetched — e.g. a CI
    runner with no network — so offline environments still run the rest of
    the suite. Downloaded locally, this exercises the real backend end to end.
    """
    from visionplay.vision.inference import model_catalog
    from visionplay.vision.inference.model_registry import (
        HttpModelDownloader,
        ModelRegistry,
        ModelRegistryError,
        ModelSpec,
    )

    spec: ModelSpec = getattr(model_catalog, spec_name)
    cache = tmp_path_factory.mktemp("mp_models")
    registry = ModelRegistry(cache, HttpModelDownloader())
    registry.register(spec)
    try:
        return registry.ensure(spec)
    except ModelRegistryError as exc:
        pytest.skip(f"{spec.model_id} model unavailable (no network?): {exc}")


@pytest.fixture(scope="session")
def hand_landmarker_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Real MediaPipe hand-landmarker model, downloaded once per session."""
    return _ensure_real_model("HAND_LANDMARKER", tmp_path_factory)


@pytest.fixture(scope="session")
def pose_landmarker_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Real MediaPipe pose-landmarker model, downloaded once per session."""
    return _ensure_real_model("POSE_LANDMARKER", tmp_path_factory)


@pytest.fixture(scope="session")
def face_landmarker_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Real MediaPipe face-landmarker model, downloaded once per session."""
    return _ensure_real_model("FACE_LANDMARKER", tmp_path_factory)
