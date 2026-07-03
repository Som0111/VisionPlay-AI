"""Fixtures shared by the unit and integration suites."""

from pathlib import Path

import pytest

#: Directory holding committed test assets (tiny ONNX models, fixture apps, ...).
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def hand_landmarker_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Real MediaPipe hand-landmarker model, downloaded once per session.

    Skips (rather than fails) when the model cannot be fetched — e.g. a CI
    runner with no network — so offline environments still run the rest of
    the suite. Downloaded locally, this exercises the real backend end to end.
    """
    from visionplay.vision.inference.model_catalog import HAND_LANDMARKER
    from visionplay.vision.inference.model_registry import (
        HttpModelDownloader,
        ModelRegistry,
        ModelRegistryError,
    )

    cache = tmp_path_factory.mktemp("mp_models")
    registry = ModelRegistry(cache, HttpModelDownloader())
    registry.register(HAND_LANDMARKER)
    try:
        return registry.ensure(HAND_LANDMARKER)
    except ModelRegistryError as exc:
        pytest.skip(f"hand_landmarker model unavailable (no network?): {exc}")
