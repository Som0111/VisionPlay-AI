"""Fixture manifest: an app that requires a backend named ``fake.hands``."""

from visionplay.core.plugin_base import CURRENT_API_VERSION, AppManifest

MANIFEST = AppManifest(
    id="results_app",
    name="Results App",
    category="ai_demos",
    version="0.1.0",
    api_version=CURRENT_API_VERSION,
    required_backends=("fake.hands",),
)
