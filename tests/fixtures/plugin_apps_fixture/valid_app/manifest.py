"""Fixture manifest: a normal, fully valid app."""

from visionplay.core.plugin_base import CURRENT_API_VERSION, AppManifest

MANIFEST = AppManifest(
    id="valid_app",
    name="Valid App",
    category="ai_demos",
    version="0.1.0",
    api_version=CURRENT_API_VERSION,
)
