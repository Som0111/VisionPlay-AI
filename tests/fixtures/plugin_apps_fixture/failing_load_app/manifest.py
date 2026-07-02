"""Fixture manifest: a valid app whose plugin raises in on_load()."""

from visionplay.core.plugin_base import CURRENT_API_VERSION, AppManifest

MANIFEST = AppManifest(
    id="failing_load_app",
    name="Failing Load App",
    category="ai_demos",
    version="0.1.0",
    api_version=CURRENT_API_VERSION,
)
