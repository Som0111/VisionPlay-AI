"""Fixture manifest: a valid app whose plugin raises in on_frame()."""

from visionplay.core.plugin_base import CURRENT_API_VERSION, AppManifest

MANIFEST = AppManifest(
    id="failing_frame_app",
    name="Failing Frame App",
    category="ai_demos",
    version="0.1.0",
    api_version=CURRENT_API_VERSION,
)
