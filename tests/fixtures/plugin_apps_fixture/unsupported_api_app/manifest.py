"""Fixture manifest: targets an api_version the registry does not support."""

from visionplay.core.plugin_base import CURRENT_API_VERSION, AppManifest

MANIFEST = AppManifest(
    id="unsupported_api_app",
    name="Unsupported API App",
    category="ai_demos",
    version="0.1.0",
    api_version=CURRENT_API_VERSION + 999,
)
