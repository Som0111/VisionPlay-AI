"""Fixture manifest: requires a backend no manager will ever satisfy."""

from visionplay.core.plugin_base import CURRENT_API_VERSION, AppManifest

MANIFEST = AppManifest(
    id="unavailable_app",
    name="Unavailable App",
    category="ai_demos",
    version="0.1.0",
    api_version=CURRENT_API_VERSION,
    required_backends=("missing.backend",),
)
