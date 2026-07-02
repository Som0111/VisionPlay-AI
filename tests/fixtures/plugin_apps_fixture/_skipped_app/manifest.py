"""Fixture manifest for an underscore-prefixed app package.

Otherwise fully valid — this app must never be discovered because of its
folder name (``_skipped_app``), the same rule that skips ``_template`` in
the real ``apps/`` tree.
"""

from visionplay.core.plugin_base import CURRENT_API_VERSION, AppManifest

MANIFEST = AppManifest(
    id="skipped_app",
    name="Skipped App",
    category="ai_demos",
    version="0.1.0",
    api_version=CURRENT_API_VERSION,
)
