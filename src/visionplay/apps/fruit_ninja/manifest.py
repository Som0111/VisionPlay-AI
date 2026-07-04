"""Fruit Ninja manifest.

``required_backends=("mediapipe.hands",)`` makes the pipeline run hand
landmark inference for this app; the launcher greys the app out when the
backend can't be satisfied (``docs/architecture.md`` §4/§5).
"""

from visionplay.core.plugin_base import CURRENT_API_VERSION, AppManifest

MANIFEST = AppManifest(
    id="fruit_ninja",
    name="Fruit Ninja",
    category="gesture_games",
    version="0.1.0",
    api_version=CURRENT_API_VERSION,
    required_backends=("mediapipe.hands",),
    icon="assets/icon.png",
)
