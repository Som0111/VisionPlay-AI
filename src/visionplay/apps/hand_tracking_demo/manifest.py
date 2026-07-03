"""Hand-tracking demo manifest.

``required_backends=("mediapipe.hands",)`` is what makes the pipeline run
real MediaPipe hand-landmark inference for this app (``docs/architecture.md``
§4/§5): the launcher checks the name against actual backend availability and
greys the app out if it can't be satisfied.
"""

from visionplay.core.plugin_base import CURRENT_API_VERSION, AppManifest

MANIFEST = AppManifest(
    id="hand_tracking_demo",
    name="Hand Tracking Demo",
    category="ai_demos",
    version="0.1.0",
    api_version=CURRENT_API_VERSION,
    required_backends=("mediapipe.hands",),
    icon="assets/icon.png",
)
