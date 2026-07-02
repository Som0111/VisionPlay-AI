"""Hand-tracking demo manifest.

``required_backends=("mediapipe.hands",)`` declares the dependency for
when Phase 2 makes it real (``docs/architecture.md`` §5) — capability
negotiation against actual backend availability is explicitly Phase 2
(``docs/roadmap.md``), so this app renders as launchable regardless of
whether that backend exists yet.
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
