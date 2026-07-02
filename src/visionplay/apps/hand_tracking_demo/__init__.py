"""Hand-tracking demo — Phase 1's concrete pipeline-wiring proof (M1.7).

The first real, registrable app (not underscore-prefixed, so
``PluginRegistry.discover`` picks it up). It declares
``required_backends=("mediapipe.hands",)`` but runs with no real backend
behind that key yet — Phase 2 wires an actual MediaPipe hands backend.
Until then this app exercises registry discovery, the full ``AppPlugin``
lifecycle, launcher listing, and the frame pipeline's ``on_frame`` seam
(M1.1-M1.6) end to end, exactly like ``apps/_template`` but as a real,
launchable app rather than a scaffold.
"""
