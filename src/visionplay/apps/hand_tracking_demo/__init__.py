"""Hand-tracking demo — Phase 1's concrete pipeline-wiring proof (M1.7).

The first real, registrable app (not underscore-prefixed, so
``PluginRegistry.discover`` picks it up). It declares
``required_backends=("mediapipe.hands",)``, so since Phase 2 the pipeline
runs real MediaPipe hand-landmark inference and delivers a
``HandLandmarkResult`` under that key in ``frame.results`` — with zero code
changes to this app relative to Phase 1, proving the plugin contract held.
It exercises registry discovery, the full ``AppPlugin`` lifecycle, launcher
capability negotiation, and the capture → backend → ``on_frame`` → publish
path end to end.
"""
