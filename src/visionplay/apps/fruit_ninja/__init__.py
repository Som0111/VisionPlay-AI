"""Fruit Ninja — gesture-controlled slicing game (Phase 3, M3.4).

Swipe a fast fingertip motion across the frame to slice falling fruit;
avoid the bombs. All game simulation (spawning, physics, slice collision,
scoring, lives, state machine) lives in ``processor.py`` (headless-testable,
built on :mod:`visionplay.vision.gestures`); Qt-specific rendering lives in
``widget.py``.
"""
