"""Air Canvas — gesture-controlled whiteboard (Phase 3 flagship, M3.2).

Draw in the air with your index finger: pinch to put the pen down, release
to lift it. A gesture toolbar across the top of the frame switches color,
brush size, and eraser, and clears the canvas — selected by pinching (or
dwelling) on a region. All drawing/gesture logic lives in ``processor.py``
(headless-testable, built on :mod:`visionplay.vision.gestures`); Qt-specific
rendering lives in ``widget.py``.
"""
