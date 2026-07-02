"""Core/platform layer: paths, config, logging, event bus, plugin registry.

This layer must stay importable headless — no Qt, no OpenCV, and no imports
from ``vision/``, ``apps/``, or ``ui/`` (see ``docs/architecture.md`` §1).
"""
