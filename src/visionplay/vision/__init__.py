"""Vision/AI layer: camera abstraction, frame pipeline, inference backends.

Depends only on ``core/`` and third-party CV libraries — never on ``apps/``
or ``ui/``, and never imports Qt (see ``docs/architecture.md`` §1). Must be
importable and testable headless.
"""
