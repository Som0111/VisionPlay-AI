"""Entry point for ``python -m visionplay``.

M0.6: launches the Qt application (window + live camera feed). The actual
bootstrap lives in :mod:`visionplay.app`; this module only re-exports
``main`` for the console-script entry in ``pyproject.toml``.
"""

import sys

from visionplay.app import main

__all__ = ["main"]

if __name__ == "__main__":
    sys.exit(main())
