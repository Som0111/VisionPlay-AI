"""Entry point for ``python -m visionplay``.

Phase 0 / M0.1: prints a banner and exits. The Qt application bootstrap
(``app.py``) replaces the body of ``main`` in M0.6.
"""

import sys

from visionplay import __version__


def main() -> int:
    # ASCII only: the Windows console default code page mangles non-ASCII output.
    print(f"VisionPlay AI v{__version__} - Phase 0 scaffold (no UI yet)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
