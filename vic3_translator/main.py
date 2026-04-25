"""Entry point for the Victoria 3 translator GUI.

Run as::

    python -m vic3_translator.main
"""

from __future__ import annotations

import logging
import sys


def main() -> int:
    try:
        from .gui import launch
    except Exception as e:  # noqa: BLE001
        logging.basicConfig(level=logging.ERROR)
        logging.exception("Failed to initialise GUI: %s", e)
        print(f"起動に失敗しました: {e}", file=sys.stderr)
        return 1

    launch()
    return 0


if __name__ == "__main__":
    sys.exit(main())
