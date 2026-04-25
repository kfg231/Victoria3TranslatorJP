"""Logging helpers shared by the CLI and GUI."""

from __future__ import annotations

import logging
import queue
import sys
from logging.handlers import QueueHandler
from pathlib import Path
from typing import Optional


_DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DEFAULT_DATEFMT = "%H:%M:%S"


def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    gui_queue: Optional[queue.Queue] = None,
) -> None:
    """Set up root logging with optional file and GUI queue handlers.

    Calling this multiple times replaces previous handlers so the GUI
    can re-attach when the window is recreated.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Remove any previously-attached handlers so we don't duplicate lines.
    for h in list(root.handlers):
        root.removeHandler(h)

    formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)

    # Console
    stream = logging.StreamHandler(stream=sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    # Optional rotating file
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        root.addHandler(fh)

    # Optional GUI queue handler (message is the already-formatted string)
    if gui_queue is not None:
        qh = QueueHandler(gui_queue)
        qh.setFormatter(formatter)
        root.addHandler(qh)
