"""
Lightweight logging configuration for the pricer package.
"""

import logging
import sys
from typing import Optional


def configure_logging(level: int = logging.INFO, stream: Optional[object] = None) -> None:
    """
    Initialize the root logger with a compact formatter suitable for notebooks,
    scripts, and CLI runs.

    Args:
        level: Logging level to set globally (e.g., logging.INFO).
        stream: Optional stream to write to; defaults to stdout so callers can
            redirect logs to in-memory buffers or files when embedding.

    Returns:
        None. The global root logger is configured in place.
    """
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

    root = logging.getLogger()
    root.setLevel(level)
    # Remove default handlers to avoid duplicate lines when reconfiguring
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
