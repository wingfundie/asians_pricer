"""
Lightweight logging configuration for the pricer package.
"""

import logging
import sys
from typing import Optional


def configure_logging(level: int = logging.INFO, stream: Optional[object] = None) -> None:
    """
    Configure root logger with a simple formatter.
    """
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

    root = logging.getLogger()
    root.setLevel(level)
    # Remove default handlers to avoid duplicate lines when reconfiguring
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
