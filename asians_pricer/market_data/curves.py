"""
Lightweight helpers for flat discount and forward curves.
"""

import math
from typing import Callable


def flat_discount(rate: float) -> Callable[[float], float]:
    """
    Return a simple flat discount curve D(t) = exp(-r * t).
    """
    def _disc(t: float) -> float:
        return math.exp(-rate * float(t))

    return _disc


def flat_forward(forward_price: float) -> Callable[[float], float]:
    """
    Simple constant forward curve F(t) = F0.
    """
    def _f(_: float) -> float:
        return float(forward_price)

    return _f
