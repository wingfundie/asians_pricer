"""
Lightweight helpers for flat discount and forward curves.
"""

import math
from typing import Callable


def flat_discount(rate: float) -> Callable[[float], float]:
    """
    Build a discount factor curve assuming a constant continuously compounded rate.

    Args:
        rate: Annualized risk-free rate expressed in continuous compounding.

    Returns:
        Callable that maps a year-fraction maturity to a discount factor
        ``exp(-rate * t)``, convenient for pricing examples and tests.
    """
    def _disc(t: float) -> float:
        """
        Compute the discount factor for maturity ``t`` (in years).

        Returns:
            Discount factor ``exp(-r * t)``.
        """
        return math.exp(-rate * float(t))

    return _disc


def flat_forward(forward_price: float) -> Callable[[float], float]:
    """
    Create a constant forward price curve.

    Args:
        forward_price: Level to return for any maturity.

    Returns:
        Callable that ignores the input maturity and yields the supplied forward
        level, useful when wiring the pricer to deterministic forwards.
    """
    def _f(_: float) -> float:
        """
        Return the constant forward price irrespective of time.

        Returns:
            The constant forward level passed to ``flat_forward``.
        """
        return float(forward_price)

    return _f
