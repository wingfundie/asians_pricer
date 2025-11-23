"""
Volatility surface placeholder utilities.

For production, hook this into live implied vol surfaces; here we expose a
flat-vol callable for demonstration and testing.
"""

from typing import Callable


def flat_vol(vol: float) -> Callable[[float, float], float]:
    """
    Return a function vol(T, K) = constant.
    """
    def _vol(_: float, __: float) -> float:
        return float(vol)

    return _vol
