"""
Volatility surface placeholder utilities.

For production, hook this into live implied vol surfaces; here we expose a
flat-vol callable for demonstration and testing.
"""

from typing import Callable


def flat_vol(vol: float) -> Callable[[float, float], float]:
    """
    Produce a trivial implied volatility surface that is flat in both expiry and strike.

    Args:
        vol: Constant volatility level to return.

    Returns:
        Callable accepting ``(T, K)`` and returning the supplied volatility, handy
        for deterministic scenarios or when wiring in a placeholder surface.
    """
    def _vol(_: float, __: float) -> float:
        """
        Return the constant volatility irrespective of expiry or strike.

        Returns:
            The flat volatility level provided to ``flat_vol``.
        """
        return float(vol)

    return _vol
