from dataclasses import dataclass, replace
from typing import Dict


@dataclass
class HestonParams:
    """
    Container for Heston model parameters.

    Attributes:
        v0: Initial variance.
        kappa: Mean reversion speed of the variance process.
        theta: Long-run variance level.
        sigma: Volatility of variance (vol of vol).
        rho: Correlation between price and variance Brownian motions.
    """

    v0: float
    kappa: float
    theta: float
    sigma: float
    rho: float

    def as_dict(self) -> Dict[str, float]:
        """Return the parameters as a plain dictionary for serialization."""
        return {
            "v0": self.v0,
            "kappa": self.kappa,
            "theta": self.theta,
            "sigma": self.sigma,
            "rho": self.rho,
        }

    def bumped(self, **kwargs) -> "HestonParams":
        """
        Return a shallow copy with selected fields bumped.

        Useful for finite-difference Greeks where only a subset of parameters
        change between valuations.

        Returns:
            New ``HestonParams`` instance with overridden fields.
        """
        return replace(self, **kwargs)


def validate_feller_condition(params: HestonParams) -> bool:
    """
    Check the Feller positivity condition ``2*kappa*theta > sigma^2``.

    Returns:
        True when the condition holds, indicating the variance process stays
        strictly positive; False otherwise.
    """
    return 2.0 * params.kappa * params.theta > params.sigma ** 2


def clamp_correlation(rho: float, eps: float = 1e-6) -> float:
    """
    Constrain the correlation to the open interval (-1, 1) to avoid degeneracy.

    Args:
        rho: Input correlation value.
        eps: Small buffer to keep the output away from the boundaries.

    Returns:
        Clamped correlation in ``(-1 + eps, 1 - eps)``.
    """
    lo = -1.0 + eps
    hi = 1.0 - eps
    return max(lo, min(hi, rho))
