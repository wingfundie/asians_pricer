from dataclasses import dataclass
from typing import Optional


@dataclass
class AsianOption:
    """
    Simple arithmetic Asian option definition.

    Attributes:
        strike: Strike price.
        maturity: Time to maturity in years.
        is_call: True for call, False for put.
        notional: Optional scaling factor for payoffs.
        averaging_observations: Optional explicit number of monitoring points.
    """

    strike: float
    maturity: float
    is_call: bool = True
    notional: float = 1.0
    averaging_observations: Optional[int] = None

    def __post_init__(self) -> None:
        """
        Validate basic input ranges after dataclass initialization.

        Raises:
            ValueError: If strike or maturity are not strictly positive.
        """
        if self.maturity <= 0:
            raise ValueError("maturity must be positive (year fraction)")
        if self.strike <= 0:
            raise ValueError("strike must be positive")

    def payoff(self, average: float) -> float:
        """
        Compute the intrinsic payoff given a realized arithmetic average.

        Args:
            average: Observed average price of the underlying over the monitoring grid.

        Returns:
            Discount-free payoff scaled by ``notional`` for either a call or put,
            clipped at zero for out-of-the-money scenarios.
        """
        intrinsic = average - self.strike if self.is_call else self.strike - average
        return max(intrinsic, 0.0) * self.notional
