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
        if self.maturity <= 0:
            raise ValueError("maturity must be positive (year fraction)")
        if self.strike <= 0:
            raise ValueError("strike must be positive")

    def payoff(self, average: float) -> float:
        """
        Payoff for a given realized average.
        """
        intrinsic = average - self.strike if self.is_call else self.strike - average
        return max(intrinsic, 0.0) * self.notional
