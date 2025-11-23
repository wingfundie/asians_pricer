from dataclasses import dataclass


@dataclass
class VarianceGammaParams:
    """
    Variance Gamma parameters.

    theta: Drift of the Brownian motion component.
    sigma: Diffusion coefficient of the Brownian component.
    nu: Variance of the Gamma subordinator.
    """

    theta: float
    sigma: float
    nu: float


@dataclass
class NIGParams:
    """
    Normal Inverse Gaussian parameters in the (alpha, beta, delta, mu) form.

    alpha: Tail heaviness parameter (alpha > |beta|).
    beta: Skew parameter.
    delta: Scale parameter.
    mu: Location parameter.
    """

    alpha: float
    beta: float
    delta: float
    mu: float = 0.0
