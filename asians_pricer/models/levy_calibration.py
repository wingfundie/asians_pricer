"""
Levy calibrator (Variance Gamma / NIG) using Monte Carlo vanilla pricing.

This is intentionally light-weight and uses scipy.optimize to fit model
parameters to observed vanilla option prices. It prices vanillas with a
one-step terminal simulation under the chosen process.
"""

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

try:
    from scipy.optimize import minimize  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    minimize = None

from .levy import NIGParams, VarianceGammaParams


def _price_vanilla_vg(spot: float, rate: float, T: float, strike: float, params: VarianceGammaParams, n_paths: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    Y = rng.gamma(shape=T / params.nu, scale=params.nu, size=n_paths)  # Gamma subordinator
    Z = rng.standard_normal(n_paths)
    log_S = np.log(spot) + params.theta * Y + params.sigma * np.sqrt(Y) * Z
    ST = np.exp(log_S)
    payoff = np.maximum(ST - strike, 0.0)
    return float(np.exp(-rate * T) * np.mean(payoff))


def _price_vanilla_nig(spot: float, rate: float, T: float, strike: float, params: NIGParams, n_paths: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    gamma_val = np.sqrt(max(params.alpha ** 2 - params.beta ** 2, 1e-12))
    mean_ig = params.delta * T / gamma_val
    scale_ig = (params.delta ** 2) * T
    Y = rng.wald(mean=mean_ig, scale=scale_ig, size=n_paths)
    Z = rng.standard_normal(n_paths)
    log_S = np.log(spot) + params.mu * T + params.beta * Y + np.sqrt(Y) * Z
    ST = np.exp(log_S)
    payoff = np.maximum(ST - strike, 0.0)
    return float(np.exp(-rate * T) * np.mean(payoff))


@dataclass
class LevyCalibrator:
    spot: float
    rate: float
    n_paths: int = 20000
    seed: int = 123

    def calibrate_vg(self, market_quotes: Iterable[Tuple[float, float, float]], initial: VarianceGammaParams = VarianceGammaParams(theta=0.0, sigma=0.2, nu=0.2)) -> VarianceGammaParams:
        """
        Calibrate VG parameters to vanilla prices.

        market_quotes: iterable of (strike, maturity_years, market_price)
        """
        if minimize is None:
            raise ImportError("scipy is required for Levy calibration")

        quotes = list(market_quotes)

        def objective(x):
            theta, sigma, nu = x
            if sigma <= 0 or nu <= 0:
                return 1e9
            params = VarianceGammaParams(theta=theta, sigma=sigma, nu=nu)
            errs: List[float] = []
            for K, T, mkt in quotes:
                mdl = _price_vanilla_vg(self.spot, self.rate, T, K, params, self.n_paths, self.seed)
                errs.append(mdl - mkt)
            return np.sum(np.square(errs))

        res = minimize(objective, x0=np.array([initial.theta, initial.sigma, initial.nu]), method="Nelder-Mead")
        theta, sigma, nu = res.x
        return VarianceGammaParams(theta=float(theta), sigma=float(abs(sigma)), nu=float(abs(nu)))

    def calibrate_nig(self, market_quotes: Iterable[Tuple[float, float, float]], initial: NIGParams = NIGParams(alpha=5.0, beta=-2.0, delta=0.5, mu=0.0)) -> NIGParams:
        """
        Calibrate NIG parameters to vanilla prices.

        market_quotes: iterable of (strike, maturity_years, market_price)
        """
        if minimize is None:
            raise ImportError("scipy is required for Levy calibration")

        quotes = list(market_quotes)

        def objective(x):
            alpha, beta, delta, mu = x
            if alpha <= abs(beta) or delta <= 0:
                return 1e9
            params = NIGParams(alpha=alpha, beta=beta, delta=delta, mu=mu)
            errs: List[float] = []
            for K, T, mkt in quotes:
                mdl = _price_vanilla_nig(self.spot, self.rate, T, K, params, self.n_paths, self.seed)
                errs.append(mdl - mkt)
            return np.sum(np.square(errs))

        res = minimize(objective, x0=np.array([initial.alpha, initial.beta, initial.delta, initial.mu]), method="Nelder-Mead")
        alpha, beta, delta, mu = res.x
        # enforce alpha > |beta|
        alpha = max(alpha, abs(beta) + 1e-6)
        return NIGParams(alpha=float(alpha), beta=float(beta), delta=float(abs(delta)), mu=float(mu))
