"""
Vectorized Monte Carlo engine for arithmetic Asian options under the Heston model.

Key features:
* Euler-Maruyama with full truncation for the variance process.
* Antithetic variates for variance reduction.
* Control variate using a geometric Asian approximation.
"""

from dataclasses import dataclass
from math import erf, sqrt
from typing import Optional

import numpy as np

from ..models.heston import HestonParams, clamp_correlation


@dataclass
class SimulationResult:
    time_grid: np.ndarray
    asset_paths: np.ndarray
    variance_paths: np.ndarray


class VectorizedHestonEngine:
    def __init__(self, params: HestonParams, risk_free_rate: float, steps_per_year: int = 252):
        self.params = params
        self.r = risk_free_rate
        self.steps_per_year = max(1, int(steps_per_year))
        self.params = params.bumped(rho=clamp_correlation(params.rho))

    def simulate(
        self,
        S0: float,
        T: float,
        n_paths: int,
        antithetic: bool = True,
        seed: Optional[int] = None,
    ) -> SimulationResult:
        """
        Simulate Heston price and variance paths.
        """
        n_paths = int(n_paths)
        if n_paths <= 0:
            raise ValueError("n_paths must be positive")

        n_steps = max(1, int(np.ceil(T * self.steps_per_year)))
        dt = T / float(n_steps)
        time_grid = np.linspace(0.0, T, n_steps + 1)

        rng = np.random.default_rng(seed)

        if antithetic and n_paths % 2 != 0:
            n_paths += 1

        half = n_paths // 2 if antithetic else n_paths
        Z1 = rng.standard_normal((half, n_steps))
        Z2 = rng.standard_normal((half, n_steps))
        if antithetic:
            Z1 = np.concatenate([Z1, -Z1], axis=0)
            Z2 = np.concatenate([Z2, -Z2], axis=0)

        rho = self.params.rho
        Z_v = Z1
        Z_s = rho * Z1 + np.sqrt(1.0 - rho ** 2) * Z2

        v = np.zeros((n_paths, n_steps + 1))
        S = np.zeros((n_paths, n_steps + 1))
        v[:, 0] = self.params.v0
        S[:, 0] = float(S0)

        kappa = self.params.kappa
        theta = self.params.theta
        sigma = self.params.sigma

        sqrt_dt = np.sqrt(dt)

        for t in range(n_steps):
            v_pos = np.maximum(v[:, t], 0.0)
            sqrt_v = np.sqrt(v_pos)

            dv = kappa * (theta - v_pos) * dt + sigma * sqrt_v * sqrt_dt * Z_v[:, t]
            v[:, t + 1] = v[:, t] + dv

            dlnS = -0.5 * v_pos * dt + sqrt_v * sqrt_dt * Z_s[:, t]
            S[:, t + 1] = S[:, t] * np.exp(dlnS)

        return SimulationResult(time_grid=time_grid, asset_paths=S, variance_paths=v)

    def _geometric_asian_bs_price(self, S0: float, K: float, T: float) -> float:
        """
        Approximate geometric Asian price using a Black-Scholes style formula.
        Uses long-run variance as the volatility proxy.
        """
        vol_approx = np.sqrt(max(self.params.theta, 0.0))
        sig_geo = vol_approx / np.sqrt(3.0)
        if sig_geo <= 0:
            return max(S0 - K, 0.0) * np.exp(-self.r * T)

        b_geo = -0.5 * (vol_approx ** 2) / 6.0
        d1 = (np.log(S0 / K) + (b_geo + 0.5 * sig_geo ** 2) * T) / (sig_geo * np.sqrt(T))
        d2 = d1 - sig_geo * np.sqrt(T)

        Nd1 = 0.5 * (1.0 + erf(d1 / sqrt(2.0)))
        Nd2 = 0.5 * (1.0 + erf(d2 / sqrt(2.0)))
        return np.exp(-self.r * T) * (S0 * np.exp(b_geo * T) * Nd1 - K * Nd2)

    def price_asian(
        self,
        option,
        S0: float,
        n_paths: int,
        antithetic: bool = True,
        control_variate: bool = True,
        seed: Optional[int] = None,
        diag_samples: int = 0,
    ) -> dict:
        """
        Price an arithmetic Asian option via Monte Carlo with an optional control variate.

        Args:
            option: AsianOption instance or any object exposing strike, maturity, and is_call.
            S0: Initial futures price.
            n_paths: Number of Monte Carlo paths.
            antithetic: Use antithetic variates if True.
            control_variate: Apply geometric Asian control variate if True.
            seed: Optional RNG seed for reproducibility (used for CRNs in Greeks).
        """
        result = self.simulate(S0, option.maturity, n_paths, antithetic=antithetic, seed=seed)
        S = result.asset_paths
        T = option.maturity
        discount = np.exp(-self.r * T)

        # Arithmetic average over monitoring dates (drop t0)
        arith_avg = np.mean(S[:, 1:], axis=1)
        if option.is_call:
            payoff_arith = np.maximum(arith_avg - option.strike, 0.0)
        else:
            payoff_arith = np.maximum(option.strike - arith_avg, 0.0)

        crude_price = discount * np.mean(payoff_arith)
        crude_se = discount * np.std(payoff_arith, ddof=1) / np.sqrt(S.shape[0])

        diagnostics = None
        if diag_samples > 0:
            take = min(diag_samples, S.shape[0])
            diagnostics = {
                "time_grid": result.time_grid.tolist(),
                "asset_paths": S[:take].tolist(),
                "variance_paths": result.variance_paths[:take].tolist(),
                "arith_avg": arith_avg[:take].tolist(),
                "payoff_arith": payoff_arith[:take].tolist(),
            }

        if not control_variate:
            return {
                "price": crude_price,
                "std_error": crude_se,
                "crude_price": crude_price,
                "crude_std_error": crude_se,
                "control_variate_price": crude_price,
                "control_variate_std_error": crude_se,
                "beta": 0.0,
                "variance_reduction": 1.0,
                "n_paths": S.shape[0],
                "n_steps": len(result.time_grid) - 1,
                "diagnostics": diagnostics,
            }

        # Control variate using geometric average
        geo_avg = np.exp(np.mean(np.log(S[:, 1:]), axis=1))
        if option.is_call:
            payoff_geo = np.maximum(geo_avg - option.strike, 0.0)
        else:
            payoff_geo = np.maximum(option.strike - geo_avg, 0.0)

        exact_geo = self._geometric_asian_bs_price(S0, option.strike, T)

        cov_matrix = np.cov(payoff_arith, payoff_geo, ddof=1)
        covariance = cov_matrix[0, 1]
        variance_geo = cov_matrix[1, 1]
        beta = covariance / variance_geo if variance_geo > 0 else 0.0

        adjusted_payoff = payoff_arith - beta * (payoff_geo - exact_geo / discount)
        cv_price = discount * np.mean(adjusted_payoff)
        cv_se = discount * np.std(adjusted_payoff, ddof=1) / np.sqrt(S.shape[0])

        vr_factor = (crude_se / cv_se) ** 2 if cv_se > 0 else np.inf

        return {
            "price": cv_price,
            "std_error": cv_se,
            "crude_price": crude_price,
            "crude_std_error": crude_se,
            "control_variate_price": cv_price,
            "control_variate_std_error": cv_se,
            "beta": beta,
            "variance_reduction": vr_factor,
            "n_paths": S.shape[0],
            "n_steps": len(result.time_grid) - 1,
            "diagnostics": diagnostics,
        }
