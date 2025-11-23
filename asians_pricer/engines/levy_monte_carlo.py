"""
Monte Carlo engine for Levy-driven asset dynamics (Variance Gamma and NIG).

The implementation follows the subordinated Brownian motion construction:
X_{t+dt} = theta * Y_dt + sigma * sqrt(Y_dt) * Z for VG,
and X_{t+dt} = mu*dt + beta * Y_dt + sqrt(Y_dt) * Z for NIG,
where Y_dt is drawn from the corresponding subordinator distribution.
"""

from dataclasses import dataclass
from math import erf, sqrt
from typing import Optional

import numpy as np

from ..models.levy import NIGParams, VarianceGammaParams


@dataclass
class LevySimulationResult:
    time_grid: np.ndarray
    asset_paths: np.ndarray


class LevyMonteCarloEngine:
    def __init__(self, risk_free_rate: float, steps_per_year: int = 252):
        self.r = risk_free_rate
        self.steps_per_year = max(1, int(steps_per_year))

    def _simulate_vg(
        self,
        params: VarianceGammaParams,
        S0: float,
        T: float,
        n_paths: int,
        antithetic: bool,
        seed: Optional[int],
    ) -> LevySimulationResult:
        n_steps = max(1, int(np.ceil(T * self.steps_per_year)))
        dt = T / float(n_steps)
        time_grid = np.linspace(0.0, T, n_steps + 1)

        rng = np.random.default_rng(seed)
        if antithetic and n_paths % 2 != 0:
            n_paths += 1
        half = n_paths // 2 if antithetic else n_paths

        gamma_increments = rng.gamma(shape=dt / params.nu, scale=params.nu, size=(half, n_steps))
        Z = rng.standard_normal((half, n_steps))
        if antithetic:
            gamma_increments = np.concatenate([gamma_increments, gamma_increments], axis=0)
            Z = np.concatenate([Z, -Z], axis=0)

        dX = params.theta * gamma_increments + params.sigma * np.sqrt(gamma_increments) * Z
        log_S = np.cumsum(dX, axis=1)
        log_S = np.concatenate([np.zeros((n_paths, 1)), log_S], axis=1)
        S = S0 * np.exp(log_S)
        return LevySimulationResult(time_grid=time_grid, asset_paths=S)

    def _simulate_nig(
        self,
        params: NIGParams,
        S0: float,
        T: float,
        n_paths: int,
        antithetic: bool,
        seed: Optional[int],
    ) -> LevySimulationResult:
        n_steps = max(1, int(np.ceil(T * self.steps_per_year)))
        dt = T / float(n_steps)
        time_grid = np.linspace(0.0, T, n_steps + 1)

        rng = np.random.default_rng(seed)
        if antithetic and n_paths % 2 != 0:
            n_paths += 1
        half = n_paths // 2 if antithetic else n_paths

        gamma_val = np.sqrt(max(params.alpha ** 2 - params.beta ** 2, 1e-12))
        mean_ig = params.delta * dt / gamma_val
        shape_ig = (params.delta ** 2) * dt

        Y = rng.wald(mean=mean_ig, scale=shape_ig, size=(half, n_steps))
        Z = rng.standard_normal((half, n_steps))
        if antithetic:
            Y = np.concatenate([Y, Y], axis=0)
            Z = np.concatenate([Z, -Z], axis=0)

        dX = params.mu * dt + params.beta * Y + np.sqrt(Y) * Z
        log_S = np.cumsum(dX, axis=1)
        log_S = np.concatenate([np.zeros((n_paths, 1)), log_S], axis=1)
        S = S0 * np.exp(log_S)
        return LevySimulationResult(time_grid=time_grid, asset_paths=S)

    def _estimate_effective_vol(self, paths: np.ndarray, T: float) -> float:
        """
        Rough volatility estimate from simulated log returns for control variates.
        """
        log_returns = np.diff(np.log(paths), axis=1)
        if log_returns.size == 0:
            return 0.0
        sigma_hat = np.std(log_returns)
        return float(sigma_hat * np.sqrt(self.steps_per_year))

    def price_asian(
        self,
        option,
        S0: float,
        n_paths: int,
        params,
        process: str = "vg",
        antithetic: bool = True,
        control_variate: bool = True,
        seed: Optional[int] = None,
        diag_samples: int = 0,
    ) -> dict:
        """
        Price an arithmetic Asian option under a Levy process (VG or NIG).
        """
        if process.lower() == "vg":
            sim = self._simulate_vg(params, S0, option.maturity, n_paths, antithetic, seed)
        elif process.lower() == "nig":
            sim = self._simulate_nig(params, S0, option.maturity, n_paths, antithetic, seed)
        else:
            raise ValueError("process must be 'vg' or 'nig'")

        S = sim.asset_paths
        discount = np.exp(-self.r * option.maturity)
        arith_avg = np.mean(S[:, 1:], axis=1)
        payoff_arith = (
            np.maximum(arith_avg - option.strike, 0.0)
            if option.is_call
            else np.maximum(option.strike - arith_avg, 0.0)
        )

        crude_price = discount * np.mean(payoff_arith)
        crude_se = discount * np.std(payoff_arith, ddof=1) / np.sqrt(S.shape[0])

        diagnostics = None
        if diag_samples > 0:
            take = min(diag_samples, S.shape[0])
            diagnostics = {
                "time_grid": sim.time_grid.tolist(),
                "asset_paths": S[:take].tolist(),
                "arith_avg": arith_avg[:take].tolist(),
                "payoff_arith": payoff_arith[:take].tolist(),
            }

        if not control_variate:
            return {
                "price": crude_price,
                "std_error": crude_se,
                "crude_price": crude_price,
                "crude_std_error": crude_se,
                "variance_reduction": 1.0,
                "beta": 0.0,
                "n_paths": S.shape[0],
                "n_steps": len(sim.time_grid) - 1,
                "diagnostics": diagnostics,
            }

        # Control variate using geometric average with a BS-style proxy
        geo_avg = np.exp(np.mean(np.log(S[:, 1:]), axis=1))
        payoff_geo = (
            np.maximum(geo_avg - option.strike, 0.0)
            if option.is_call
            else np.maximum(option.strike - geo_avg, 0.0)
        )

        vol_eff = self._estimate_effective_vol(S, option.maturity)
        sig_geo = vol_eff / np.sqrt(3.0) if vol_eff > 0 else 0.0
        if sig_geo > 0:
            d1 = (np.log(S0 / option.strike) + 0.5 * sig_geo ** 2 * option.maturity) / (
                sig_geo * np.sqrt(option.maturity)
            )
            d2 = d1 - sig_geo * np.sqrt(option.maturity)
            Nd1 = 0.5 * (1.0 + erf(d1 / sqrt(2.0)))
            Nd2 = 0.5 * (1.0 + erf(d2 / sqrt(2.0)))
            geo_exact = discount * (S0 * Nd1 - option.strike * Nd2)
        else:
            geo_exact = discount * np.mean(payoff_geo)

        cov_matrix = np.cov(payoff_arith, payoff_geo, ddof=1)
        covariance = cov_matrix[0, 1]
        variance_geo = cov_matrix[1, 1]
        beta = covariance / variance_geo if variance_geo > 0 else 0.0

        adjusted_payoff = payoff_arith - beta * (payoff_geo - geo_exact / discount)
        cv_price = discount * np.mean(adjusted_payoff)
        cv_se = discount * np.std(adjusted_payoff, ddof=1) / np.sqrt(S.shape[0])
        vr_factor = (crude_se / cv_se) ** 2 if cv_se > 0 else np.inf

        return {
            "price": cv_price,
            "std_error": cv_se,
            "crude_price": crude_price,
            "crude_std_error": crude_se,
            "variance_reduction": vr_factor,
            "beta": beta,
            "n_paths": S.shape[0],
            "n_steps": len(sim.time_grid) - 1,
            "diagnostics": diagnostics,
        }
