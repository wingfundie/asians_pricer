"""
Finite-difference Greeks for Heston Asian options using common random numbers.
"""

from dataclasses import replace
from typing import Dict, Optional

import numpy as np

from ..engines.monte_carlo import VectorizedHestonEngine
from ..models.heston import HestonParams
from ..instruments.asian_option import AsianOption


class GreekEngine:
    def __init__(
        self,
        engine: VectorizedHestonEngine,
        spot_bump_pct: float = 0.01,
        vol_bump: float = 0.01,
        time_bump: float = 1.0 / 252.0,
        seed: Optional[int] = 7,
    ):
        self.engine = engine
        self.spot_bump_pct = spot_bump_pct
        self.vol_bump = vol_bump
        self.time_bump = time_bump
        self.seed = seed

    def _price(
        self, option: AsianOption, S0: float, n_paths: int, params: Optional[HestonParams] = None
    ) -> Dict[str, float]:
        if params is None:
            return self.engine.price_asian(
                option, S0, n_paths, antithetic=True, control_variate=True, seed=self.seed
            )
        temp_engine = VectorizedHestonEngine(
            params=params, risk_free_rate=self.engine.r, steps_per_year=self.engine.steps_per_year
        )
        return temp_engine.price_asian(
            option, S0, n_paths, antithetic=True, control_variate=True, seed=self.seed
        )

    def calculate(
        self, option: AsianOption, S0: float, n_paths: int
    ) -> Dict[str, float]:
        """
        Compute Delta, Gamma, Vega (via v0), Vanna, Volga, and a simple Theta.
        """
        eps_S = self.spot_bump_pct * S0
        eps_vol = self.vol_bump
        if eps_S <= 0:
            raise ValueError("spot bump must be positive")

        base_res = self._price(option, S0, n_paths)
        base_price = base_res["price"]

        # Spot bumps for Delta/Gamma
        up_res = self._price(option, S0 + eps_S, n_paths)
        dn_res = self._price(option, S0 - eps_S, n_paths)

        delta = (up_res["price"] - dn_res["price"]) / (2.0 * eps_S)
        gamma = (up_res["price"] - 2.0 * base_price + dn_res["price"]) / (eps_S ** 2)

        # Vega via bumping sqrt(v0)
        sigma0 = np.sqrt(max(self.engine.params.v0, 0.0))
        sigma_up = sigma0 + eps_vol
        sigma_dn = max(sigma0 - eps_vol, 1e-8)
        v0_up = sigma_up ** 2
        v0_dn = sigma_dn ** 2

        params_up = self.engine.params.bumped(v0=v0_up)
        params_dn = self.engine.params.bumped(v0=v0_dn)

        price_vol_up = self._price(option, S0, n_paths, params=params_up)["price"]
        price_vol_dn = self._price(option, S0, n_paths, params=params_dn)["price"]

        vega = (price_vol_up - price_vol_dn) / (2.0 * eps_vol)
        volga = (price_vol_up - 2.0 * base_price + price_vol_dn) / (eps_vol ** 2)

        # Vanna: sensitivity of Delta to volatility bump
        delta_vol_up = (
            self._price(option, S0 + eps_S, n_paths, params=params_up)["price"]
            - self._price(option, S0 - eps_S, n_paths, params=params_up)["price"]
        ) / (2.0 * eps_S)
        delta_vol_dn = (
            self._price(option, S0 + eps_S, n_paths, params=params_dn)["price"]
            - self._price(option, S0 - eps_S, n_paths, params=params_dn)["price"]
        ) / (2.0 * eps_S)
        vanna = (delta_vol_up - delta_vol_dn) / (2.0 * eps_vol)

        # Theta: forward difference by shortening maturity
        short_T = max(option.maturity - self.time_bump, 1e-6)
        short_opt = replace(option, maturity=short_T)
        price_short = self._price(short_opt, S0, n_paths)["price"]
        theta = (price_short - base_price) / self.time_bump

        return {
            "price": base_price,
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "vanna": vanna,
            "volga": volga,
            "theta": theta,
        }
