# engines

Monte Carlo pricing engines.

## Heston (`monte_carlo.py`)
- `SimulationResult`: container `(time_grid, asset_paths, variance_paths)`.
- `VectorizedHestonEngine(params, risk_free_rate, steps_per_year=252)`
  - `simulate(S0, T, n_paths, antithetic=True, seed=None)` -> `SimulationResult`
    - Euler-Maruyama with full truncation for variance; log-Euler for asset; correlated shocks from rho; antithetic variates optional.
  - `_geometric_asian_bs_price(S0, K, T)` -> control variate analytic proxy using long-run variance.
  - `price_asian(option, S0, n_paths, antithetic=True, control_variate=True, seed=None, diag_samples=0)`
    - Returns dict: `price, std_error, crude_price, crude_std_error, beta, variance_reduction, n_paths, n_steps, diagnostics(optional)`.
    - `diagnostics` includes sampled paths/averages/payoffs if `diag_samples>0` for dashboards.

## Lévy (`levy_monte_carlo.py`)
- `LevySimulationResult`: container `(time_grid, asset_paths)`.
- `LevyMonteCarloEngine(risk_free_rate, steps_per_year=252)`
  - Private simulators: `_simulate_vg(...)` and `_simulate_nig(...)` for VG/NIG paths via subordinated Brownian motion.
  - `_estimate_effective_vol(paths, T)` helper for control variate vol guess.
  - `price_asian(option, S0, n_paths, params, process="vg"|"nig", antithetic=True, control_variate=True, seed=None, diag_samples=0)`
    - Params: `VarianceGammaParams` (VG) or `NIGParams` (NIG).
    - Returns same keys as Heston engine plus optional diagnostics.
    - Control variate: geometric average with BS-style proxy using effective vol.

## Usage
```python
engine = VectorizedHestonEngine(params, risk_free_rate=0.03, steps_per_year=252)
res = engine.price_asian(option, S0=spot, n_paths=50_000, diag_samples=200)
```
```python
levy_engine = LevyMonteCarloEngine(risk_free_rate=0.03)
res = levy_engine.price_asian(option, S0=spot, params=vg_params, process="vg", diag_samples=200)
```
