# engines

Monte Carlo pricing engines.

- `monte_carlo.py` – `VectorizedHestonEngine` for Heston paths with full truncation, antithetic variates, geometric control variate; returns price, std error, beta, variance reduction, and optional diagnostics samples for dashboards.
- `levy_monte_carlo.py` – `LevyMonteCarloEngine` for Variance Gamma and NIG subordinated Brownian motion paths with similar outputs and diagnostics.
- `SimulationResult` / `LevySimulationResult` – containers for simulated grids and paths.

Usage:
```python
engine = VectorizedHestonEngine(params, risk_free_rate=0.03, steps_per_year=252)
res = engine.price_asian(option, S0=spot, n_paths=50000, diag_samples=200)
```
For Lévy:
```python
levy_engine = LevyMonteCarloEngine(risk_free_rate=0.03)
res = levy_engine.price_asian(option, S0=spot, params=vg_params, process="vg")
```
