# asians_pricer package

Python toolkit for pricing arithmetic Asian options in ASX energy markets under Heston and Lévy (VG/NIG) models.

## Key modules
- `engines.monte_carlo.VectorizedHestonEngine`: Heston MC with full truncation, antithetic variates, geometric control variate, optional diagnostics.
- `engines.levy_monte_carlo.LevyMonteCarloEngine`: MC for Variance Gamma / NIG with control variates.
- `models.heston`, `models.levy`: parameter dataclasses and helpers (`validate_feller_condition`, `clamp_correlation`).
- `models.calibration.HestonCalibrator` (optional QuantLib): calibrate Heston to vanilla quotes.
- `analytics.greeks.GreekEngine`: finite-difference Greeks using common random numbers.
- `instruments.asian_option.AsianOption`: contract definition.
- `storage.run_store`: record and load pricing runs (JSONL).
- `visualization.plotting`: Plotly dashboards for diagnostics.

## Typical flow
1. Prepare market data (spot, rate, vanilla vols).
2. Calibrate Heston (or set parameters manually).
3. Instantiate `AsianOption`.
4. Price with `VectorizedHestonEngine.price_asian(...)` or `LevyMonteCarloEngine.price_asian(...)`.
5. (Optional) Compute Greeks with `GreekEngine`.
6. Persist with `record_run(...)`.
7. Render a dashboard with `render_dashboard(...)`.

See `notebooks/example_pricing.ipynb` and `app.py` for end-to-end examples.
