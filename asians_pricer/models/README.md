# models

Model parameter containers and calibration utilities.

## heston.py
- `HestonParams(v0, kappa, theta, sigma, rho)` – dataclass with `as_dict()` and `bumped(**kwargs)`.
- `validate_feller_condition(params)` – check positivity condition.
- `clamp_correlation(rho)` – clip rho into (-1,1).
  - `bumped(**kwargs)` returns a copy with overrides for scenario/Greeks.

## levy.py
- `VarianceGammaParams(theta, sigma, nu)`
- `NIGParams(alpha, beta, delta, mu)`

## calibration.py (optional QuantLib)
- `HestonCalibrator(valuation_date, spot_price, risk_free_rate)`  
  `calibrate(market_quotes)` where `market_quotes` is iterable of `(strike, expiry, implied_vol)`; returns `HestonParams`.

## levy_calibration.py (scipy-based)
- `LevyCalibrator(spot, rate, n_paths=20000, seed=123)`
  - `calibrate_vg(market_quotes, initial=VarianceGammaParams(...))`  
    Fits VG params to vanilla prices via MC + Nelder-Mead. `market_quotes`: `(strike, maturity_years, market_price)`.
  - `calibrate_nig(market_quotes, initial=NIGParams(...))`  
    Fits NIG params similarly.
  - Uses one-step terminal simulation for vanillas; requires `scipy.optimize`.

### Examples
```python
params = HestonParams(v0=0.04, kappa=2.0, theta=0.09, sigma=0.5, rho=-0.4)
validate_feller_condition(params)
```
```python
from datetime import date
cal = HestonCalibrator(date.today(), spot_price=spot, risk_free_rate=rate)
heston_params = cal.calibrate([(100, 0.25, 0.6)])
```
```python
levy_cal = LevyCalibrator(spot=105.0, rate=0.03)
vg_params = levy_cal.calibrate_vg([(100, 0.25, 6.0), (110, 0.25, 3.5)])
nig_params = levy_cal.calibrate_nig([(100, 0.25, 6.0), (110, 0.25, 3.5)])
```
