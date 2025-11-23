# models

Model parameter containers and calibration.

- `heston.py`: `HestonParams`, `validate_feller_condition`, `clamp_correlation`.
- `levy.py`: `VarianceGammaParams`, `NIGParams`.
- `calibration.py`: `HestonCalibrator` (requires QuantLib) to fit Heston to vanilla quotes.

Example:
```python
params = HestonParams(v0=0.04, kappa=2.0, theta=0.09, sigma=0.5, rho=-0.4)
validate_feller_condition(params)
```
Calibration (if QuantLib installed):
```python
cal = HestonCalibrator(valuation_date=date.today(), spot_price=spot, risk_free_rate=rate)
heston_params = cal.calibrate([(100, 0.25, 0.6)])
```
