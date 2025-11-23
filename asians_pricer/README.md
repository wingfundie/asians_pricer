# asians_pricer

Python toolkit for pricing arithmetic Asian options (ASX energy focus) under Heston and Lévy (Variance Gamma, NIG) models. Includes Monte Carlo engines, optional calibration, Greeks, run logging, and Plotly dashboards.

## Features
- Heston MC with full truncation, antithetic variates, geometric control variate, and diagnostics.
- VG/NIG MC via subordinated Brownian motion with control variates.
- Calibration helpers: QuantLib-based Heston calibrator; scipy-based VG/NIG calibrator.
- Finite-difference Greeks (Delta, Gamma, Vega via v0, Vanna, Volga, Theta) with common random numbers.
- Run logging to JSONL plus Plotly dashboards for paths, payoffs, and summary metrics.
- CLI, scripts, and notebook for quick starts; simple logging config helper.

## Install (local)
```bash
pip install -e .
# optional extras
pip install plotly
pip install QuantLib-Python
```

## Layout
```
asians_pricer/
  analytics/greeks.py          # FD Greeks engine
  engines/monte_carlo.py       # Heston MC
  engines/levy_monte_carlo.py  # VG/NIG MC
  instruments/asian_option.py  # Contract definition
  models/heston.py, levy.py    # Params and helpers
  models/calibration.py        # QuantLib Heston calibrator (optional)
  models/levy_calibration.py   # Scipy VG/NIG calibrator
  logging_utils.py             # configure_logging helper
  storage/run_store.py         # JSONL run logging
  visualization/plotting.py    # Plotly dashboards
app.py                         # CLI
demo_heston.py, demo_levy.py   # Small scripts
notebooks/example_pricing.ipynb# End-to-end walkthrough
```

## Quick start
```python
from asians_pricer import AsianOption, HestonParams, VectorizedHestonEngine

option = AsianOption(strike=100.0, maturity=0.25, is_call=True)
params = HestonParams(v0=0.04, kappa=2.0, theta=0.09, sigma=0.5, rho=-0.4)
engine = VectorizedHestonEngine(params=params, risk_free_rate=0.03, steps_per_year=252)

res = engine.price_asian(option=option, S0=105.0, n_paths=50_000, antithetic=True, control_variate=True)
print(res["price"], res["std_error"])
```

## End-to-end workflow
```python
import logging
from pathlib import Path
from datetime import date
from asians_pricer import (
    AsianOption, HestonParams, VectorizedHestonEngine,
    LevyMonteCarloEngine, VarianceGammaParams, NIGParams,
    GreekEngine, record_run
)
from asians_pricer.visualization.plotting import render_dashboard
from asians_pricer.logging_utils import configure_logging

configure_logging(logging.INFO)
spot, rate = 105.0, 0.03

# 1) Calibrate or set Heston params (and optionally VG/NIG via LevyCalibrator)
try:
    from asians_pricer.models import HestonCalibrator
    cal = HestonCalibrator(date.today(), spot_price=spot, risk_free_rate=rate)
    heston_params = cal.calibrate([(100, 0.25, 0.6)])
except Exception:
    heston_params = HestonParams(v0=0.04, kappa=2.0, theta=0.09, sigma=0.5, rho=-0.4)
from asians_pricer import LevyCalibrator, VarianceGammaParams, NIGParams
levy_cal = LevyCalibrator(spot=spot, rate=rate, n_paths=10_000)
vg_guess = levy_cal.calibrate_vg([(100, 0.25, 6.0)])
nig_guess = levy_cal.calibrate_nig([(100, 0.25, 6.0)])

# 2) Contract
opt = AsianOption(strike=100.0, maturity=0.25, is_call=True)

# 3) Heston MC + Greeks
hes_engine = VectorizedHestonEngine(heston_params, risk_free_rate=rate, steps_per_year=252)
hes_res = hes_engine.price_asian(opt, S0=spot, n_paths=30_000, antithetic=True, control_variate=True, seed=42, diag_samples=200)
greeks = GreekEngine(hes_engine, seed=42).calculate(opt, S0=spot, n_paths=10_000)

# 4) Lévy MC (VG, NIG)
levy_engine = LevyMonteCarloEngine(risk_free_rate=rate, steps_per_year=252)
vg_res = levy_engine.price_asian(opt, S0=spot, n_paths=30_000, params=VarianceGammaParams(theta=0.0, sigma=0.25, nu=0.2), process="vg", seed=7, diag_samples=200)
nig_res = levy_engine.price_asian(opt, S0=spot, n_paths=30_000, params=NIGParams(alpha=5.0, beta=-2.0, delta=0.5, mu=0.0), process="nig", seed=7, diag_samples=200)

# 5) Persist runs
record_run("heston", opt, heston_params, {k: v for k, v in hes_res.items() if k != "diagnostics"})
record_run("variance_gamma", opt, VarianceGammaParams(theta=0.0, sigma=0.25, nu=0.2), {k: v for k, v in vg_res.items() if k != "diagnostics"})
record_run("nig", opt, NIGParams(alpha=5.0, beta=-2.0, delta=0.5, mu=0.0), {k: v for k, v in nig_res.items() if k != "diagnostics"})

# 6) Dashboards (needs plotly)
Path("runs").mkdir(exist_ok=True)
render_dashboard(hes_res["diagnostics"], hes_res, "runs/heston_dashboard.html")
render_dashboard(vg_res["diagnostics"], vg_res, "runs/vg_dashboard.html")
render_dashboard(nig_res["diagnostics"], nig_res, "runs/nig_dashboard.html")

print("Heston price", hes_res["price"], "Greeks", greeks)
```

## CLI
```bash
python app.py --model heston --strike 100 --maturity 0.25 --spot 105 --paths 50000 --diag-samples 200 --dashboard-path runs/heston.html
python app.py --model vg --strike 100 --maturity 0.25 --spot 105 --paths 50000 --diag-samples 200 --dashboard-path runs/vg.html
```

## Notebook
- `notebooks/example_pricing.ipynb` replicates the workflow interactively (calibration fallback, pricing, Greeks, logging, dashboards).

## Dependencies
- Required: numpy.
- Optional: plotly (dashboards), QuantLib-Python (calibration).

## Support
Use the CLI or notebook for quickest start. For custom integrations, import engines and utilities directly as shown above.
