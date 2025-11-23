# analytics

Finite-difference Greeks for Asian options.

## greeks.py
- `GreekEngine(engine, spot_bump_pct=0.01, vol_bump=0.01, time_bump=1/252, seed=None)`
  - `calculate(option, S0, n_paths)` -> returns dict with:
    - `price`: base price (re-priced via engine).
    - `delta`, `gamma`: central-difference on spot with common random numbers.
    - `vega`: central-difference bump on sqrt(v0) (resets engine v0).
    - `vanna`: sensitivity of Delta to vol bump.
    - `volga`: second-order vol sensitivity.
    - `theta`: forward-difference by reducing maturity by `time_bump`.
  - Internal helper `_price(...)` supports temporary parameter overrides and seeds to reuse CRNs.

Usage:
```python
from asians_pricer.analytics import GreekEngine
greeks = GreekEngine(engine, seed=42).calculate(option=asian_opt, S0=spot, n_paths=20_000)
```
