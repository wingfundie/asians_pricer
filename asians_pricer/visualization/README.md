# visualization

Plotly-based diagnostics.

## plotting.py
- `plot_paths(time_grid, asset_paths, variance_paths=None, num_paths=50)` – quick path plots.
- `plot_convergence(path_counts, prices)` – convergence line chart.
- `plot_payoff_histogram(payoffs, bins=50)` – payoff distribution.
- `render_dashboard(diagnostics, price_result, output_path="runs/dashboard.html", title=...)` – combined HTML with paths, variance, payoff histogram, and summary table. Requires `diagnostics` produced by engines with `diag_samples>0`.
  - `diagnostics` keys consumed: `time_grid`, `asset_paths`, optionally `variance_paths`, `payoff_arith`.
  - Summary table uses `price_result` keys (`price`, `std_error`, `crude_price`, `variance_reduction`, `n_paths`, `n_steps`).

Example:
```python
from asians_pricer.visualization.plotting import render_dashboard
render_dashboard(res["diagnostics"], res, output_path="runs/dashboard.html")
```
