# visualization

Plotly-based diagnostics.

- `plotting.py`: `plot_paths`, `plot_convergence`, `plot_payoff_histogram`, and `render_dashboard` to produce an HTML summary combining paths, payoffs, and metrics.

Example:
```python
from asians_pricer.visualization.plotting import render_dashboard
render_dashboard(res["diagnostics"], res, output_path="runs/dashboard.html")
```
