"""
Plotting helpers for Monte Carlo diagnostics.
"""

from typing import Iterable, Sequence, Tuple


def _require_plotly():
    """
    Import Plotly lazily and raise a helpful error if it is missing.

    Returns:
        The ``plotly.graph_objects`` module once successfully imported.
    """
    try:
        import plotly.graph_objects as go  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("plotly is required for visualization utilities") from exc
    return go


def plot_paths(time_grid, asset_paths, variance_paths=None, num_paths: int = 50):
    """
    Plot a subset of simulated asset (and optional variance) paths.

    Args:
        time_grid: 1D iterable of time stamps for each column in ``asset_paths``.
        asset_paths: 2D array-like of simulated price paths.
        variance_paths: Optional 2D array-like of variance paths to plot alongside prices.
        num_paths: Maximum number of paths to render to keep the figure readable.

    Returns:
        A Plotly ``Figure`` for asset paths, or a tuple of (asset_fig, variance_fig)
        when variance paths are supplied. Useful for quick diagnostics after simulation.
    """
    go = _require_plotly()
    fig = go.Figure()
    n = min(num_paths, asset_paths.shape[0])
    for i in range(n):
        fig.add_trace(
            go.Scatter(
                x=time_grid,
                y=asset_paths[i],
                mode="lines",
                line=dict(width=0.5, color="steelblue"),
                opacity=0.4,
                showlegend=False,
            )
        )
    fig.update_layout(
        title="Heston Monte Carlo Asset Paths",
        xaxis_title="Time (years)",
        yaxis_title="Price",
    )

    if variance_paths is not None:
        fig_var = go.Figure()
        for i in range(n):
            fig_var.add_trace(
                go.Scatter(
                    x=time_grid,
                    y=variance_paths[i],
                    mode="lines",
                    line=dict(width=0.5, color="firebrick"),
                    opacity=0.4,
                    showlegend=False,
                )
            )
        fig_var.update_layout(
            title="Variance Paths",
            xaxis_title="Time (years)",
            yaxis_title="Variance",
        )
        return fig, fig_var
    return fig


def plot_convergence(path_counts: Sequence[int], prices: Sequence[float]):
    """
    Visualize Monte Carlo convergence as the number of simulated paths increases.

    Args:
        path_counts: Sequence of path counts tested.
        prices: Sequence of estimated prices corresponding to ``path_counts``.

    Returns:
        Plotly ``Figure`` showing the price trajectory, helping assess stability
        and choose an appropriate path budget.
    """
    go = _require_plotly()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(path_counts),
            y=list(prices),
            mode="lines+markers",
            line=dict(color="darkgreen"),
        )
    )
    fig.update_layout(
        title="Monte Carlo Convergence",
        xaxis_title="Number of paths",
        yaxis_title="Asian option price",
    )
    return fig


def plot_payoff_histogram(payoffs: Sequence[float], bins: int = 50):
    """
    Draw a histogram of simulated payoffs to inspect distribution shape and tails.

    Args:
        payoffs: Sequence of simulated payoff realizations.
        bins: Number of histogram bins to render.

    Returns:
        Plotly ``Figure`` summarizing payoff dispersion, helpful when debugging
        variance issues or extreme scenario behavior.
    """
    go = _require_plotly()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=list(payoffs), nbinsx=bins, marker_color="teal"))
    fig.update_layout(title="Payoff Distribution", xaxis_title="Payoff", yaxis_title="Frequency")
    return fig


def render_dashboard(
    diagnostics: dict,
    price_result: dict,
    output_path: str = "runs/dashboard.html",
    title: str = "Asian Option Pricing Diagnostics",
):
    """
    Build an HTML dashboard combining paths, payoff histogram, and summary metrics.

    Args:
        diagnostics: Output diagnostics from pricing (time grid, sampled paths, payoffs).
        price_result: Pricing result dictionary including summary statistics.
        output_path: Destination HTML file written with ``plotly.io.write_html``.
        title: Title to display on the dashboard.

    Returns:
        Path to the generated HTML file, which can be opened in a browser or shared
        with stakeholders for quick inspection of a run.
    """
    go = _require_plotly()
    from plotly.subplots import make_subplots  # type: ignore

    if diagnostics is None:
        raise ValueError("Diagnostics are required to render dashboard (set diag_samples>0 in pricing).")

    fig = make_subplots(rows=2, cols=2, subplot_titles=("Asset Paths", "Variance Paths", "Payoff Histogram", "Summary"))

    # Asset paths
    for path in diagnostics.get("asset_paths", [])[:50]:
        fig.add_trace(
            go.Scatter(x=diagnostics["time_grid"], y=path, mode="lines", line=dict(width=0.7, color="steelblue"), opacity=0.6, showlegend=False),
            row=1,
            col=1,
        )

    # Variance paths if available
    if "variance_paths" in diagnostics:
        for vpath in diagnostics["variance_paths"][:50]:
            fig.add_trace(
                go.Scatter(x=diagnostics["time_grid"], y=vpath, mode="lines", line=dict(width=0.7, color="firebrick"), opacity=0.5, showlegend=False),
                row=1,
                col=2,
            )
    else:
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=price_result.get("variance_reduction", 1.0),
                title={"text": "Variance Reduction"},
            ),
            row=1,
            col=2,
        )

    # Payoff histogram
    fig.add_trace(
        go.Histogram(x=diagnostics.get("payoff_arith", []), nbinsx=40, marker_color="teal", showlegend=False),
        row=2,
        col=1,
    )

    # Summary box
    summary_text = f"Price: {price_result.get('price', 0):.4f}<br>"
    summary_text += f"Std Err: {price_result.get('std_error', 0):.4f}<br>"
    summary_text += f"Crude: {price_result.get('crude_price', 0):.4f}<br>"
    summary_text += f"VR Factor: {price_result.get('variance_reduction', 1):.1f}<br>"
    summary_text += f"Paths: {price_result.get('n_paths', 0)}<br>"
    summary_text += f"Steps: {price_result.get('n_steps', 0)}"
    fig.add_trace(
        go.Table(
            header=dict(values=["Metric", "Value"]),
            cells=dict(
                values=[
                    ["Price", "Std Error", "Crude", "VR Factor", "Paths", "Steps"],
                    [
                        f"{price_result.get('price', 0):.4f}",
                        f"{price_result.get('std_error', 0):.6f}",
                        f"{price_result.get('crude_price', 0):.4f}",
                        f"{price_result.get('variance_reduction', 1):.1f}",
                        str(price_result.get("n_paths", 0)),
                        str(price_result.get("n_steps", 0)),
                    ],
                ]
            ),
        ),
        row=2,
        col=2,
    )

    fig.update_layout(title=title, bargap=0.1, height=800)
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path
