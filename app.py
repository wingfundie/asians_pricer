import argparse
import logging
from pathlib import Path
from typing import Optional

from asians_pricer import (
    AsianOption,
    HestonParams,
    LevyMonteCarloEngine,
    NIGParams,
    VarianceGammaParams,
    VectorizedHestonEngine,
    configure_logging,
    record_run,
)
from asians_pricer.visualization.plotting import render_dashboard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASX energy Asian option pricer CLI.")
    parser.add_argument("--model", choices=["heston", "vg", "nig"], default="heston")
    parser.add_argument("--strike", type=float, required=True)
    parser.add_argument("--maturity", type=float, required=True, help="Maturity in years.")
    parser.add_argument("--call", action="store_true", help="Price call (default).")
    parser.add_argument("--put", action="store_true", help="Price put.")
    parser.add_argument("--spot", type=float, required=True, help="Current futures price S0.")
    parser.add_argument("--rate", type=float, default=0.03, help="Risk-free rate.")
    parser.add_argument("--paths", type=int, default=50000, help="Number of Monte Carlo paths.")
    parser.add_argument("--steps-per-year", type=int, default=252, help="Time steps per year.")
    parser.add_argument("--seed", type=int, default=123, help="RNG seed.")
    parser.add_argument("--diag-samples", type=int, default=200, help="Number of paths to store for diagnostics.")
    parser.add_argument("--no-dashboard", action="store_true", help="Skip dashboard generation.")
    parser.add_argument("--dashboard-path", type=str, default=None, help="Output HTML for dashboard.")
    parser.add_argument("--log-path", type=str, default=None, help="Run log JSONL path.")
    parser.add_argument("--no-control-variate", action="store_true", help="Disable control variate.")
    parser.add_argument("--no-antithetic", action="store_true", help="Disable antithetic variates.")

    # Heston params
    parser.add_argument("--heston-v0", type=float, default=0.04)
    parser.add_argument("--heston-kappa", type=float, default=2.0)
    parser.add_argument("--heston-theta", type=float, default=0.09)
    parser.add_argument("--heston-sigma", type=float, default=0.5)
    parser.add_argument("--heston-rho", type=float, default=-0.4)

    # VG params
    parser.add_argument("--vg-theta", type=float, default=0.0)
    parser.add_argument("--vg-sigma", type=float, default=0.25)
    parser.add_argument("--vg-nu", type=float, default=0.2)

    # NIG params
    parser.add_argument("--nig-alpha", type=float, default=5.0)
    parser.add_argument("--nig-beta", type=float, default=-2.0)
    parser.add_argument("--nig-delta", type=float, default=0.5)
    parser.add_argument("--nig-mu", type=float, default=0.0)

    return parser.parse_args()


def main():
    args = parse_args()
    configure_logging()

    is_call = True
    if args.put:
        is_call = False
    elif args.call:
        is_call = True

    option = AsianOption(strike=args.strike, maturity=args.maturity, is_call=is_call)

    if args.model == "heston":
        params = HestonParams(
            v0=args.heston_v0,
            kappa=args.heston_kappa,
            theta=args.heston_theta,
            sigma=args.heston_sigma,
            rho=args.heston_rho,
        )
        engine = VectorizedHestonEngine(
            params=params, risk_free_rate=args.rate, steps_per_year=args.steps_per_year
        )
        price_result = engine.price_asian(
            option=option,
            S0=args.spot,
            n_paths=args.paths,
            antithetic=not args.no_antithetic,
            control_variate=not args.no_control_variate,
            seed=args.seed,
            diag_samples=args.diag_samples if not args.no_dashboard else 0,
        )
        model_label = "heston"
    elif args.model == "vg":
        params = VarianceGammaParams(theta=args.vg_theta, sigma=args.vg_sigma, nu=args.vg_nu)
        engine = LevyMonteCarloEngine(risk_free_rate=args.rate, steps_per_year=args.steps_per_year)
        price_result = engine.price_asian(
            option=option,
            S0=args.spot,
            n_paths=args.paths,
            params=params,
            process="vg",
            antithetic=not args.no_antithetic,
            control_variate=not args.no_control_variate,
            seed=args.seed,
            diag_samples=args.diag_samples if not args.no_dashboard else 0,
        )
        model_label = "variance_gamma"
    else:  # NIG
        params = NIGParams(
            alpha=args.nig_alpha,
            beta=args.nig_beta,
            delta=args.nig_delta,
            mu=args.nig_mu,
        )
        engine = LevyMonteCarloEngine(risk_free_rate=args.rate, steps_per_year=args.steps_per_year)
        price_result = engine.price_asian(
            option=option,
            S0=args.spot,
            n_paths=args.paths,
            params=params,
            process="nig",
            antithetic=not args.no_antithetic,
            control_variate=not args.no_control_variate,
            seed=args.seed,
            diag_samples=args.diag_samples if not args.no_dashboard else 0,
        )
        model_label = "nig"

    logging.info(
        "%s price=%.4f (std_error=%.6f) crude=%.4f VR=%.1f paths=%d",
        model_label,
        price_result["price"],
        price_result["std_error"],
        price_result.get("crude_price", price_result["price"]),
        price_result.get("variance_reduction", 1.0),
        price_result.get("n_paths", args.paths),
    )

    if not args.no_dashboard and price_result.get("diagnostics") is not None:
        dash_path = args.dashboard_path or f"runs/dashboard_{model_label}.html"
        try:
            render_dashboard(price_result["diagnostics"], price_result, dash_path)
            logging.info("Dashboard written to %s", dash_path)
        except Exception as exc:  # pragma: no cover - plotting optional
            logging.warning("Dashboard rendering failed: %s", exc)

    # Persist run
    log_path = Path(args.log_path) if args.log_path else None
    try:
        record_run(
            model=model_label,
            option=option,
            engine_params=params,
            price_result={k: v for k, v in price_result.items() if k != "diagnostics"},
            storage_path=log_path or Path("runs") / "pricing_runs.jsonl",
            meta={
                "rate": args.rate,
                "paths": args.paths,
                "steps_per_year": args.steps_per_year,
                "seed": args.seed,
                "control_variate": not args.no_control_variate,
                "antithetic": not args.no_antithetic,
            },
        )
    except Exception as exc:  # pragma: no cover - best effort persistence
        logging.warning("Failed to record run: %s", exc)


if __name__ == "__main__":
    main()
