from pprint import pprint

from asians_pricer import AsianOption, LevyMonteCarloEngine, NIGParams, VarianceGammaParams


def main():
    """
    Demonstrate Asian option pricing under Variance Gamma and NIG Levy processes
    using the dedicated Monte Carlo engine with common settings.
    """
    option = AsianOption(strike=100.0, maturity=0.25, is_call=True)
    engine = LevyMonteCarloEngine(risk_free_rate=0.03, steps_per_year=252)

    vg_params = VarianceGammaParams(theta=0.0, sigma=0.25, nu=0.2)
    vg_res = engine.price_asian(
        option=option, S0=105.0, n_paths=40_000, params=vg_params, process="vg", seed=99
    )
    print("Variance Gamma Asian price:")
    pprint(vg_res)

    nig_params = NIGParams(alpha=5.0, beta=-2.0, delta=0.5, mu=0.0)
    nig_res = engine.price_asian(
        option=option, S0=105.0, n_paths=40_000, params=nig_params, process="nig", seed=99
    )
    print("\nNIG Asian price:")
    pprint(nig_res)


if __name__ == "__main__":
    main()
