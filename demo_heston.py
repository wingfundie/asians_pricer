from pprint import pprint

from asians_pricer import (
    AsianOption,
    GreekEngine,
    HestonParams,
    VectorizedHestonEngine,
)


def main():
    option = AsianOption(strike=100.0, maturity=0.25, is_call=True)
    params = HestonParams(v0=0.04, kappa=2.0, theta=0.09, sigma=0.5, rho=-0.4)
    engine = VectorizedHestonEngine(params=params, risk_free_rate=0.03, steps_per_year=252)

    price_result = engine.price_asian(
        option=option, S0=105.0, n_paths=50_000, antithetic=True, control_variate=True, seed=123
    )
    print("Heston Asian price with control variate:")
    pprint(price_result)

    greek_engine = GreekEngine(engine, seed=123)
    greeks = greek_engine.calculate(option=option, S0=105.0, n_paths=20_000)
    print("\nGreeks (finite difference with common random numbers):")
    pprint(greeks)


if __name__ == "__main__":
    main()
