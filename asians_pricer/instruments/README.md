# instruments

Contract definitions.

## asian_option.py
- `AsianOption(strike, maturity, is_call=True, notional=1.0, averaging_observations=None)`
  - `payoff(average)` -> intrinsic payoff scaled by `notional`.
  - Basic validation on strike and maturity.
   - `averaging_observations` placeholder for metadata on sampling frequency (not used directly by engines but can be stored for reporting).

Used as the payoff input to all pricing engines.
