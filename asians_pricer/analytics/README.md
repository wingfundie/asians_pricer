# analytics

Finite-difference Greeks.

- `greeks.py`: `GreekEngine` computes Delta, Gamma, Vega (via v0), Vanna, Volga, Theta using common random numbers.

Example:
```python
greeks = GreekEngine(engine, seed=42).calculate(option=asian_opt, S0=spot, n_paths=20000)
```
