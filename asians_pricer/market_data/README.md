# market_data

Lightweight placeholders for curves and surfaces.

## curves.py
- `flat_discount(rate)` -> returns D(t) = exp(-r t).
- `flat_forward(forward_price)` -> constant forward curve F(t) = F0.
  - These are callable factories to plug into calibration/pricing when full curve bootstrapping is unnecessary.

## surfaces.py
- `flat_vol(vol)` -> function vol(T, K) = constant.

Replace with real loaders/bootstrappers as needed.
