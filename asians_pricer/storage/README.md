# storage

Run persistence utilities.

## run_store.py
- `record_run(model, option, engine_params, price_result, storage_path=..., meta=None)`  
  Writes JSONL entry with timestamp, model name, inputs, result, and optional meta.
- `load_runs(storage_path=...)` -> list of `RunRecord`.
- `RunRecord`: dataclass structure for stored runs.
- `DEFAULT_LOG_PATH`: default `runs/pricing_runs.jsonl`.
  - Run inputs are serialized via `as_dict` where available; option details are stored from its attributes.

Example:
```python
from pathlib import Path
from asians_pricer.storage import record_run
record_run("heston", option, params, price_result, storage_path=Path("runs/pricing_runs.jsonl"))
```
