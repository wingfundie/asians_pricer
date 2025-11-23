# storage

Run persistence utilities.

- `run_store.py`: `record_run(model, option, engine_params, price_result, storage_path=..., meta=None)` writes JSONL entries; `load_runs` loads them; `RunRecord` dataclass describes the structure.
- `DEFAULT_LOG_PATH`: default `runs/pricing_runs.jsonl`.

Example:
```python
from pathlib import Path
from asians_pricer.storage import record_run
record_run("heston", option, params, price_result, storage_path=Path("runs/pricing_runs.jsonl"))
```
