"""
Run storage utilities for recording pricing runs.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..models.heston import HestonParams
from ..models.levy import NIGParams, VarianceGammaParams
from ..instruments.asian_option import AsianOption

DEFAULT_LOG_PATH = Path("runs") / "pricing_runs.jsonl"


@dataclass
class RunRecord:
    timestamp: str
    model: str
    inputs: Dict[str, Any]
    result: Dict[str, Any]
    meta: Dict[str, Any] = field(default_factory=dict)


def _ensure_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _to_serializable(obj: Any) -> Any:
    if hasattr(obj, "as_dict"):
        return obj.as_dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return obj


def record_run(
    model: str,
    option: AsianOption,
    engine_params: Any,
    price_result: Dict[str, Any],
    storage_path: Path = DEFAULT_LOG_PATH,
    meta: Optional[Dict[str, Any]] = None,
) -> RunRecord:
    """
    Persist a pricing run as JSONL. Returns the RunRecord.
    """
    timestamp = datetime.utcnow().isoformat()
    record = RunRecord(
        timestamp=timestamp,
        model=model,
        inputs={
            "option": option.__dict__,
            "engine_params": _to_serializable(engine_params),
        },
        result=price_result,
        meta=meta or {},
    )
    _ensure_path(storage_path)
    with storage_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record)) + "\n")
    logging.getLogger(__name__).info("Recorded run at %s to %s", timestamp, storage_path)
    return record


def load_runs(storage_path: Path = DEFAULT_LOG_PATH) -> List[RunRecord]:
    """
    Load all stored runs.
    """
    if not storage_path.exists():
        return []
    records: List[RunRecord] = []
    with storage_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                records.append(RunRecord(**data))
            except Exception:
                continue
    return records
