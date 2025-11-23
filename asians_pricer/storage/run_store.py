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
    """
    Structured record capturing a single pricing run for later inspection or replay.

    Attributes:
        timestamp: ISO8601 UTC timestamp when the run was recorded.
        model: Name of the pricing model used.
        inputs: Serialized option and engine parameter inputs.
        result: Pricing outputs (typically excluding heavy diagnostics).
        meta: Optional additional metadata such as seeds or path counts.
    """
    timestamp: str
    model: str
    inputs: Dict[str, Any]
    result: Dict[str, Any]
    meta: Dict[str, Any] = field(default_factory=dict)


def _ensure_path(path: Path) -> None:
    """
    Create parent directories for the storage path if they do not exist.

    Args:
        path: Full file path whose parents should be created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def _to_serializable(obj: Any) -> Any:
    """
    Convert rich objects to a JSON-serializable representation.

    Attempts to use ``as_dict`` or ``__dict__`` when available, falling back to the
    original object for primitives.

    Args:
        obj: Object to convert to a serializable structure.

    Returns:
        JSON-friendly representation of the object.
    """
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
    Persist a pricing run to a JSON Lines log and return the recorded entry.

    Args:
        model: Name of the pricing model (e.g., ``\"heston\"`` or ``\"nig\"``).
        option: AsianOption instance describing the contract.
        engine_params: Model/engine parameters to be serialized alongside the run.
        price_result: Pricing outputs (excluding large diagnostics blobs).
        storage_path: Path to the JSONL file; created if missing.
        meta: Optional extra metadata such as path counts or seeds.

    Returns:
        The ``RunRecord`` that was written to disk, useful for immediate inspection.
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
    Load all previously recorded runs from a JSONL log.

    Args:
        storage_path: Path to the JSONL log file.

    Returns:
        List of ``RunRecord`` entries in the order they were written; returns an
        empty list if the file is missing or unreadable lines are skipped.
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
