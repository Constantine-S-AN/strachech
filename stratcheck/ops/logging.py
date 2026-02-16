"""Structured JSON logging helpers for long-running runners."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from stratcheck.ops.secrets import sanitize_logging_payload


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


class JsonEventLogger:
    """Append-only JSON-lines logger."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(
        self,
        *,
        level: str,
        event: str,
        run_id: str | None = None,
        **fields: Any,
    ) -> None:
        """Write a JSON line with timestamp, level, event, and fields."""
        payload: dict[str, Any] = {
            "timestamp": _utc_now().isoformat(),
            "level": str(level).lower(),
            "event": str(event),
        }
        if run_id is not None:
            payload["run_id"] = str(run_id)

        safe_fields = sanitize_logging_payload(fields)
        for key, value in safe_fields.items():
            payload[key] = _to_jsonable(value)

        with self.path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _to_jsonable(raw) for key, raw in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return str(value)
