"""Parquet cache helpers with an on-disk cache index."""

from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


def parquet_engine_available() -> bool:
    """Return whether a parquet engine is available in current environment."""
    return bool(importlib.util.find_spec("pyarrow") or importlib.util.find_spec("fastparquet"))


def load_or_build_parquet_cache(
    symbol: str,
    csv_path: Path,
    cache_dir: Path,
    index_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load bars from parquet cache or build cache from CSV on miss."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_data = _load_index(index_path)

    symbol_key = symbol.strip()
    cache_file_path = cache_dir / f"{symbol_key}.parquet"
    source_signature = _source_signature(csv_path)
    index_entry = dict(index_data.get(symbol_key, {}))

    if not parquet_engine_available():
        index_entry.update(
            {
                "source_signature": source_signature,
                "cache_file": str(cache_file_path),
                "status": "engine_unavailable",
                "updated_at": _utc_now_text(),
            }
        )
        index_data[symbol_key] = index_entry
        _save_index(index_path=index_path, index_data=index_data)
        raw_bars = pd.read_csv(csv_path)
        return raw_bars, {
            "cache_used": False,
            "cache_hit": False,
            "status": "engine_unavailable",
            "index_path": str(index_path),
            "cache_file": str(cache_file_path),
        }

    index_signature = index_entry.get("source_signature")
    if (
        isinstance(index_signature, dict)
        and _same_signature(index_signature, source_signature)
        and cache_file_path.exists()
    ):
        try:
            raw_bars = pd.read_parquet(cache_file_path)
            hit_count = int(index_entry.get("hit_count", 0)) + 1
            index_entry.update(
                {
                    "source_signature": source_signature,
                    "cache_file": str(cache_file_path),
                    "status": "hit",
                    "hit_count": hit_count,
                    "updated_at": _utc_now_text(),
                }
            )
            index_data[symbol_key] = index_entry
            _save_index(index_path=index_path, index_data=index_data)
            return raw_bars, {
                "cache_used": True,
                "cache_hit": True,
                "status": "hit",
                "hit_count": hit_count,
                "index_path": str(index_path),
                "cache_file": str(cache_file_path),
            }
        except Exception:
            pass

    raw_bars = pd.read_csv(csv_path)
    raw_bars.to_parquet(cache_file_path, index=False)
    build_count = int(index_entry.get("build_count", 0)) + 1
    index_entry.update(
        {
            "source_signature": source_signature,
            "cache_file": str(cache_file_path),
            "status": "built",
            "build_count": build_count,
            "updated_at": _utc_now_text(),
        }
    )
    index_data[symbol_key] = index_entry
    _save_index(index_path=index_path, index_data=index_data)
    return raw_bars, {
        "cache_used": True,
        "cache_hit": False,
        "status": "built",
        "build_count": build_count,
        "index_path": str(index_path),
        "cache_file": str(cache_file_path),
    }


def _source_signature(path: Path) -> dict[str, int]:
    stats = path.stat()
    return {
        "size": int(stats.st_size),
        "mtime_ns": int(stats.st_mtime_ns),
    }


def _same_signature(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return int(left.get("size", -1)) == int(right.get("size", -1)) and int(
        left.get("mtime_ns", -1)
    ) == int(right.get("mtime_ns", -1))


def _load_index(index_path: Path) -> dict[str, dict[str, Any]]:
    if not index_path.exists():
        return {}
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    cleaned: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, dict):
            cleaned[key] = value
    return cleaned


def _save_index(index_path: Path, index_data: dict[str, dict[str, Any]]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index_data, indent=2, ensure_ascii=False), encoding="utf-8")


def _utc_now_text() -> str:
    return datetime.now(UTC).isoformat()
