from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from stratcheck.core.data import BarsSchema, CSVDataProvider
from stratcheck.perf import parquet_engine_available


def test_csv_provider_parquet_cache_tracks_miss_and_hit(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_sample_csv(data_dir=data_dir, symbol="BTCUSDT")

    cache_dir = tmp_path / "cache" / "parquet"
    index_path = cache_dir / "index.json"
    provider = CSVDataProvider(
        data_dir=data_dir,
        use_parquet_cache=True,
        cache_dir=cache_dir,
        cache_index_path=index_path,
    )

    bars_first = provider.get_bars(symbol="BTCUSDT", timeframe="1d")
    bars_second = provider.get_bars(symbol="BTCUSDT", timeframe="1d")
    assert list(bars_first.columns) == list(BarsSchema.COLUMNS)
    assert bars_first.equals(bars_second)

    assert index_path.exists()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert "BTCUSDT" in payload
    meta = provider.get_cache_meta("BTCUSDT")
    assert meta is not None

    if parquet_engine_available():
        cache_file = cache_dir / "BTCUSDT.parquet"
        assert cache_file.exists()
        assert meta["cache_used"] is True
        assert meta["cache_hit"] is True
        assert meta["status"] == "hit"
        assert int(payload["BTCUSDT"].get("hit_count", 0)) >= 1
    else:
        assert meta["cache_used"] is False
        assert meta["status"] == "engine_unavailable"


@pytest.mark.skipif(not parquet_engine_available(), reason="parquet engine unavailable")
def test_csv_provider_reads_direct_symbol_parquet(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = data_dir / "BTCUSDT.parquet"

    timestamps = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps.astype(str),
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [1_000, 1_100, 1_200, 1_300],
        }
    )
    frame.to_parquet(parquet_path, index=False)

    provider = CSVDataProvider(data_dir=data_dir, use_parquet_cache=True)
    bars = provider.get_bars(symbol="BTCUSDT", timeframe="1d")

    assert list(bars.columns) == list(BarsSchema.COLUMNS)
    assert len(bars) == 4
    cache_meta = provider.get_cache_meta("BTCUSDT")
    assert cache_meta is not None
    assert cache_meta["status"] == "direct_parquet"


def _write_sample_csv(data_dir: Path, symbol: str) -> None:
    timestamps = pd.date_range("2024-01-01", periods=20, freq="D", tz="UTC")
    close_prices = np.linspace(100.0, 120.0, len(timestamps))
    bars = pd.DataFrame(
        {
            "timestamp": timestamps.astype(str),
            "open": close_prices - 0.1,
            "high": close_prices + 1.0,
            "low": close_prices - 1.0,
            "close": close_prices,
            "volume": np.full(len(timestamps), 1_000),
        }
    )
    bars.to_csv(data_dir / f"{symbol}.csv", index=False)
