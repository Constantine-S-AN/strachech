from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from stratcheck.core.data import BarsSchema
from stratcheck.core.data_sources import StooqCSVProvider

SAMPLE_STOOQ_CSV = """Date,Open,High,Low,Close,Volume
2024-01-03,101,103,100,102,1000
2024-01-01,100,101,99,100,900
2024-01-02,100.5,102,100,101,950
"""

INVALID_STOOQ_CSV = """Date,Open,High,Low,Close
2024-01-01,100,101,99,100
"""


def test_stooq_provider_downloads_on_cache_miss_and_reuses_cache(tmp_path: Path) -> None:
    provider = StooqCSVProvider(cache_dir=tmp_path / "cache", auto_download=True)
    download_calls = {"count": 0}

    def fake_download(source_symbol: str, interval_code: str) -> str:
        del source_symbol, interval_code
        download_calls["count"] += 1
        return SAMPLE_STOOQ_CSV

    provider._download_csv_text = fake_download

    first_bars = provider.get_bars(symbol="QQQ", timeframe="1d")
    second_bars = provider.get_bars(symbol="QQQ", timeframe="1d")

    assert download_calls["count"] == 1
    assert list(first_bars.columns) == list(BarsSchema.COLUMNS)
    assert first_bars.index.is_monotonic_increasing
    assert len(first_bars) == 3
    assert second_bars.equals(first_bars)


def test_stooq_provider_raises_on_cache_miss_when_auto_download_disabled(tmp_path: Path) -> None:
    provider = StooqCSVProvider(cache_dir=tmp_path / "cache", auto_download=False)
    with pytest.raises(FileNotFoundError):
        provider.get_bars(symbol="QQQ", timeframe="1d")


def test_stooq_provider_validates_bars_schema_from_cache(tmp_path: Path) -> None:
    provider = StooqCSVProvider(cache_dir=tmp_path / "cache", auto_download=False)
    cache_file = provider.cache_dir / "qqq_us_d.csv"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(INVALID_STOOQ_CSV, encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        provider.get_bars(symbol="QQQ", timeframe="1d")


def test_stooq_provider_respects_time_range_filter(tmp_path: Path) -> None:
    provider = StooqCSVProvider(cache_dir=tmp_path / "cache", auto_download=True)
    provider._download_csv_text = lambda source_symbol, interval_code: SAMPLE_STOOQ_CSV

    bars = provider.get_bars(
        symbol="QQQ",
        start="2024-01-02",
        end="2024-01-02",
        timeframe="1d",
    )

    assert isinstance(bars.index, pd.DatetimeIndex)
    assert len(bars) == 1
    assert bars.index[0] == pd.Timestamp("2024-01-02", tz="UTC")


def test_stooq_provider_raises_when_price_adjustment_is_requested(tmp_path: Path) -> None:
    provider = StooqCSVProvider(
        cache_dir=tmp_path / "cache",
        auto_download=True,
        adjust_prices=True,
    )

    with pytest.raises(NotImplementedError, match="not supported"):
        provider.get_bars(symbol="QQQ", timeframe="1d")
