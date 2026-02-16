from __future__ import annotations

from pathlib import Path

import pandas as pd
from stratcheck.core.data import BarsSchema, CSVDataProvider

FIXTURE_DATA_DIR = Path(__file__).parent / "fixtures" / "data"


def test_csv_data_provider_returns_standardized_bars() -> None:
    provider = CSVDataProvider(data_dir=FIXTURE_DATA_DIR)

    bars = provider.get_bars(
        symbol="BTCUSDT",
        start="2024-01-01 00:00:00",
        end="2024-01-03 00:00:00",
        timeframe="1d",
    )

    assert list(bars.columns) == list(BarsSchema.COLUMNS)
    assert isinstance(bars.index, pd.DatetimeIndex)
    assert bars.index.name == BarsSchema.INDEX_NAME
    assert bars.index.tz is not None
    assert str(bars.index.tz) == "UTC"
    assert bars.index.is_monotonic_increasing
    assert len(bars) == 3

    duplicate_timestamp = pd.Timestamp("2024-01-02 00:00:00", tz="UTC")
    assert float(bars.loc[duplicate_timestamp, "close"]) == 103.0
