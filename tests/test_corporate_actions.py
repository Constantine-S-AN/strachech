from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from stratcheck.core.corporate_actions import (
    CorporateAction,
    apply_corporate_actions_to_bars,
)
from stratcheck.core.data import CSVDataProvider


def test_apply_split_adjusts_historical_price_and_volume_consistently() -> None:
    timestamps = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    bars = pd.DataFrame(
        {
            "open": [198.0, 100.0, 101.0],
            "high": [202.0, 101.0, 103.0],
            "low": [197.0, 98.0, 99.0],
            "close": [200.0, 100.0, 102.0],
            "volume": [1_000.0, 2_000.0, 1_800.0],
        },
        index=timestamps,
    )
    split_actions = [
        CorporateAction(
            timestamp=timestamps[1],
            action_type="split",
            value=2.0,
        )
    ]

    adjusted_bars = apply_corporate_actions_to_bars(
        bars=bars,
        actions=split_actions,
    )

    assert adjusted_bars.loc[timestamps[0], "close"] == pytest.approx(100.0, rel=1e-9)
    assert adjusted_bars.loc[timestamps[0], "volume"] == pytest.approx(2_000.0, rel=1e-9)
    assert adjusted_bars.loc[timestamps[1], "close"] == pytest.approx(100.0, rel=1e-9)
    assert adjusted_bars.loc[timestamps[2], "close"] == pytest.approx(102.0, rel=1e-9)

    original_notional = bars.loc[timestamps[0], "close"] * bars.loc[timestamps[0], "volume"]
    adjusted_notional = (
        adjusted_bars.loc[timestamps[0], "close"] * adjusted_bars.loc[timestamps[0], "volume"]
    )
    assert adjusted_notional == pytest.approx(original_notional, rel=1e-9)


def test_csv_data_provider_reads_actions_and_applies_split_adjustment(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    actions_dir = data_dir / "actions"
    data_dir.mkdir(parents=True, exist_ok=True)
    actions_dir.mkdir(parents=True, exist_ok=True)

    bars_csv = data_dir / "QQQ.csv"
    bars_csv.write_text(
        "\n".join(
            [
                "timestamp,open,high,low,close,volume",
                "2024-01-01,198,202,197,200,1000",
                "2024-01-02,100,101,98,100,2000",
                "2024-01-03,101,103,99,102,1800",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    actions_csv = actions_dir / "QQQ.csv"
    actions_csv.write_text(
        "\n".join(
            [
                "date,type,value,note",
                "2024-01-02,split,2,2-for-1 split",
                "2024-01-03,dividend,0.8,regular cash",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    provider = CSVDataProvider(data_dir=data_dir)
    bars = provider.get_bars(symbol="QQQ", timeframe="1d")
    actions_summary = provider.get_corporate_actions_summary(symbol="QQQ")

    first_timestamp = pd.Timestamp("2024-01-01", tz="UTC")
    assert bars.loc[first_timestamp, "close"] == pytest.approx(100.0, rel=1e-9)
    assert bars.loc[first_timestamp, "volume"] == pytest.approx(2_000.0, rel=1e-9)

    assert len(actions_summary) == 2
    assert actions_summary[0]["type"] == "split"
    assert actions_summary[0]["date"] == "2024-01-02"
    assert "ratio=2" in actions_summary[0]["details"]
