from __future__ import annotations

import pandas as pd
import pytest
from stratcheck.analysis import compute_execution_quality_scorecard
from stratcheck.core.backtest import OrderRecord
from stratcheck.core.strategy import Fill


def test_compute_execution_quality_scorecard_returns_expected_metrics() -> None:
    timestamps = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    bars = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.0, 101.0, 102.0, 103.0],
            "volume": [1000.0, 1000.0, 1000.0, 1000.0],
        },
        index=timestamps,
    )

    orders = [
        OrderRecord(
            created_at=timestamps[0],
            side="buy",
            qty=1.0,
            limit_price=None,
            market=True,
            filled=True,
            fill_time=timestamps[1],
            fill_price=101.5,
        ),
        OrderRecord(
            created_at=timestamps[1],
            side="sell",
            qty=2.0,
            limit_price=None,
            market=True,
            filled=True,
            fill_time=timestamps[2],
            fill_price=100.5,
        ),
        OrderRecord(
            created_at=timestamps[2],
            side="buy",
            qty=1.0,
            limit_price=None,
            market=True,
            filled=False,
            fill_time=None,
            fill_price=None,
        ),
    ]
    trades = [
        Fill(side="buy", qty=1.0, price=101.5, timestamp=timestamps[1]),
        Fill(side="sell", qty=1.0, price=100.5, timestamp=timestamps[2]),
    ]

    scorecard = compute_execution_quality_scorecard(
        orders=orders,
        bars=bars,
        trades=trades,
    )

    assert len(scorecard) == 1
    row = scorecard.iloc[0]
    assert int(row["orders_total"]) == 3
    assert int(row["filled_orders"]) == 2
    assert int(row["canceled_orders"]) == 1
    assert float(row["cancel_rate"]) == pytest.approx(1.0 / 3.0, rel=1e-6)
    assert int(row["partially_filled_orders"]) == 1
    assert float(row["partial_fill_ratio"]) == pytest.approx(1.0 / 3.0, rel=1e-6)
    assert float(row["avg_slippage_bps"]) == pytest.approx(99.752475, rel=1e-6)
    assert float(row["median_slippage_bps"]) == pytest.approx(99.752475, rel=1e-6)
    assert float(row["avg_latency_seconds"]) == pytest.approx(86400.0)
    assert float(row["median_latency_seconds"]) == pytest.approx(86400.0)
    assert float(row["avg_latency_bars"]) == pytest.approx(1.0)
    assert float(row["median_latency_bars"]) == pytest.approx(1.0)


def test_compute_execution_quality_scorecard_handles_empty_orders() -> None:
    timestamps = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")
    bars = pd.DataFrame(
        {"close": [100.0, 101.0]},
        index=timestamps,
    )

    scorecard = compute_execution_quality_scorecard(
        orders=[],
        bars=bars,
        trades=[],
    )

    assert len(scorecard) == 1
    row = scorecard.iloc[0]
    assert int(row["orders_total"]) == 0
    assert float(row["cancel_rate"]) == 0.0
    assert float(row["partial_fill_ratio"]) == 0.0
    assert float(row["avg_slippage_bps"]) == 0.0
    assert float(row["avg_latency_seconds"]) == 0.0


def test_compute_execution_quality_scorecard_requires_signal_price_column() -> None:
    bars = pd.DataFrame(
        {"open": [100.0, 101.0]},
        index=pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"),
    )

    with pytest.raises(ValueError, match="signal price column"):
        compute_execution_quality_scorecard(orders=[], bars=bars)
