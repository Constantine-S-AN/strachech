from __future__ import annotations

import pandas as pd
import pytest
from stratcheck.core.metrics import compute_metrics
from stratcheck.core.strategy import Fill


def _sample_equity_curve() -> pd.Series:
    timestamps = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    return pd.Series([100.0, 100.1, 100.0, 100.2], index=timestamps, dtype=float)


def test_compute_metrics_calculates_cagr_volatility_sharpe_and_drawdown() -> None:
    metrics = compute_metrics(
        equity_curve=_sample_equity_curve(),
        trades=[],
        bars_freq="1D",
    )

    assert metrics["cagr"] == pytest.approx(0.1827381585264416, rel=1e-9)
    assert metrics["annual_volatility"] == pytest.approx(0.017742920728189525, rel=1e-9)
    assert metrics["sharpe"] == pytest.approx(10.299215181416644, rel=1e-9)
    assert metrics["max_drawdown"] == pytest.approx(-0.0009990009990009652, rel=1e-9)


def test_compute_metrics_calculates_turnover_win_rate_and_avg_trade_pnl() -> None:
    timestamps = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    trades = [
        Fill(side="buy", qty=1.0, price=100.0, timestamp=timestamps[0]),
        Fill(side="sell", qty=1.0, price=110.0, timestamp=timestamps[1]),
        Fill(side="buy", qty=1.0, price=120.0, timestamp=timestamps[2]),
        Fill(side="sell", qty=1.0, price=110.0, timestamp=timestamps[3]),
    ]

    metrics = compute_metrics(
        equity_curve=_sample_equity_curve(),
        trades=trades,
        bars_freq="1D",
    )

    assert metrics["turnover"] == pytest.approx(4.396702473145141, rel=1e-9)
    assert metrics["win_rate"] == pytest.approx(0.5, rel=1e-9)
    assert metrics["avg_trade_pnl"] == pytest.approx(0.0, abs=1e-12)
