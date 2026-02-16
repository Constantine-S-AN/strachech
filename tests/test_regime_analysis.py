from __future__ import annotations

import numpy as np
import pandas as pd
from stratcheck.analysis import classify_market_regimes, compute_regime_scorecard
from stratcheck.core.strategy import Fill


def test_classify_market_regimes_assigns_expected_labels() -> None:
    bars = _build_regime_bars()
    regime_frame = classify_market_regimes(
        bars=bars,
        vol_window=12,
        trend_window=12,
        trend_threshold=0.015,
    )

    assert "regime" in regime_frame.columns
    assert len(regime_frame) == len(bars)
    assert regime_frame["regime"].nunique() >= 2
    assert regime_frame["regime"].nunique() <= 5


def test_compute_regime_scorecard_returns_metrics_table() -> None:
    bars = _build_regime_bars()
    equity_curve = pd.Series(
        100_000.0 * (1.0 + bars["close"].pct_change().fillna(0.0)).cumprod(),
        index=bars.index,
        name="equity",
    )
    trades = [
        Fill(
            side="buy",
            qty=1.0,
            price=float(bars["close"].iloc[20]),
            timestamp=bars.index[20],
        ),
        Fill(
            side="sell",
            qty=1.0,
            price=float(bars["close"].iloc[75]),
            timestamp=bars.index[75],
        ),
    ]

    scorecard = compute_regime_scorecard(
        bars=bars,
        equity_curve=equity_curve,
        trades=trades,
        vol_window=12,
        trend_window=12,
        trend_threshold=0.015,
    )

    assert not scorecard.empty
    assert {
        "regime",
        "bars_count",
        "mean_return",
        "total_return",
        "max_drawdown",
        "win_rate",
        "trade_count",
    }.issubset(scorecard.columns)
    assert scorecard["bars_count"].ge(1).all()
    assert scorecard["win_rate"].between(0.0, 1.0).all()


def _build_regime_bars() -> pd.DataFrame:
    total_periods = 120
    timestamps = pd.date_range("2023-01-01", periods=total_periods, freq="D", tz="UTC")
    random_generator = np.random.default_rng(23)

    uptrend = np.linspace(100.0, 125.0, total_periods // 2)
    downtrend = np.linspace(125.0, 95.0, total_periods // 2)
    close_prices = np.concatenate([uptrend, downtrend])
    close_prices = close_prices + random_generator.normal(0.0, 1.2, size=total_periods)

    bars = pd.DataFrame(
        {
            "open": close_prices - 0.4,
            "high": close_prices + 1.4,
            "low": close_prices - 1.4,
            "close": close_prices,
            "volume": np.full(total_periods, 8_000),
        },
        index=timestamps,
    )
    return bars
