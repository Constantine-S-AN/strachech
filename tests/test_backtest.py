from __future__ import annotations

from stratcheck.core import generate_random_ohlcv, run_moving_average_backtest


def test_generate_random_ohlcv_has_expected_shape() -> None:
    market_data = generate_random_ohlcv(periods=120, seed=7)

    assert list(market_data.columns) == ["open", "high", "low", "close", "volume"]
    assert len(market_data) == 120
    assert (market_data["high"] >= market_data[["open", "close"]].max(axis=1)).all()
    assert (market_data["low"] <= market_data[["open", "close"]].min(axis=1)).all()


def test_backtest_returns_metrics() -> None:
    market_data = generate_random_ohlcv(periods=250, seed=13)
    result = run_moving_average_backtest(market_data=market_data)

    expected_metric_keys = {
        "total_return",
        "annual_return",
        "annual_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "win_rate",
        "trade_entries",
    }
    assert expected_metric_keys.issubset(result.metrics.keys())
    assert len(result.equity_curve) == 250
