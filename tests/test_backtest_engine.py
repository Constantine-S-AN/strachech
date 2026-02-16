from __future__ import annotations

import pandas as pd
import pytest
from stratcheck.core.backtest import BacktestEngine, FixedBpsCostModel
from stratcheck.core.strategy import OrderIntent, PortfolioState


class FixedSignalStrategy:
    def generate_orders(
        self,
        bars: pd.DataFrame,
        portfolio_state: PortfolioState,
    ) -> list[OrderIntent]:
        del portfolio_state
        if len(bars) == 1:
            return [OrderIntent(side="buy", qty=1.0, market=True)]
        if len(bars) == 2:
            return [OrderIntent(side="sell", qty=1.0, market=True)]
        return []


def test_backtest_engine_market_order_fill_and_equity_calculation() -> None:
    timestamps = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    bars = pd.DataFrame(
        {
            "open": [100.0, 110.0, 120.0],
            "high": [101.0, 113.0, 121.0],
            "low": [99.0, 109.0, 117.0],
            "close": [100.0, 112.0, 118.0],
            "volume": [1_000, 1_100, 1_200],
        },
        index=timestamps,
    )
    strategy = FixedSignalStrategy()
    engine = BacktestEngine()

    result = engine.run(
        strategy=strategy,
        bars=bars,
        initial_cash=1000.0,
        cost_model=FixedBpsCostModel(commission_bps=10.0, slippage_bps=20.0),
    )

    assert len(result.orders) == 2
    assert len(result.trades) == 2
    assert result.orders[0].created_at == timestamps[0]
    assert result.orders[0].fill_time == timestamps[1]
    assert result.orders[1].created_at == timestamps[1]
    assert result.orders[1].fill_time == timestamps[2]

    assert result.trades[0].side == "buy"
    assert result.trades[0].price == pytest.approx(110.22, rel=1e-9)
    assert result.trades[0].fee == pytest.approx(0.11022, rel=1e-9)
    assert result.trades[0].cost == pytest.approx(0.33022, rel=1e-9)

    assert result.trades[1].side == "sell"
    assert result.trades[1].price == pytest.approx(119.76, rel=1e-9)
    assert result.trades[1].fee == pytest.approx(0.11976, rel=1e-9)
    assert result.trades[1].cost == pytest.approx(0.35976, rel=1e-9)

    assert result.positions.loc[timestamps[0]] == pytest.approx(0.0, rel=1e-9)
    assert result.positions.loc[timestamps[1]] == pytest.approx(1.0, rel=1e-9)
    assert result.positions.loc[timestamps[2]] == pytest.approx(0.0, rel=1e-9)

    assert result.equity_curve.loc[timestamps[0]] == pytest.approx(1000.0, rel=1e-9)
    assert result.equity_curve.loc[timestamps[1]] == pytest.approx(1001.66978, rel=1e-9)
    assert result.equity_curve.loc[timestamps[2]] == pytest.approx(1009.31002, rel=1e-9)
