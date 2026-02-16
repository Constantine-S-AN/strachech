from __future__ import annotations

import pandas as pd
import pytest
from stratcheck.core.backtest import FixedBpsCostModel
from stratcheck.core.strategy import OrderIntent, PortfolioState
from stratcheck.sim.engine import BacktestEngineV2


class SingleLimitBuyStrategy:
    def __init__(self, limit_price: float, qty: float) -> None:
        self.limit_price = limit_price
        self.qty = qty

    def generate_orders(
        self,
        bars: pd.DataFrame,
        portfolio_state: PortfolioState,
    ) -> list[OrderIntent]:
        del portfolio_state
        if len(bars) == 1:
            return [
                OrderIntent(
                    side="buy",
                    qty=self.qty,
                    market=False,
                    limit_price=self.limit_price,
                )
            ]
        return []


class SingleMarketBuyStrategy:
    def __init__(self, qty: float) -> None:
        self.qty = qty

    def generate_orders(
        self,
        bars: pd.DataFrame,
        portfolio_state: PortfolioState,
    ) -> list[OrderIntent]:
        del portfolio_state
        if len(bars) == 1:
            return [OrderIntent(side="buy", qty=self.qty, market=True)]
        return []


def test_backtest_engine_v2_limit_order_fills_when_price_touches() -> None:
    timestamps = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    bars = pd.DataFrame(
        {
            "open": [100.0, 102.0, 101.0, 100.0],
            "high": [101.0, 103.0, 102.0, 101.0],
            "low": [99.0, 100.0, 98.0, 99.0],
            "close": [100.0, 101.0, 99.5, 100.5],
            "volume": [10_000.0, 10_000.0, 10_000.0, 10_000.0],
        },
        index=timestamps,
    )
    strategy = SingleLimitBuyStrategy(limit_price=99.0, qty=1.0)
    engine = BacktestEngineV2(
        max_volume_share=1.0,
        default_time_in_force="GTC",
    )

    result = engine.run(
        strategy=strategy,
        bars=bars,
        initial_cash=10_000.0,
        cost_model=FixedBpsCostModel(commission_bps=0.0, slippage_bps=0.0),
    )

    assert len(result.orders) == 1
    assert len(result.trades) == 1
    assert result.trades[0].timestamp == timestamps[2]
    assert result.trades[0].price == pytest.approx(99.0, rel=1e-9)
    assert result.trades[0].qty == pytest.approx(1.0, rel=1e-9)

    order = result.orders[0]
    assert order.filled is True
    assert order.fill_time == timestamps[2]
    assert order.fill_price == pytest.approx(99.0, rel=1e-9)


def test_backtest_engine_v2_market_order_partial_fills_across_bars() -> None:
    timestamps = pd.date_range("2024-02-01", periods=4, freq="D", tz="UTC")
    bars = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [1_000.0, 10.0, 10.0, 40.0],
        },
        index=timestamps,
    )
    strategy = SingleMarketBuyStrategy(qty=12.0)
    engine = BacktestEngineV2(
        max_volume_share=0.25,
        default_time_in_force="GTC",
    )

    result = engine.run(
        strategy=strategy,
        bars=bars,
        initial_cash=1_000_000.0,
        cost_model=FixedBpsCostModel(commission_bps=0.0, slippage_bps=0.0),
    )

    assert len(result.orders) == 1
    assert len(result.trades) == 3
    assert [trade.timestamp for trade in result.trades] == [
        timestamps[1],
        timestamps[2],
        timestamps[3],
    ]
    assert [trade.qty for trade in result.trades] == pytest.approx([2.5, 2.5, 7.0], rel=1e-9)
    assert sum(trade.qty for trade in result.trades) == pytest.approx(12.0, rel=1e-9)

    order = result.orders[0]
    expected_average_price = (2.5 * 101.0 + 2.5 * 102.0 + 7.0 * 103.0) / 12.0
    assert order.filled is True
    assert order.fill_time == timestamps[3]
    assert order.fill_price == pytest.approx(expected_average_price, rel=1e-9)


def test_backtest_engine_v2_day_limit_order_expires() -> None:
    timestamps = pd.date_range("2024-03-01", periods=4, freq="D", tz="UTC")
    bars = pd.DataFrame(
        {
            "open": [100.0, 101.0, 101.0, 101.0],
            "high": [101.0, 102.0, 102.0, 102.0],
            "low": [99.5, 100.5, 98.0, 97.5],
            "close": [100.0, 101.0, 100.0, 99.0],
            "volume": [1_000.0, 1_000.0, 1_000.0, 1_000.0],
        },
        index=timestamps,
    )
    strategy = SingleLimitBuyStrategy(limit_price=99.0, qty=1.0)
    engine = BacktestEngineV2(
        max_volume_share=1.0,
        default_time_in_force="DAY",
    )

    result = engine.run(
        strategy=strategy,
        bars=bars,
        initial_cash=10_000.0,
        cost_model=FixedBpsCostModel(commission_bps=0.0, slippage_bps=0.0),
    )

    assert len(result.orders) == 1
    assert result.orders[0].filled is False
    assert result.orders[0].fill_time is None
    assert len(result.trades) == 0
