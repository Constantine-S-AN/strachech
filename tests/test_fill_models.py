from __future__ import annotations

import pandas as pd
import pytest
from stratcheck.core.backtest import FixedBpsCostModel
from stratcheck.core.strategy import OrderIntent, PortfolioState
from stratcheck.sim.engine import BacktestEngineV2
from stratcheck.sim.fill_models import SpreadAwareFillModel, VolumeParticipationFillModel


class OneShotMarketBuyStrategy:
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


def test_spread_aware_fill_model_produces_higher_buy_price_than_bar_fill_model() -> None:
    timestamps = pd.date_range("2024-04-01", periods=3, freq="D", tz="UTC")
    bars = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0],
            "high": [101.0, 102.0, 101.0],
            "low": [99.0, 98.0, 99.0],
            "close": [100.0, 100.0, 100.0],
            "volume": [1_000.0, 1_000.0, 1_000.0],
        },
        index=timestamps,
    )
    strategy = OneShotMarketBuyStrategy(qty=1.0)
    zero_cost_model = FixedBpsCostModel(commission_bps=0.0, slippage_bps=0.0)

    bar_result = BacktestEngineV2(max_volume_share=1.0).run(
        strategy=strategy,
        bars=bars,
        initial_cash=10_000.0,
        cost_model=zero_cost_model,
    )
    spread_result = BacktestEngineV2(
        max_volume_share=1.0,
        fill_model=SpreadAwareFillModel(
            spread_bps=100.0,
            use_bar_range=False,
        ),
    ).run(
        strategy=strategy,
        bars=bars,
        initial_cash=10_000.0,
        cost_model=zero_cost_model,
    )

    assert len(bar_result.trades) == 1
    assert len(spread_result.trades) == 1
    assert spread_result.trades[0].price > bar_result.trades[0].price
    assert spread_result.trades[0].price == pytest.approx(100.5, rel=1e-9)


def test_volume_participation_fill_model_limits_total_executed_quantity() -> None:
    timestamps = pd.date_range("2024-05-01", periods=4, freq="D", tz="UTC")
    bars = pd.DataFrame(
        {
            "open": [100.0, 100.0, 101.0, 102.0],
            "high": [101.0, 101.0, 102.0, 103.0],
            "low": [99.0, 99.0, 100.0, 101.0],
            "close": [100.0, 100.5, 101.5, 102.5],
            "volume": [10_000.0, 10.0, 10.0, 10.0],
        },
        index=timestamps,
    )
    strategy = OneShotMarketBuyStrategy(qty=9.0)
    zero_cost_model = FixedBpsCostModel(commission_bps=0.0, slippage_bps=0.0)

    bar_result = BacktestEngineV2(max_volume_share=1.0).run(
        strategy=strategy,
        bars=bars,
        initial_cash=100_000.0,
        cost_model=zero_cost_model,
    )
    participation_result = BacktestEngineV2(
        max_volume_share=1.0,
        fill_model=VolumeParticipationFillModel(participation_rate=0.2),
    ).run(
        strategy=strategy,
        bars=bars,
        initial_cash=100_000.0,
        cost_model=zero_cost_model,
    )

    bar_filled_qty = sum(trade.qty for trade in bar_result.trades)
    participation_filled_qty = sum(trade.qty for trade in participation_result.trades)

    assert bar_filled_qty == pytest.approx(9.0, rel=1e-9)
    assert participation_filled_qty == pytest.approx(6.0, rel=1e-9)
    assert participation_filled_qty < bar_filled_qty
    assert bar_result.orders[0].filled is True
    assert participation_result.orders[0].filled is False
