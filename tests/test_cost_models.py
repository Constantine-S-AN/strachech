from __future__ import annotations

import pandas as pd
import pytest
from stratcheck.core.backtest import (
    BacktestEngine,
    FixedBpsCostModel,
    MarketImpactToyModel,
    SpreadCostModel,
    build_cost_model,
)
from stratcheck.core.strategy import OrderIntent, PortfolioState


class BuyOnceStrategy:
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


class BuySellStrategy:
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
        if len(bars) == 2:
            return [OrderIntent(side="sell", qty=self.qty, market=True)]
        return []


def test_spread_cost_model_uses_high_low_range() -> None:
    bars = _sample_bars()
    engine = BacktestEngine()
    strategy = BuyOnceStrategy(qty=1.0)

    result = engine.run(
        strategy=strategy,
        bars=bars,
        initial_cash=10_000.0,
        cost_model=SpreadCostModel(
            commission_bps=0.0,
            use_bar_range=True,
            range_multiplier=0.10,
        ),
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.price == pytest.approx(101.0, rel=1e-9)
    assert trade.cost == pytest.approx(1.0, rel=1e-9)


def test_market_impact_toy_model_cost_increases_with_order_size() -> None:
    bars = _sample_bars()
    engine = BacktestEngine()
    cost_model = MarketImpactToyModel(
        commission_bps=0.0,
        base_slippage_bps=0.0,
        impact_factor=0.1,
    )

    small_result = engine.run(
        strategy=BuyOnceStrategy(qty=1.0),
        bars=bars,
        initial_cash=10_000.0,
        cost_model=cost_model,
    )
    large_result = engine.run(
        strategy=BuyOnceStrategy(qty=20.0),
        bars=bars,
        initial_cash=100_000.0,
        cost_model=cost_model,
    )

    small_trade = small_result.trades[0]
    large_trade = large_result.trades[0]

    assert large_trade.price > small_trade.price
    assert large_trade.cost > small_trade.cost


def test_fixed_bps_cost_model_reduces_equity_vs_zero_cost() -> None:
    bars = _sample_bars()
    strategy = BuySellStrategy(qty=1.0)
    engine = BacktestEngine()

    no_cost_result = engine.run(
        strategy=strategy,
        bars=bars,
        initial_cash=1_000.0,
        cost_model=FixedBpsCostModel(commission_bps=0.0, slippage_bps=0.0),
    )
    bps_cost_result = engine.run(
        strategy=strategy,
        bars=bars,
        initial_cash=1_000.0,
        cost_model=FixedBpsCostModel(commission_bps=10.0, slippage_bps=20.0),
    )

    assert bps_cost_result.equity_curve.iloc[-1] < no_cost_result.equity_curve.iloc[-1]
    assert sum(trade.cost for trade in bps_cost_result.trades) > 0.0


def test_build_cost_model_dispatches_types() -> None:
    fixed = build_cost_model({"commission_bps": 5, "slippage_bps": 3})
    spread = build_cost_model({"type": "spread", "spread_bps": 10})
    impact = build_cost_model({"type": "market_impact", "impact_factor": 0.2})

    assert isinstance(fixed, FixedBpsCostModel)
    assert isinstance(spread, SpreadCostModel)
    assert isinstance(impact, MarketImpactToyModel)


def _sample_bars() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "open": [100.0, 100.0, 102.0],
            "high": [101.0, 110.0, 103.0],
            "low": [99.0, 90.0, 101.0],
            "close": [100.0, 101.0, 102.0],
            "volume": [1_000.0, 100.0, 1_000.0],
        },
        index=timestamps,
    )
