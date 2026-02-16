from __future__ import annotations

import pandas as pd
from stratcheck.core.backtest import BacktestEngine, FixedBpsCostModel
from stratcheck.core.data import generate_random_ohlcv
from stratcheck.core.strategy import MovingAverageCrossStrategy, OrderIntent, PortfolioState
from stratcheck.strategies import BuyAndHoldStrategy
from stratcheck.validation import validate_against_vectorized_baseline


class PassiveStrategy:
    def generate_orders(
        self,
        bars: pd.DataFrame,
        portfolio_state: PortfolioState,
    ) -> list[OrderIntent]:
        del bars, portfolio_state
        return []


def test_validation_buy_and_hold_matches_vectorized_reference() -> None:
    bars = generate_random_ohlcv(periods=160, seed=11)
    strategy = BuyAndHoldStrategy(target_position_qty=1.0)
    cost_model = FixedBpsCostModel(commission_bps=5.0, slippage_bps=3.0)
    engine = BacktestEngine()
    result = engine.run(
        strategy=strategy,
        bars=bars,
        initial_cash=100_000.0,
        cost_model=cost_model,
    )

    summary = validate_against_vectorized_baseline(
        strategy=strategy,
        bars=bars,
        engine_equity_curve=result.equity_curve,
        initial_cash=100_000.0,
        cost_model=cost_model,
        tolerance_abs=1e-6,
    )

    assert len(summary) == 1
    assert summary[0]["status"] == "pass"
    assert float(summary[0]["max_abs_error"]) <= 1e-6


def test_validation_moving_average_cross_matches_vectorized_reference() -> None:
    bars = generate_random_ohlcv(periods=220, seed=17)
    strategy = MovingAverageCrossStrategy(short_window=10, long_window=30, target_position_qty=1.0)
    cost_model = FixedBpsCostModel(commission_bps=4.0, slippage_bps=2.0)
    engine = BacktestEngine()
    result = engine.run(
        strategy=strategy,
        bars=bars,
        initial_cash=100_000.0,
        cost_model=cost_model,
    )

    summary = validate_against_vectorized_baseline(
        strategy=strategy,
        bars=bars,
        engine_equity_curve=result.equity_curve,
        initial_cash=100_000.0,
        cost_model=cost_model,
        tolerance_abs=1e-6,
    )

    assert len(summary) == 1
    assert summary[0]["status"] == "pass"
    assert float(summary[0]["max_abs_error"]) <= 1e-6


def test_validation_unknown_strategy_returns_skipped() -> None:
    bars = generate_random_ohlcv(periods=90, seed=3)
    strategy = PassiveStrategy()
    cost_model = FixedBpsCostModel(commission_bps=0.0, slippage_bps=0.0)
    engine = BacktestEngine()
    result = engine.run(
        strategy=strategy,
        bars=bars,
        initial_cash=100_000.0,
        cost_model=cost_model,
    )

    summary = validate_against_vectorized_baseline(
        strategy=strategy,
        bars=bars,
        engine_equity_curve=result.equity_curve,
        initial_cash=100_000.0,
        cost_model=cost_model,
    )

    assert len(summary) == 1
    assert summary[0]["status"] == "skipped"
