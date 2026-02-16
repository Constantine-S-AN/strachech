from __future__ import annotations

import pandas as pd
from stratcheck.core.strategy import PortfolioState
from stratcheck.strategies import (
    BuyAndHoldStrategy,
    MeanReversionZScoreStrategy,
    VolatilityTargetStrategy,
)


def _bars_from_closes(close_values: list[float]) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=len(close_values), freq="D", tz="UTC")
    return pd.DataFrame({"close": close_values}, index=timestamps)


def test_buy_and_hold_strategy_buys_target_quantity_from_flat() -> None:
    bars = _bars_from_closes([100.0, 101.0, 102.0])
    strategy = BuyAndHoldStrategy(target_position_qty=2.0)
    portfolio_state = PortfolioState(cash=100_000.0, position_qty=0.0)

    intents = strategy.generate_orders(bars=bars, portfolio_state=portfolio_state)

    assert len(intents) == 1
    assert intents[0].side == "buy"
    assert intents[0].qty == 2.0


def test_volatility_target_strategy_reduces_position_when_volatility_is_high() -> None:
    bars = _bars_from_closes(
        [
            100.0,
            90.0,
            99.0,
            89.1,
            98.01,
            88.209,
            97.0299,
            87.32691,
            96.059601,
            86.4536409,
            95.09900499,
            85.589104491,
        ]
    )
    strategy = VolatilityTargetStrategy(
        target_volatility=0.2,
        lookback=10,
        base_position_qty=1.0,
        max_leverage=2.0,
        bars_freq="1d",
        min_volatility=0.0001,
        rebalance_threshold=0.01,
    )
    portfolio_state = PortfolioState(cash=100_000.0, position_qty=1.0)

    intents = strategy.generate_orders(bars=bars, portfolio_state=portfolio_state)

    assert len(intents) == 1
    assert intents[0].side == "sell"
    assert intents[0].qty > 0.5


def test_mean_reversion_zscore_strategy_emits_buy_and_exit_signals() -> None:
    strategy = MeanReversionZScoreStrategy(
        lookback=20,
        entry_z=1.0,
        exit_z=0.0,
        target_position_qty=1.0,
    )

    entry_bars = _bars_from_closes([100.0] * 19 + [90.0])
    flat_state = PortfolioState(cash=100_000.0, position_qty=0.0)
    entry_intents = strategy.generate_orders(bars=entry_bars, portfolio_state=flat_state)

    assert len(entry_intents) == 1
    assert entry_intents[0].side == "buy"
    assert entry_intents[0].qty == 1.0

    exit_bars = _bars_from_closes([100.0] * 18 + [90.0, 100.0])
    long_state = PortfolioState(cash=90_000.0, position_qty=1.0)
    exit_intents = strategy.generate_orders(bars=exit_bars, portfolio_state=long_state)

    assert len(exit_intents) == 1
    assert exit_intents[0].side == "sell"
    assert exit_intents[0].qty == 1.0
