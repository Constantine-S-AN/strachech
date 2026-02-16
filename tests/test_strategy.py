from __future__ import annotations

import pandas as pd
from stratcheck.core.strategy import MovingAverageCrossStrategy, PortfolioState


def _bars_from_closes(close_values: list[float]) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=len(close_values), freq="D", tz="UTC")
    return pd.DataFrame({"close": close_values}, index=timestamps)


def test_moving_average_cross_strategy_emits_buy_intent_on_bullish_cross() -> None:
    bars = _bars_from_closes([10.0, 9.0, 8.0, 9.0, 11.0])
    strategy = MovingAverageCrossStrategy(short_window=2, long_window=3, target_position_qty=1.0)
    portfolio_state = PortfolioState(cash=10_000.0, position_qty=0.0)

    intents = strategy.generate_orders(bars=bars, portfolio_state=portfolio_state)

    assert len(intents) == 1
    assert intents[0].side == "buy"
    assert intents[0].qty == 1.0
    assert intents[0].market is True
    assert intents[0].limit_price is None


def test_moving_average_cross_strategy_emits_sell_intent_on_bearish_cross() -> None:
    bars = _bars_from_closes([10.0, 11.0, 12.0, 11.0, 9.0])
    strategy = MovingAverageCrossStrategy(short_window=2, long_window=3, target_position_qty=1.0)
    portfolio_state = PortfolioState(cash=10_000.0, position_qty=1.0)

    intents = strategy.generate_orders(bars=bars, portfolio_state=portfolio_state)

    assert len(intents) == 1
    assert intents[0].side == "sell"
    assert intents[0].qty == 1.0
    assert intents[0].market is True


def test_moving_average_cross_strategy_returns_no_intent_without_cross() -> None:
    bars = _bars_from_closes([10.0, 10.1, 10.2, 10.25, 10.3])
    strategy = MovingAverageCrossStrategy(short_window=2, long_window=3, target_position_qty=1.0)
    portfolio_state = PortfolioState(cash=10_000.0, position_qty=0.0)

    intents = strategy.generate_orders(bars=bars, portfolio_state=portfolio_state)

    assert intents == []
