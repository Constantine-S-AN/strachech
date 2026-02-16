from __future__ import annotations

import pandas as pd
from stratcheck.core.strategy import PortfolioState
from stratcheck.sdk import StrategySignal, StrategyTemplate


class MomentumTemplateStrategy(StrategyTemplate):
    def build_signals(
        self,
        bars: pd.DataFrame,
        portfolio_state: PortfolioState,
    ) -> list[StrategySignal]:
        if len(bars) < 3:
            return []
        close_prices = bars["close"].astype(float)
        last_change = float(close_prices.iloc[-1] - close_prices.iloc[-2])
        if last_change > 0 and portfolio_state.position_qty <= 0:
            return [self.signal(side="buy", qty=1.0, reason="positive_momentum")]
        if last_change <= 0 and portfolio_state.position_qty > 0:
            return [
                self.signal(side="sell", qty=portfolio_state.position_qty, reason="exit_signal")
            ]
        return []


def test_strategy_template_records_signals_and_converts_to_orders() -> None:
    timestamps = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    bars = pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [100, 101, 102, 103, 104],
            "volume": [1000, 1000, 1000, 1000, 1000],
        },
        index=timestamps,
    )
    strategy = MomentumTemplateStrategy(strategy_name="MomentumTemplate")
    portfolio_state = PortfolioState(
        cash=100_000.0,
        position_qty=0.0,
        average_entry_price=0.0,
        equity=100_000.0,
    )

    orders = strategy.generate_orders(bars=bars, portfolio_state=portfolio_state)

    assert len(orders) == 1
    assert orders[0].side == "buy"
    assert orders[0].qty == 1.0

    signal_records = strategy.get_signal_records()
    assert len(signal_records) == 1
    assert signal_records[0].reason == "positive_momentum"
    assert signal_records[0].strategy == "MomentumTemplate"

    signal_frame = strategy.get_signal_frame()
    assert len(signal_frame) == 1
    assert signal_frame.loc[0, "side"] == "buy"
    assert signal_frame.loc[0, "reason"] == "positive_momentum"


def test_strategy_template_clear_signal_records() -> None:
    strategy = MomentumTemplateStrategy()
    strategy.clear_signal_records()
    signal_frame = strategy.get_signal_frame()
    assert signal_frame.empty
