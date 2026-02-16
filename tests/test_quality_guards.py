from __future__ import annotations

import pandas as pd
import pytest
from stratcheck.core.backtest import BacktestEngine, FixedBpsCostModel
from stratcheck.core.strategy import OrderIntent, PortfolioState
from stratcheck.quality import GuardViolationError


class FutureShiftLeakStrategy:
    def generate_orders(
        self,
        bars: pd.DataFrame,
        portfolio_state: PortfolioState,
    ) -> list[OrderIntent]:
        del portfolio_state
        if len(bars) < 3:
            return []

        future_shift_value = bars["close"].shift(-1).iloc[-2]
        current_close = float(bars["close"].iloc[-1])
        if float(future_shift_value) > current_close:
            return [OrderIntent(side="buy", qty=1.0, market=True)]
        return []


class PassiveStrategy:
    def generate_orders(
        self,
        bars: pd.DataFrame,
        portfolio_state: PortfolioState,
    ) -> list[OrderIntent]:
        del bars, portfolio_state
        return []


def test_lookahead_guard_blocks_negative_shift_strategy() -> None:
    bars = _sample_bars()
    engine = BacktestEngine()

    with pytest.raises(GuardViolationError) as error_info:
        engine.run(
            strategy=FutureShiftLeakStrategy(),
            bars=bars,
            initial_cash=10_000.0,
            cost_model=FixedBpsCostModel(),
        )

    flag_checks = {flag.check for flag in error_info.value.flags}
    assert "LookaheadGuard" in flag_checks
    assert "shift(-1)" in str(error_info.value)


def test_data_leak_guard_blocks_future_feature_timestamp() -> None:
    bars = _sample_bars()
    bars["alpha_ts"] = (bars.index + pd.Timedelta(days=1)).astype(str)
    engine = BacktestEngine()

    with pytest.raises(GuardViolationError) as error_info:
        engine.run(
            strategy=PassiveStrategy(),
            bars=bars,
            initial_cash=10_000.0,
            cost_model=FixedBpsCostModel(),
        )

    flag_checks = {flag.check for flag in error_info.value.flags}
    assert "DataLeakGuard" in flag_checks
    assert "alpha_ts" in str(error_info.value)


def _sample_bars() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 103.0, 102.0, 104.0, 105.0],
            "high": [101.0, 102.0, 104.0, 103.0, 105.0, 106.0],
            "low": [99.0, 100.0, 102.0, 101.0, 103.0, 104.0],
            "close": [100.5, 101.5, 103.5, 102.5, 104.5, 105.5],
            "volume": [1_000, 1_100, 1_050, 1_200, 1_150, 1_180],
        },
        index=timestamps,
    )
