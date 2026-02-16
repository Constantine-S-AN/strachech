from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from stratcheck.connectors import PaperBrokerConnector
from stratcheck.core.backtest import BacktestEngine, FixedBpsCostModel
from stratcheck.core.strategy import OrderIntent, PortfolioState

_FLOAT_STRATEGY = st.floats(
    min_value=5.0,
    max_value=1_000.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)


class RoundTripStrategy:
    def __init__(self, qty: float) -> None:
        self.qty = float(qty)

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


class NoTradeStrategy:
    def generate_orders(
        self,
        bars: pd.DataFrame,
        portfolio_state: PortfolioState,
    ) -> list[OrderIntent]:
        del bars, portfolio_state
        return []


@settings(
    max_examples=80,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    initial_cash=st.floats(
        min_value=100.0, max_value=100_000.0, allow_nan=False, allow_infinity=False
    ),
    order_qty=st.floats(min_value=0.01, max_value=200.0, allow_nan=False, allow_infinity=False),
    reference_price=st.floats(
        min_value=5.0, max_value=500.0, allow_nan=False, allow_infinity=False
    ),
    commission_bps=st.floats(
        min_value=0.0,
        max_value=80.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    slippage_bps=st.floats(
        min_value=0.0,
        max_value=80.0,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_property_cash_does_not_increase_from_round_trip_costs(
    initial_cash: float,
    order_qty: float,
    reference_price: float,
    commission_bps: float,
    slippage_bps: float,
) -> None:
    bars = _constant_bars(price=reference_price, periods=4)
    strategy = RoundTripStrategy(qty=order_qty)
    engine = BacktestEngine()

    result = engine.run(
        strategy=strategy,
        bars=bars,
        initial_cash=float(initial_cash),
        cost_model=FixedBpsCostModel(
            commission_bps=float(commission_bps),
            slippage_bps=float(slippage_bps),
        ),
    )

    final_equity = float(result.equity_curve.iloc[-1])
    total_cost = float(sum(trade.cost for trade in result.trades))

    # With non-negative costs and no alpha in a flat-price round trip, equity should not increase.
    assert final_equity <= float(initial_cash) + 1e-6
    assert total_cost >= -1e-9


@settings(
    max_examples=80,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    close_prices=st.lists(
        _FLOAT_STRATEGY,
        min_size=3,
        max_size=50,
    ),
    initial_cash=st.floats(
        min_value=100.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False
    ),
    commission_bps=st.floats(
        min_value=0.0,
        max_value=50.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    slippage_bps=st.floats(
        min_value=0.0,
        max_value=50.0,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_property_equity_constant_when_no_position_and_no_trades(
    close_prices: list[float],
    initial_cash: float,
    commission_bps: float,
    slippage_bps: float,
) -> None:
    bars = _bars_from_close_prices(close_prices)
    strategy = NoTradeStrategy()
    engine = BacktestEngine()

    result = engine.run(
        strategy=strategy,
        bars=bars,
        initial_cash=float(initial_cash),
        cost_model=FixedBpsCostModel(
            commission_bps=float(commission_bps),
            slippage_bps=float(slippage_bps),
        ),
    )

    assert len(result.trades) == 0
    assert float(result.positions.abs().max()) == 0.0
    assert np.allclose(result.equity_curve.to_numpy(dtype=float), float(initial_cash), atol=1e-9)


@settings(
    max_examples=80,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    initial_cash=st.floats(
        min_value=1_000.0, max_value=100_000.0, allow_nan=False, allow_infinity=False
    ),
    order_qty=st.floats(
        min_value=0.1,
        max_value=500.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    max_fill_ratio=st.floats(
        min_value=0.1,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    max_volume_share=st.floats(
        min_value=0.1,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    bars_input=st.lists(
        st.tuples(
            st.floats(min_value=5.0, max_value=500.0, allow_nan=False, allow_infinity=False),
            st.floats(min_value=1.0, max_value=100_000.0, allow_nan=False, allow_infinity=False),
        ),
        min_size=1,
        max_size=30,
    ),
    should_cancel=st.booleans(),
    cancel_step=st.integers(min_value=0, max_value=29),
)
def test_property_order_fill_never_exceeds_order_qty_and_status_is_monotonic(
    initial_cash: float,
    order_qty: float,
    max_fill_ratio: float,
    max_volume_share: float,
    bars_input: list[tuple[float, float]],
    should_cancel: bool,
    cancel_step: int,
) -> None:
    connector = PaperBrokerConnector(
        initial_cash=float(initial_cash),
        max_fill_ratio_per_step=float(max_fill_ratio),
        max_volume_share=float(max_volume_share),
        allow_short=False,
    )
    placed_order = connector.place(symbol="AAPL", side="buy", qty=float(order_qty), market=True)
    updates = list(connector.stream_updates())

    for step_index, (price_value, volume_value) in enumerate(bars_input):
        connector.step_market(
            symbol="AAPL",
            bar=pd.Series(
                {
                    "open": float(price_value),
                    "high": float(price_value) * 1.01,
                    "low": float(price_value) * 0.99,
                    "close": float(price_value),
                    "volume": float(volume_value),
                }
            ),
            timestamp=pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=step_index),
        )
        updates.extend(connector.stream_updates())

        if should_cancel and step_index == min(cancel_step, len(bars_input) - 1):
            connector.cancel(placed_order.order_id)
            updates.extend(connector.stream_updates())
            break

    order_snapshot = connector.get_orders()[0]
    fill_qty_sum = float(sum(max(update.fill_qty, 0.0) for update in updates))

    assert fill_qty_sum <= float(order_qty) + 1e-8
    assert float(order_snapshot.filled_qty) <= float(order_qty) + 1e-8
    assert connector.cash <= float(initial_cash) + 1e-8

    status_rank = {"new": 0, "partially_filled": 1, "filled": 2, "canceled": 2}
    status_sequence = [update.status for update in updates]
    assert status_sequence
    assert status_sequence[0] == "new"
    for left_status, right_status in zip(status_sequence, status_sequence[1:], strict=False):
        assert status_rank[left_status] <= status_rank[right_status]

    if "canceled" in status_sequence:
        canceled_index = status_sequence.index("canceled")
        assert "filled" not in status_sequence[canceled_index + 1 :]

    for update in updates:
        assert 0.0 <= float(update.filled_qty) <= float(order_qty) + 1e-8
        assert 0.0 <= float(update.remaining_qty) <= float(order_qty) + 1e-8


def _constant_bars(price: float, periods: int) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="D", tz="UTC")
    bars = pd.DataFrame(
        {
            "open": float(price),
            "high": float(price),
            "low": float(price),
            "close": float(price),
            "volume": 1_000.0,
        },
        index=index,
    )
    return bars


def _bars_from_close_prices(close_prices: list[float]) -> pd.DataFrame:
    periods = len(close_prices)
    index = pd.date_range("2024-01-01", periods=periods, freq="D", tz="UTC")
    close_series = pd.Series(close_prices, dtype=float, index=index)
    bars = pd.DataFrame(
        {
            "open": close_series,
            "high": close_series * 1.01,
            "low": close_series * 0.99,
            "close": close_series,
            "volume": np.full(periods, 1_000.0),
        },
        index=index,
    )
    return bars
