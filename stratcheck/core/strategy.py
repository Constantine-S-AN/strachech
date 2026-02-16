"""Strategy plugin interfaces and reference implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import pandas as pd


@dataclass(slots=True)
class OrderIntent:
    """An order request emitted by a strategy."""

    side: Literal["buy", "sell"]
    qty: float
    limit_price: float | None = None
    market: bool = True

    def __post_init__(self) -> None:
        if self.qty <= 0:
            msg = "Order qty must be positive."
            raise ValueError(msg)
        if self.side not in {"buy", "sell"}:
            msg = "Order side must be either 'buy' or 'sell'."
            raise ValueError(msg)
        if not self.market and self.limit_price is None:
            msg = "Limit price is required when market is False."
            raise ValueError(msg)


@dataclass(slots=True)
class Fill:
    """Execution result for an accepted order."""

    side: Literal["buy", "sell"]
    qty: float
    price: float
    timestamp: pd.Timestamp
    fee: float = 0.0
    cost: float = 0.0


@dataclass(slots=True)
class PortfolioState:
    """Current portfolio snapshot passed into strategies."""

    cash: float
    position_qty: float
    average_entry_price: float = 0.0
    equity: float | None = None


@runtime_checkable
class Strategy(Protocol):
    """Strategy plugin interface."""

    def generate_orders(
        self,
        bars: pd.DataFrame,
        portfolio_state: PortfolioState,
    ) -> list[OrderIntent]:
        """Generate order intents from bars and current portfolio state."""


class MovingAverageCrossStrategy:
    """A long-only moving-average cross strategy."""

    def __init__(
        self,
        short_window: int = 20,
        long_window: int = 50,
        target_position_qty: float = 1.0,
    ) -> None:
        if short_window < 1:
            msg = "short_window must be >= 1."
            raise ValueError(msg)
        if long_window <= short_window:
            msg = "long_window must be greater than short_window."
            raise ValueError(msg)
        if target_position_qty <= 0:
            msg = "target_position_qty must be positive."
            raise ValueError(msg)

        self.short_window = short_window
        self.long_window = long_window
        self.target_position_qty = target_position_qty

    def generate_orders(
        self,
        bars: pd.DataFrame,
        portfolio_state: PortfolioState,
    ) -> list[OrderIntent]:
        """Generate buy/sell intent when fast/slow averages cross."""
        if "close" not in bars.columns:
            msg = "bars must include a 'close' column."
            raise ValueError(msg)
        if len(bars) < self.long_window + 1:
            return []

        close_prices = bars["close"].astype(float)
        short_average = close_prices.rolling(
            window=self.short_window,
            min_periods=self.short_window,
        ).mean()
        long_average = close_prices.rolling(
            window=self.long_window,
            min_periods=self.long_window,
        ).mean()

        previous_short = short_average.iloc[-2]
        current_short = short_average.iloc[-1]
        previous_long = long_average.iloc[-2]
        current_long = long_average.iloc[-1]

        if (
            pd.isna(previous_short)
            or pd.isna(current_short)
            or pd.isna(previous_long)
            or pd.isna(current_long)
        ):
            return []

        bullish_cross = previous_short <= previous_long and current_short > current_long
        bearish_cross = previous_short >= previous_long and current_short < current_long

        current_position_qty = portfolio_state.position_qty

        orders: list[OrderIntent] = []
        if bullish_cross:
            buy_qty = self.target_position_qty - current_position_qty
            if buy_qty > 0:
                orders.append(OrderIntent(side="buy", qty=float(buy_qty), market=True))
            return orders

        if bearish_cross and current_position_qty > 0:
            orders.append(OrderIntent(side="sell", qty=float(current_position_qty), market=True))

        return orders
