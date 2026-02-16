"""Broker connector interfaces and shared order data types."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import pandas as pd

OrderSide = Literal["buy", "sell"]
OrderStatus = Literal["new", "partially_filled", "filled", "canceled"]
OrderType = Literal["market", "limit"]


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


@dataclass(slots=True)
class BrokerOrder:
    """Broker order snapshot with lifecycle fields."""

    order_id: str
    symbol: str
    side: OrderSide
    qty: float
    order_type: OrderType
    limit_price: float | None
    status: OrderStatus = "new"
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    created_at: pd.Timestamp = field(default_factory=_utc_now)
    updated_at: pd.Timestamp = field(default_factory=_utc_now)
    canceled_at: pd.Timestamp | None = None

    def __post_init__(self) -> None:
        if self.qty <= 0:
            msg = "Order qty must be positive."
            raise ValueError(msg)
        if self.side not in {"buy", "sell"}:
            msg = "Order side must be 'buy' or 'sell'."
            raise ValueError(msg)
        if self.order_type not in {"market", "limit"}:
            msg = "Order type must be 'market' or 'limit'."
            raise ValueError(msg)
        if self.order_type == "limit" and self.limit_price is None:
            msg = "limit_price is required for limit orders."
            raise ValueError(msg)

        self.symbol = str(self.symbol).upper()
        self.qty = float(self.qty)
        self.filled_qty = float(self.filled_qty)
        self.avg_fill_price = float(self.avg_fill_price)
        if self.filled_qty < 0:
            msg = "filled_qty cannot be negative."
            raise ValueError(msg)
        if self.filled_qty - self.qty > 1e-12:
            msg = "filled_qty cannot exceed total qty."
            raise ValueError(msg)

    @property
    def remaining_qty(self) -> float:
        """Unfilled quantity remaining."""
        return max(float(self.qty - self.filled_qty), 0.0)


@dataclass(slots=True)
class BrokerPosition:
    """Current symbol position tracked by broker connector."""

    symbol: str
    qty: float
    average_price: float = 0.0


@dataclass(slots=True)
class OrderUpdate:
    """Single order lifecycle update emitted by connector stream."""

    order_id: str
    symbol: str
    status: OrderStatus
    timestamp: pd.Timestamp
    filled_qty: float
    remaining_qty: float
    fill_qty: float = 0.0
    fill_price: float | None = None
    note: str = ""


@runtime_checkable
class BrokerConnector(Protocol):
    """Common paper/live broker connector interface."""

    def place(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        *,
        market: bool = True,
        limit_price: float | None = None,
    ) -> BrokerOrder:
        """Place a new order."""

    def cancel(self, order_id: str) -> BrokerOrder:
        """Cancel an open order."""

    def get_orders(self) -> list[BrokerOrder]:
        """Return current order snapshots."""

    def get_positions(self) -> dict[str, BrokerPosition]:
        """Return current positions keyed by symbol."""

    def stream_updates(self) -> Iterator[OrderUpdate]:
        """Stream pending order updates and clear connector queue."""
