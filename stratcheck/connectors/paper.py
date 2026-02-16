"""Local paper broker connector with deterministic partial-fill simulation."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import replace
from typing import Literal

import pandas as pd

from stratcheck.connectors.base import (
    BrokerOrder,
    BrokerPosition,
    OrderSide,
    OrderStatus,
    OrderUpdate,
)

_EPSILON = 1e-12


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


class PaperBrokerConnector:
    """In-process paper broker for order lifecycle and state synchronization.

    Fill assumptions:
    - Market orders are filled starting from the next `step_market(...)` call
      using bar `open` as reference.
    - Limit buy fills when `bar.low <= limit_price`.
    - Limit sell fills when `bar.high >= limit_price`.
    - Each market step can fill only a fraction of order qty controlled by
      `max_fill_ratio_per_step`, and by optional bar-volume participation cap.
    """

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        max_fill_ratio_per_step: float = 1.0,
        max_volume_share: float = 1.0,
        allow_short: bool = False,
    ) -> None:
        if initial_cash < 0:
            msg = "initial_cash must be non-negative."
            raise ValueError(msg)
        if not (0 < max_fill_ratio_per_step <= 1.0):
            msg = "max_fill_ratio_per_step must be in (0, 1]."
            raise ValueError(msg)
        if not (0 < max_volume_share <= 1.0):
            msg = "max_volume_share must be in (0, 1]."
            raise ValueError(msg)

        self.cash = float(initial_cash)
        self.max_fill_ratio_per_step = float(max_fill_ratio_per_step)
        self.max_volume_share = float(max_volume_share)
        self.allow_short = bool(allow_short)

        self._order_sequence = 1
        self._orders_by_id: dict[str, BrokerOrder] = {}
        self._positions: dict[str, BrokerPosition] = {}
        self._pending_updates: deque[OrderUpdate] = deque()

    def place(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        *,
        market: bool = True,
        limit_price: float | None = None,
    ) -> BrokerOrder:
        """Create new broker order and enqueue `new` update."""
        order_type: Literal["market", "limit"] = "market" if market else "limit"
        now = _utc_now()
        order_id = str(self._order_sequence)
        self._order_sequence += 1

        order = BrokerOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            qty=float(qty),
            order_type=order_type,
            limit_price=limit_price,
            status="new",
            created_at=now,
            updated_at=now,
        )
        self._orders_by_id[order_id] = order
        self._emit_update(
            order=order,
            status="new",
            fill_qty=0.0,
            fill_price=None,
            note="order_accepted",
            timestamp=now,
        )
        return replace(order)

    def cancel(self, order_id: str) -> BrokerOrder:
        """Cancel order when still open."""
        if order_id not in self._orders_by_id:
            msg = f"Unknown order_id={order_id!r}"
            raise ValueError(msg)

        order = self._orders_by_id[order_id]
        if order.status in {"filled", "canceled"}:
            return replace(order)

        now = _utc_now()
        order.status = "canceled"
        order.canceled_at = now
        order.updated_at = now
        self._emit_update(
            order=order,
            status="canceled",
            fill_qty=0.0,
            fill_price=None,
            note="order_canceled",
            timestamp=now,
        )
        return replace(order)

    def get_orders(self) -> list[BrokerOrder]:
        """Return immutable snapshots of all orders."""
        ordered = sorted(
            self._orders_by_id.values(),
            key=lambda order_snapshot: (
                order_snapshot.created_at.value,
                int(order_snapshot.order_id),
            ),
        )
        return [replace(order) for order in ordered]

    def get_positions(self) -> dict[str, BrokerPosition]:
        """Return immutable snapshots of positions."""
        return {symbol: replace(position) for symbol, position in self._positions.items()}

    def stream_updates(self) -> Iterator[OrderUpdate]:
        """Yield queued updates once and clear internal queue."""
        while self._pending_updates:
            yield self._pending_updates.popleft()

    def step_market(
        self,
        symbol: str,
        bar: pd.Series,
        timestamp: pd.Timestamp | str | None = None,
    ) -> None:
        """Advance paper broker matching by one market bar."""
        current_time = self._normalize_timestamp(timestamp)
        symbol_name = str(symbol).upper()
        open_price = float(bar.get("open", bar.get("close", 0.0)))
        high_price = float(bar.get("high", open_price))
        low_price = float(bar.get("low", open_price))
        volume_value = float(bar.get("volume", 0.0))

        for order in self._iter_open_orders(symbol=symbol_name):
            match_result = self._simulate_match(
                order=order,
                open_price=open_price,
                high_price=high_price,
                low_price=low_price,
                volume_value=volume_value,
            )
            if match_result is None:
                continue
            fill_qty, fill_price = match_result
            if fill_qty <= _EPSILON:
                continue
            self._apply_fill(
                order=order,
                fill_qty=fill_qty,
                fill_price=fill_price,
                timestamp=current_time,
            )

    def _iter_open_orders(self, symbol: str) -> list[BrokerOrder]:
        return [
            order
            for order in self._orders_by_id.values()
            if order.symbol == symbol and order.status in {"new", "partially_filled"}
        ]

    def _simulate_match(
        self,
        order: BrokerOrder,
        open_price: float,
        high_price: float,
        low_price: float,
        volume_value: float,
    ) -> tuple[float, float] | None:
        if order.order_type == "market":
            reference_price = max(open_price, 0.0)
        else:
            if order.limit_price is None:
                return None
            if order.side == "buy" and low_price <= order.limit_price:
                reference_price = float(order.limit_price)
            elif order.side == "sell" and high_price >= order.limit_price:
                reference_price = float(order.limit_price)
            else:
                return None

        step_fraction_cap = order.qty * self.max_fill_ratio_per_step
        volume_cap = (
            abs(volume_value) * self.max_volume_share if volume_value > 0 else order.remaining_qty
        )
        raw_fill_qty = min(order.remaining_qty, step_fraction_cap, volume_cap)
        if raw_fill_qty <= _EPSILON:
            return None

        if order.side == "buy":
            affordable_qty = self.cash / reference_price if reference_price > 0 else 0.0
            raw_fill_qty = min(raw_fill_qty, affordable_qty)
        else:
            available_qty = self._positions.get(order.symbol, BrokerPosition(order.symbol, 0.0)).qty
            if not self.allow_short:
                raw_fill_qty = min(raw_fill_qty, max(available_qty, 0.0))

        if raw_fill_qty <= _EPSILON:
            return None
        return float(raw_fill_qty), float(reference_price)

    def _apply_fill(
        self,
        order: BrokerOrder,
        fill_qty: float,
        fill_price: float,
        timestamp: pd.Timestamp,
    ) -> None:
        position = self._positions.get(order.symbol, BrokerPosition(order.symbol, 0.0, 0.0))
        fill_notional = fill_qty * fill_price

        if order.side == "buy":
            self.cash -= fill_notional
            new_qty = position.qty + fill_qty
            if new_qty <= _EPSILON:
                position = BrokerPosition(order.symbol, 0.0, 0.0)
            else:
                weighted_value = (position.qty * position.average_price) + fill_notional
                position = BrokerPosition(order.symbol, new_qty, weighted_value / new_qty)
        else:
            self.cash += fill_notional
            new_qty = position.qty - fill_qty
            if abs(new_qty) <= _EPSILON:
                position = BrokerPosition(order.symbol, 0.0, 0.0)
            else:
                next_avg = position.average_price if new_qty > 0 else fill_price
                position = BrokerPosition(order.symbol, new_qty, next_avg)

        if abs(position.qty) <= _EPSILON:
            self._positions.pop(order.symbol, None)
        else:
            self._positions[order.symbol] = position

        previous_filled = order.filled_qty
        total_filled = previous_filled + fill_qty
        order.filled_qty = total_filled
        if total_filled > 0:
            previous_notional = previous_filled * order.avg_fill_price
            order.avg_fill_price = (previous_notional + fill_notional) / total_filled
        order.updated_at = timestamp

        if order.remaining_qty <= _EPSILON:
            order.status = "filled"
            update_status: OrderStatus = "filled"
        else:
            order.status = "partially_filled"
            update_status = "partially_filled"

        self._emit_update(
            order=order,
            status=update_status,
            fill_qty=fill_qty,
            fill_price=fill_price,
            note="order_matched",
            timestamp=timestamp,
        )

    def _emit_update(
        self,
        order: BrokerOrder,
        status: OrderStatus,
        fill_qty: float,
        fill_price: float | None,
        note: str,
        timestamp: pd.Timestamp,
    ) -> None:
        self._pending_updates.append(
            OrderUpdate(
                order_id=order.order_id,
                symbol=order.symbol,
                status=status,
                timestamp=timestamp,
                filled_qty=float(order.filled_qty),
                remaining_qty=float(order.remaining_qty),
                fill_qty=float(fill_qty),
                fill_price=None if fill_price is None else float(fill_price),
                note=note,
            )
        )

    @staticmethod
    def _normalize_timestamp(value: pd.Timestamp | str | None) -> pd.Timestamp:
        if value is None:
            return _utc_now()
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            return timestamp.tz_localize("UTC")
        return timestamp.tz_convert("UTC")
