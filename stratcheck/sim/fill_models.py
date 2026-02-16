"""Fill model plugins for event-driven matching assumptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from stratcheck.sim.events import TimeInForce

_ORDER_EPSILON = 1e-12


@dataclass(slots=True)
class FillModelOrder:
    """Order snapshot consumed by fill models."""

    order_id: int
    side: Literal["buy", "sell"]
    market: bool
    limit_price: float | None
    remaining_qty: float
    first_eligible_bar_index: int
    last_eligible_bar_index: int | None
    time_in_force: TimeInForce


@dataclass(slots=True)
class FillModelState:
    """Runtime state available to fill models at one bar."""

    bar_index: int
    timestamp: pd.Timestamp
    cash: float
    position_qty: float
    max_volume_share: float


@dataclass(slots=True)
class FillInstruction:
    """Fill model decision for one order on current bar."""

    order_id: int
    qty: float
    reference_price: float | None = None
    expire: bool = False


@runtime_checkable
class FillModel(Protocol):
    """Pluggable fill simulation interface."""

    def simulate_fills(
        self,
        orders: list[FillModelOrder],
        bars_slice: pd.DataFrame,
        state: FillModelState,
    ) -> list[FillInstruction]:
        """Return fill instructions for current bar."""


class BarFillModel:
    """Default bar-based matching model."""

    def simulate_fills(
        self,
        orders: list[FillModelOrder],
        bars_slice: pd.DataFrame,
        state: FillModelState,
    ) -> list[FillInstruction]:
        if not orders or bars_slice.empty:
            return []

        current_bar = bars_slice.iloc[-1]
        fill_capacity = self._bar_fill_capacity(
            bar=current_bar,
            max_volume_share=state.max_volume_share,
        )

        instructions: list[FillInstruction] = []
        for order in orders:
            if state.bar_index < order.first_eligible_bar_index:
                continue
            if (
                order.last_eligible_bar_index is not None
                and state.bar_index > order.last_eligible_bar_index
            ):
                instructions.append(
                    FillInstruction(
                        order_id=order.order_id,
                        qty=0.0,
                        reference_price=None,
                        expire=True,
                    )
                )
                continue

            if not order.market and not self._is_limit_touched(order=order, bar=current_bar):
                if order.time_in_force in {"DAY", "IOC"}:
                    instructions.append(
                        FillInstruction(
                            order_id=order.order_id,
                            qty=0.0,
                            reference_price=None,
                            expire=True,
                        )
                    )
                continue

            proposed_qty = min(order.remaining_qty, fill_capacity)
            if proposed_qty <= _ORDER_EPSILON:
                if order.time_in_force in {"DAY", "IOC"}:
                    instructions.append(
                        FillInstruction(
                            order_id=order.order_id,
                            qty=0.0,
                            reference_price=None,
                            expire=True,
                        )
                    )
                continue

            instructions.append(
                FillInstruction(
                    order_id=order.order_id,
                    qty=float(proposed_qty),
                    reference_price=self._reference_price(order=order, bar=current_bar),
                    expire=order.time_in_force in {"DAY", "IOC"},
                )
            )

        return instructions

    def _bar_fill_capacity(
        self,
        bar: pd.Series,
        max_volume_share: float,
    ) -> float:
        volume_value = float(bar["volume"])
        if not np.isfinite(volume_value) or volume_value <= 0:
            return 0.0
        return volume_value * max_volume_share

    def _is_limit_touched(
        self,
        order: FillModelOrder,
        bar: pd.Series,
    ) -> bool:
        if order.limit_price is None:
            return False
        limit_price = float(order.limit_price)
        bar_low = float(bar["low"])
        bar_high = float(bar["high"])
        if order.side == "buy":
            return bar_low <= limit_price
        return bar_high >= limit_price

    def _reference_price(
        self,
        order: FillModelOrder,
        bar: pd.Series,
    ) -> float:
        bar_open = float(bar["open"])
        if order.market:
            return bar_open

        if order.limit_price is None:
            msg = "limit_price is required for limit order fills."
            raise ValueError(msg)
        limit_price = float(order.limit_price)
        if order.side == "buy":
            return min(bar_open, limit_price)
        return max(bar_open, limit_price)


class SpreadAwareFillModel(BarFillModel):
    """Bar fill model with bid/ask spread approximation."""

    def __init__(
        self,
        spread_bps: float = 5.0,
        use_bar_range: bool = True,
        range_multiplier: float = 0.10,
    ) -> None:
        if spread_bps < 0:
            msg = "spread_bps must be non-negative."
            raise ValueError(msg)
        if range_multiplier < 0:
            msg = "range_multiplier must be non-negative."
            raise ValueError(msg)
        self.spread_bps = float(spread_bps)
        self.use_bar_range = bool(use_bar_range)
        self.range_multiplier = float(range_multiplier)

    def _reference_price(
        self,
        order: FillModelOrder,
        bar: pd.Series,
    ) -> float:
        baseline_price = super()._reference_price(order=order, bar=bar)
        spread_value = self._estimate_spread(
            baseline_price=baseline_price,
            bar=bar,
        )
        half_spread = spread_value / 2.0
        if order.side == "buy":
            adjusted_price = baseline_price + half_spread
            if order.limit_price is not None:
                adjusted_price = min(adjusted_price, float(order.limit_price))
            return adjusted_price

        adjusted_price = baseline_price - half_spread
        if order.limit_price is not None:
            adjusted_price = max(adjusted_price, float(order.limit_price))
        return adjusted_price

    def _estimate_spread(
        self,
        baseline_price: float,
        bar: pd.Series,
    ) -> float:
        if self.use_bar_range:
            bar_high = float(bar["high"])
            bar_low = float(bar["low"])
            range_value = max(bar_high - bar_low, 0.0)
            if range_value > 0:
                return range_value * self.range_multiplier
        return baseline_price * (self.spread_bps / 10_000.0)


class VolumeParticipationFillModel(BarFillModel):
    """Bar fill model with explicit participation-rate cap."""

    def __init__(self, participation_rate: float = 0.25) -> None:
        if participation_rate <= 0 or participation_rate > 1.0:
            msg = "participation_rate must be in (0, 1]."
            raise ValueError(msg)
        self.participation_rate = float(participation_rate)

    def _bar_fill_capacity(
        self,
        bar: pd.Series,
        max_volume_share: float,
    ) -> float:
        shared_limit = min(max_volume_share, self.participation_rate)
        return super()._bar_fill_capacity(
            bar=bar,
            max_volume_share=shared_limit,
        )
