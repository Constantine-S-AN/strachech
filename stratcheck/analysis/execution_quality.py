"""Execution-quality analysis for slippage, latency, and fill outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from stratcheck.core.backtest import OrderRecord
from stratcheck.core.strategy import Fill

_EPSILON = 1e-12


@dataclass(slots=True)
class _OrderExecutionView:
    created_at: pd.Timestamp
    side: str
    requested_qty: float
    signal_price: float | None
    fill_time: pd.Timestamp | None
    fill_price: float | None
    filled_qty: float
    is_filled: bool
    is_canceled: bool
    is_partially_filled: bool
    latency_seconds: float | None
    latency_bars: float | None
    slippage_bps: float | None


def compute_execution_quality_scorecard(
    orders: list[OrderRecord] | list[object],
    bars: pd.DataFrame,
    trades: list[Fill] | None = None,
    signal_price_column: str = "close",
) -> pd.DataFrame:
    """Compute execution quality summary from order lifecycle and fills."""
    if not isinstance(bars.index, pd.DatetimeIndex):
        msg = "bars must use a DatetimeIndex."
        raise ValueError(msg)
    if signal_price_column not in bars.columns:
        msg = f"bars must include signal price column: {signal_price_column!r}."
        raise ValueError(msg)

    if not orders:
        return _empty_scorecard()

    signal_price_series = bars[signal_price_column].astype(float)
    aligned_trades = list(trades or [])
    order_views = _build_order_views(
        orders=orders,
        trades=aligned_trades,
        signal_price_series=signal_price_series,
        bar_index=bars.index,
    )
    if not order_views:
        return _empty_scorecard()

    slippage_values = [view.slippage_bps for view in order_views if view.slippage_bps is not None]
    latency_seconds_values = [
        view.latency_seconds for view in order_views if view.latency_seconds is not None
    ]
    latency_bars_values = [
        view.latency_bars for view in order_views if view.latency_bars is not None
    ]

    total_orders = float(len(order_views))
    filled_orders = float(sum(1 for view in order_views if view.is_filled))
    canceled_orders = float(sum(1 for view in order_views if view.is_canceled))
    partial_orders = float(sum(1 for view in order_views if view.is_partially_filled))

    scorecard = pd.DataFrame(
        [
            {
                "orders_total": int(total_orders),
                "filled_orders": int(filled_orders),
                "canceled_orders": int(canceled_orders),
                "cancel_rate": float(canceled_orders / total_orders) if total_orders > 0 else 0.0,
                "partially_filled_orders": int(partial_orders),
                "partial_fill_ratio": float(partial_orders / total_orders)
                if total_orders > 0
                else 0.0,
                "avg_slippage_bps": _safe_mean(slippage_values),
                "median_slippage_bps": _safe_median(slippage_values),
                "avg_latency_seconds": _safe_mean(latency_seconds_values),
                "median_latency_seconds": _safe_median(latency_seconds_values),
                "avg_latency_bars": _safe_mean(latency_bars_values),
                "median_latency_bars": _safe_median(latency_bars_values),
            }
        ]
    )
    return scorecard


def _build_order_views(
    orders: list[object],
    trades: list[Fill],
    signal_price_series: pd.Series,
    bar_index: pd.DatetimeIndex,
) -> list[_OrderExecutionView]:
    views: list[_OrderExecutionView] = []
    next_trade_position = 0

    for order in orders:
        created_at_raw = getattr(order, "created_at", None)
        if created_at_raw is None:
            continue
        created_at = pd.Timestamp(created_at_raw)
        side = str(getattr(order, "side", "")).lower()
        requested_qty = float(getattr(order, "qty", 0.0))

        fill_time = _optional_timestamp(
            getattr(order, "fill_time", None) or getattr(order, "updated_at", None)
        )
        fill_price = _optional_float(
            getattr(order, "fill_price", None) or getattr(order, "avg_fill_price", None)
        )

        explicit_filled = bool(getattr(order, "filled", False))
        status_text = str(getattr(order, "status", "")).lower()
        filled_qty_value = _optional_float(getattr(order, "filled_qty", None))

        if filled_qty_value is None and explicit_filled and requested_qty > 0:
            if next_trade_position < len(trades):
                trade_item = trades[next_trade_position]
                next_trade_position += 1
                filled_qty_value = float(trade_item.qty)
                if fill_time is None:
                    fill_time = pd.Timestamp(trade_item.timestamp)
                if fill_price is None:
                    fill_price = float(trade_item.price)
            else:
                filled_qty_value = requested_qty

        inferred_filled = (
            explicit_filled
            or status_text in {"filled", "partially_filled", "partiallyfilled"}
            or (filled_qty_value is not None and filled_qty_value > _EPSILON)
        )
        normalized_filled_qty = float(filled_qty_value or 0.0)
        is_partially_filled = (
            inferred_filled
            and requested_qty > _EPSILON
            and normalized_filled_qty > _EPSILON
            and normalized_filled_qty + _EPSILON < requested_qty
        )
        is_canceled = status_text in {"canceled", "cancelled", "rejected", "expired"} or (
            not inferred_filled
        )

        signal_price = _lookup_signal_price(
            signal_price_series=signal_price_series,
            timestamp=created_at,
        )
        latency_seconds = _latency_seconds(created_at=created_at, fill_time=fill_time)
        latency_bars = _latency_bars(
            bar_index=bar_index,
            created_at=created_at,
            fill_time=fill_time,
        )
        slippage_bps = _slippage_bps(
            side=side,
            signal_price=signal_price,
            fill_price=fill_price,
            is_filled=inferred_filled,
        )

        views.append(
            _OrderExecutionView(
                created_at=created_at,
                side=side,
                requested_qty=requested_qty,
                signal_price=signal_price,
                fill_time=fill_time,
                fill_price=fill_price,
                filled_qty=normalized_filled_qty,
                is_filled=inferred_filled,
                is_canceled=is_canceled,
                is_partially_filled=is_partially_filled,
                latency_seconds=latency_seconds,
                latency_bars=latency_bars,
                slippage_bps=slippage_bps,
            )
        )

    return views


def _lookup_signal_price(signal_price_series: pd.Series, timestamp: pd.Timestamp) -> float | None:
    aligned_timestamp = _align_timestamp_to_index(
        timestamp=timestamp,
        index=signal_price_series.index,
    )

    if aligned_timestamp in signal_price_series.index:
        value = signal_price_series.loc[aligned_timestamp]
        return float(value)

    earlier_prices = signal_price_series.loc[:aligned_timestamp]
    if not earlier_prices.empty:
        return float(earlier_prices.iloc[-1])

    later_prices = signal_price_series.loc[aligned_timestamp:]
    if not later_prices.empty:
        return float(later_prices.iloc[0])
    return None


def _align_timestamp_to_index(timestamp: pd.Timestamp, index: pd.Index) -> pd.Timestamp:
    aligned_timestamp = pd.Timestamp(timestamp)
    if not isinstance(index, pd.DatetimeIndex):
        return aligned_timestamp

    if index.tz is None:
        if aligned_timestamp.tzinfo is None:
            return aligned_timestamp
        return aligned_timestamp.tz_convert("UTC").tz_localize(None)

    if aligned_timestamp.tzinfo is None:
        return aligned_timestamp.tz_localize(index.tz)
    return aligned_timestamp.tz_convert(index.tz)


def _optional_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    return pd.Timestamp(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _latency_seconds(created_at: pd.Timestamp, fill_time: pd.Timestamp | None) -> float | None:
    if fill_time is None:
        return None
    start = _to_utc_naive(created_at)
    end = _to_utc_naive(fill_time)
    delay_seconds = float((end - start).total_seconds())
    return max(delay_seconds, 0.0)


def _latency_bars(
    bar_index: pd.DatetimeIndex,
    created_at: pd.Timestamp,
    fill_time: pd.Timestamp | None,
) -> float | None:
    if fill_time is None or len(bar_index) == 0:
        return None

    aligned_created = _align_timestamp_to_index(timestamp=created_at, index=bar_index)
    aligned_fill = _align_timestamp_to_index(timestamp=fill_time, index=bar_index)
    if aligned_created not in bar_index or aligned_fill not in bar_index:
        return None

    created_pos = bar_index.get_loc(aligned_created)
    fill_pos = bar_index.get_loc(aligned_fill)
    if not isinstance(created_pos, int) or not isinstance(fill_pos, int):
        return None
    return float(max(fill_pos - created_pos, 0))


def _to_utc_naive(timestamp: pd.Timestamp) -> pd.Timestamp:
    parsed = pd.Timestamp(timestamp)
    if parsed.tzinfo is None:
        return parsed
    return parsed.tz_convert("UTC").tz_localize(None)


def _slippage_bps(
    side: str,
    signal_price: float | None,
    fill_price: float | None,
    is_filled: bool,
) -> float | None:
    if not is_filled:
        return None
    if signal_price is None or fill_price is None:
        return None
    if signal_price <= _EPSILON:
        return None

    if side == "buy":
        return float((fill_price - signal_price) / signal_price * 10_000.0)
    if side == "sell":
        return float((signal_price - fill_price) / signal_price * 10_000.0)
    return None


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def _safe_median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.median(values))


def _empty_scorecard() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "orders_total": 0,
                "filled_orders": 0,
                "canceled_orders": 0,
                "cancel_rate": 0.0,
                "partially_filled_orders": 0,
                "partial_fill_ratio": 0.0,
                "avg_slippage_bps": 0.0,
                "median_slippage_bps": 0.0,
                "avg_latency_seconds": 0.0,
                "median_latency_seconds": 0.0,
                "avg_latency_bars": 0.0,
                "median_latency_bars": 0.0,
            }
        ]
    )
