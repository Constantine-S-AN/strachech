"""Baseline validation utilities for backtest equity-curve comparison."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from stratcheck.core.backtest import CostModel, FixedBpsCostModel
from stratcheck.core.strategy import MovingAverageCrossStrategy, OrderIntent, Strategy
from stratcheck.strategies import BuyAndHoldStrategy


@dataclass(slots=True)
class ValidationSummary:
    """Equity-curve comparison summary against a lightweight baseline."""

    strategy: str
    baseline: str
    status: str
    compared_points: int
    tolerance_abs: float
    max_abs_error: float
    mean_abs_error: float
    rmse: float
    message: str

    def to_record(self) -> dict[str, Any]:
        """Serialize summary to report-friendly dict."""
        return {
            "strategy": self.strategy,
            "baseline": self.baseline,
            "status": self.status,
            "compared_points": int(self.compared_points),
            "tolerance_abs": float(self.tolerance_abs),
            "max_abs_error": float(self.max_abs_error),
            "mean_abs_error": float(self.mean_abs_error),
            "rmse": float(self.rmse),
            "message": self.message,
        }


def validate_against_vectorized_baseline(
    strategy: Strategy,
    bars: pd.DataFrame,
    engine_equity_curve: pd.Series,
    initial_cash: float,
    cost_model: CostModel | None,
    tolerance_abs: float = 1e-6,
) -> list[dict[str, Any]]:
    """Compare backtest equity curve vs a vectorized reference baseline."""
    if tolerance_abs < 0:
        msg = "tolerance_abs must be non-negative."
        raise ValueError(msg)
    if initial_cash <= 0:
        msg = "initial_cash must be positive."
        raise ValueError(msg)

    normalized_bars = _normalize_bars(bars)
    model = cost_model or FixedBpsCostModel()
    baseline_label = "vectorized_reference"
    strategy_label = type(strategy).__name__

    if isinstance(strategy, BuyAndHoldStrategy):
        baseline_curve = _simulate_buy_and_hold_equity(
            bars=normalized_bars,
            initial_cash=initial_cash,
            cost_model=model,
            target_position_qty=float(strategy.target_position_qty),
            rebalance_tolerance=float(strategy.rebalance_tolerance),
        )
        baseline_label = "vectorized_buy_and_hold"
    elif isinstance(strategy, MovingAverageCrossStrategy):
        baseline_curve = _simulate_ma_cross_equity(
            bars=normalized_bars,
            initial_cash=initial_cash,
            cost_model=model,
            short_window=int(strategy.short_window),
            long_window=int(strategy.long_window),
            target_position_qty=float(strategy.target_position_qty),
        )
        baseline_label = "vectorized_moving_average_cross"
    else:
        skipped_summary = ValidationSummary(
            strategy=strategy_label,
            baseline=baseline_label,
            status="skipped",
            compared_points=0,
            tolerance_abs=tolerance_abs,
            max_abs_error=math.nan,
            mean_abs_error=math.nan,
            rmse=math.nan,
            message="No vectorized baseline available for this strategy class.",
        )
        return [skipped_summary.to_record()]

    summary = _compare_equity_curves(
        strategy_label=strategy_label,
        baseline_label=baseline_label,
        baseline_curve=baseline_curve,
        engine_curve=engine_equity_curve,
        tolerance_abs=tolerance_abs,
    )
    return [summary.to_record()]


def _simulate_buy_and_hold_equity(
    bars: pd.DataFrame,
    initial_cash: float,
    cost_model: CostModel,
    target_position_qty: float,
    rebalance_tolerance: float,
) -> pd.Series:
    def order_factory(bar_index: int, current_position: float) -> list[OrderIntent]:
        quantity_gap = target_position_qty - current_position
        if abs(quantity_gap) <= rebalance_tolerance:
            return []
        if quantity_gap > 0:
            return [OrderIntent(side="buy", qty=float(quantity_gap), market=True)]
        return [OrderIntent(side="sell", qty=float(abs(quantity_gap)), market=True)]

    return _simulate_equity_from_order_factory(
        bars=bars,
        initial_cash=initial_cash,
        cost_model=cost_model,
        order_factory=order_factory,
    )


def _simulate_ma_cross_equity(
    bars: pd.DataFrame,
    initial_cash: float,
    cost_model: CostModel,
    short_window: int,
    long_window: int,
    target_position_qty: float,
) -> pd.Series:
    close_prices = bars["close"].astype(float)
    short_average = close_prices.rolling(window=short_window, min_periods=short_window).mean()
    long_average = close_prices.rolling(window=long_window, min_periods=long_window).mean()

    def order_factory(bar_index: int, current_position: float) -> list[OrderIntent]:
        if bar_index < long_window:
            return []

        previous_short = short_average.iloc[bar_index - 1]
        current_short = short_average.iloc[bar_index]
        previous_long = long_average.iloc[bar_index - 1]
        current_long = long_average.iloc[bar_index]
        if (
            pd.isna(previous_short)
            or pd.isna(current_short)
            or pd.isna(previous_long)
            or pd.isna(current_long)
        ):
            return []

        bullish_cross = previous_short <= previous_long and current_short > current_long
        bearish_cross = previous_short >= previous_long and current_short < current_long

        if bullish_cross:
            buy_qty = target_position_qty - current_position
            if buy_qty > 0:
                return [OrderIntent(side="buy", qty=float(buy_qty), market=True)]
            return []
        if bearish_cross and current_position > 0:
            return [OrderIntent(side="sell", qty=float(current_position), market=True)]
        return []

    return _simulate_equity_from_order_factory(
        bars=bars,
        initial_cash=initial_cash,
        cost_model=cost_model,
        order_factory=order_factory,
    )


def _simulate_equity_from_order_factory(
    bars: pd.DataFrame,
    initial_cash: float,
    cost_model: CostModel,
    order_factory: Callable[[int, float], list[OrderIntent]],
) -> pd.Series:
    cash = float(initial_cash)
    position_qty = 0.0
    pending_orders: list[OrderIntent] = []
    equity_points: list[float] = []
    timestamps: list[pd.Timestamp] = []

    bar_count = len(bars)
    for bar_index, (timestamp, bar_row) in enumerate(bars.iterrows()):
        open_price = float(bar_row["open"])
        close_price = float(bar_row["close"])

        if pending_orders:
            for order in pending_orders:
                if order.side == "buy":
                    filled_qty = _resolve_affordable_buy_qty(
                        requested_qty=float(order.qty),
                        cash=cash,
                        reference_price=open_price,
                        bar_row=bar_row,
                        cost_model=cost_model,
                    )
                else:
                    filled_qty = min(float(order.qty), position_qty)

                if filled_qty <= 0:
                    continue

                quote = cost_model.quote_fill(
                    side=order.side,
                    qty=filled_qty,
                    reference_price=open_price,
                    bar=bar_row,
                )
                fill_price = float(quote.fill_price)
                fee = float(quote.fee)
                notional = filled_qty * fill_price

                if order.side == "buy":
                    cash -= notional + fee
                    position_qty += filled_qty
                else:
                    cash += notional - fee
                    position_qty -= filled_qty

        equity_points.append(float(cash + position_qty * close_price))
        timestamps.append(timestamp)

        if bar_index >= bar_count - 1:
            pending_orders = []
            continue
        pending_orders = order_factory(bar_index, position_qty)

    return pd.Series(
        data=equity_points,
        index=pd.DatetimeIndex(timestamps),
        name="baseline_equity",
        dtype=float,
    )


def _resolve_affordable_buy_qty(
    requested_qty: float,
    cash: float,
    reference_price: float,
    bar_row: pd.Series,
    cost_model: CostModel,
) -> float:
    if requested_qty <= 0 or cash <= 0:
        return 0.0

    full_quote = cost_model.quote_fill(
        side="buy",
        qty=requested_qty,
        reference_price=reference_price,
        bar=bar_row,
    )
    full_required_cash = requested_qty * float(full_quote.fill_price) + float(full_quote.fee)
    if full_required_cash <= cash:
        return requested_qty

    lower_qty = 0.0
    upper_qty = requested_qty
    for _ in range(40):
        middle_qty = (lower_qty + upper_qty) / 2.0
        if middle_qty <= 0:
            break
        quote = cost_model.quote_fill(
            side="buy",
            qty=middle_qty,
            reference_price=reference_price,
            bar=bar_row,
        )
        required_cash = middle_qty * float(quote.fill_price) + float(quote.fee)
        if required_cash <= cash:
            lower_qty = middle_qty
        else:
            upper_qty = middle_qty
    return lower_qty


def _compare_equity_curves(
    strategy_label: str,
    baseline_label: str,
    baseline_curve: pd.Series,
    engine_curve: pd.Series,
    tolerance_abs: float,
) -> ValidationSummary:
    baseline_series = pd.Series(baseline_curve, dtype=float).dropna()
    engine_series = pd.Series(engine_curve, dtype=float).dropna()
    joined = pd.concat(
        [engine_series.rename("engine"), baseline_series.rename("baseline")],
        axis=1,
        join="inner",
    ).dropna()

    if joined.empty:
        return ValidationSummary(
            strategy=strategy_label,
            baseline=baseline_label,
            status="skipped",
            compared_points=0,
            tolerance_abs=tolerance_abs,
            max_abs_error=math.nan,
            mean_abs_error=math.nan,
            rmse=math.nan,
            message="No overlapping equity points for baseline comparison.",
        )

    abs_error = (joined["engine"] - joined["baseline"]).abs()
    max_abs_error = float(abs_error.max())
    mean_abs_error = float(abs_error.mean())
    rmse = float(np.sqrt(np.mean(np.square(abs_error.to_numpy(dtype=float)))))
    compared_points = int(len(abs_error))
    passed = max_abs_error <= tolerance_abs
    status = "pass" if passed else "fail"
    message = (
        f"max_abs_error={max_abs_error:.10f}, tolerance={tolerance_abs:.10f}, "
        f"points={compared_points}"
    )
    return ValidationSummary(
        strategy=strategy_label,
        baseline=baseline_label,
        status=status,
        compared_points=compared_points,
        tolerance_abs=tolerance_abs,
        max_abs_error=max_abs_error,
        mean_abs_error=mean_abs_error,
        rmse=rmse,
        message=message,
    )


def _normalize_bars(bars: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(bars.index, pd.DatetimeIndex):
        msg = "bars must use a DatetimeIndex."
        raise ValueError(msg)
    required_columns = {"open", "close"}
    missing_columns = required_columns.difference(bars.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        msg = f"bars is missing required columns: {missing_text}"
        raise ValueError(msg)

    normalized = bars.sort_index().copy()
    normalized = normalized[~normalized.index.duplicated(keep="last")]
    return normalized
