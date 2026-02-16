"""Incremental metric and plot-series calculations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from stratcheck.core.calendar import normalize_bars_freq, periods_per_year
from stratcheck.core.strategy import Fill


@dataclass(slots=True)
class IncrementalMetricsState:
    """Running state for incremental metrics updates."""

    bars_freq: str
    processed_equity_points: int = 0
    equity_sum: float = 0.0
    start_equity: float | None = None
    end_equity: float | None = None
    last_equity: float | None = None
    running_peak: float | None = None
    max_drawdown: float = 0.0
    return_count: int = 0
    return_mean: float = 0.0
    return_m2: float = 0.0
    processed_trades: int = 0
    total_notional: float = 0.0
    open_position_qty: float = 0.0
    average_entry_cost: float = 0.0
    realized_trade_count: int = 0
    realized_win_count: int = 0
    realized_trade_pnl_sum: float = 0.0


@dataclass(slots=True)
class IncrementalPlotState:
    """Running state for incremental drawdown/return series."""

    processed_equity_points: int = 0
    running_peak: float | None = None
    last_equity: float | None = None
    drawdown_index: list[pd.Timestamp] = field(default_factory=list)
    drawdown_values: list[float] = field(default_factory=list)
    return_index: list[pd.Timestamp] = field(default_factory=list)
    return_values: list[float] = field(default_factory=list)


def compute_metrics_incremental(
    equity_curve: pd.Series,
    trades: list[Fill] | tuple[Fill, ...],
    bars_freq: str,
    state: IncrementalMetricsState | None = None,
) -> tuple[dict[str, float], IncrementalMetricsState]:
    """Compute metrics by incrementally consuming newly appended data points."""
    normalized_equity = _normalize_equity_curve(equity_curve)
    trade_list = list(trades)
    canonical_freq = normalize_bars_freq(bars_freq)

    if state is None or state.bars_freq != canonical_freq:
        state = IncrementalMetricsState(bars_freq=canonical_freq)
    if state.processed_equity_points > len(normalized_equity):
        state = IncrementalMetricsState(bars_freq=canonical_freq)
    if state.processed_trades > len(trade_list):
        state = IncrementalMetricsState(bars_freq=canonical_freq)

    new_equity = normalized_equity.iloc[state.processed_equity_points :]
    for timestamp, equity_value in new_equity.items():
        del timestamp
        current_equity = float(equity_value)
        if state.processed_equity_points == 0 and state.start_equity is None:
            state.start_equity = current_equity

        state.processed_equity_points += 1
        state.equity_sum += current_equity
        state.end_equity = current_equity

        if state.running_peak is None:
            state.running_peak = current_equity
        else:
            state.running_peak = max(state.running_peak, current_equity)
        if state.running_peak > 0:
            drawdown_value = current_equity / state.running_peak - 1.0
            state.max_drawdown = min(state.max_drawdown, float(drawdown_value))

        if state.last_equity is None:
            return_value = 0.0
        elif state.last_equity != 0:
            return_value = current_equity / state.last_equity - 1.0
        else:
            return_value = 0.0

        state.return_count += 1
        delta = return_value - state.return_mean
        state.return_mean += delta / state.return_count
        delta2 = return_value - state.return_mean
        state.return_m2 += delta * delta2
        state.last_equity = current_equity

    new_trades = trade_list[state.processed_trades :]
    for trade in new_trades:
        qty = float(trade.qty)
        price = float(trade.price)
        fee = float(trade.fee)
        if qty <= 0:
            state.processed_trades += 1
            continue

        state.total_notional += abs(qty * price)
        if trade.side == "buy":
            total_cost = price * qty + fee
            new_position_qty = state.open_position_qty + qty
            if new_position_qty > 0:
                state.average_entry_cost = (
                    state.average_entry_cost * state.open_position_qty + total_cost
                ) / new_position_qty
            state.open_position_qty = new_position_qty
        else:
            closed_qty = min(qty, state.open_position_qty)
            if closed_qty > 0 and qty > 0:
                proceeds_after_fee = price * closed_qty - fee * (closed_qty / qty)
                trade_pnl = proceeds_after_fee - state.average_entry_cost * closed_qty
                state.realized_trade_count += 1
                state.realized_trade_pnl_sum += float(trade_pnl)
                if trade_pnl > 0:
                    state.realized_win_count += 1

                state.open_position_qty -= closed_qty
                if state.open_position_qty <= 1e-12:
                    state.open_position_qty = 0.0
                    state.average_entry_cost = 0.0

        state.processed_trades += 1

    metrics = _metrics_from_state(state)
    return metrics, state


def prepare_incremental_plot_series(
    equity_curve: pd.Series,
    returns: pd.Series | None = None,
    state: IncrementalPlotState | None = None,
) -> tuple[pd.Series, pd.Series, IncrementalPlotState]:
    """Build drawdown and returns series incrementally from equity updates."""
    normalized_equity = _normalize_equity_curve(equity_curve)
    if state is None:
        state = IncrementalPlotState()
    if state.processed_equity_points > len(normalized_equity):
        state = IncrementalPlotState()

    new_equity = normalized_equity.iloc[state.processed_equity_points :]
    for timestamp, equity_value in new_equity.items():
        current_equity = float(equity_value)
        if state.running_peak is None:
            state.running_peak = current_equity
        else:
            state.running_peak = max(state.running_peak, current_equity)

        drawdown_value = (
            current_equity / state.running_peak - 1.0
            if state.running_peak and state.running_peak > 0
            else 0.0
        )
        state.drawdown_index.append(timestamp)
        state.drawdown_values.append(float(drawdown_value))

        if state.last_equity is not None and state.last_equity != 0:
            return_value = current_equity / state.last_equity - 1.0
            state.return_index.append(timestamp)
            state.return_values.append(float(return_value))
        state.last_equity = current_equity
        state.processed_equity_points += 1

    drawdown_curve = pd.Series(
        data=state.drawdown_values,
        index=pd.DatetimeIndex(state.drawdown_index),
        name="drawdown",
        dtype=float,
    )

    if returns is None:
        returns_series = pd.Series(
            data=state.return_values,
            index=pd.DatetimeIndex(state.return_index),
            name="returns",
            dtype=float,
        )
    else:
        returns_series = returns.astype(float)
        if len(returns_series) == len(normalized_equity):
            returns_series = returns_series.iloc[1:]
        returns_series = returns_series.dropna()

    return drawdown_curve, returns_series, state


def _metrics_from_state(state: IncrementalMetricsState) -> dict[str, float]:
    if state.start_equity is None or state.end_equity is None or state.processed_equity_points <= 0:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "sharpe": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "turnover": 0.0,
            "win_rate": 0.0,
            "avg_trade_pnl": 0.0,
        }

    start_equity = float(state.start_equity)
    end_equity = float(state.end_equity)
    total_points = int(state.processed_equity_points)
    average_equity = float(state.equity_sum / total_points) if total_points > 0 else 0.0
    annualization_periods = periods_per_year(state.bars_freq)
    return_intervals = max(total_points - 1, 0)

    total_return = end_equity / start_equity - 1.0 if start_equity > 0 else 0.0
    if start_equity <= 0 or return_intervals == 0:
        cagr = 0.0
    else:
        cagr = float(
            (end_equity / start_equity) ** (annualization_periods / return_intervals) - 1.0
        )

    if state.return_count > 0:
        return_variance = state.return_m2 / state.return_count
        annual_volatility = float(
            math.sqrt(max(return_variance, 0.0)) * math.sqrt(annualization_periods)
        )
    else:
        annual_volatility = 0.0
    sharpe = float(cagr / annual_volatility) if annual_volatility > 0 else 0.0

    turnover = float(state.total_notional / average_equity) if average_equity > 0 else 0.0
    if state.realized_trade_count > 0:
        win_rate = float(state.realized_win_count / state.realized_trade_count)
        avg_trade_pnl = float(state.realized_trade_pnl_sum / state.realized_trade_count)
    else:
        win_rate = 0.0
        avg_trade_pnl = 0.0

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "annual_return": float(cagr),
        "annual_volatility": float(annual_volatility),
        "sharpe": float(sharpe),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(state.max_drawdown),
        "turnover": float(turnover),
        "win_rate": float(win_rate),
        "avg_trade_pnl": float(avg_trade_pnl),
    }


def _normalize_equity_curve(equity_curve: pd.Series) -> pd.Series:
    if len(equity_curve) == 0:
        msg = "equity_curve cannot be empty."
        raise ValueError(msg)
    if not isinstance(equity_curve.index, pd.DatetimeIndex):
        msg = "equity_curve must use a DatetimeIndex."
        raise ValueError(msg)

    normalized_equity = equity_curve.astype(float).sort_index()
    normalized_equity = normalized_equity[~normalized_equity.index.duplicated(keep="last")]
    return normalized_equity
