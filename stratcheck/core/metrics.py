"""Performance metrics computed from equity curve and trade history."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from stratcheck.core.calendar import normalize_bars_freq, periods_per_year
from stratcheck.core.strategy import Fill
from stratcheck.perf import IncrementalMetricsState, compute_metrics_incremental


def compute_metrics(
    equity_curve: pd.Series,
    trades: Iterable[Fill],
    bars_freq: str,
    incremental_state: IncrementalMetricsState | None = None,
    return_incremental_state: bool = False,
) -> dict[str, float] | tuple[dict[str, float], IncrementalMetricsState]:
    """Compute performance metrics from equity curve and trades.

    Turnover definition:
    - turnover = total traded notional / average equity
    - traded notional is `sum(abs(qty * price))` across all fills.
    """
    normalized_equity = _normalize_equity_curve(equity_curve)
    trade_list = list(trades)

    if incremental_state is not None or return_incremental_state:
        metrics, resolved_state = compute_metrics_incremental(
            equity_curve=normalized_equity,
            trades=trade_list,
            bars_freq=bars_freq,
            state=incremental_state,
        )
        if return_incremental_state:
            return metrics, resolved_state
        return metrics

    canonical_freq = normalize_bars_freq(bars_freq)
    annualization_periods = periods_per_year(canonical_freq)
    strategy_returns = normalized_equity.pct_change().fillna(0.0)

    start_equity = float(normalized_equity.iloc[0])
    end_equity = float(normalized_equity.iloc[-1])
    total_return = (end_equity / start_equity - 1.0) if start_equity > 0 else 0.0

    return_intervals = max(len(normalized_equity) - 1, 0)
    if start_equity <= 0 or return_intervals == 0:
        cagr = 0.0
    else:
        cagr = float(
            (end_equity / start_equity) ** (annualization_periods / return_intervals) - 1.0
        )

    annual_volatility = float(strategy_returns.std(ddof=0) * np.sqrt(annualization_periods))
    sharpe = float(cagr / annual_volatility) if annual_volatility > 0 else 0.0

    running_peak = normalized_equity.cummax()
    drawdown_curve = normalized_equity / running_peak - 1.0
    max_drawdown = float(drawdown_curve.min())

    average_equity = float(normalized_equity.mean())
    total_notional = float(sum(abs(trade.qty * trade.price) for trade in trade_list))
    turnover = float(total_notional / average_equity) if average_equity > 0 else 0.0

    trade_pnl_values = _realized_trade_pnl(trade_list)
    if trade_pnl_values:
        win_rate = float(sum(1 for pnl in trade_pnl_values if pnl > 0.0) / len(trade_pnl_values))
        avg_trade_pnl = float(np.mean(trade_pnl_values))
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
        "max_drawdown": float(max_drawdown),
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


def _realized_trade_pnl(trades: list[Fill]) -> list[float]:
    """Compute realized PnL for completed long trades."""
    current_position_qty = 0.0
    average_entry_cost = 0.0
    realized_values: list[float] = []

    for trade in trades:
        qty = float(trade.qty)
        if qty <= 0:
            continue

        if trade.side == "buy":
            total_cost = trade.price * qty + float(trade.fee)
            new_position_qty = current_position_qty + qty
            average_entry_cost = (
                average_entry_cost * current_position_qty + total_cost
            ) / new_position_qty
            current_position_qty = new_position_qty
            continue

        closed_qty = min(qty, current_position_qty)
        if closed_qty <= 0:
            continue

        proceeds_after_fee = trade.price * closed_qty - float(trade.fee) * (closed_qty / qty)
        realized_values.append(proceeds_after_fee - average_entry_cost * closed_qty)
        current_position_qty -= closed_qty

        if current_position_qty <= 1e-12:
            current_position_qty = 0.0
            average_entry_cost = 0.0

    return realized_values
