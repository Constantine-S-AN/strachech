"""Matplotlib-based plot generation for report assets."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib
import pandas as pd

from stratcheck.core.strategy import Fill
from stratcheck.perf import IncrementalPlotState, prepare_incremental_plot_series

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_performance_plots(
    equity_curve: pd.Series,
    trades: Iterable[Fill] | None = None,
    returns: pd.Series | None = None,
    output_dir: str | Path = "reports/assets",
    prefix: str = "performance",
    incremental_state: IncrementalPlotState | None = None,
    return_incremental_state: bool = False,
) -> list[Path] | tuple[list[Path], IncrementalPlotState]:
    """Generate equity, drawdown, and return-distribution PNG charts."""
    normalized_equity = _normalize_equity_curve(equity_curve)
    if incremental_state is None:
        returns_series = _resolve_returns(returns=returns, equity_curve=normalized_equity)
        drawdown_curve = normalized_equity / normalized_equity.cummax() - 1.0
        resolved_state = None
    else:
        drawdown_curve, returns_series, resolved_state = prepare_incremental_plot_series(
            equity_curve=normalized_equity,
            returns=returns,
            state=incremental_state,
        )
    trade_count = len(list(trades)) if trades is not None else 0

    asset_dir = Path(output_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        _plot_equity(
            equity_curve=normalized_equity,
            output_path=asset_dir / f"{prefix}_equity.png",
            trade_count=trade_count,
        ),
        _plot_drawdown(
            drawdown_curve=drawdown_curve,
            output_path=asset_dir / f"{prefix}_drawdown.png",
        ),
        _plot_returns_histogram(
            returns_series=returns_series,
            output_path=asset_dir / f"{prefix}_returns_hist.png",
        ),
    ]
    if return_incremental_state:
        if resolved_state is None:
            _, _, resolved_state = prepare_incremental_plot_series(
                equity_curve=normalized_equity,
                returns=returns,
                state=IncrementalPlotState(),
            )
        return paths, resolved_state
    return paths


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


def _resolve_returns(returns: pd.Series | None, equity_curve: pd.Series) -> pd.Series:
    if returns is None:
        return equity_curve.pct_change().dropna()

    returns_series = returns.astype(float)
    if len(returns_series) == len(equity_curve):
        returns_series = returns_series.iloc[1:]
    return returns_series.dropna()


def _plot_equity(
    equity_curve: pd.Series,
    output_path: Path,
    trade_count: int,
) -> Path:
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(equity_curve.index, equity_curve.values, color="#1f77b4", linewidth=2.0)
    axis.set_title(f"Equity Curve (trades={trade_count})")
    axis.set_xlabel("Time")
    axis.set_ylabel("Equity")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def _plot_drawdown(drawdown_curve: pd.Series, output_path: Path) -> Path:
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(drawdown_curve.index, drawdown_curve.values, color="#d62728", linewidth=1.8)
    axis.fill_between(
        drawdown_curve.index,
        drawdown_curve.values,
        0.0,
        color="#fca5a5",
        alpha=0.5,
    )
    axis.set_title("Drawdown Curve")
    axis.set_xlabel("Time")
    axis.set_ylabel("Drawdown")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def _plot_returns_histogram(returns_series: pd.Series, output_path: Path) -> Path:
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.hist(
        returns_series.values,
        bins=30,
        color="#10b981",
        edgecolor="#065f46",
        alpha=0.75,
    )
    axis.set_title("Return Distribution")
    axis.set_xlabel("Return")
    axis.set_ylabel("Frequency")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path
