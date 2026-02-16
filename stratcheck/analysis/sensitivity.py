"""Automatic cost/slippage/spread sensitivity scan."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd

from stratcheck.core.backtest import BacktestEngine, FillQuote
from stratcheck.core.strategy import Strategy


@dataclass(slots=True)
class CombinedBpsCostModel:
    """Combined commission + slippage + spread model for sensitivity analysis."""

    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    spread_bps: float = 0.0

    def __post_init__(self) -> None:
        for key, value in (
            ("commission_bps", self.commission_bps),
            ("slippage_bps", self.slippage_bps),
            ("spread_bps", self.spread_bps),
        ):
            if value < 0:
                msg = f"{key} must be non-negative."
                raise ValueError(msg)

    def quote_fill(
        self,
        side: Literal["buy", "sell"],
        qty: float,
        reference_price: float,
        bar: pd.Series,
    ) -> FillQuote:
        del bar
        slippage_rate = self.slippage_bps / 10_000.0
        half_spread_rate = (self.spread_bps / 10_000.0) / 2.0
        direction_rate = slippage_rate + half_spread_rate

        if side == "buy":
            fill_price = reference_price * (1.0 + direction_rate)
        else:
            fill_price = reference_price * (1.0 - direction_rate)
        fill_price = max(fill_price, 0.0)

        fee = qty * fill_price * (self.commission_bps / 10_000.0)
        return FillQuote(fill_price=float(fill_price), fee=float(fee))

    def describe(self) -> dict[str, float | str]:
        return {
            "cost_model_type": "combined_bps_scan",
            "commission_bps": float(self.commission_bps),
            "slippage_bps": float(self.slippage_bps),
            "spread_bps": float(self.spread_bps),
        }


def run_cost_sensitivity_scan(
    strategy: Strategy,
    bars: pd.DataFrame,
    initial_cash: float,
    commission_grid: list[float] | tuple[float, ...] = (0.0, 5.0, 10.0),
    slippage_grid: list[float] | tuple[float, ...] = (0.0, 5.0, 10.0),
    spread_grid: list[float] | tuple[float, ...] = (0.0, 5.0, 10.0),
) -> pd.DataFrame:
    """Scan key metrics under commission/slippage/spread parameter grid."""
    commission_values = _normalize_grid("commission_grid", commission_grid)
    slippage_values = _normalize_grid("slippage_grid", slippage_grid)
    spread_values = _normalize_grid("spread_grid", spread_grid)

    records: list[dict[str, float | str]] = []
    engine = BacktestEngine()
    for commission_bps, slippage_bps, spread_bps in product(
        commission_values,
        slippage_values,
        spread_values,
    ):
        model = CombinedBpsCostModel(
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            spread_bps=spread_bps,
        )
        result = engine.run(
            strategy=strategy,
            bars=bars,
            initial_cash=initial_cash,
            cost_model=model,
        )
        metrics = result.metrics
        records.append(
            {
                "cost_assumption": _cost_assumption_label(
                    commission_bps=commission_bps,
                    slippage_bps=slippage_bps,
                    spread_bps=spread_bps,
                ),
                "commission_bps": float(commission_bps),
                "slippage_bps": float(slippage_bps),
                "spread_bps": float(spread_bps),
                "total_cost_bps": float(commission_bps + slippage_bps + spread_bps),
                "total_return": float(metrics.get("total_return", 0.0)),
                "cagr": float(metrics.get("cagr", 0.0)),
                "sharpe": float(metrics.get("sharpe", 0.0)),
                "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
                "turnover": float(metrics.get("turnover", 0.0)),
            }
        )

    sensitivity_frame = pd.DataFrame(records)
    if sensitivity_frame.empty:
        return sensitivity_frame

    return sensitivity_frame.sort_values(
        by=["total_cost_bps", "commission_bps", "slippage_bps", "spread_bps"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


def plot_cost_sensitivity(
    sensitivity_frame: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Plot simple line chart for sensitivity trends."""
    if sensitivity_frame.empty:
        msg = "sensitivity_frame cannot be empty."
        raise ValueError(msg)

    required_columns = {"total_cost_bps", "sharpe", "cagr"}
    missing_columns = required_columns.difference(sensitivity_frame.columns)
    if missing_columns:
        msg = f"sensitivity_frame missing required columns: {sorted(missing_columns)}"
        raise ValueError(msg)

    rendered = (
        sensitivity_frame.groupby("total_cost_bps", as_index=False)[["sharpe", "cagr"]]
        .mean(numeric_only=True)
        .sort_values("total_cost_bps")
    )
    if rendered.empty:
        msg = "No values available to plot sensitivity lines."
        raise ValueError(msg)

    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(
        rendered["total_cost_bps"],
        rendered["sharpe"],
        marker="o",
        linewidth=1.8,
        label="Sharpe",
    )
    axis.plot(
        rendered["total_cost_bps"],
        rendered["cagr"],
        marker="o",
        linewidth=1.8,
        label="CAGR",
    )
    axis.set_xlabel("Total Cost (bps)")
    axis.set_ylabel("Metric Value")
    axis.set_title("Cost Sensitivity Scan")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()

    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target_path, dpi=144)
    plt.close(figure)
    return target_path


def _normalize_grid(name: str, values: list[float] | tuple[float, ...]) -> list[float]:
    if not values:
        msg = f"{name} must not be empty."
        raise ValueError(msg)

    normalized: list[float] = []
    for raw_value in values:
        value = float(raw_value)
        if value < 0:
            msg = f"{name} values must be non-negative."
            raise ValueError(msg)
        normalized.append(value)
    return sorted(set(normalized))


def _cost_assumption_label(
    commission_bps: float,
    slippage_bps: float,
    spread_bps: float,
) -> str:
    return f"comm={commission_bps:.1f}bps; slip={slippage_bps:.1f}bps; spread={spread_bps:.1f}bps"
