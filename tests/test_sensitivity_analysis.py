from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from stratcheck.analysis import plot_cost_sensitivity, run_cost_sensitivity_scan
from stratcheck.core.strategy import MovingAverageCrossStrategy


def test_run_cost_sensitivity_scan_outputs_expected_grid_rows() -> None:
    bars = _build_sample_bars(periods=120)
    strategy = MovingAverageCrossStrategy(short_window=5, long_window=15, target_position_qty=1.0)

    sensitivity_frame = run_cost_sensitivity_scan(
        strategy=strategy,
        bars=bars,
        initial_cash=100_000.0,
        commission_grid=(0.0, 5.0),
        slippage_grid=(0.0, 5.0),
        spread_grid=(0.0, 5.0),
    )

    assert len(sensitivity_frame) == 8
    assert {
        "cost_assumption",
        "commission_bps",
        "slippage_bps",
        "spread_bps",
        "total_cost_bps",
        "sharpe",
        "cagr",
        "max_drawdown",
    }.issubset(sensitivity_frame.columns)
    assert float(sensitivity_frame["total_cost_bps"].min()) == 0.0
    assert float(sensitivity_frame["total_cost_bps"].max()) == 15.0


def test_plot_cost_sensitivity_writes_png_file(tmp_path: Path) -> None:
    bars = _build_sample_bars(periods=90)
    strategy = MovingAverageCrossStrategy(short_window=5, long_window=15, target_position_qty=1.0)
    sensitivity_frame = run_cost_sensitivity_scan(
        strategy=strategy,
        bars=bars,
        initial_cash=100_000.0,
        commission_grid=(0.0, 5.0),
        slippage_grid=(0.0, 5.0),
        spread_grid=(0.0, 5.0),
    )

    output_path = tmp_path / "cost_sensitivity.png"
    rendered_path = plot_cost_sensitivity(
        sensitivity_frame=sensitivity_frame, output_path=output_path
    )

    assert rendered_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def _build_sample_bars(periods: int) -> pd.DataFrame:
    timestamps = pd.date_range("2023-01-01", periods=periods, freq="D", tz="UTC")
    random_generator = np.random.default_rng(17)
    base_curve = 100.0 + np.sin(np.linspace(0.0, 16.0, periods)) * 4.0
    trend_curve = np.linspace(0.0, 10.0, periods)
    close_prices = base_curve + trend_curve + random_generator.normal(0.0, 0.3, size=periods)
    bars = pd.DataFrame(
        {
            "open": close_prices - 0.2,
            "high": close_prices + 1.1,
            "low": close_prices - 1.1,
            "close": close_prices,
            "volume": np.full(periods, 10_000),
        },
        index=timestamps,
    )
    return bars
