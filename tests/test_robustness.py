from __future__ import annotations

import numpy as np
import pandas as pd
from stratcheck.core.robustness import bootstrap_sharpe_ci, parameter_sweep


def test_bootstrap_sharpe_ci_returns_ordered_bounds() -> None:
    returns = pd.Series([0.01, -0.005, 0.004, 0.002, -0.001, 0.003], dtype=float)
    low, high = bootstrap_sharpe_ci(
        returns=returns,
        n=200,
        confidence=0.90,
        random_seed=7,
    )

    assert low <= high
    assert np.isfinite(low)
    assert np.isfinite(high)


def test_parameter_sweep_outputs_grid_rows_and_metric() -> None:
    timestamps = pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC")
    close_prices = np.linspace(100.0, 130.0, 60)
    bars = pd.DataFrame(
        {
            "open": close_prices,
            "high": close_prices + 1.0,
            "low": close_prices - 1.0,
            "close": close_prices,
            "volume": np.full(60, 1_000),
        },
        index=timestamps,
    )

    sweep_frame = parameter_sweep(
        bars=bars,
        strategy_reference="stratcheck.core.strategy:MovingAverageCrossStrategy",
        base_params={"short_window": 4, "long_window": 10, "target_position_qty": 1.0},
        param_grid={"short_window": [3, 5], "long_window": [9, 12]},
        initial_cash=100_000.0,
        metric_key="sharpe",
    )

    assert len(sweep_frame) == 4
    assert {"short_window", "long_window", "sharpe", "status", "error"}.issubset(
        sweep_frame.columns
    )
    assert (sweep_frame["status"] == "ok").all()
