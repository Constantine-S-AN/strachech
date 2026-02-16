from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from stratcheck.tuning import plot_tuning_robustness, tune_strategy_parameters


def test_tune_strategy_parameters_grid_returns_best_params() -> None:
    bars = _build_bars(periods=140)
    result = tune_strategy_parameters(
        bars=bars,
        strategy_reference="stratcheck.core.strategy:MovingAverageCrossStrategy",
        base_params={"short_window": 5, "long_window": 20, "target_position_qty": 1.0},
        search_space={
            "short_window": [4, 6],
            "long_window": [16, 24],
        },
        initial_cash=100_000.0,
        method="grid",
        window_size="60D",
        step_size="30D",
        drawdown_penalty=1.5,
    )

    assert result.method == "grid"
    assert result.total_trials == 4
    assert len(result.trials) == 4
    assert result.successful_trials > 0
    assert {"short_window", "long_window"}.issubset(result.best_params.keys())
    assert np.isfinite(result.best_score)
    assert {"objective_score", "worst_window_sharpe", "worst_window_drawdown"}.issubset(
        result.trials.columns
    )


def test_tune_strategy_parameters_random_respects_n_iter() -> None:
    bars = _build_bars(periods=140)
    result = tune_strategy_parameters(
        bars=bars,
        strategy_reference="stratcheck.core.strategy:MovingAverageCrossStrategy",
        base_params={"short_window": 5, "long_window": 20, "target_position_qty": 1.0},
        search_space={
            "short_window": [3, 4, 5, 6],
            "long_window": [16, 20, 24, 28],
        },
        initial_cash=100_000.0,
        method="random",
        n_iter=5,
        random_seed=7,
        window_size="60D",
        step_size="30D",
    )

    assert result.method == "random"
    assert result.total_trials == 5
    assert len(result.trials) == 5
    assert result.successful_trials >= 1


def test_plot_tuning_robustness_writes_png(tmp_path: Path) -> None:
    trials = pd.DataFrame(
        [
            {
                "trial_index": 0,
                "objective_score": 0.10,
                "worst_window_sharpe": 0.22,
                "worst_window_drawdown": -0.06,
                "status": "ok",
            },
            {
                "trial_index": 1,
                "objective_score": 0.14,
                "worst_window_sharpe": 0.28,
                "worst_window_drawdown": -0.07,
                "status": "ok",
            },
            {
                "trial_index": 2,
                "objective_score": np.nan,
                "worst_window_sharpe": np.nan,
                "worst_window_drawdown": np.nan,
                "status": "failed",
            },
        ]
    )
    path = tmp_path / "tuning.png"
    rendered = plot_tuning_robustness(trials=trials, output_path=path)

    assert rendered == path
    assert path.exists()
    assert path.stat().st_size > 0


def _build_bars(periods: int) -> pd.DataFrame:
    timestamps = pd.date_range("2023-01-01", periods=periods, freq="D", tz="UTC")
    random_generator = np.random.default_rng(33)
    close_prices = 100.0 + np.cumsum(random_generator.normal(loc=0.06, scale=0.9, size=periods))
    bars = pd.DataFrame(
        {
            "open": close_prices - 0.3,
            "high": close_prices + 1.0,
            "low": close_prices - 1.1,
            "close": close_prices,
            "volume": np.full(periods, 1_000),
        },
        index=timestamps,
    )
    return bars
