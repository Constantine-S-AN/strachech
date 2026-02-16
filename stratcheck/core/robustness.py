"""Robustness utilities: uncertainty intervals and parameter sweep."""

from __future__ import annotations

import importlib
import itertools
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from stratcheck.core.backtest import BacktestEngine, CostModel
from stratcheck.core.strategy import Strategy


def bootstrap_sharpe_ci(
    returns: pd.Series,
    n: int = 1000,
    confidence: float = 0.95,
    periods_per_year: float = 252.0,
    random_seed: int = 42,
) -> tuple[float, float]:
    """Estimate Sharpe ratio confidence interval via bootstrap resampling."""
    if n <= 0:
        msg = "n must be positive."
        raise ValueError(msg)
    if not (0.0 < confidence < 1.0):
        msg = "confidence must be between 0 and 1."
        raise ValueError(msg)
    if periods_per_year <= 0:
        msg = "periods_per_year must be positive."
        raise ValueError(msg)

    clean_returns = pd.Series(returns, dtype=float).dropna()
    if len(clean_returns) < 2:
        return 0.0, 0.0

    rng = np.random.default_rng(random_seed)
    values = clean_returns.to_numpy(dtype=float)
    sharpe_samples: list[float] = []

    for _ in range(n):
        sampled = rng.choice(values, size=len(values), replace=True)
        sample_std = float(np.std(sampled, ddof=0))
        if sample_std <= 1e-12:
            sharpe_samples.append(0.0)
            continue
        sample_mean = float(np.mean(sampled))
        sharpe_value = sample_mean / sample_std * np.sqrt(periods_per_year)
        sharpe_samples.append(float(sharpe_value))

    lower_quantile = (1.0 - confidence) / 2.0
    upper_quantile = 1.0 - lower_quantile
    low = float(np.quantile(sharpe_samples, lower_quantile))
    high = float(np.quantile(sharpe_samples, upper_quantile))
    return low, high


def parameter_sweep(
    bars: pd.DataFrame,
    strategy_reference: str,
    base_params: Mapping[str, Any],
    param_grid: Mapping[str, Sequence[Any]],
    initial_cash: float,
    cost_model: CostModel | None = None,
    metric_key: str = "sharpe",
) -> pd.DataFrame:
    """Run grid-search backtests and return row-wise heatmap data."""
    if not param_grid:
        msg = "param_grid cannot be empty."
        raise ValueError(msg)

    param_names = list(param_grid.keys())
    param_values: list[list[Any]] = []
    for name in param_names:
        values = list(param_grid[name])
        if not values:
            msg = f"Parameter '{name}' has an empty candidate list."
            raise ValueError(msg)
        param_values.append(values)

    strategy_class = _load_strategy_class(strategy_reference)
    engine = BacktestEngine()
    records: list[dict[str, Any]] = []

    for combo in itertools.product(*param_values):
        combo_params = dict(zip(param_names, combo, strict=True))
        run_params = dict(base_params)
        run_params.update(combo_params)

        if _is_invalid_moving_average_combo(run_params):
            records.append(
                {
                    **combo_params,
                    metric_key: np.nan,
                    "status": "failed",
                    "error": "short_window must be smaller than long_window.",
                }
            )
            continue

        try:
            strategy_instance = strategy_class(**run_params)
            if not isinstance(strategy_instance, Strategy):
                msg = "Loaded strategy must implement generate_orders(bars, portfolio_state)."
                raise TypeError(msg)

            backtest_result = engine.run(
                strategy=strategy_instance,
                bars=bars,
                initial_cash=initial_cash,
                cost_model=cost_model,
            )
            records.append(
                {
                    **combo_params,
                    metric_key: float(backtest_result.metrics.get(metric_key, np.nan)),
                    "status": "ok",
                    "error": "",
                }
            )
        except Exception as error:
            records.append(
                {
                    **combo_params,
                    metric_key: np.nan,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    return pd.DataFrame(records)


def _load_strategy_class(strategy_reference: str):
    if ":" not in strategy_reference:
        msg = "strategy_reference must be in 'module:Class' format."
        raise ValueError(msg)

    module_name, class_name = strategy_reference.split(":", maxsplit=1)
    strategy_module = importlib.import_module(module_name)
    strategy_class = getattr(strategy_module, class_name, None)
    if strategy_class is None:
        msg = f"Strategy class not found: {strategy_reference}"
        raise ValueError(msg)
    return strategy_class


def _is_invalid_moving_average_combo(params: Mapping[str, Any]) -> bool:
    if "short_window" not in params or "long_window" not in params:
        return False

    short_window = params.get("short_window")
    long_window = params.get("long_window")
    if not isinstance(short_window, (int, float)) or not isinstance(long_window, (int, float)):
        return False
    return int(short_window) >= int(long_window)
