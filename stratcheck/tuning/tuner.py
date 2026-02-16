"""Grid/random tuning with walk-forward anti-overfitting objective."""

from __future__ import annotations

import importlib
import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stratcheck.core.backtest import BacktestEngine, CostModel
from stratcheck.core.calendar import MarketCalendar, normalize_bars_freq
from stratcheck.core.metrics import compute_metrics
from stratcheck.core.strategy import Strategy


@dataclass(slots=True)
class TuningResult:
    """Tuning run output bundle."""

    method: Literal["grid", "random"]
    objective_name: str
    trials: pd.DataFrame
    best_params: dict[str, Any]
    best_score: float
    successful_trials: int
    total_trials: int


def tune_strategy_parameters(
    bars: pd.DataFrame,
    strategy_reference: str,
    base_params: Mapping[str, Any],
    search_space: Mapping[str, Sequence[Any]],
    initial_cash: float,
    cost_model: CostModel | None = None,
    method: Literal["grid", "random"] = "grid",
    n_iter: int = 20,
    random_seed: int = 42,
    window_size: str = "126D",
    step_size: str = "63D",
    bars_freq: str = "1d",
    drawdown_penalty: float = 2.0,
) -> TuningResult:
    """Tune strategy parameters by maximizing worst-window Sharpe with drawdown penalty.

    Objective:
    score = worst_window_sharpe - drawdown_penalty * abs(min(worst_window_drawdown, 0))
    """
    if not search_space:
        msg = "search_space cannot be empty."
        raise ValueError(msg)
    if n_iter <= 0:
        msg = "n_iter must be positive."
        raise ValueError(msg)
    if drawdown_penalty < 0:
        msg = "drawdown_penalty must be non-negative."
        raise ValueError(msg)
    if method not in {"grid", "random"}:
        msg = "method must be 'grid' or 'random'."
        raise ValueError(msg)

    param_names, candidate_values = _normalize_search_space(search_space)
    candidates = _build_candidates(
        param_names=param_names,
        candidate_values=candidate_values,
        method=method,
        n_iter=n_iter,
        random_seed=random_seed,
    )
    strategy_class = _load_strategy_class(strategy_reference)

    records: list[dict[str, Any]] = []
    for trial_index, candidate in enumerate(candidates):
        run_params = dict(base_params)
        run_params.update(candidate)

        if _is_invalid_moving_average_combo(run_params):
            records.append(
                {
                    "trial_index": trial_index,
                    **candidate,
                    "objective_score": np.nan,
                    "worst_window_sharpe": np.nan,
                    "worst_window_drawdown": np.nan,
                    "mean_window_sharpe": np.nan,
                    "status": "failed",
                    "error": "short_window must be smaller than long_window.",
                    "params_json": json.dumps(run_params, ensure_ascii=False, sort_keys=True),
                }
            )
            continue

        try:
            strategy = strategy_class(**run_params)
            if not isinstance(strategy, Strategy):
                msg = "Loaded strategy must implement generate_orders(bars, portfolio_state)."
                raise TypeError(msg)

            (
                objective_score,
                worst_window_sharpe,
                worst_window_drawdown,
                mean_window_sharpe,
            ) = _evaluate_walk_forward_objective(
                strategy=strategy,
                bars=bars,
                initial_cash=initial_cash,
                cost_model=cost_model,
                window_size=window_size,
                step_size=step_size,
                bars_freq=bars_freq,
                drawdown_penalty=drawdown_penalty,
            )
            records.append(
                {
                    "trial_index": trial_index,
                    **candidate,
                    "objective_score": objective_score,
                    "worst_window_sharpe": worst_window_sharpe,
                    "worst_window_drawdown": worst_window_drawdown,
                    "mean_window_sharpe": mean_window_sharpe,
                    "status": "ok",
                    "error": "",
                    "params_json": json.dumps(run_params, ensure_ascii=False, sort_keys=True),
                }
            )
        except Exception as error:
            records.append(
                {
                    "trial_index": trial_index,
                    **candidate,
                    "objective_score": np.nan,
                    "worst_window_sharpe": np.nan,
                    "worst_window_drawdown": np.nan,
                    "mean_window_sharpe": np.nan,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "params_json": json.dumps(run_params, ensure_ascii=False, sort_keys=True),
                }
            )

    trials = pd.DataFrame(records)
    objective_name = "worst_window_sharpe_minus_drawdown_penalty"
    successful = trials[trials["status"] == "ok"] if not trials.empty else pd.DataFrame()
    successful_count = int(len(successful))
    if successful.empty:
        best_params: dict[str, Any] = {}
        best_score = float("nan")
    else:
        best_row = successful.sort_values(
            by=["objective_score", "worst_window_sharpe"],
            ascending=[False, False],
        ).iloc[0]
        best_params = dict(base_params)
        for name in param_names:
            best_params[name] = best_row[name]
        best_score = float(best_row["objective_score"])

    return TuningResult(
        method=method,
        objective_name=objective_name,
        trials=trials,
        best_params=best_params,
        best_score=best_score,
        successful_trials=successful_count,
        total_trials=int(len(trials)),
    )


def plot_tuning_robustness(
    trials: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Plot robustness chart for tuning trials."""
    if trials.empty:
        msg = "trials cannot be empty."
        raise ValueError(msg)
    required_columns = {"trial_index", "objective_score", "worst_window_sharpe", "status"}
    missing = required_columns.difference(trials.columns)
    if missing:
        msg = f"trials missing required columns: {sorted(missing)}"
        raise ValueError(msg)

    ok_trials = trials[trials["status"] == "ok"].copy()
    if ok_trials.empty:
        msg = "No successful trials to plot."
        raise ValueError(msg)

    ok_trials = ok_trials.sort_values("trial_index").reset_index(drop=True)
    ok_trials["best_so_far"] = ok_trials["objective_score"].cummax()

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    axes[0].plot(
        ok_trials["trial_index"],
        ok_trials["objective_score"],
        marker="o",
        linewidth=1.6,
        color="#2563eb",
        label="objective_score",
    )
    axes[0].plot(
        ok_trials["trial_index"],
        ok_trials["best_so_far"],
        linewidth=1.8,
        color="#059669",
        label="best_so_far",
    )
    axes[0].set_title("Tuning Objective by Trial")
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel("Objective Score")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    x_values = ok_trials["worst_window_drawdown"].abs()
    y_values = ok_trials["worst_window_sharpe"]
    colors = ok_trials["objective_score"]
    scatter = axes[1].scatter(
        x_values,
        y_values,
        c=colors,
        cmap="viridis",
        edgecolor="black",
        linewidth=0.4,
    )
    axes[1].set_title("Worst Sharpe vs Drawdown")
    axes[1].set_xlabel("|Worst Window Drawdown|")
    axes[1].set_ylabel("Worst Window Sharpe")
    axes[1].grid(alpha=0.25)
    figure.colorbar(scatter, ax=axes[1], label="Objective Score")

    figure.tight_layout()
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target_path, dpi=150)
    plt.close(figure)
    return target_path


def _evaluate_walk_forward_objective(
    strategy: Strategy,
    bars: pd.DataFrame,
    initial_cash: float,
    cost_model: CostModel | None,
    window_size: str,
    step_size: str,
    bars_freq: str,
    drawdown_penalty: float,
) -> tuple[float, float, float, float]:
    normalized_bars = _normalize_bars(bars)
    canonical_freq = normalize_bars_freq(bars_freq)
    calendar = MarketCalendar()
    engine = BacktestEngine()

    window_sharpes: list[float] = []
    window_drawdowns: list[float] = []
    for window in calendar.split_rolling_windows(
        bars=normalized_bars,
        window_size=window_size,
        step_size=step_size,
        bars_freq=canonical_freq,
    ):
        result = engine.run(
            strategy=strategy,
            bars=window.bars,
            initial_cash=initial_cash,
            cost_model=cost_model,
        )
        metrics = compute_metrics(
            equity_curve=result.equity_curve,
            trades=result.trades,
            bars_freq=canonical_freq,
        )
        window_sharpes.append(float(metrics.get("sharpe", 0.0)))
        window_drawdowns.append(float(metrics.get("max_drawdown", 0.0)))

    if not window_sharpes:
        msg = "No walk-forward windows generated for objective."
        raise ValueError(msg)

    worst_window_sharpe = float(min(window_sharpes))
    worst_window_drawdown = float(min(window_drawdowns))
    drawdown_size = abs(min(worst_window_drawdown, 0.0))
    objective_score = worst_window_sharpe - drawdown_penalty * drawdown_size
    mean_window_sharpe = float(np.mean(window_sharpes))
    return objective_score, worst_window_sharpe, worst_window_drawdown, mean_window_sharpe


def _normalize_search_space(
    search_space: Mapping[str, Sequence[Any]],
) -> tuple[list[str], list[list[Any]]]:
    param_names = list(search_space.keys())
    if not param_names:
        msg = "search_space must contain at least one parameter."
        raise ValueError(msg)

    candidate_values: list[list[Any]] = []
    for name in param_names:
        values = list(search_space[name])
        if not values:
            msg = f"Parameter '{name}' has an empty candidate list."
            raise ValueError(msg)
        candidate_values.append(values)
    return param_names, candidate_values


def _build_candidates(
    param_names: list[str],
    candidate_values: list[list[Any]],
    method: Literal["grid", "random"],
    n_iter: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    all_combinations = [
        dict(zip(param_names, combo, strict=True)) for combo in itertools.product(*candidate_values)
    ]
    if method == "grid":
        return all_combinations

    if len(all_combinations) <= n_iter:
        return all_combinations

    random_generator = np.random.default_rng(random_seed)
    selected_indices = random_generator.choice(
        len(all_combinations),
        size=n_iter,
        replace=False,
    )
    return [all_combinations[int(index)] for index in selected_indices]


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


def _normalize_bars(bars: pd.DataFrame) -> pd.DataFrame:
    if len(bars) == 0:
        msg = "bars cannot be empty."
        raise ValueError(msg)
    if not isinstance(bars.index, pd.DatetimeIndex):
        msg = "bars must use DatetimeIndex."
        raise ValueError(msg)

    normalized = bars.sort_index().copy()
    normalized = normalized[~normalized.index.duplicated(keep="last")]
    return normalized


def _is_invalid_moving_average_combo(params: Mapping[str, Any]) -> bool:
    if "short_window" not in params or "long_window" not in params:
        return False
    short_window = params.get("short_window")
    long_window = params.get("long_window")
    if not isinstance(short_window, (int, float)) or not isinstance(long_window, (int, float)):
        return False
    return int(short_window) >= int(long_window)
