"""Command-line interface for stratcheck."""

from __future__ import annotations

import argparse
import importlib
import re
import shlex
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from stratcheck.analysis import (
    compute_execution_quality_scorecard,
    compute_regime_scorecard,
    plot_cost_sensitivity,
    run_cost_sensitivity_scan,
)
from stratcheck.audit import RunAuditStore
from stratcheck.core import (
    BacktestEngine,
    CSVDataProvider,
    FixedBpsCostModel,
    MovingAverageCrossStrategy,
    bootstrap_sharpe_ci,
    build_cost_model,
    evaluate_overfit_risk,
    generate_random_ohlcv,
    parameter_sweep,
    run_healthcheck,
)
from stratcheck.core.backtest import CostModel, OrderRecord
from stratcheck.core.bundle import (
    RunSnapshot,
    build_run_snapshot,
    bundle_snapshot,
    detect_source_data_path,
    reproduce_snapshot,
)
from stratcheck.core.strategy import Fill, Strategy
from stratcheck.dashboard import build_dashboard_site
from stratcheck.quality import GuardViolationError
from stratcheck.report import ReportBuilder, generate_performance_plots
from stratcheck.tuning import plot_tuning_robustness, tune_strategy_parameters
from stratcheck.validation import validate_against_vectorized_baseline


@dataclass(slots=True)
class AppConfig:
    """Configuration required for `run` and `healthcheck` commands."""

    symbol: str
    data_path: Path
    timeframe: str
    strategy_reference: str
    strategy_params: dict[str, Any]
    initial_cash: float
    cost_model: CostModel
    window_size: str
    step_size: str
    bars_freq: str
    report_name: str
    report_dir: Path
    summary_json_path: Path
    start: str | None
    end: str | None
    bootstrap_samples: int
    bootstrap_confidence: float
    sweep_param_grid: dict[str, list[Any]]
    sweep_metric: str
    use_parquet_cache: bool
    parquet_cache_dir: Path | None
    tuning_enabled: bool
    tuning_method: Literal["grid", "random"]
    tuning_n_iter: int
    tuning_random_seed: int
    tuning_window_size: str
    tuning_step_size: str
    tuning_bars_freq: str
    tuning_drawdown_penalty: float
    tuning_search_space: dict[str, list[Any]]
    raw_config: dict[str, Any]


@dataclass(slots=True)
class RunExecution:
    """Execution outputs for one CLI run/healthcheck command."""

    report_path: Path
    run_id: str
    run_dir: Path
    summary_json_path: Path | None = None


def run_demo(output_path: Path, periods: int = 400, seed: int = 42) -> Path:
    """Run demo with random bars, built-in strategy, and HTML report output."""
    bars = generate_random_ohlcv(periods=periods, seed=seed)
    strategy = MovingAverageCrossStrategy(short_window=20, long_window=50, target_position_qty=1.0)
    engine = BacktestEngine()
    cost_model = FixedBpsCostModel(commission_bps=5.0, slippage_bps=5.0)
    demo_config = {
        "report_name": output_path.stem,
        "symbol": "DEMO",
        "periods": periods,
        "seed": seed,
        "strategy": "stratcheck.core.strategy:MovingAverageCrossStrategy",
        "initial_cash": 100_000.0,
        **cost_model.describe(),
    }
    try:
        result = engine.run(
            strategy=strategy,
            bars=bars,
            initial_cash=100_000.0,
            cost_model=cost_model,
        )
    except GuardViolationError as guard_error:
        report_builder = ReportBuilder(output_dir=output_path.parent)
        failure_report_path = _build_guard_failure_report(
            builder=report_builder,
            report_config=demo_config,
            full_config=demo_config,
            bars=bars,
            guard_error=guard_error,
            reproduce_command=(
                f"python -m stratcheck demo --output {shlex.quote(str(output_path))} "
                f"--periods {periods} --seed {seed}"
            ),
        )
        msg = f"{guard_error} Guard report generated: {failure_report_path.resolve()}"
        raise RuntimeError(msg) from guard_error

    plot_paths = generate_performance_plots(
        equity_curve=result.equity_curve,
        trades=result.trades,
        output_dir=output_path.parent / "assets",
        prefix=output_path.stem,
    )
    window_metrics = _single_window_metrics_table(bars=bars, metrics=result.metrics)
    returns = result.equity_curve.pct_change().dropna()
    sharpe_ci_low, sharpe_ci_high = bootstrap_sharpe_ci(returns=returns, n=500)
    overfit_summary, risk_flags = evaluate_overfit_risk(
        returns=returns,
        window_metrics_df=window_metrics,
    )
    sweep_results_df = parameter_sweep(
        bars=bars,
        strategy_reference="stratcheck.core.strategy:MovingAverageCrossStrategy",
        base_params={
            "short_window": 20,
            "long_window": 50,
            "target_position_qty": 1.0,
        },
        param_grid={
            "short_window": [10, 20, 30],
            "long_window": [40, 50, 60],
        },
        initial_cash=100_000.0,
        cost_model=cost_model,
        metric_key="sharpe",
    )
    risk_flag_records = _risk_flag_records(risk_flags)
    validation_summary = validate_against_vectorized_baseline(
        strategy=strategy,
        bars=bars,
        engine_equity_curve=result.equity_curve,
        initial_cash=100_000.0,
        cost_model=cost_model,
    )
    risk_flag_records.extend(_validation_flag_records(validation_summary))
    sensitivity_metrics_df, sensitivity_plot_paths = _compute_sensitivity_outputs(
        strategy=strategy,
        bars=bars,
        initial_cash=100_000.0,
        report_dir=output_path.parent,
        report_name=output_path.stem,
    )
    regime_scorecard_df = compute_regime_scorecard(
        bars=bars,
        equity_curve=result.equity_curve,
        trades=result.trades,
    )
    execution_quality_df = compute_execution_quality_scorecard(
        orders=result.orders,
        bars=bars,
        trades=result.trades,
    )

    builder = ReportBuilder(output_dir=output_path.parent)
    return builder.build(
        overall_metrics=result.metrics,
        window_metrics_df=window_metrics,
        plot_paths=plot_paths,
        robustness_summary={
            "bootstrap_samples": 500,
            "sharpe_ci_low": sharpe_ci_low,
            "sharpe_ci_high": sharpe_ci_high,
            **overfit_summary,
        },
        sweep_results_df=sweep_results_df,
        sensitivity_metrics_df=sensitivity_metrics_df,
        sensitivity_plot_paths=sensitivity_plot_paths,
        regime_scorecard_df=regime_scorecard_df,
        execution_quality_df=execution_quality_df,
        risk_flags=risk_flag_records,
        validation_summary=validation_summary,
        reproduce_command=(
            f"python -m stratcheck demo --output {shlex.quote(str(output_path))} "
            f"--periods {periods} --seed {seed}"
        ),
        full_config=demo_config,
        config=demo_config,
    )


def run_with_config(config_path: Path) -> Path:
    """Run a single backtest from TOML config and generate HTML report."""
    execution = _run_with_config_execution(config_path=config_path)
    return execution.report_path


def run_healthcheck_with_config(config_path: Path) -> tuple[Path, Path]:
    """Run walk-forward healthcheck from TOML config, report + summary JSON."""
    execution = _run_healthcheck_with_config_execution(config_path=config_path)
    if execution.summary_json_path is None:
        msg = "healthcheck execution did not produce a summary JSON path."
        raise RuntimeError(msg)
    return execution.report_path, execution.summary_json_path


def bundle_run(
    run_id: str,
    runs_dir: Path = Path("reports/runs"),
    output_path: Path | None = None,
) -> Path:
    """Export one existing run snapshot to a zip file."""
    return bundle_snapshot(
        run_id=run_id,
        runs_dir=runs_dir,
        output_path=output_path,
    )


def reproduce_bundle(
    bundle_path: Path,
    output_dir: Path = Path("reports/reproduced"),
) -> tuple[str, Path]:
    """Reproduce report from bundle archive and return run-id + report path."""
    reproduction = reproduce_snapshot(
        bundle_path=bundle_path,
        output_dir=output_dir,
    )
    return reproduction.run_id, reproduction.report_path


def _run_with_config_execution(config_path: Path) -> RunExecution:
    app_config = load_app_config(config_path)
    bars, actions_summary = _load_bars(app_config)
    strategy = _load_strategy(
        strategy_reference=app_config.strategy_reference,
        strategy_params=app_config.strategy_params,
    )

    engine = BacktestEngine()
    try:
        result = engine.run(
            strategy=strategy,
            bars=bars,
            initial_cash=app_config.initial_cash,
            cost_model=app_config.cost_model,
        )
    except GuardViolationError as guard_error:
        report_builder = ReportBuilder(output_dir=app_config.report_dir)
        failure_report_path = _build_guard_failure_report(
            builder=report_builder,
            report_config=_config_for_report(app_config),
            full_config=app_config.raw_config,
            bars=bars,
            guard_error=guard_error,
            reproduce_command=f"python -m stratcheck run --config {shlex.quote(str(config_path))}",
            corporate_actions_summary=actions_summary,
        )
        msg = f"{guard_error} Guard report generated: {failure_report_path.resolve()}"
        raise RuntimeError(msg) from guard_error

    plot_paths = generate_performance_plots(
        equity_curve=result.equity_curve,
        trades=result.trades,
        output_dir=app_config.report_dir / "assets",
        prefix=app_config.report_name,
    )
    window_metrics = _single_window_metrics_table(bars=bars, metrics=result.metrics)
    robustness_summary, sweep_results_df = _compute_robustness_outputs(
        app_config=app_config,
        bars=bars,
        equity_curve=result.equity_curve,
        window_metrics_df=window_metrics,
    )
    risk_flag_records = _risk_flag_records(robustness_summary.pop("risk_flags", []))
    validation_summary = validate_against_vectorized_baseline(
        strategy=strategy,
        bars=bars,
        engine_equity_curve=result.equity_curve,
        initial_cash=app_config.initial_cash,
        cost_model=app_config.cost_model,
    )
    risk_flag_records.extend(_validation_flag_records(validation_summary))
    sensitivity_metrics_df, sensitivity_plot_paths = _compute_sensitivity_outputs(
        strategy=strategy,
        bars=bars,
        initial_cash=app_config.initial_cash,
        report_dir=app_config.report_dir,
        report_name=app_config.report_name,
    )
    tuning_best_params, tuning_trials_df, tuning_plot_paths = _compute_tuning_outputs(
        app_config=app_config,
        bars=bars,
    )
    regime_scorecard_df = compute_regime_scorecard(
        bars=bars,
        equity_curve=result.equity_curve,
        trades=result.trades,
    )
    execution_quality_df = compute_execution_quality_scorecard(
        orders=result.orders,
        bars=bars,
        trades=result.trades,
    )

    builder = ReportBuilder(output_dir=app_config.report_dir)
    report_path = builder.build(
        overall_metrics=result.metrics,
        window_metrics_df=window_metrics,
        plot_paths=plot_paths,
        robustness_summary=robustness_summary,
        sweep_results_df=sweep_results_df,
        sensitivity_metrics_df=sensitivity_metrics_df,
        sensitivity_plot_paths=sensitivity_plot_paths,
        regime_scorecard_df=regime_scorecard_df,
        execution_quality_df=execution_quality_df,
        tuning_best_params=tuning_best_params,
        tuning_trials_df=tuning_trials_df,
        tuning_plot_paths=tuning_plot_paths,
        risk_flags=risk_flag_records,
        validation_summary=validation_summary,
        reproduce_command=f"python -m stratcheck run --config {shlex.quote(str(config_path))}",
        full_config=app_config.raw_config,
        config=_config_for_report(app_config),
        corporate_actions_summary=actions_summary,
    )

    run_snapshot = _persist_run_snapshot(
        run_mode="run",
        config_path=config_path,
        app_config=app_config,
        bars=bars,
        equity_curve=result.equity_curve,
        orders=result.orders,
        trades=result.trades,
        overall_metrics=result.metrics,
        window_metrics_df=window_metrics,
        summary_json_path=None,
        report_path=report_path,
        asset_paths=[*plot_paths, *sensitivity_plot_paths, *tuning_plot_paths],
    )

    return RunExecution(
        report_path=report_path,
        run_id=run_snapshot.run_id,
        run_dir=run_snapshot.run_dir,
    )


def _run_healthcheck_with_config_execution(config_path: Path) -> RunExecution:
    app_config = load_app_config(config_path)
    bars, actions_summary = _load_bars(app_config)
    strategy = _load_strategy(
        strategy_reference=app_config.strategy_reference,
        strategy_params=app_config.strategy_params,
    )

    engine = BacktestEngine()
    try:
        full_result = engine.run(
            strategy=strategy,
            bars=bars,
            initial_cash=app_config.initial_cash,
            cost_model=app_config.cost_model,
        )

        window_metrics, summary_json_path = run_healthcheck(
            strategy=strategy,
            bars=bars,
            window_size=app_config.window_size,
            step_size=app_config.step_size,
            initial_cash=app_config.initial_cash,
            cost_model=app_config.cost_model,
            bars_freq=app_config.bars_freq,
            output_json_path=app_config.summary_json_path,
        )
    except GuardViolationError as guard_error:
        report_builder = ReportBuilder(output_dir=app_config.report_dir)
        failure_report_path = _build_guard_failure_report(
            builder=report_builder,
            report_config=_config_for_report(app_config),
            full_config=app_config.raw_config,
            bars=bars,
            guard_error=guard_error,
            reproduce_command=(
                f"python -m stratcheck healthcheck --config {shlex.quote(str(config_path))}"
            ),
            corporate_actions_summary=actions_summary,
        )
        msg = f"{guard_error} Guard report generated: {failure_report_path.resolve()}"
        raise RuntimeError(msg) from guard_error

    plot_paths = generate_performance_plots(
        equity_curve=full_result.equity_curve,
        trades=full_result.trades,
        output_dir=app_config.report_dir / "assets",
        prefix=app_config.report_name,
    )
    robustness_summary, sweep_results_df = _compute_robustness_outputs(
        app_config=app_config,
        bars=bars,
        equity_curve=full_result.equity_curve,
        window_metrics_df=window_metrics,
    )
    risk_flag_records = _risk_flag_records(robustness_summary.pop("risk_flags", []))
    validation_summary = validate_against_vectorized_baseline(
        strategy=strategy,
        bars=bars,
        engine_equity_curve=full_result.equity_curve,
        initial_cash=app_config.initial_cash,
        cost_model=app_config.cost_model,
    )
    risk_flag_records.extend(_validation_flag_records(validation_summary))
    sensitivity_metrics_df, sensitivity_plot_paths = _compute_sensitivity_outputs(
        strategy=strategy,
        bars=bars,
        initial_cash=app_config.initial_cash,
        report_dir=app_config.report_dir,
        report_name=app_config.report_name,
    )
    tuning_best_params, tuning_trials_df, tuning_plot_paths = _compute_tuning_outputs(
        app_config=app_config,
        bars=bars,
    )
    regime_scorecard_df = compute_regime_scorecard(
        bars=bars,
        equity_curve=full_result.equity_curve,
        trades=full_result.trades,
    )
    execution_quality_df = compute_execution_quality_scorecard(
        orders=full_result.orders,
        bars=bars,
        trades=full_result.trades,
    )

    builder = ReportBuilder(output_dir=app_config.report_dir)
    report_path = builder.build(
        overall_metrics=full_result.metrics,
        window_metrics_df=window_metrics,
        plot_paths=plot_paths,
        robustness_summary=robustness_summary,
        sweep_results_df=sweep_results_df,
        sensitivity_metrics_df=sensitivity_metrics_df,
        sensitivity_plot_paths=sensitivity_plot_paths,
        regime_scorecard_df=regime_scorecard_df,
        execution_quality_df=execution_quality_df,
        tuning_best_params=tuning_best_params,
        tuning_trials_df=tuning_trials_df,
        tuning_plot_paths=tuning_plot_paths,
        risk_flags=risk_flag_records,
        validation_summary=validation_summary,
        reproduce_command=(
            f"python -m stratcheck healthcheck --config {shlex.quote(str(config_path))}"
        ),
        full_config=app_config.raw_config,
        config=_config_for_report(app_config),
        corporate_actions_summary=actions_summary,
    )

    run_snapshot = _persist_run_snapshot(
        run_mode="healthcheck",
        config_path=config_path,
        app_config=app_config,
        bars=bars,
        equity_curve=full_result.equity_curve,
        orders=full_result.orders,
        trades=full_result.trades,
        overall_metrics=full_result.metrics,
        window_metrics_df=window_metrics,
        summary_json_path=summary_json_path,
        report_path=report_path,
        asset_paths=[*plot_paths, *sensitivity_plot_paths, *tuning_plot_paths],
    )

    return RunExecution(
        report_path=report_path,
        run_id=run_snapshot.run_id,
        run_dir=run_snapshot.run_dir,
        summary_json_path=summary_json_path,
    )


def replay_run(run_id: str, sqlite_path: Path = Path("reports/paper_trading.sqlite")) -> str:
    """Replay run timeline from sqlite audit store."""
    audit_store = RunAuditStore(sqlite_path=sqlite_path)
    return audit_store.render_replay(run_id=run_id)


def run_dashboard(
    results_jsonl_path: Path = Path("reports/results.jsonl"),
    sqlite_path: Path = Path("reports/paper_trading.sqlite"),
    output_path: Path = Path("reports/dashboard.html"),
    reports_dir: Path = Path("reports"),
) -> Path:
    """Build static dashboard page from experiment and run data."""
    return build_dashboard_site(
        results_jsonl_path=results_jsonl_path,
        sqlite_path=sqlite_path,
        output_path=output_path,
        reports_dir=reports_dir,
    )


def create_strategy(
    strategy_name: str,
    strategy_dir: Path = Path("stratcheck/strategies"),
    config_dir: Path = Path("configs/examples"),
    force: bool = False,
) -> tuple[Path, Path]:
    """Create strategy plugin scaffold and a matching example config."""
    class_name = _normalize_strategy_class_name(strategy_name)
    module_name = _to_snake_case(class_name)

    strategy_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    _ensure_package_marker(strategy_dir)

    strategy_path = strategy_dir / f"{module_name}.py"
    config_path = config_dir / f"{module_name}.toml"

    if strategy_path.exists() and not force:
        msg = f"Strategy file already exists: {strategy_path}"
        raise FileExistsError(msg)
    if config_path.exists() and not force:
        msg = f"Config file already exists: {config_path}"
        raise FileExistsError(msg)

    strategy_reference = _strategy_reference_from_path(strategy_path, class_name)
    strategy_content = _render_strategy_template(class_name)
    config_content = _render_strategy_config_template(
        report_name=module_name,
        strategy_reference=strategy_reference,
    )

    strategy_path.write_text(strategy_content, encoding="utf-8")
    config_path.write_text(config_content, encoding="utf-8")
    return strategy_path, config_path


def load_app_config(config_path: Path) -> AppConfig:
    """Load and validate command config from TOML."""
    if not config_path.exists():
        msg = f"Config file not found: {config_path}"
        raise FileNotFoundError(msg)

    with config_path.open("rb") as config_file:
        raw_config = tomllib.load(config_file)

    required_keys = {"symbol", "data_path", "strategy", "initial_cash", "cost_model", "windows"}
    missing = sorted(required_keys.difference(raw_config))
    if missing:
        msg = f"config is missing required keys: {', '.join(missing)}"
        raise ValueError(msg)

    base_dir = config_path.parent
    data_path = _resolve_path(base_dir=base_dir, raw_path=str(raw_config["data_path"]))
    report_dir = _resolve_path(
        base_dir=base_dir, raw_path=str(raw_config.get("report_dir", "reports"))
    )

    cost_model_raw = raw_config["cost_model"]
    if not isinstance(cost_model_raw, dict):
        msg = "`cost_model` must be a TOML table."
        raise ValueError(msg)
    cost_model = build_cost_model(cost_model_raw)

    windows_raw = raw_config["windows"]
    if not isinstance(windows_raw, dict):
        msg = "`windows` must be a TOML table."
        raise ValueError(msg)

    window_size = str(windows_raw.get("window_size", "")).strip()
    step_size = str(windows_raw.get("step_size", "")).strip()
    if not window_size or not step_size:
        msg = "`windows.window_size` and `windows.step_size` are required."
        raise ValueError(msg)

    summary_json_raw = raw_config.get("summary_json_path")
    if summary_json_raw is None:
        summary_json_path = report_dir / "healthcheck_summary.json"
    else:
        summary_json_candidate = Path(str(summary_json_raw))
        if summary_json_candidate.is_absolute():
            summary_json_path = summary_json_candidate
        else:
            summary_json_path = report_dir / summary_json_candidate

    strategy_params = raw_config.get("strategy_params", {})
    if not isinstance(strategy_params, dict):
        msg = "`strategy_params` must be a TOML table when provided."
        raise ValueError(msg)

    (
        bootstrap_samples,
        bootstrap_confidence,
        sweep_metric,
        sweep_param_grid,
    ) = _parse_robustness_config(raw_config)
    use_parquet_cache, parquet_cache_dir = _parse_performance_config(
        raw_config=raw_config,
        base_dir=base_dir,
    )
    symbol = str(raw_config["symbol"])
    timeframe = str(raw_config.get("timeframe", "1d"))
    bars_freq = str(raw_config.get("bars_freq", timeframe))
    report_name = str(raw_config.get("report_name", f"{symbol.lower()}_report"))

    (
        tuning_enabled,
        tuning_method,
        tuning_n_iter,
        tuning_random_seed,
        tuning_window_size,
        tuning_step_size,
        tuning_bars_freq,
        tuning_drawdown_penalty,
        tuning_search_space,
    ) = _parse_tuning_config(
        raw_config=raw_config,
        default_window_size=window_size,
        default_step_size=step_size,
        default_bars_freq=bars_freq,
    )

    return AppConfig(
        symbol=symbol,
        data_path=data_path,
        timeframe=timeframe,
        strategy_reference=str(raw_config["strategy"]),
        strategy_params=dict(strategy_params),
        initial_cash=float(raw_config["initial_cash"]),
        cost_model=cost_model,
        window_size=window_size,
        step_size=step_size,
        bars_freq=bars_freq,
        report_name=report_name,
        report_dir=report_dir,
        summary_json_path=summary_json_path,
        start=_as_optional_string(raw_config.get("start")),
        end=_as_optional_string(raw_config.get("end")),
        bootstrap_samples=bootstrap_samples,
        bootstrap_confidence=bootstrap_confidence,
        sweep_param_grid=sweep_param_grid,
        sweep_metric=sweep_metric,
        use_parquet_cache=use_parquet_cache,
        parquet_cache_dir=parquet_cache_dir,
        tuning_enabled=tuning_enabled,
        tuning_method=tuning_method,
        tuning_n_iter=tuning_n_iter,
        tuning_random_seed=tuning_random_seed,
        tuning_window_size=tuning_window_size,
        tuning_step_size=tuning_step_size,
        tuning_bars_freq=tuning_bars_freq,
        tuning_drawdown_penalty=tuning_drawdown_penalty,
        tuning_search_space=tuning_search_space,
        raw_config=dict(raw_config),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        prog="stratcheck",
        description="Generate strategy health-check reports.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser("demo", help="Run demo backtest and generate report.")
    demo_parser.add_argument(
        "--output",
        default="reports/demo.html",
        help="Output HTML report path. Default: reports/demo.html",
    )
    demo_parser.add_argument(
        "--periods",
        type=int,
        default=400,
        help="Number of generated candles. Default: 400",
    )
    demo_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data generation. Default: 42",
    )

    run_parser = subparsers.add_parser(
        "run", help="Run backtest from config.toml and generate report."
    )
    run_parser.add_argument(
        "--config",
        required=True,
        help="Path to config.toml",
    )

    healthcheck_parser = subparsers.add_parser(
        "healthcheck",
        help="Run walk-forward healthcheck from config.toml.",
    )
    healthcheck_parser.add_argument(
        "--config",
        required=True,
        help="Path to config.toml",
    )

    bundle_parser = subparsers.add_parser(
        "bundle",
        help="Export one previous run snapshot into a reproducible zip package.",
    )
    bundle_parser.add_argument(
        "--run-id",
        required=True,
        help="Run ID under reports/runs/<run-id>.",
    )
    bundle_parser.add_argument(
        "--runs-dir",
        default="reports/runs",
        help="Run snapshots directory. Default: reports/runs",
    )
    bundle_parser.add_argument(
        "--output",
        default=None,
        help="Output zip path. Default: reports/bundles/<run-id>.zip",
    )

    reproduce_parser = subparsers.add_parser(
        "reproduce",
        help="Reproduce one run report from a bundle archive.",
    )
    reproduce_parser.add_argument(
        "bundle_path",
        help="Path to bundle zip file.",
    )
    reproduce_parser.add_argument(
        "--output-dir",
        default="reports/reproduced",
        help="Directory to extract and reproduce report. Default: reports/reproduced",
    )

    replay_parser = subparsers.add_parser(
        "replay",
        help="Replay one audited run timeline from sqlite by run-id.",
    )
    replay_parser.add_argument(
        "--run-id",
        required=True,
        help="Run ID recorded in audit database.",
    )
    replay_parser.add_argument(
        "--db",
        default="reports/paper_trading.sqlite",
        help="Path to sqlite audit database. Default: reports/paper_trading.sqlite",
    )

    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Build a local static dashboard page for experiments and paper runs.",
    )
    dashboard_parser.add_argument(
        "--results-jsonl",
        default="reports/results.jsonl",
        help="Path to experiment results JSONL. Default: reports/results.jsonl",
    )
    dashboard_parser.add_argument(
        "--db",
        default="reports/paper_trading.sqlite",
        help="Path to sqlite audit database. Default: reports/paper_trading.sqlite",
    )
    dashboard_parser.add_argument(
        "--output",
        default="reports/dashboard.html",
        help="Output dashboard HTML path. Default: reports/dashboard.html",
    )
    dashboard_parser.add_argument(
        "--reports-dir",
        default="reports",
        help="Reports directory for report links. Default: reports",
    )

    scaffold_parser = subparsers.add_parser(
        "create-strategy",
        help="Generate strategy plugin template and example config.",
    )
    scaffold_parser.add_argument(
        "name",
        help="Strategy class name, e.g. MyStrategy",
    )
    scaffold_parser.add_argument(
        "--strategy-dir",
        default="stratcheck/strategies",
        help="Directory to write strategy module. Default: stratcheck/strategies",
    )
    scaffold_parser.add_argument(
        "--config-dir",
        default="configs/examples",
        help="Directory to write example config. Default: configs/examples",
    )
    scaffold_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite files if they already exist.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "demo":
            report_path = run_demo(
                output_path=Path(args.output),
                periods=args.periods,
                seed=args.seed,
            )
            print(f"Report generated: {report_path.resolve()}")
            return 0

        if args.command == "run":
            execution = _run_with_config_execution(config_path=Path(args.config))
            print(f"Report generated: {execution.report_path.resolve()}")
            print(f"Run ID: {execution.run_id}")
            print(f"Run snapshot: {execution.run_dir.resolve()}")
            return 0

        if args.command == "healthcheck":
            execution = _run_healthcheck_with_config_execution(
                config_path=Path(args.config),
            )
            print(f"Report generated: {execution.report_path.resolve()}")
            if execution.summary_json_path is not None:
                print(f"Healthcheck summary: {execution.summary_json_path.resolve()}")
            print(f"Run ID: {execution.run_id}")
            print(f"Run snapshot: {execution.run_dir.resolve()}")
            return 0

        if args.command == "bundle":
            output_path = Path(str(args.output)) if args.output else None
            bundle_path = bundle_run(
                run_id=str(args.run_id),
                runs_dir=Path(args.runs_dir),
                output_path=output_path,
            )
            print(f"Bundle generated: {bundle_path.resolve()}")
            return 0

        if args.command == "reproduce":
            run_id, report_path = reproduce_bundle(
                bundle_path=Path(args.bundle_path),
                output_dir=Path(args.output_dir),
            )
            print(f"Reproduced run: {run_id}")
            print(f"Report generated: {report_path.resolve()}")
            return 0

        if args.command == "replay":
            timeline_text = replay_run(
                run_id=str(args.run_id),
                sqlite_path=Path(args.db),
            )
            print(timeline_text)
            return 0

        if args.command == "dashboard":
            dashboard_path = run_dashboard(
                results_jsonl_path=Path(args.results_jsonl),
                sqlite_path=Path(args.db),
                output_path=Path(args.output),
                reports_dir=Path(args.reports_dir),
            )
            print(f"Dashboard generated: {dashboard_path.resolve()}")
            return 0

        if args.command == "create-strategy":
            strategy_path, config_path = create_strategy(
                strategy_name=str(args.name),
                strategy_dir=Path(args.strategy_dir),
                config_dir=Path(args.config_dir),
                force=bool(args.force),
            )
            print(f"Strategy template created: {strategy_path.resolve()}")
            print(f"Example config created: {config_path.resolve()}")
            return 0
    except Exception as error:
        parser.exit(status=1, message=f"Error: {error}\n")

    parser.print_help()
    return 1


def _resolve_path(base_dir: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def _load_bars(config: AppConfig) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    provider = CSVDataProvider(
        data_dir=config.data_path,
        use_parquet_cache=config.use_parquet_cache,
        cache_dir=config.parquet_cache_dir,
    )
    bars = provider.get_bars(
        symbol=config.symbol,
        start=config.start,
        end=config.end,
        timeframe=config.timeframe,
    )
    actions_summary = provider.get_corporate_actions_summary(
        symbol=config.symbol,
        start=config.start,
        end=config.end,
    )
    return bars, actions_summary


def _load_strategy(
    strategy_reference: str,
    strategy_params: dict[str, Any],
) -> Strategy:
    if ":" not in strategy_reference:
        msg = "strategy must be in 'module:Class' format."
        raise ValueError(msg)

    module_name, class_name = strategy_reference.split(":", maxsplit=1)
    strategy_module = importlib.import_module(module_name)
    strategy_class = getattr(strategy_module, class_name, None)
    if strategy_class is None:
        msg = f"Strategy class not found: {strategy_reference}"
        raise ValueError(msg)

    strategy_instance = strategy_class(**strategy_params)
    if not isinstance(strategy_instance, Strategy):
        msg = "Loaded strategy must implement generate_orders(bars, portfolio_state)."
        raise TypeError(msg)
    return strategy_instance


def _single_window_metrics_table(
    bars: pd.DataFrame,
    metrics: dict[str, float | int],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "window_index": 0,
                "window_start": bars.index[0],
                "window_end": bars.index[-1],
                "bars_count": int(len(bars)),
                **metrics,
            }
        ]
    )


def _config_for_report(config: AppConfig) -> dict[str, Any]:
    report_config: dict[str, Any] = {
        "report_name": config.report_name,
        "symbol": config.symbol,
        "data_path": str(config.data_path),
        "strategy": config.strategy_reference,
        "initial_cash": config.initial_cash,
        "timeframe": config.timeframe,
        "bars_freq": config.bars_freq,
        "window_size": config.window_size,
        "step_size": config.step_size,
        "start": config.start or "",
        "end": config.end or "",
        "bootstrap_samples": config.bootstrap_samples,
        "bootstrap_confidence": config.bootstrap_confidence,
        "sweep_metric": config.sweep_metric,
        "use_parquet_cache": config.use_parquet_cache,
        "parquet_cache_dir": str(config.parquet_cache_dir) if config.parquet_cache_dir else "",
        "tuning_enabled": config.tuning_enabled,
        "tuning_method": config.tuning_method,
        "tuning_n_iter": config.tuning_n_iter,
        "tuning_window_size": config.tuning_window_size,
        "tuning_step_size": config.tuning_step_size,
        "tuning_bars_freq": config.tuning_bars_freq,
        "tuning_drawdown_penalty": config.tuning_drawdown_penalty,
        "tuning_search_space_size": len(config.tuning_search_space),
    }
    report_config.update(config.cost_model.describe())
    return report_config


def _persist_run_snapshot(
    *,
    run_mode: Literal["run", "healthcheck", "demo"],
    config_path: Path | None,
    app_config: AppConfig,
    bars: pd.DataFrame,
    equity_curve: pd.Series,
    orders: list[OrderRecord],
    trades: list[Fill],
    overall_metrics: dict[str, float | int],
    window_metrics_df: pd.DataFrame | None,
    summary_json_path: Path | None,
    report_path: Path,
    asset_paths: list[Path],
) -> RunSnapshot:
    source_data_path = detect_source_data_path(
        data_path=app_config.data_path,
        symbol=app_config.symbol,
    )
    return build_run_snapshot(
        run_mode=run_mode,
        report_path=report_path,
        report_dir=app_config.report_dir,
        report_name=app_config.report_name,
        config_path=config_path,
        raw_config=app_config.raw_config,
        symbol=app_config.symbol,
        strategy_reference=app_config.strategy_reference,
        bars=bars,
        equity_curve=equity_curve,
        orders=orders,
        trades=trades,
        overall_metrics=overall_metrics,
        asset_paths=asset_paths,
        window_metrics_df=window_metrics_df,
        summary_json_path=summary_json_path,
        source_data_path=source_data_path,
    )


def _parse_robustness_config(
    raw_config: dict[str, Any],
) -> tuple[int, float, str, dict[str, list[Any]]]:
    robustness_raw = raw_config.get("robustness", {})
    if not isinstance(robustness_raw, dict):
        msg = "`robustness` must be a TOML table when provided."
        raise ValueError(msg)

    bootstrap_samples = int(robustness_raw.get("bootstrap_samples", 1000))
    bootstrap_confidence = float(robustness_raw.get("confidence", 0.95))

    sweep_raw = robustness_raw.get("parameter_sweep", {})
    if not isinstance(sweep_raw, dict):
        msg = "`robustness.parameter_sweep` must be a TOML table when provided."
        raise ValueError(msg)

    sweep_metric = str(sweep_raw.get("metric", "sharpe"))
    grid_raw = sweep_raw.get("grid", {})
    if not isinstance(grid_raw, dict):
        msg = "`robustness.parameter_sweep.grid` must be a TOML table when provided."
        raise ValueError(msg)

    sweep_param_grid: dict[str, list[Any]] = {}
    for param_name, values in grid_raw.items():
        if not isinstance(values, list):
            msg = f"`robustness.parameter_sweep.grid.{param_name}` must be an array."
            raise ValueError(msg)
        sweep_param_grid[str(param_name)] = list(values)

    return bootstrap_samples, bootstrap_confidence, sweep_metric, sweep_param_grid


def _parse_performance_config(
    raw_config: dict[str, Any],
    base_dir: Path,
) -> tuple[bool, Path | None]:
    performance_raw = raw_config.get("performance", {})
    if not isinstance(performance_raw, dict):
        msg = "`performance` must be a TOML table when provided."
        raise ValueError(msg)

    use_parquet_cache = bool(performance_raw.get("use_parquet_cache", False))
    cache_dir_raw = performance_raw.get("parquet_cache_dir")
    if cache_dir_raw is None:
        cache_dir = None
    else:
        cache_dir = _resolve_path(base_dir=base_dir, raw_path=str(cache_dir_raw))
    return use_parquet_cache, cache_dir


def _parse_tuning_config(
    raw_config: dict[str, Any],
    default_window_size: str,
    default_step_size: str,
    default_bars_freq: str,
) -> tuple[
    bool,
    Literal["grid", "random"],
    int,
    int,
    str,
    str,
    str,
    float,
    dict[str, list[Any]],
]:
    tuning_raw = raw_config.get("tuning", {})
    if not isinstance(tuning_raw, dict):
        msg = "`tuning` must be a TOML table when provided."
        raise ValueError(msg)

    enabled = bool(tuning_raw.get("enabled", False))
    method_text = str(tuning_raw.get("method", "grid")).strip().lower()
    if method_text not in {"grid", "random"}:
        msg = "`tuning.method` must be one of: grid, random."
        raise ValueError(msg)
    method: Literal["grid", "random"] = "grid" if method_text == "grid" else "random"

    n_iter = int(tuning_raw.get("n_iter", 20))
    random_seed = int(tuning_raw.get("random_seed", 42))
    drawdown_penalty = float(tuning_raw.get("drawdown_penalty", 2.0))
    window_size = str(tuning_raw.get("window_size", default_window_size))
    step_size = str(tuning_raw.get("step_size", default_step_size))
    bars_freq = str(tuning_raw.get("bars_freq", default_bars_freq))

    if n_iter <= 0:
        msg = "`tuning.n_iter` must be positive."
        raise ValueError(msg)
    if drawdown_penalty < 0:
        msg = "`tuning.drawdown_penalty` must be non-negative."
        raise ValueError(msg)

    search_space_raw = tuning_raw.get("search_space", {})
    if not isinstance(search_space_raw, dict):
        msg = "`tuning.search_space` must be a TOML table when provided."
        raise ValueError(msg)
    search_space: dict[str, list[Any]] = {}
    for param_name, values in search_space_raw.items():
        if not isinstance(values, list):
            msg = f"`tuning.search_space.{param_name}` must be an array."
            raise ValueError(msg)
        search_space[str(param_name)] = list(values)

    return (
        enabled,
        method,
        n_iter,
        random_seed,
        window_size,
        step_size,
        bars_freq,
        drawdown_penalty,
        search_space,
    )


def _compute_sensitivity_outputs(
    strategy: Strategy,
    bars: pd.DataFrame,
    initial_cash: float,
    report_dir: Path,
    report_name: str,
) -> tuple[pd.DataFrame, list[Path]]:
    sensitivity_metrics_df = run_cost_sensitivity_scan(
        strategy=strategy,
        bars=bars,
        initial_cash=initial_cash,
        commission_grid=(0.0, 5.0),
        slippage_grid=(0.0, 5.0),
        spread_grid=(0.0, 5.0),
    )
    if sensitivity_metrics_df.empty:
        return sensitivity_metrics_df, []

    sensitivity_plot_path = plot_cost_sensitivity(
        sensitivity_frame=sensitivity_metrics_df,
        output_path=report_dir / "assets" / f"{report_name}_cost_sensitivity.png",
    )
    return sensitivity_metrics_df, [sensitivity_plot_path]


def _compute_tuning_outputs(
    app_config: AppConfig,
    bars: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, list[Path]]:
    if not app_config.tuning_enabled or not app_config.tuning_search_space:
        return {}, pd.DataFrame(), []

    tuning_result = tune_strategy_parameters(
        bars=bars,
        strategy_reference=app_config.strategy_reference,
        base_params=app_config.strategy_params,
        search_space=app_config.tuning_search_space,
        initial_cash=app_config.initial_cash,
        cost_model=app_config.cost_model,
        method=app_config.tuning_method,
        n_iter=app_config.tuning_n_iter,
        random_seed=app_config.tuning_random_seed,
        window_size=app_config.tuning_window_size,
        step_size=app_config.tuning_step_size,
        bars_freq=app_config.tuning_bars_freq,
        drawdown_penalty=app_config.tuning_drawdown_penalty,
    )
    trials = tuning_result.trials
    if trials.empty or (trials["status"] == "ok").sum() == 0:
        best_params = dict(tuning_result.best_params)
        best_params.update(
            {
                "objective_name": tuning_result.objective_name,
                "best_score": tuning_result.best_score,
                "total_trials": tuning_result.total_trials,
                "successful_trials": tuning_result.successful_trials,
                "tuning_method": tuning_result.method,
            }
        )
        return best_params, trials, []

    plot_path = plot_tuning_robustness(
        trials=trials,
        output_path=app_config.report_dir / "assets" / f"{app_config.report_name}_tuning.png",
    )
    best_params = dict(tuning_result.best_params)
    best_params.update(
        {
            "objective_name": tuning_result.objective_name,
            "best_score": tuning_result.best_score,
            "total_trials": tuning_result.total_trials,
            "successful_trials": tuning_result.successful_trials,
            "tuning_method": tuning_result.method,
        }
    )
    return best_params, trials, [plot_path]


def _compute_robustness_outputs(
    app_config: AppConfig,
    bars: pd.DataFrame,
    equity_curve: pd.Series,
    window_metrics_df: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    strategy_params = dict(app_config.strategy_params)
    returns = equity_curve.pct_change().dropna()
    sharpe_ci_low, sharpe_ci_high = bootstrap_sharpe_ci(
        returns=returns,
        n=app_config.bootstrap_samples,
        confidence=app_config.bootstrap_confidence,
    )

    sweep_grid = app_config.sweep_param_grid or _default_sweep_grid(
        strategy_reference=app_config.strategy_reference,
        strategy_params=strategy_params,
    )
    if sweep_grid:
        sweep_results_df = parameter_sweep(
            bars=bars,
            strategy_reference=app_config.strategy_reference,
            base_params=strategy_params,
            param_grid=sweep_grid,
            initial_cash=app_config.initial_cash,
            cost_model=app_config.cost_model,
            metric_key=app_config.sweep_metric,
        )
    else:
        sweep_results_df = pd.DataFrame()

    overfit_summary, risk_flags = evaluate_overfit_risk(
        returns=returns,
        window_metrics_df=window_metrics_df,
    )

    robustness_summary = {
        "bootstrap_samples": app_config.bootstrap_samples,
        "bootstrap_confidence": app_config.bootstrap_confidence,
        "sharpe_ci_low": sharpe_ci_low,
        "sharpe_ci_high": sharpe_ci_high,
        **overfit_summary,
        "risk_flags": risk_flags,
    }
    return robustness_summary, sweep_results_df


def _default_sweep_grid(
    strategy_reference: str,
    strategy_params: dict[str, Any],
) -> dict[str, list[Any]]:
    if not strategy_reference.endswith("MovingAverageCrossStrategy"):
        return {}
    short_window = strategy_params.get("short_window")
    long_window = strategy_params.get("long_window")
    if not isinstance(short_window, int) or not isinstance(long_window, int):
        return {}

    short_candidates = sorted({max(1, short_window - 5), short_window, short_window + 5})
    long_candidates = sorted({max(2, long_window - 10), long_window, long_window + 10})
    return {
        "short_window": short_candidates,
        "long_window": long_candidates,
    }


def _normalize_strategy_class_name(name: str) -> str:
    class_name = str(name).strip()
    if not class_name:
        msg = "strategy_name cannot be empty."
        raise ValueError(msg)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", class_name):
        msg = (
            "strategy_name must be a valid Python class name "
            "(letters/digits/underscore, not starting with a digit)."
        )
        raise ValueError(msg)
    return class_name


def _to_snake_case(class_name: str) -> str:
    intermediate = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", class_name)
    snake_case = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", intermediate).lower()
    return snake_case.strip("_")


def _ensure_package_marker(strategy_dir: Path) -> None:
    init_file = strategy_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text(
            '"""User strategy modules generated by stratcheck create-strategy."""\n',
            encoding="utf-8",
        )


def _strategy_reference_from_path(strategy_path: Path, class_name: str) -> str:
    module_path = strategy_path.with_suffix("")
    try:
        relative_module = module_path.resolve().relative_to(Path.cwd().resolve())
        module_tokens = [
            _to_snake_case(token) if not token.isidentifier() else token
            for token in relative_module.parts
        ]
    except ValueError:
        module_tokens = [_to_snake_case(module_path.name)]
    module_name = ".".join(token for token in module_tokens if token)
    return f"{module_name}:{class_name}"


def _render_strategy_template(class_name: str) -> str:
    template = f"""
\"\"\"Strategy plugin scaffold generated by `stratcheck create-strategy`.\"\"\"

from __future__ import annotations

import pandas as pd

from stratcheck.core.strategy import PortfolioState
from stratcheck.sdk import StrategySignal, StrategyTemplate


class {class_name}(StrategyTemplate):
    \"\"\"Example momentum-style strategy.

    Replace the signal rule in `build_signals(...)` with your own logic.
    \"\"\"

    def __init__(
        self,
        lookback: int = 20,
        entry_threshold: float = 0.01,
        target_position_qty: float = 1.0,
    ) -> None:
        super().__init__(strategy_name="{class_name}")
        if lookback < 2:
            msg = "lookback must be >= 2."
            raise ValueError(msg)
        if target_position_qty <= 0:
            msg = "target_position_qty must be positive."
            raise ValueError(msg)

        self.lookback = int(lookback)
        self.entry_threshold = float(entry_threshold)
        self.target_position_qty = float(target_position_qty)

    def build_signals(
        self,
        bars: pd.DataFrame,
        portfolio_state: PortfolioState,
    ) -> list[StrategySignal]:
        if len(bars) < self.lookback + 1:
            return []

        close_prices = bars["close"].astype(float)
        momentum = float(close_prices.iloc[-1] / close_prices.iloc[-self.lookback] - 1.0)
        current_position_qty = float(portfolio_state.position_qty)

        signals: list[StrategySignal] = []
        if momentum > self.entry_threshold and current_position_qty < self.target_position_qty:
            buy_qty = self.target_position_qty - current_position_qty
            signals.append(
                self.signal(
                    side="buy",
                    qty=buy_qty,
                    reason=(
                        f"momentum={{momentum:.4f}} > "
                        f"threshold={{self.entry_threshold:.4f}}"
                    ),
                )
            )
        elif momentum <= 0.0 and current_position_qty > 0:
            signals.append(
                self.signal(
                    side="sell",
                    qty=current_position_qty,
                    reason=f"momentum={{momentum:.4f}} <= 0 exit",
                )
            )
        return signals
"""
    return template.strip() + "\n"


def _render_strategy_config_template(
    report_name: str,
    strategy_reference: str,
) -> str:
    template = f"""
symbol = "BTCUSDT"
data_path = "data"
strategy = "{strategy_reference}"
initial_cash = 100000
timeframe = "1d"
bars_freq = "1d"
report_name = "{report_name}"
report_dir = "reports"

[cost_model]
type = "fixed_bps"
commission_bps = 5
slippage_bps = 3

[windows]
window_size = "6M"
step_size = "3M"

[strategy_params]
lookback = 20
entry_threshold = 0.01
target_position_qty = 1.0
"""
    return template.strip() + "\n"


def _as_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _build_guard_failure_report(
    builder: ReportBuilder,
    report_config: dict[str, Any],
    full_config: dict[str, Any] | str,
    bars: pd.DataFrame,
    guard_error: GuardViolationError,
    reproduce_command: str,
    corporate_actions_summary: list[dict[str, str]] | None = None,
) -> Path:
    overall_metrics: dict[str, float | int] = {
        "cagr": 0.0,
        "annual_volatility": 0.0,
        "sharpe": 0.0,
        "max_drawdown": 0.0,
        "turnover": 0.0,
        "guard_failure_count": int(len(guard_error.flags)),
    }

    window_metrics = pd.DataFrame()
    if len(bars) > 0 and isinstance(bars.index, pd.DatetimeIndex):
        window_metrics = pd.DataFrame(
            [
                {
                    "window_index": 0,
                    "window_start": bars.index[0],
                    "window_end": bars.index[-1],
                    "bars_count": int(len(bars)),
                    "sharpe": 0.0,
                    "max_drawdown": 0.0,
                }
            ]
        )

    return builder.build(
        overall_metrics=overall_metrics,
        window_metrics_df=window_metrics,
        plot_paths=[],
        config=report_config,
        robustness_summary={
            "guard_violation": str(guard_error),
            "guard_flag_count": float(len(guard_error.flags)),
        },
        sweep_results_df=pd.DataFrame(),
        risk_flags=_risk_flag_records(guard_error.flags),
        reproduce_command=reproduce_command,
        full_config=full_config,
        corporate_actions_summary=corporate_actions_summary,
    )


def _risk_flag_records(flags: list[Any]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for flag in flags:
        check = str(getattr(flag, "check", "Unknown"))
        level = str(getattr(flag, "level", "yellow"))
        message = str(getattr(flag, "message", ""))
        records.append(
            {
                "check": check,
                "level": level,
                "message": message,
            }
        )
    return records


def _validation_flag_records(validation_summary: list[dict[str, Any]]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for row in validation_summary:
        status = str(row.get("status", "")).lower()
        strategy_name = str(row.get("strategy", "unknown"))
        baseline_name = str(row.get("baseline", "vectorized_reference"))
        if status == "fail":
            records.append(
                {
                    "check": "Backtest Baseline Validation",
                    "level": "red",
                    "message": (
                        f"{strategy_name} vs {baseline_name} failed: "
                        f"{row.get('message', 'equity error exceeds tolerance')}"
                    ),
                }
            )
        elif status == "skipped":
            records.append(
                {
                    "check": "Backtest Baseline Validation",
                    "level": "yellow",
                    "message": (
                        f"{strategy_name} skipped baseline validation: "
                        f"{row.get('message', 'no available baseline')}"
                    ),
                }
            )
        elif status == "pass":
            records.append(
                {
                    "check": "Backtest Baseline Validation",
                    "level": "green",
                    "message": (
                        f"{strategy_name} vs {baseline_name} passed: {row.get('message', '')}"
                    ),
                }
            )
    return records
