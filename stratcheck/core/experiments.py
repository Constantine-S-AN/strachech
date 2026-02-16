"""Batch experiment runner for multiple strategy configs."""

from __future__ import annotations

import html
import importlib
import json
import tomllib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import repeat
from pathlib import Path
from typing import Any

import pandas as pd

from stratcheck.core.backtest import BacktestEngine, CostModel, build_cost_model
from stratcheck.core.data import CSVDataProvider
from stratcheck.core.healthcheck import run_healthcheck
from stratcheck.core.strategy import Strategy
from stratcheck.quality import GuardViolationError
from stratcheck.report import ReportBuilder, generate_performance_plots
from stratcheck.validation import validate_against_vectorized_baseline


@dataclass(slots=True)
class ExperimentConfig:
    """Typed configuration for one experiment run."""

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
    start: str | None
    end: str | None
    use_parquet_cache: bool
    parquet_cache_dir: Path | None


class ExperimentRunner:
    """Run all TOML configs in a directory and build an experiment index page."""

    def __init__(
        self,
        configs_dir: str | Path,
        output_dir: str | Path = "reports",
        parallel: bool = False,
        max_workers: int | None = None,
    ) -> None:
        self.configs_dir = Path(configs_dir)
        self.output_dir = Path(output_dir)
        self.parallel = bool(parallel)
        self.max_workers = max_workers

    def run_all(self) -> tuple[pd.DataFrame, Path]:
        """Run every config in `configs_dir` and generate `index.html` + `results.jsonl`."""
        config_paths = sorted(self.configs_dir.glob("*.toml"))
        if not config_paths:
            msg = f"No config files found under: {self.configs_dir}"
            raise ValueError(msg)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, object]]
        if self.parallel and len(config_paths) > 1:
            records = _run_configs_in_parallel(
                config_paths=config_paths,
                output_dir=self.output_dir,
                max_workers=self.max_workers,
            )
        else:
            records = [
                _run_single_config_with_output(config_path=config_path, output_dir=self.output_dir)
                for config_path in config_paths
            ]

        summary_frame = pd.DataFrame(records)
        if not summary_frame.empty:
            summary_frame = summary_frame.sort_values("experiment").reset_index(drop=True)

        results_path = self._write_results_jsonl(records=records)
        index_path = self._build_index_html(summary_frame=summary_frame, results_path=results_path)
        return summary_frame, index_path

    def _write_results_jsonl(self, records: list[dict[str, object]]) -> Path:
        results_path = self.output_dir / "results.jsonl"
        with results_path.open("w", encoding="utf-8") as result_file:
            for record in records:
                line = json.dumps(record, ensure_ascii=False)
                result_file.write(line + "\n")
        return results_path

    def _build_index_html(self, summary_frame: pd.DataFrame, results_path: Path) -> Path:
        index_path = self.output_dir / "index.html"
        generated_time = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

        if summary_frame.empty:
            table_html = "<p>No experiments were executed.</p>"
            sorting_controls = ""
        else:
            rendered = summary_frame.copy()
            rendered["report"] = rendered["report_path"].map(_as_report_link)
            rendered = rendered.drop(columns=["report_path"])
            preferred_order = [
                "experiment",
                "status",
                "cost_assumption",
                "total_return",
                "cagr",
                "sharpe",
                "max_drawdown",
                "worst_window_sharpe",
                "worst_window_drawdown",
                "report",
                "summary_json_path",
                "error",
            ]
            ordered_columns = [name for name in preferred_order if name in rendered.columns]
            ordered_columns.extend(
                name for name in rendered.columns if name not in set(ordered_columns)
            )
            rendered = rendered.loc[:, ordered_columns]
            for column_name in rendered.columns:
                if pd.api.types.is_float_dtype(rendered[column_name]):
                    rendered[column_name] = rendered[column_name].map(
                        lambda value: "-" if pd.isna(value) else f"{value:.6f}"
                    )
            table_html = rendered.to_html(
                index=False,
                escape=False,
                border=0,
                table_id="experimentTable",
            )
            sort_columns = {
                "experiment": rendered.columns.get_loc("experiment"),
                "status": rendered.columns.get_loc("status"),
                "cost_assumption": rendered.columns.get_loc("cost_assumption"),
                "sharpe": rendered.columns.get_loc("sharpe"),
                "max_drawdown": rendered.columns.get_loc("max_drawdown"),
                "total_return": rendered.columns.get_loc("total_return"),
            }
            sorting_controls = f"""
      <div class="sort-controls">
        <label for="sortKey">Sort by:</label>
        <select id="sortKey">
          <option value="experiment">experiment</option>
          <option value="cost_assumption">cost_assumption</option>
          <option value="sharpe">sharpe</option>
          <option value="max_drawdown">max_drawdown</option>
          <option value="total_return">total_return</option>
          <option value="status">status</option>
        </select>
        <label for="sortDirection">Direction:</label>
        <select id="sortDirection">
          <option value="asc">asc</option>
          <option value="desc">desc</option>
        </select>
        <button id="applySort" type="button">Apply</button>
      </div>
      <script>
        const columnIndexByKey = {json.dumps(sort_columns)};
        const sortButton = document.getElementById("applySort");
        const sortKeySelect = document.getElementById("sortKey");
        const sortDirectionSelect = document.getElementById("sortDirection");
        const experimentTable = document.getElementById("experimentTable");

        function parseSortValue(text) {{
          const numeric = Number(text.replace(/[^0-9.+-]/g, ""));
          if (!Number.isNaN(numeric) && text !== "-") {{
            return numeric;
          }}
          return text.toLowerCase();
        }}

        function applySort() {{
          const sortKey = sortKeySelect.value;
          const direction = sortDirectionSelect.value;
          const columnIndex = columnIndexByKey[sortKey];
          if (columnIndex === undefined || !experimentTable || !experimentTable.tBodies.length) {{
            return;
          }}
          const body = experimentTable.tBodies[0];
          const rows = Array.from(body.rows);
          rows.sort((left, right) => {{
            const leftValue = parseSortValue(left.cells[columnIndex].textContent.trim());
            const rightValue = parseSortValue(right.cells[columnIndex].textContent.trim());
            let result = 0;
            if (typeof leftValue === "number" && typeof rightValue === "number") {{
              result = leftValue - rightValue;
            }} else {{
              result = String(leftValue).localeCompare(String(rightValue));
            }}
            return direction === "desc" ? -result : result;
          }});
          for (const row of rows) {{
            body.appendChild(row);
          }}
        }}

        if (sortButton) {{
          sortButton.addEventListener("click", applySort);
        }}
      </script>
"""

        document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Stratcheck Experiments Index</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #131722;
      background: #f3f5fb;
    }}
    .container {{
      max-width: 1200px;
      margin: 24px auto;
      padding: 0 16px 24px;
    }}
    .panel {{
      background: #ffffff;
      border-radius: 12px;
      padding: 18px;
      box-shadow: 0 8px 20px rgba(19, 23, 34, 0.08);
      margin-bottom: 14px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      text-align: left;
      border-bottom: 1px solid #e6ebf3;
      padding: 8px 10px;
      vertical-align: top;
    }}
    .meta {{
      color: #4b5565;
      margin: 0;
    }}
    .sort-controls {{
      margin-bottom: 12px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }}
    .sort-controls label {{
      font-size: 13px;
      color: #334155;
    }}
    .sort-controls select, .sort-controls button {{
      padding: 6px 8px;
      border-radius: 8px;
      border: 1px solid #d5dceb;
      background: #ffffff;
      color: #111827;
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <main class="container">
    <section class="panel">
      <h1>Stratcheck Experiments</h1>
      <p class="meta">Generated at {generated_time}</p>
      <p class="meta">Configs dir: {html.escape(str(self.configs_dir))}</p>
      <p class="meta">Results JSONL: {html.escape(results_path.name)}</p>
    </section>
    <section class="panel">
      <h2>Experiment Summary</h2>
      {sorting_controls}
      {table_html}
    </section>
  </main>
</body>
</html>
"""
        index_path.write_text(document, encoding="utf-8")
        return index_path


def _run_configs_in_parallel(
    config_paths: list[Path],
    output_dir: Path,
    max_workers: int | None,
) -> list[dict[str, object]]:
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            records = list(
                executor.map(
                    _run_single_config_worker,
                    (str(config_path) for config_path in config_paths),
                    repeat(str(output_dir)),
                )
            )
        return records
    except (NotImplementedError, PermissionError, OSError):
        return [
            _run_single_config_with_output(config_path=config_path, output_dir=output_dir)
            for config_path in config_paths
        ]


def _run_single_config_worker(config_path_text: str, output_dir_text: str) -> dict[str, object]:
    return _run_single_config_with_output(
        config_path=Path(config_path_text),
        output_dir=Path(output_dir_text),
    )


def _run_single_config_with_output(config_path: Path, output_dir: Path) -> dict[str, object]:
    experiment_name = config_path.stem
    report_name = _safe_name(experiment_name)
    summary_name = f"{report_name}_healthcheck_summary.json"
    summary_json_path = output_dir / summary_name
    actions_summary: list[dict[str, str]] = []
    experiment_config: ExperimentConfig | None = None

    try:
        experiment_config = _load_experiment_config(config_path=config_path)
        bars, actions_summary = _load_bars(config=experiment_config)
        strategy = _load_strategy(
            strategy_reference=experiment_config.strategy_reference,
            strategy_params=experiment_config.strategy_params,
        )

        engine = BacktestEngine()
        full_result = engine.run(
            strategy=strategy,
            bars=bars,
            initial_cash=experiment_config.initial_cash,
            cost_model=experiment_config.cost_model,
        )

        window_metrics_df, exported_summary_path = run_healthcheck(
            strategy=strategy,
            bars=bars,
            window_size=experiment_config.window_size,
            step_size=experiment_config.step_size,
            initial_cash=experiment_config.initial_cash,
            cost_model=experiment_config.cost_model,
            bars_freq=experiment_config.bars_freq,
            output_json_path=summary_json_path,
        )
        plot_paths = generate_performance_plots(
            equity_curve=full_result.equity_curve,
            trades=full_result.trades,
            output_dir=output_dir / "assets",
            prefix=report_name,
        )

        report_builder = ReportBuilder(output_dir=output_dir)
        validation_summary = validate_against_vectorized_baseline(
            strategy=strategy,
            bars=bars,
            engine_equity_curve=full_result.equity_curve,
            initial_cash=experiment_config.initial_cash,
            cost_model=experiment_config.cost_model,
        )
        from stratcheck.analysis import (
            compute_execution_quality_scorecard,
            compute_regime_scorecard,
        )

        sensitivity_metrics_df, sensitivity_plot_paths = _compute_sensitivity_outputs(
            strategy=strategy,
            bars=bars,
            initial_cash=experiment_config.initial_cash,
            output_dir=output_dir,
            report_name=report_name,
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
        report_path = report_builder.build(
            overall_metrics=full_result.metrics,
            window_metrics_df=window_metrics_df,
            plot_paths=plot_paths,
            config=_config_for_report(
                config=experiment_config,
                config_path=config_path,
                report_name=report_name,
            ),
            sensitivity_metrics_df=sensitivity_metrics_df,
            sensitivity_plot_paths=sensitivity_plot_paths,
            regime_scorecard_df=regime_scorecard_df,
            execution_quality_df=execution_quality_df,
            validation_summary=validation_summary,
            corporate_actions_summary=actions_summary,
        )

        return {
            "experiment": experiment_name,
            "status": "success",
            "config_path": str(config_path),
            "cost_assumption": _cost_assumption_text(experiment_config.cost_model),
            "report_path": report_path.name,
            "summary_json_path": exported_summary_path.name,
            "total_return": _as_float(full_result.metrics.get("total_return")),
            "cagr": _as_float(full_result.metrics.get("cagr")),
            "sharpe": _as_float(full_result.metrics.get("sharpe")),
            "max_drawdown": _as_float(full_result.metrics.get("max_drawdown")),
            "worst_window_sharpe": _frame_column_min(window_metrics_df, "sharpe"),
            "worst_window_drawdown": _frame_column_min(window_metrics_df, "max_drawdown"),
            "error": "",
        }
    except GuardViolationError as error:
        report_builder = ReportBuilder(output_dir=output_dir)
        failure_report_path = report_builder.build(
            overall_metrics={
                "cagr": 0.0,
                "annual_volatility": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "turnover": 0.0,
                "guard_failure_count": int(len(error.flags)),
            },
            window_metrics_df=pd.DataFrame(),
            plot_paths=[],
            config={
                "report_name": report_name,
                "symbol": experiment_config.symbol if experiment_config is not None else "",
                "config_path": str(config_path),
            },
            robustness_summary={
                "guard_violation": str(error),
                "guard_flag_count": float(len(error.flags)),
            },
            risk_flags=[
                {"check": flag.check, "level": flag.level, "message": flag.message}
                for flag in error.flags
            ],
            reproduce_command=f"python -m stratcheck run --config {config_path}",
            corporate_actions_summary=actions_summary,
        )
        return {
            "experiment": experiment_name,
            "status": "failed",
            "config_path": str(config_path),
            "cost_assumption": (
                _cost_assumption_text(experiment_config.cost_model)
                if experiment_config is not None
                else ""
            ),
            "report_path": failure_report_path.name,
            "summary_json_path": "",
            "total_return": None,
            "cagr": None,
            "sharpe": None,
            "max_drawdown": None,
            "worst_window_sharpe": None,
            "worst_window_drawdown": None,
            "error": f"{type(error).__name__}: {error}",
        }
    except Exception as error:
        return {
            "experiment": experiment_name,
            "status": "failed",
            "config_path": str(config_path),
            "cost_assumption": (
                _cost_assumption_text(experiment_config.cost_model)
                if experiment_config is not None
                else ""
            ),
            "report_path": "",
            "summary_json_path": "",
            "total_return": None,
            "cagr": None,
            "sharpe": None,
            "max_drawdown": None,
            "worst_window_sharpe": None,
            "worst_window_drawdown": None,
            "error": f"{type(error).__name__}: {error}",
        }


def _resolve_path(base_dir: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def _load_experiment_config(config_path: Path) -> ExperimentConfig:
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

    strategy_params = raw_config.get("strategy_params", {})
    if not isinstance(strategy_params, dict):
        msg = "`strategy_params` must be a TOML table when provided."
        raise ValueError(msg)

    performance_raw = raw_config.get("performance", {})
    if not isinstance(performance_raw, dict):
        msg = "`performance` must be a TOML table when provided."
        raise ValueError(msg)

    use_parquet_cache = bool(performance_raw.get("use_parquet_cache", False))
    parquet_cache_dir_raw = performance_raw.get("parquet_cache_dir")
    if parquet_cache_dir_raw is None:
        parquet_cache_dir = None
    else:
        parquet_cache_dir = _resolve_path(base_dir=base_dir, raw_path=str(parquet_cache_dir_raw))

    symbol = str(raw_config["symbol"])
    timeframe = str(raw_config.get("timeframe", "1d"))
    bars_freq = str(raw_config.get("bars_freq", timeframe))

    return ExperimentConfig(
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
        report_name=str(raw_config.get("report_name", symbol.lower())),
        start=_as_optional_string(raw_config.get("start")),
        end=_as_optional_string(raw_config.get("end")),
        use_parquet_cache=use_parquet_cache,
        parquet_cache_dir=parquet_cache_dir,
    )


def _load_bars(config: ExperimentConfig) -> tuple[pd.DataFrame, list[dict[str, str]]]:
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


def _config_for_report(
    config: ExperimentConfig,
    config_path: Path,
    report_name: str,
) -> dict[str, Any]:
    report_config: dict[str, Any] = {
        "report_name": report_name,
        "experiment": config_path.stem,
        "config_path": str(config_path),
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
        "use_parquet_cache": config.use_parquet_cache,
        "parquet_cache_dir": str(config.parquet_cache_dir) if config.parquet_cache_dir else "",
    }
    report_config.update(config.cost_model.describe())
    return report_config


def _as_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _as_report_link(path_text: str) -> str:
    if not path_text:
        return "-"
    escaped_path = html.escape(path_text)
    return f'<a href="{escaped_path}">Open</a>'


def _safe_name(name: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_" for character in name
    )
    return cleaned or "report"


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    return None


def _frame_column_min(frame: pd.DataFrame, column_name: str) -> float | None:
    if column_name not in frame.columns or frame.empty:
        return None

    series = pd.to_numeric(frame[column_name], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.min())


def _cost_assumption_text(cost_model: CostModel) -> str:
    model_config = cost_model.describe()
    commission_bps = float(model_config.get("commission_bps", 0.0))
    slippage_bps = float(
        model_config.get("slippage_bps", model_config.get("base_slippage_bps", 0.0))
    )
    spread_bps = float(model_config.get("spread_bps", 0.0))
    return f"comm={commission_bps:.1f}bps; slip={slippage_bps:.1f}bps; spread={spread_bps:.1f}bps"


def _compute_sensitivity_outputs(
    strategy: Strategy,
    bars: pd.DataFrame,
    initial_cash: float,
    output_dir: Path,
    report_name: str,
) -> tuple[pd.DataFrame, list[Path]]:
    from stratcheck.analysis import plot_cost_sensitivity, run_cost_sensitivity_scan

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
        output_path=output_dir / "assets" / f"{report_name}_cost_sensitivity.png",
    )
    return sensitivity_metrics_df, [sensitivity_plot_path]
