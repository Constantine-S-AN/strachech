"""HTML report builder for healthcheck outputs."""

from __future__ import annotations

import html
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


class ReportBuilder:
    """Build a self-contained HTML report with metrics, windows, and chart images."""

    def __init__(self, output_dir: str | Path = "reports") -> None:
        self.output_dir = Path(output_dir)

    def build(
        self,
        overall_metrics: dict[str, float | int],
        window_metrics_df: pd.DataFrame,
        plot_paths: list[str | Path],
        config: dict[str, Any],
        robustness_summary: dict[str, Any] | None = None,
        sweep_results_df: pd.DataFrame | None = None,
        sensitivity_metrics_df: pd.DataFrame | None = None,
        sensitivity_plot_paths: list[str | Path] | None = None,
        regime_scorecard_df: pd.DataFrame | None = None,
        execution_quality_df: pd.DataFrame | None = None,
        tuning_best_params: dict[str, Any] | None = None,
        tuning_trials_df: pd.DataFrame | None = None,
        tuning_plot_paths: list[str | Path] | None = None,
        risk_flags: list[dict[str, str]] | None = None,
        validation_summary: list[dict[str, Any]] | None = None,
        reproduce_command: str | None = None,
        full_config: dict[str, Any] | str | None = None,
        corporate_actions_summary: list[dict[str, str]] | None = None,
        universe_summary_df: pd.DataFrame | None = None,
    ) -> Path:
        """Generate an HTML report at `reports/<name>.html`."""
        report_name = str(config.get("report_name", "report"))
        safe_name = _safe_report_name(report_name)
        output_path = self.output_dir / f"{safe_name}.html"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        generated_time = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        summary_cards = _render_summary_cards(overall_metrics)
        config_table = _render_key_value_table(config)
        actions_table = _render_actions_summary_table(corporate_actions_summary)
        universe_table = _render_dataframe_table(
            frame=universe_summary_df,
            empty_message="No universe history available.",
        )
        overall_table = _render_key_value_table(overall_metrics)
        windows_table = _render_window_metrics_table(window_metrics_df)
        robustness_table = _render_key_value_table(robustness_summary or {})
        sweep_table = _render_dataframe_table(
            frame=sweep_results_df,
            empty_message="No parameter sweep results available.",
        )
        sensitivity_table = _render_dataframe_table(
            frame=sensitivity_metrics_df,
            empty_message="No cost sensitivity scan results available.",
        )
        risk_flags_table = _render_risk_flags_table(risk_flags)
        validation_table = _render_validation_summary_table(validation_summary)
        regime_table = _render_dataframe_table(
            frame=regime_scorecard_df,
            empty_message="No regime scorecard available.",
        )
        execution_quality_table = _render_dataframe_table(
            frame=execution_quality_df,
            empty_message="No execution quality metrics available.",
        )
        tuning_best_params_table = _render_key_value_table(tuning_best_params or {})
        tuning_trials_table = _render_dataframe_table(
            frame=tuning_trials_df,
            empty_message="No tuning trials available.",
        )
        tuning_images_html = _render_images(
            plot_paths=tuning_plot_paths or [],
            html_path=output_path,
        )
        sensitivity_images_html = _render_images(
            plot_paths=sensitivity_plot_paths or [],
            html_path=output_path,
        )
        images_html = _render_images(plot_paths=plot_paths, html_path=output_path)
        reproduce_command_html = _render_reproduce_command(
            reproduce_command or _default_reproduce_command(config)
        )
        resolved_full_config = full_config if full_config is not None else config
        full_config_html = _render_full_config_text(resolved_full_config)

        document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Stratcheck Report - {html.escape(safe_name)}</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #131722;
      background: #f3f5fb;
    }}
    .container {{
      max-width: 1100px;
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
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
    }}
    h2 {{
      margin: 0 0 10px;
      font-size: 20px;
    }}
    h3 {{
      margin: 12px 0 8px;
      font-size: 16px;
      color: #243046;
    }}
    .meta {{
      color: #4b5565;
      margin: 0;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }}
    .summary-card {{
      border: 1px solid #e6ebf3;
      border-radius: 10px;
      padding: 12px;
      background: #fbfcff;
    }}
    .summary-label {{
      margin: 0;
      font-size: 12px;
      letter-spacing: 0.02em;
      text-transform: uppercase;
      color: #5f6b81;
    }}
    .summary-value {{
      margin: 6px 0 0;
      font-size: 22px;
      font-weight: 700;
      color: #111827;
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
    th {{
      width: 240px;
      font-weight: 600;
      color: #2f3a4f;
    }}
    .image-grid {{
      display: grid;
      gap: 12px;
      grid-template-columns: 1fr;
    }}
    .image-grid img {{
      width: 100%;
      border: 1px solid #e6ebf3;
      border-radius: 10px;
      background: #ffffff;
    }}
    .flag-badge {{
      display: inline-block;
      border-radius: 999px;
      padding: 2px 10px;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.02em;
    }}
    .flag-red {{
      background: #fee2e2;
      color: #b91c1c;
    }}
    .flag-yellow {{
      background: #fef3c7;
      color: #92400e;
    }}
    .flag-green {{
      background: #dcfce7;
      color: #166534;
    }}
    .window-row-worst td {{
      background: #fff1f2;
    }}
    .worst-note {{
      margin: 0 0 10px;
      color: #9f1239;
      font-weight: 600;
    }}
    .command-block, .config-block {{
      margin: 0;
      padding: 12px;
      border-radius: 8px;
      border: 1px solid #e6ebf3;
      background: #f8fafc;
      overflow-x: auto;
      font-size: 13px;
      line-height: 1.4;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    details {{
      margin-top: 8px;
    }}
    details summary {{
      cursor: pointer;
      color: #243046;
      font-weight: 600;
    }}
    @media (max-width: 860px) {{
      .summary-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
    }}
  </style>
</head>
<body>
  <main class="container">
    <section class="panel">
      <h1>Stratcheck Report</h1>
      <p class="meta">Generated at {generated_time}</p>
      <p class="meta">Report name: {html.escape(safe_name)}</p>
    </section>
    <section class="panel">
      <h2>Summary</h2>
      {summary_cards}
    </section>
    <section class="panel">
      <h2>Config</h2>
      {config_table}
    </section>
    <section class="panel">
      <h2>Corporate Actions</h2>
      {actions_table}
    </section>
    <section class="panel">
      <h2>Universe Dynamics</h2>
      {universe_table}
    </section>
    <section class="panel">
      <h2>Overall Metrics</h2>
      {overall_table}
    </section>
    <section class="panel">
      <h2>Window Metrics</h2>
      {windows_table}
    </section>
    <section class="panel">
      <h2>Robustness</h2>
      <h3>Bootstrap Sharpe CI</h3>
      {robustness_table}
      <h3>Parameter Sweep</h3>
      {sweep_table}
      <h3>Risk Flags</h3>
      {risk_flags_table}
    </section>
    <section class="panel">
      <h2>Cost/Slippage Sensitivity</h2>
      <h3>Metrics by Cost Assumption</h3>
      {sensitivity_table}
      <h3>Sensitivity Trend</h3>
      {sensitivity_images_html}
    </section>
    <section class="panel">
      <h2>Regime Scorecard</h2>
      {regime_table}
    </section>
    <section class="panel">
      <h2>Execution Quality</h2>
      {execution_quality_table}
    </section>
    <section class="panel">
      <h2>Parameter Tuning</h2>
      <h3>Best Parameters</h3>
      {tuning_best_params_table}
      <h3>Trial Results</h3>
      {tuning_trials_table}
      <h3>Robustness Plot</h3>
      {tuning_images_html}
    </section>
    <section class="panel">
      <h2>Backtest Validation</h2>
      {validation_table}
    </section>
    <section class="panel">
      <h2>Charts</h2>
      {images_html}
    </section>
    <section class="panel">
      <h2>Reproducibility</h2>
      <h3>Reproduce Command</h3>
      {reproduce_command_html}
      <h3>Full Config</h3>
      {full_config_html}
    </section>
  </main>
</body>
</html>
"""

        output_path.write_text(document, encoding="utf-8")
        return output_path


def _safe_report_name(name: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_" for character in name
    )
    return cleaned or "report"


def _render_key_value_table(values: dict[str, Any]) -> str:
    if not values:
        return "<p>No metrics available.</p>"

    rows: list[str] = []
    for key, raw_value in values.items():
        metric_name = _display_name(key)
        metric_value = _format_value(key, raw_value)
        rows.append(
            f"<tr><th>{html.escape(metric_name)}</th><td>{html.escape(metric_value)}</td></tr>"
        )
    return f"<table><tbody>{''.join(rows)}</tbody></table>"


def _render_window_metrics_table(window_metrics_df: pd.DataFrame) -> str:
    if window_metrics_df.empty:
        return "<p>No window metrics available.</p>"

    display_frame = _format_dataframe_for_display(window_metrics_df)
    worst_row_position = _find_worst_window_row(window_metrics_df)
    if worst_row_position is None:
        return _render_table_html(display_frame)

    note_text = ""
    if "Window Index" in display_frame.columns and 0 <= worst_row_position < len(display_frame):
        worst_window_index = display_frame.iloc[worst_row_position]["Window Index"]
        note_text = (
            '<p class="worst-note">'
            "Worst window highlighted (Window Index: "
            f"{html.escape(str(worst_window_index))})."
            "</p>"
        )

    return note_text + _render_table_html(display_frame, highlight_row=worst_row_position)


def _render_dataframe_table(
    frame: pd.DataFrame | None,
    empty_message: str = "No table data available.",
) -> str:
    if frame is None or frame.empty:
        return f"<p>{html.escape(empty_message)}</p>"

    display_frame = _format_dataframe_for_display(frame)
    return _render_table_html(display_frame)


def _render_images(plot_paths: list[str | Path], html_path: Path) -> str:
    if not plot_paths:
        return "<p>No charts generated.</p>"

    image_tags: list[str] = []
    for path in plot_paths:
        original_path = Path(path)
        if original_path.is_absolute():
            relative_path = Path(os.path.relpath(original_path, start=html_path.parent))
        else:
            relative_path = original_path
        escaped_path = html.escape(relative_path.as_posix())
        image_tags.append(f'<img src="{escaped_path}" alt="{escaped_path}" loading="lazy" />')
    return f'<div class="image-grid">{"".join(image_tags)}</div>'


def _render_risk_flags_table(risk_flags: list[dict[str, str]] | None) -> str:
    if not risk_flags:
        return "<p>No risk flags available.</p>"

    rows: list[str] = []
    for flag in risk_flags:
        check_name = html.escape(str(flag.get("check", "")))
        level = str(flag.get("level", "yellow")).lower()
        level_text = html.escape(level)
        message = html.escape(str(flag.get("message", "")))
        css_class = "flag-yellow"
        if level == "red":
            css_class = "flag-red"
        elif level == "green":
            css_class = "flag-green"
        level_badge = f'<span class="flag-badge {css_class}">{level_text}</span>'
        rows.append(f"<tr><td>{check_name}</td><td>{level_badge}</td><td>{message}</td></tr>")

    return (
        "<table><thead><tr><th>Check</th><th>Level</th><th>Message</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _render_actions_summary_table(actions_summary: list[dict[str, str]] | None) -> str:
    if not actions_summary:
        return "<p>No corporate actions recorded.</p>"

    rows: list[str] = []
    for action in actions_summary:
        date_text = html.escape(str(action.get("date", "")))
        action_type = html.escape(str(action.get("type", "")))
        details = html.escape(str(action.get("details", "")))
        rows.append(f"<tr><td>{date_text}</td><td>{action_type}</td><td>{details}</td></tr>")

    return (
        "<table><thead><tr><th>Date</th><th>Type</th><th>Details</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _render_validation_summary_table(validation_summary: list[dict[str, Any]] | None) -> str:
    if not validation_summary:
        return "<p>No baseline validation summary available.</p>"

    validation_frame = pd.DataFrame(validation_summary)
    if validation_frame.empty:
        return "<p>No baseline validation summary available.</p>"

    display_frame = _format_dataframe_for_display(validation_frame)
    return _render_table_html(display_frame)


def _render_summary_cards(overall_metrics: dict[str, float | int]) -> str:
    summary_keys = ["cagr", "sharpe", "max_drawdown", "turnover"]
    cards: list[str] = []
    for key in summary_keys:
        title = _display_name(key)
        if key in overall_metrics:
            value_text = _format_value(key, overall_metrics[key])
        else:
            value_text = "-"
        cards.append(
            '<article class="summary-card">'
            f'<p class="summary-label">{html.escape(title)}</p>'
            f'<p class="summary-value">{html.escape(value_text)}</p>'
            "</article>"
        )
    return f'<div class="summary-grid">{"".join(cards)}</div>'


def _format_dataframe_for_display(frame: pd.DataFrame) -> pd.DataFrame:
    formatted = frame.copy()
    for column_name in formatted.columns:
        if pd.api.types.is_datetime64_any_dtype(formatted[column_name]):
            formatted[column_name] = formatted[column_name].astype(str)
            continue
        if pd.api.types.is_float_dtype(formatted[column_name]):
            formatted[column_name] = formatted[column_name].map(lambda value: f"{value:.6f}")
            continue
        if pd.api.types.is_numeric_dtype(formatted[column_name]):
            formatted[column_name] = formatted[column_name].astype(str)

    renamed_columns = {column_name: _display_name(column_name) for column_name in formatted.columns}
    return formatted.rename(columns=renamed_columns)


def _render_table_html(display_frame: pd.DataFrame, highlight_row: int | None = None) -> str:
    header_cells = "".join(
        f"<th>{html.escape(str(column_name))}</th>" for column_name in display_frame.columns
    )
    body_rows: list[str] = []
    for row_position in range(len(display_frame)):
        row = display_frame.iloc[row_position]
        row_class = ' class="window-row-worst"' if highlight_row == row_position else ""
        value_cells = "".join(f"<td>{html.escape(str(value))}</td>" for value in row.tolist())
        body_rows.append(f"<tr{row_class}>{value_cells}</tr>")

    return (
        f"<table><thead><tr>{header_cells}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"
    )


def _find_worst_window_row(window_metrics_df: pd.DataFrame) -> int | None:
    analysis_frame = window_metrics_df.reset_index(drop=True)
    if analysis_frame.empty:
        return None

    if "sharpe" in analysis_frame.columns:
        sharpe_series = pd.to_numeric(analysis_frame["sharpe"], errors="coerce")
        if sharpe_series.notna().any():
            return int(sharpe_series.idxmin())

    if "max_drawdown" in analysis_frame.columns:
        drawdown_series = pd.to_numeric(analysis_frame["max_drawdown"], errors="coerce")
        if drawdown_series.notna().any():
            return int(drawdown_series.idxmin())

    return None


def _render_reproduce_command(command: str | None) -> str:
    if not command:
        return "<p>No reproduce command available.</p>"
    return f'<pre class="command-block"><code>{html.escape(command)}</code></pre>'


def _default_reproduce_command(config: dict[str, Any]) -> str:
    report_name = str(config.get("report_name", "report"))
    return f"python -m stratcheck run --config config.toml  # report={report_name}"


def _render_full_config_text(full_config: dict[str, Any] | str) -> str:
    if isinstance(full_config, str):
        payload = full_config
    else:
        payload = json.dumps(full_config, indent=2, ensure_ascii=False, default=str)
    return (
        "<details>"
        "<summary>Show Full Config</summary>"
        f'<pre class="config-block">{html.escape(payload)}</pre>'
        "</details>"
    )


def _display_name(name: str) -> str:
    special_names = {
        "cagr": "CAGR",
        "max_drawdown": "Max Drawdown",
        "win_rate": "Win Rate",
        "avg_trade_pnl": "Avg Trade PnL",
        "window_index": "Window Index",
        "window_start": "Window Start",
        "window_end": "Window End",
        "bars_count": "Bars Count",
        "sharpe_ci_low": "Sharpe CI Low",
        "sharpe_ci_high": "Sharpe CI High",
        "bootstrap_samples": "Bootstrap Samples",
        "autocorr_lag1": "Autocorr Lag1",
        "hit_rate_zscore": "Hit-Rate Z-Score",
        "sharpe_variance": "Sharpe Variance",
        "worst_window_sharpe": "Worst Window Sharpe",
        "stability_score": "Stability Score",
        "risk_red_count": "Risk Red Count",
        "risk_yellow_count": "Risk Yellow Count",
        "risk_green_count": "Risk Green Count",
        "orders_total": "Orders Total",
        "filled_orders": "Filled Orders",
        "canceled_orders": "Canceled Orders",
        "cancel_rate": "Cancel Rate",
        "partially_filled_orders": "Partially Filled Orders",
        "partial_fill_ratio": "Partial Fill Ratio",
        "avg_slippage_bps": "Avg Slippage (bps)",
        "median_slippage_bps": "Median Slippage (bps)",
        "avg_latency_seconds": "Avg Latency Seconds",
        "median_latency_seconds": "Median Latency Seconds",
        "avg_latency_bars": "Avg Latency Bars",
        "median_latency_bars": "Median Latency Bars",
        "compared_points": "Compared Points",
        "tolerance_abs": "Tolerance Abs",
        "max_abs_error": "Max Abs Error",
        "mean_abs_error": "Mean Abs Error",
        "rmse": "RMSE",
    }
    if name in special_names:
        return special_names[name]
    return name.replace("_", " ").title()


def _format_value(metric_key: str, value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        percentage_keys = {
            "total_return",
            "annual_return",
            "annual_volatility",
            "max_drawdown",
            "win_rate",
            "cagr",
        }
        if metric_key in percentage_keys:
            return f"{float(value) * 100:.2f}%"
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):.6f}"
    return str(value)
