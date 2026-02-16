"""Render a healthcheck summary markdown from healthcheck_summary.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def render_summary_markdown(summary_payload: dict[str, Any]) -> str:
    """Render markdown summary with overall and worst-window indicators."""
    raw_windows = summary_payload.get("windows", [])
    if not isinstance(raw_windows, list) or not raw_windows:
        return "## Healthcheck Summary\n\nNo window data found.\n"

    windows = [window for window in raw_windows if isinstance(window, dict)]
    if not windows:
        return "## Healthcheck Summary\n\nNo valid window rows found.\n"

    has_sharpe = any(_to_float(window.get("sharpe")) is not None for window in windows)
    sharpe_key = "sharpe" if has_sharpe else "sharpe_ratio"
    cagr_values = [_to_float(window.get("cagr")) for window in windows]
    cagr_values = [value for value in cagr_values if value is not None]
    sharpe_values = [_to_float(window.get(sharpe_key)) for window in windows]
    sharpe_values = [value for value in sharpe_values if value is not None]
    drawdown_values = [_to_float(window.get("max_drawdown")) for window in windows]
    drawdown_values = [value for value in drawdown_values if value is not None]

    worst_drawdown_window = _select_worst_window(
        windows=windows,
        metric_key="max_drawdown",
    )
    worst_sharpe_window = _select_worst_window(
        windows=windows,
        metric_key=sharpe_key,
    )

    lines: list[str] = ["## Healthcheck Summary", ""]
    lines.append("### Overall")
    lines.append(f"- Windows: {len(windows)}")
    if cagr_values:
        lines.append(f"- Mean CAGR: {_format_percent(sum(cagr_values) / len(cagr_values))}")
    if sharpe_values:
        lines.append(f"- Mean Sharpe: {_format_number(sum(sharpe_values) / len(sharpe_values))}")
    if drawdown_values:
        lines.append(f"- Worst Max Drawdown: {_format_percent(min(drawdown_values))}")
    lines.append("")

    lines.append("### Worst Windows")
    lines.append("| Metric | Window Index | Window Start | Window End | Value |")
    lines.append("|---|---:|---|---|---:|")
    lines.append(_render_window_row("Max Drawdown", "max_drawdown", worst_drawdown_window))
    lines.append(_render_window_row("Sharpe", sharpe_key, worst_sharpe_window))
    lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Render healthcheck summary markdown.")
    parser.add_argument(
        "summary_json",
        nargs="?",
        default="reports/healthcheck_summary.json",
        help="Path to healthcheck_summary.json",
    )
    args = parser.parse_args(argv)

    summary_path = Path(args.summary_json)
    if not summary_path.exists():
        print(f"## Healthcheck Summary\n\nNo summary file found at `{summary_path}`.\n")
        return 0

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    print(render_summary_markdown(payload))
    return 0


def _select_worst_window(windows: list[dict[str, Any]], metric_key: str) -> dict[str, Any] | None:
    comparable_windows = [
        window for window in windows if _to_float(window.get(metric_key)) is not None
    ]
    if not comparable_windows:
        return None
    return min(comparable_windows, key=lambda window: float(window[metric_key]))


def _render_window_row(
    metric_name: str,
    metric_key: str,
    window: dict[str, Any] | None,
) -> str:
    if window is None:
        return f"| {metric_name} | - | - | - | n/a |"

    metric_value = _to_float(window.get(metric_key))
    if metric_key == "max_drawdown":
        formatted_metric = _format_percent(metric_value) if metric_value is not None else "n/a"
    else:
        formatted_metric = _format_number(metric_value) if metric_value is not None else "n/a"

    window_index = window.get("window_index", "-")
    window_start = window.get("window_start", "-")
    window_end = window.get("window_end", "-")
    return (
        f"| {metric_name} | {window_index} | {window_start} | {window_end} | {formatted_metric} |"
    )


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _format_number(value: float) -> str:
    return f"{value:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
