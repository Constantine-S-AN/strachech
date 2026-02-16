"""Static HTML dashboard builder."""

from __future__ import annotations

import html
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


def build_dashboard_html(
    output_path: str | Path,
    experiments_df: pd.DataFrame,
    runs_df: pd.DataFrame,
    live_positions_df: pd.DataFrame | None = None,
    live_trades_df: pd.DataFrame | None = None,
    live_risk_df: pd.DataFrame | None = None,
    live_errors_df: pd.DataFrame | None = None,
    reports_dir: str | Path = "reports",
) -> Path:
    """Build static dashboard page with leaderboard and live-status panels."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    reports_root = Path(reports_dir)

    experiment_table = _render_experiment_table(
        frame=experiments_df,
        html_path=destination,
        reports_dir=reports_root,
    )
    leaderboard_table = _render_leaderboard_table(
        frame=experiments_df,
        html_path=destination,
        reports_dir=reports_root,
    )
    run_table = _render_generic_table(frame=runs_df, empty_message="No paper runs found.")

    positions_table = _render_live_positions_table(
        frame=live_positions_df if live_positions_df is not None else pd.DataFrame()
    )
    trades_table = _render_live_trades_table(
        frame=live_trades_df if live_trades_df is not None else pd.DataFrame()
    )
    risk_table = _render_live_risk_table(
        frame=live_risk_df if live_risk_df is not None else pd.DataFrame()
    )
    errors_table = _render_live_errors_table(
        frame=live_errors_df if live_errors_df is not None else pd.DataFrame()
    )
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Stratcheck Dashboard</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #0f172a;
      background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
    }}
    .wrap {{
      max-width: 1240px;
      margin: 24px auto;
      padding: 0 16px 28px;
    }}
    .panel {{
      background: #ffffff;
      border-radius: 14px;
      padding: 18px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
      margin-bottom: 14px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 28px;
    }}
    h2 {{
      margin: 0 0 10px;
      font-size: 20px;
    }}
    h3 {{
      margin: 14px 0 8px;
      font-size: 16px;
      color: #1f2a44;
    }}
    .meta {{
      margin: 0;
      color: #475569;
      font-size: 14px;
    }}
    .search {{
      width: 100%;
      box-sizing: border-box;
      margin: 8px 0 12px;
      padding: 10px 12px;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      font-size: 14px;
    }}
    .sort-controls {{
      margin: 8px 0 12px;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
    }}
    .sort-controls label {{
      color: #334155;
      font-size: 13px;
    }}
    .sort-controls select, .sort-controls button {{
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      background: #ffffff;
      font-size: 13px;
      padding: 6px 8px;
      color: #0f172a;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      text-align: left;
      border-bottom: 1px solid #e2e8f0;
      padding: 8px 10px;
      vertical-align: top;
    }}
    th {{
      color: #334155;
      font-weight: 600;
      background: #f8fafc;
      position: sticky;
      top: 0;
    }}
    .table-wrap {{
      max-height: 420px;
      overflow: auto;
      border: 1px solid #e2e8f0;
      border-radius: 10px;
    }}
    .live-grid {{
      display: grid;
      gap: 14px;
      grid-template-columns: 1fr;
    }}
    .live-section {{
      border: 1px solid #e2e8f0;
      border-radius: 10px;
      padding: 10px;
      background: #fcfdff;
    }}
    .pill {{
      display: inline-block;
      border-radius: 999px;
      padding: 2px 10px;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }}
    .ok {{
      background: #dcfce7;
      color: #14532d;
    }}
    .warn {{
      background: #fef3c7;
      color: #92400e;
    }}
    .bad {{
      background: #fee2e2;
      color: #991b1b;
    }}
  </style>
</head>
<body>
  <main class="wrap">
    <section class="panel">
      <h1>Stratcheck Dashboard</h1>
      <p class="meta">Generated at {generated_at}</p>
      <p class="meta">
        One-page view: experiment ranking, leaderboard, live status, and paper run history.
      </p>
    </section>
    <section class="panel">
      <h2>Leaderboard</h2>
      <div class="sort-controls">
        <label for="leaderMetric">Metric:</label>
        <select id="leaderMetric">
          <option value="sharpe">sharpe</option>
          <option value="max_drawdown">max_drawdown</option>
          <option value="worst_window_sharpe">worst_window_sharpe</option>
          <option value="worst_window_drawdown">worst_window_drawdown</option>
        </select>
        <label for="leaderDirection">Direction:</label>
        <select id="leaderDirection">
          <option value="desc">desc</option>
          <option value="asc">asc</option>
        </select>
        <button id="leaderApplySort" type="button">Apply</button>
      </div>
      {leaderboard_table}
    </section>
    <section class="panel">
      <h2>Experiment Ranking</h2>
      <input id="experimentFilter" class="search" type="text" placeholder="Filter experiments..." />
      {experiment_table}
    </section>
    <section class="panel">
      <h2>Live Status</h2>
      <div class="live-grid">
        <section class="live-section">
          <h3>Current Positions</h3>
          {positions_table}
        </section>
        <section class="live-section">
          <h3>Today&apos;s Trades</h3>
          {trades_table}
        </section>
        <section class="live-section">
          <h3>Risk Status</h3>
          {risk_table}
        </section>
        <section class="live-section">
          <h3>Recent Errors</h3>
          {errors_table}
        </section>
      </div>
    </section>
    <section class="panel">
      <h2>Paper Run Status</h2>
      <input id="runFilter" class="search" type="text" placeholder="Filter runs..." />
      {run_table}
    </section>
  </main>
  <script>
    function setupFilter(inputId, tableId) {{
      const input = document.getElementById(inputId);
      const table = document.getElementById(tableId);
      if (!input || !table) return;
      input.addEventListener("input", () => {{
        const query = input.value.toLowerCase();
        const rows = table.querySelectorAll("tbody tr");
        rows.forEach((row) => {{
          const text = row.innerText.toLowerCase();
          row.style.display = text.includes(query) ? "" : "none";
        }});
      }});
    }}

    function setupLeaderboardSort() {{
      const table = document.getElementById("leaderboardTable");
      const metricSelect = document.getElementById("leaderMetric");
      const directionSelect = document.getElementById("leaderDirection");
      const applyButton = document.getElementById("leaderApplySort");
      if (!table || !metricSelect || !directionSelect || !applyButton) return;

      function columnIndexByName(columnName) {{
        const headers = Array.from(table.querySelectorAll("thead th"));
        return headers.findIndex((header) => header.innerText.trim() === columnName);
      }}

      function parseNumeric(text) {{
        const normalized = text.replace(/,/g, "").trim();
        if (normalized === "" || normalized === "-") return null;
        const numeric = Number(normalized);
        if (Number.isNaN(numeric)) return null;
        return numeric;
      }}

      function applySort() {{
        const metricName = metricSelect.value;
        const direction = directionSelect.value;
        const metricIndex = columnIndexByName(metricName);
        if (metricIndex < 0 || !table.tBodies.length) return;

        const rankIndex = columnIndexByName("leaderboard_rank");
        const rows = Array.from(table.tBodies[0].rows);
        rows.sort((leftRow, rightRow) => {{
          const leftText = leftRow.cells[metricIndex].innerText;
          const rightText = rightRow.cells[metricIndex].innerText;
          let leftValue = parseNumeric(leftText);
          let rightValue = parseNumeric(rightText);

          if (
            metricName === "max_drawdown" ||
            metricName === "worst_window_drawdown"
          ) {{
            leftValue = leftValue === null ? null : Math.abs(leftValue);
            rightValue = rightValue === null ? null : Math.abs(rightValue);
          }}

          if (leftValue === null && rightValue === null) return 0;
          if (leftValue === null) return 1;
          if (rightValue === null) return -1;

          const result = leftValue - rightValue;
          return direction === "desc" ? -result : result;
        }});

        rows.forEach((row, rowIndex) => {{
          table.tBodies[0].appendChild(row);
          if (rankIndex >= 0) {{
            row.cells[rankIndex].innerText = String(rowIndex + 1);
          }}
        }});
      }}

      applyButton.addEventListener("click", applySort);
      applySort();
    }}

    setupFilter("experimentFilter", "experimentTable");
    setupFilter("runFilter", "runTable");
    setupLeaderboardSort();
  </script>
</body>
</html>
"""

    destination.write_text(document, encoding="utf-8")
    return destination


def _render_experiment_table(
    frame: pd.DataFrame,
    html_path: Path,
    reports_dir: Path,
) -> str:
    if frame.empty:
        return "<p>No experiment records found.</p>"

    display_frame = frame.copy()
    if "report_path" in display_frame.columns:
        display_frame["report_path"] = display_frame["report_path"].map(
            lambda value: _as_report_link(value=value, html_path=html_path, reports_dir=reports_dir)
        )
    if "status" in display_frame.columns:
        display_frame["status"] = display_frame["status"].map(_as_status_pill)
    return _render_table_from_frame(
        display_frame,
        table_id="experimentTable",
        safe_html_columns={"status", "report_path"},
    )


def _render_leaderboard_table(
    frame: pd.DataFrame,
    html_path: Path,
    reports_dir: Path,
) -> str:
    if frame.empty:
        return "<p>No experiment records found.</p>"

    source = frame.copy()
    if "status" in source.columns:
        success_only = source[source["status"] == "success"].copy()
        if not success_only.empty:
            source = success_only

    wanted_columns = [
        "leaderboard_rank",
        "experiment",
        "sharpe",
        "max_drawdown",
        "worst_window_sharpe",
        "worst_window_drawdown",
        "cagr",
        "total_return",
        "status",
        "report_path",
        "error",
    ]
    display_frame = source.copy()
    for column_name in wanted_columns:
        if column_name not in display_frame.columns:
            display_frame[column_name] = float("nan") if "drawdown" in column_name else ""

    display_frame = display_frame.sort_values(
        by=["sharpe", "experiment"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)
    display_frame["leaderboard_rank"] = [index + 1 for index in range(len(display_frame))]
    display_frame = display_frame[wanted_columns]

    display_frame["report_path"] = display_frame["report_path"].map(
        lambda value: _as_report_link(value=value, html_path=html_path, reports_dir=reports_dir)
    )
    display_frame["status"] = display_frame["status"].map(_as_status_pill)
    return _render_table_from_frame(
        display_frame,
        table_id="leaderboardTable",
        safe_html_columns={"status", "report_path"},
    )


def _render_generic_table(frame: pd.DataFrame, empty_message: str) -> str:
    if frame.empty:
        return f"<p>{html.escape(empty_message)}</p>"

    display_frame = frame.copy()
    safe_html_columns: set[str] = set()
    if "status" in display_frame.columns:
        display_frame["status"] = display_frame["status"].map(_as_status_pill)
        safe_html_columns.add("status")
    return _render_table_from_frame(
        display_frame,
        table_id="runTable",
        safe_html_columns=safe_html_columns,
    )


def _render_live_positions_table(frame: pd.DataFrame) -> str:
    return _render_live_table(
        frame=frame,
        table_id="livePositionsTable",
        empty_message="No live position snapshots found.",
    )


def _render_live_trades_table(frame: pd.DataFrame) -> str:
    return _render_live_table(
        frame=frame,
        table_id="liveTradesTable",
        empty_message="No trades found for today.",
    )


def _render_live_risk_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "<p>No live risk status found.</p>"

    display_frame = frame.copy()
    safe_html_columns: set[str] = set()
    if "status" in display_frame.columns:
        display_frame["status"] = display_frame["status"].map(_as_status_pill)
        safe_html_columns.add("status")
    if "risk_state" in display_frame.columns:
        display_frame["risk_state"] = display_frame["risk_state"].map(_as_risk_state_pill)
        safe_html_columns.add("risk_state")
    return _render_table_from_frame(
        display_frame,
        table_id="liveRiskTable",
        safe_html_columns=safe_html_columns,
    )


def _render_live_errors_table(frame: pd.DataFrame) -> str:
    return _render_live_table(
        frame=frame,
        table_id="liveErrorsTable",
        empty_message="No recent errors found.",
    )


def _render_live_table(frame: pd.DataFrame, table_id: str, empty_message: str) -> str:
    if frame.empty:
        return f"<p>{html.escape(empty_message)}</p>"
    return _render_table_from_frame(frame=frame, table_id=table_id)


def _render_table_from_frame(
    frame: pd.DataFrame,
    table_id: str,
    safe_html_columns: set[str] | None = None,
) -> str:
    rendered = frame.copy()
    html_safe = set(safe_html_columns or set())

    for column_name in rendered.columns:
        if pd.api.types.is_float_dtype(rendered[column_name]):
            rendered[column_name] = rendered[column_name].map(
                lambda value: "-" if pd.isna(value) else f"{value:.6f}"
            )
            continue

        if column_name in html_safe:
            rendered[column_name] = rendered[column_name].map(
                lambda value: "" if pd.isna(value) else str(value)
            )
            continue

        rendered[column_name] = rendered[column_name].map(
            lambda value: "" if pd.isna(value) else html.escape(str(value))
        )

    header_cells = "".join(f"<th>{html.escape(str(name))}</th>" for name in rendered.columns)
    body_rows: list[str] = []
    for _, row in rendered.iterrows():
        cells = "".join(f"<td>{value}</td>" for value in row.tolist())
        body_rows.append(f"<tr>{cells}</tr>")

    return (
        '<div class="table-wrap">'
        f'<table id="{html.escape(table_id)}">'
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
        "</div>"
    )


def _as_report_link(value: object, html_path: Path, reports_dir: Path) -> str:
    report_name = str(value or "").strip()
    if not report_name:
        return "-"
    report_path = reports_dir / report_name
    if not report_path.exists():
        return html.escape(report_name)
    relative_path = (
        report_path.relative_to(html_path.parent)
        if report_path.is_relative_to(html_path.parent)
        else report_path
    )
    url = html.escape(relative_path.as_posix())
    label = html.escape(report_name)
    return f'<a href="{url}" target="_blank" rel="noopener noreferrer">{label}</a>'


def _as_status_pill(value: object) -> str:
    status_text = str(value or "").strip().lower()
    css_class = "warn"
    if status_text in {"success", "completed", "running"}:
        css_class = "ok"
    elif status_text in {"failed", "killed"}:
        css_class = "bad"
    return f'<span class="pill {css_class}">{html.escape(status_text or "-")}</span>'


def _as_risk_state_pill(value: object) -> str:
    risk_text = str(value or "").strip().lower()
    css_class = "warn"
    if risk_text in {"ok", "healthy"}:
        css_class = "ok"
    elif risk_text in {"critical", "halted"}:
        css_class = "bad"
    return f'<span class="pill {css_class}">{html.escape(risk_text or "-")}</span>'
