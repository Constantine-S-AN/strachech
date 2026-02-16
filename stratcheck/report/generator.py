"""HTML report generation utilities."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from stratcheck.core.backtest import BacktestResult


def render_html_report(
    result: BacktestResult,
    output_path: str | Path,
    title: str = "Strategy Check Report",
) -> Path:
    """Render a complete HTML report with chart and metrics."""
    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    chart_html = _build_chart(result)
    metrics_table_html = _build_metrics_table(result.metrics)
    generated_time = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f4f6fb;
      color: #131722;
    }}
    .wrap {{
      max-width: 1120px;
      margin: 24px auto;
      padding: 0 16px 24px;
    }}
    .card {{
      background: #ffffff;
      border-radius: 14px;
      box-shadow: 0 8px 24px rgba(19, 23, 34, 0.08);
      padding: 20px;
      margin-bottom: 16px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
    }}
    p {{
      margin: 0;
      color: #4b5565;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
      font-size: 14px;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid #e7eaf2;
      text-align: left;
    }}
    th {{
      width: 240px;
      color: #30384a;
      font-weight: 600;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="card">
      <h1>{title}</h1>
      <p>Generated at {generated_time}</p>
    </section>
    <section class="card">
      <h2>Performance Metrics</h2>
      {metrics_table_html}
    </section>
    <section class="card">
      <h2>Charts</h2>
      {chart_html}
    </section>
  </div>
</body>
</html>
"""

    report_path.write_text(document, encoding="utf-8")
    return report_path


def _build_chart(result: BacktestResult) -> str:
    market_data = result.market_data

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.62, 0.38],
        subplot_titles=("Price (OHLC + Moving Averages)", "Equity Curves"),
    )

    figure.add_trace(
        go.Candlestick(
            x=market_data.index,
            open=market_data["open"],
            high=market_data["high"],
            low=market_data["low"],
            close=market_data["close"],
            name="OHLC",
            increasing_line_color="#06b6d4",
            decreasing_line_color="#f97316",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=market_data.index,
            y=market_data["fast_average"],
            mode="lines",
            name="Fast MA",
            line={"color": "#2563eb", "width": 1.6},
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=market_data.index,
            y=market_data["slow_average"],
            mode="lines",
            name="Slow MA",
            line={"color": "#7c3aed", "width": 1.6},
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=result.equity_curve.index,
            y=result.equity_curve,
            mode="lines",
            name="Strategy Equity",
            line={"color": "#16a34a", "width": 2},
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=result.benchmark_curve.index,
            y=result.benchmark_curve,
            mode="lines",
            name="Benchmark Equity",
            line={"color": "#dc2626", "width": 2, "dash": "dot"},
        ),
        row=2,
        col=1,
    )

    figure.update_layout(
        template="plotly_white",
        height=860,
        margin={"l": 40, "r": 24, "t": 50, "b": 24},
        xaxis_rangeslider_visible=False,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "left", "x": 0.0},
    )
    figure.update_yaxes(title_text="Price", row=1, col=1)
    figure.update_yaxes(title_text="Equity", row=2, col=1)

    return figure.to_html(full_html=False, include_plotlyjs="cdn")


def _build_metrics_table(metrics: dict[str, float | int]) -> str:
    metric_names = {
        "total_return": "Total Return",
        "annual_return": "Annual Return",
        "annual_volatility": "Annual Volatility",
        "sharpe_ratio": "Sharpe Ratio",
        "max_drawdown": "Max Drawdown",
        "win_rate": "Win Rate",
        "trade_entries": "Trade Entries",
    }

    rows: list[str] = []
    for metric_key, metric_name in metric_names.items():
        raw_value = metrics.get(metric_key, 0.0)
        formatted_value = _format_metric(metric_key, raw_value)
        rows.append(f"<tr><th>{metric_name}</th><td>{formatted_value}</td></tr>")

    body = "\n".join(rows)
    return f"<table><tbody>{body}</tbody></table>"


def _format_metric(metric_key: str, value: float | int) -> str:
    percentage_metrics = {
        "total_return",
        "annual_return",
        "annual_volatility",
        "max_drawdown",
        "win_rate",
    }
    if metric_key in percentage_metrics:
        return f"{float(value) * 100:.2f}%"
    if metric_key == "trade_entries":
        return f"{int(value)}"
    return f"{float(value):.3f}"
