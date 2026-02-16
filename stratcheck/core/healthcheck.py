"""Walk-forward healthcheck utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stratcheck.core.backtest import BacktestEngine, CostModel
from stratcheck.core.calendar import MarketCalendar, normalize_bars_freq
from stratcheck.core.metrics import compute_metrics
from stratcheck.core.strategy import Strategy


def run_healthcheck(
    strategy: Strategy,
    bars: pd.DataFrame,
    window_size: str,
    step_size: str,
    initial_cash: float = 100_000.0,
    cost_model: CostModel | None = None,
    bars_freq: str = "1d",
    output_json_path: str | Path = "reports/healthcheck_summary.json",
) -> tuple[pd.DataFrame, Path]:
    """Run walk-forward checks on rolling windows and export a JSON summary."""
    normalized_bars = _normalize_bars(bars)
    canonical_freq = normalize_bars_freq(bars_freq)

    engine = BacktestEngine()
    calendar = MarketCalendar()
    rows: list[dict[str, object]] = []

    for window in calendar.split_rolling_windows(
        bars=normalized_bars,
        window_size=window_size,
        step_size=step_size,
        bars_freq=canonical_freq,
    ):
        backtest_result = engine.run(
            strategy=strategy,
            bars=window.bars,
            initial_cash=initial_cash,
            cost_model=cost_model,
        )
        window_metrics = compute_metrics(
            equity_curve=backtest_result.equity_curve,
            trades=backtest_result.trades,
            bars_freq=canonical_freq,
        )

        rows.append(
            {
                "window_index": window.window_index,
                "window_start": window.bars.index[0],
                "window_end": window.bars.index[-1],
                "bars_count": int(len(window.bars)),
                **window_metrics,
            }
        )

    summary_frame = pd.DataFrame(rows)
    if not summary_frame.empty:
        summary_frame = summary_frame.sort_values("window_index").reset_index(drop=True)

    json_path = _write_summary_json(summary_frame, output_json_path)
    return summary_frame, json_path


def _normalize_bars(bars: pd.DataFrame) -> pd.DataFrame:
    if len(bars) == 0:
        msg = "bars cannot be empty."
        raise ValueError(msg)
    if not isinstance(bars.index, pd.DatetimeIndex):
        msg = "bars must use a DatetimeIndex."
        raise ValueError(msg)

    normalized = bars.sort_index().copy()
    normalized = normalized[~normalized.index.duplicated(keep="last")]
    return normalized


def _write_summary_json(summary_frame: pd.DataFrame, output_json_path: str | Path) -> Path:
    json_path = Path(output_json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    for _, row in summary_frame.iterrows():
        entry: dict[str, object] = {}
        for column_name, value in row.items():
            if isinstance(value, pd.Timestamp):
                entry[column_name] = value.isoformat()
            elif isinstance(value, (float, int)):
                entry[column_name] = float(value)
            else:
                entry[column_name] = value
        records.append(entry)

    payload = {"windows": records}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return json_path
