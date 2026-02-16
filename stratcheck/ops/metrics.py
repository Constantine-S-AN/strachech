"""Runtime metrics collection and export helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(slots=True)
class RuntimeMetricsSnapshot:
    """Runtime metrics snapshot."""

    pnl: float
    drawdown: float
    order_rate: float
    error_rate: float
    bars_processed: int
    orders_placed: int
    errors_total: int


class RuntimeMetrics:
    """Collect runner metrics and export to Prometheus text / CSV."""

    def __init__(self, initial_equity: float) -> None:
        self.initial_equity = float(initial_equity)
        self.current_equity = float(initial_equity)
        self.peak_equity = float(initial_equity)
        self.max_drawdown = 0.0

        self.start_time = pd.Timestamp.now(tz="UTC")
        self.end_time: pd.Timestamp | None = None
        self.bars_processed = 0
        self.orders_placed = 0
        self.errors_total = 0

    def record_bar(self, equity: float) -> None:
        """Record one processed bar and update drawdown/pnl state."""
        self.bars_processed += 1
        self.current_equity = float(equity)
        self.peak_equity = max(self.peak_equity, self.current_equity)
        if self.peak_equity > 0:
            drawdown_value = (self.peak_equity - self.current_equity) / self.peak_equity
            self.max_drawdown = max(self.max_drawdown, float(drawdown_value))

    def record_order(self, count: int = 1) -> None:
        """Increment order counter."""
        self.orders_placed += int(count)

    def record_error(self, count: int = 1) -> None:
        """Increment error counter."""
        self.errors_total += int(count)

    def finalize(self) -> None:
        """Mark metrics collection end timestamp."""
        self.end_time = pd.Timestamp.now(tz="UTC")

    def snapshot(self) -> RuntimeMetricsSnapshot:
        """Return computed runtime metrics."""
        elapsed_seconds = self._elapsed_seconds()
        pnl = float(self.current_equity - self.initial_equity)
        order_rate = float(self.orders_placed / elapsed_seconds) if elapsed_seconds > 0 else 0.0
        error_rate = float(self.errors_total / elapsed_seconds) if elapsed_seconds > 0 else 0.0
        return RuntimeMetricsSnapshot(
            pnl=pnl,
            drawdown=float(self.max_drawdown),
            order_rate=order_rate,
            error_rate=error_rate,
            bars_processed=int(self.bars_processed),
            orders_placed=int(self.orders_placed),
            errors_total=int(self.errors_total),
        )

    def export_prometheus(self, path: str | Path) -> Path:
        """Export metrics in simple Prometheus text format."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metrics = self.snapshot()
        lines = [
            "# HELP stratcheck_pnl Current run PnL.",
            "# TYPE stratcheck_pnl gauge",
            f"stratcheck_pnl {metrics.pnl:.10f}",
            "# HELP stratcheck_drawdown Max drawdown ratio.",
            "# TYPE stratcheck_drawdown gauge",
            f"stratcheck_drawdown {metrics.drawdown:.10f}",
            "# HELP stratcheck_order_rate Orders per second.",
            "# TYPE stratcheck_order_rate gauge",
            f"stratcheck_order_rate {metrics.order_rate:.10f}",
            "# HELP stratcheck_error_rate Errors per second.",
            "# TYPE stratcheck_error_rate gauge",
            f"stratcheck_error_rate {metrics.error_rate:.10f}",
        ]
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path

    def export_csv(self, path: str | Path) -> Path:
        """Export metrics as single-row CSV."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metrics = self.snapshot()
        fieldnames = [
            "pnl",
            "drawdown",
            "order_rate",
            "error_rate",
            "bars_processed",
            "orders_placed",
            "errors_total",
        ]
        with output_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(
                {
                    "pnl": metrics.pnl,
                    "drawdown": metrics.drawdown,
                    "order_rate": metrics.order_rate,
                    "error_rate": metrics.error_rate,
                    "bars_processed": metrics.bars_processed,
                    "orders_placed": metrics.orders_placed,
                    "errors_total": metrics.errors_total,
                }
            )
        return output_path

    def _elapsed_seconds(self) -> float:
        stop_time = self.end_time or pd.Timestamp.now(tz="UTC")
        elapsed_seconds = (stop_time - self.start_time).total_seconds()
        return max(float(elapsed_seconds), 1e-9)
