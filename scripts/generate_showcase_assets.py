"""Generate portfolio showcase images from demo outputs and snapshots."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from stratcheck.analysis import compute_execution_quality_scorecard
from stratcheck.core import BacktestEngine, CSVDataProvider, build_cost_model
from stratcheck.dashboard.query import load_live_status
from stratcheck.strategies import BuyAndHoldStrategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate showcase image set for README/docs portfolio sections.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Project root directory. Default: current directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/images/showcase"),
        help="Output image directory relative to project root.",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="QQQ",
        help="Symbol used for execution-quality showcase snapshot.",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=100000.0,
        help="Initial cash used for execution-quality showcase snapshot.",
    )
    return parser.parse_args()


def render_table_image(
    frame: pd.DataFrame,
    title: str,
    output_path: Path,
    *,
    max_rows: int = 12,
) -> None:
    """Render a dataframe as an image table for portfolio slides."""
    display_frame = frame.copy()
    if display_frame.empty:
        display_frame = pd.DataFrame([{"note": "No data available"}])

    display_frame = display_frame.head(max_rows)
    for column_name in display_frame.columns:
        if pd.api.types.is_float_dtype(display_frame[column_name]):
            display_frame[column_name] = display_frame[column_name].map(
                lambda value: f"{value:.4f}",
            )
    display_frame = display_frame.fillna("")

    row_count = len(display_frame)
    column_count = len(display_frame.columns)
    figure_width = max(10.0, min(24.0, 1.45 * column_count))
    figure_height = max(3.0, min(16.0, 0.62 * row_count + 2.4))

    figure, axis = plt.subplots(figsize=(figure_width, figure_height), dpi=180)
    axis.axis("off")
    axis.set_title(title, fontsize=14, fontweight="bold", pad=16)

    table = axis.table(
        cellText=display_frame.values,
        colLabels=display_frame.columns,
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.05, 1.25)

    for (row_index, _), cell in table.get_celld().items():
        if row_index == 0:
            cell.set_text_props(weight="bold", color="#0f172a")
            cell.set_facecolor("#e2e8f0")
        else:
            cell.set_facecolor("#f8fafc" if row_index % 2 == 0 else "#ffffff")

    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def generate_static_copies(project_root: Path, output_dir: Path) -> None:
    """Copy representative report and chart images."""
    copy_mapping = {
        project_root / "docs/images/demo-report.png": output_dir / "01_report_overview.png",
        project_root
        / "reports/assets/qqq_buy_and_hold_equity.png": output_dir / "02_equity_curve.png",
        project_root
        / "reports/assets/qqq_buy_and_hold_drawdown.png": output_dir / "03_drawdown_curve.png",
        project_root
        / "reports/assets/buy_and_hold_cost_sensitivity.png": output_dir
        / "04_cost_sensitivity.png",
    }
    for source_path, destination_path in copy_mapping.items():
        if source_path.exists():
            shutil.copy2(source_path, destination_path)


def generate_execution_quality_snapshot(
    project_root: Path,
    output_dir: Path,
    symbol: str,
    initial_cash: float,
) -> None:
    """Build execution-quality metrics table image."""
    provider = CSVDataProvider(data_dir=project_root / "data")
    bars = provider.get_bars(symbol=symbol, timeframe="1d")
    strategy = BuyAndHoldStrategy(target_position_qty=1.0)
    cost_model = build_cost_model(
        {
            "type": "fixed_bps",
            "commission_bps": 2,
            "slippage_bps": 1,
        }
    )
    backtest_result = BacktestEngine().run(
        strategy=strategy,
        bars=bars,
        initial_cash=float(initial_cash),
        cost_model=cost_model,
    )
    metrics_frame = compute_execution_quality_scorecard(
        orders=backtest_result.orders,
        bars=backtest_result.market_data,
        trades=backtest_result.trades,
    )
    render_table_image(
        frame=metrics_frame,
        title=f"Execution Quality Snapshot ({symbol})",
        output_path=output_dir / "05_execution_quality_table.png",
    )


def load_results_frame(results_path: Path) -> pd.DataFrame:
    """Load experiment rows from results JSONL."""
    if not results_path.exists():
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    with results_path.open("r", encoding="utf-8") as input_file:
        for raw_line in input_file:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            rows.append(dict(payload))
    return pd.DataFrame(rows)


def generate_leaderboard_snapshot(project_root: Path, output_dir: Path) -> None:
    """Build experiment leaderboard image."""
    results_frame = load_results_frame(project_root / "reports/results.jsonl")
    if results_frame.empty:
        render_table_image(
            frame=pd.DataFrame(),
            title="Experiment Leaderboard Snapshot",
            output_path=output_dir / "06_leaderboard_table.png",
        )
        return

    for metric_name in [
        "sharpe",
        "max_drawdown",
        "worst_window_sharpe",
        "cagr",
        "total_return",
    ]:
        if metric_name in results_frame.columns:
            results_frame[metric_name] = pd.to_numeric(
                results_frame[metric_name],
                errors="coerce",
            )

    if "status" in results_frame.columns:
        results_frame["status_sort"] = (
            results_frame["status"].map({"success": 0, "failed": 1}).fillna(2)
        )
    else:
        results_frame["status_sort"] = 2

    results_frame = results_frame.sort_values(
        by=["status_sort", "sharpe"],
        ascending=[True, False],
        na_position="last",
    )
    leaderboard_frame = results_frame.reindex(
        columns=[
            "experiment",
            "status",
            "sharpe",
            "max_drawdown",
            "worst_window_sharpe",
            "cagr",
            "total_return",
        ],
        fill_value="",
    )
    render_table_image(
        frame=leaderboard_frame,
        title="Experiment Leaderboard Snapshot",
        output_path=output_dir / "06_leaderboard_table.png",
    )


def generate_live_status_snapshot(project_root: Path, output_dir: Path) -> None:
    """Render live status tables into one image."""
    live_status = load_live_status(sqlite_path=project_root / "reports/paper_trading.sqlite")
    positions_frame = live_status.positions_df.reindex(
        columns=[
            "run_id",
            "symbol",
            "strategy",
            "position_qty",
            "close_price",
            "equity",
            "updated_at",
        ],
        fill_value="",
    )
    risk_frame = live_status.risk_status_df.reindex(
        columns=[
            "run_id",
            "symbol",
            "risk_state",
            "risk_halt_hits",
            "risk_block_hits",
            "last_risk_action",
        ],
        fill_value="",
    )

    figure, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 9), dpi=180)
    for axis in axes:
        axis.axis("off")

    axes[0].set_title(
        "Live Status Snapshot: Current Positions",
        fontsize=13,
        fontweight="bold",
        pad=8,
    )
    positions_view = positions_frame.head(8).copy()
    if positions_view.empty:
        positions_view = pd.DataFrame([{"note": "No live positions"}])
    positions_view = positions_view.fillna("")
    positions_table = axes[0].table(
        cellText=positions_view.values,
        colLabels=positions_view.columns,
        loc="center",
        cellLoc="left",
    )
    positions_table.auto_set_font_size(False)
    positions_table.set_fontsize(8)
    positions_table.scale(1.0, 1.2)

    axes[1].set_title("Live Status Snapshot: Risk", fontsize=13, fontweight="bold", pad=8)
    risk_view = risk_frame.head(8).copy()
    if risk_view.empty:
        risk_view = pd.DataFrame([{"note": "No live risk rows"}])
    risk_view = risk_view.fillna("")
    risk_table = axes[1].table(
        cellText=risk_view.values,
        colLabels=risk_view.columns,
        loc="center",
        cellLoc="left",
    )
    risk_table.auto_set_font_size(False)
    risk_table.set_fontsize(8)
    risk_table.scale(1.0, 1.2)

    for table in (positions_table, risk_table):
        for (row_index, _), cell in table.get_celld().items():
            if row_index == 0:
                cell.set_text_props(weight="bold", color="#0f172a")
                cell.set_facecolor("#e2e8f0")
            else:
                cell.set_facecolor("#f8fafc" if row_index % 2 == 0 else "#ffffff")

    figure.tight_layout(h_pad=2.0)
    figure.savefig(output_dir / "07_live_status_table.png", bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    project_root = args.root.resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_static_copies(project_root=project_root, output_dir=output_dir)
    generate_execution_quality_snapshot(
        project_root=project_root,
        output_dir=output_dir,
        symbol=str(args.symbol),
        initial_cash=float(args.initial_cash),
    )
    generate_leaderboard_snapshot(project_root=project_root, output_dir=output_dir)
    generate_live_status_snapshot(project_root=project_root, output_dir=output_dir)

    generated = sorted(path.name for path in output_dir.glob("*.png"))
    print("Generated showcase files:")
    for filename in generated:
        print(f"- {filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
