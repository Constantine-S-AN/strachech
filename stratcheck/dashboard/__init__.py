"""Static dashboard entry points."""

from pathlib import Path

from stratcheck.dashboard.builder import build_dashboard_html
from stratcheck.dashboard.query import (
    LiveStatusData,
    load_experiment_rankings,
    load_live_status,
    load_paper_run_statuses,
)


def build_dashboard_site(
    *,
    results_jsonl_path: str | Path = "reports/results.jsonl",
    sqlite_path: str | Path = "reports/paper_trading.sqlite",
    output_path: str | Path = "reports/dashboard.html",
    reports_dir: str | Path = "reports",
) -> Path:
    """Build static dashboard from experiment results and paper-run sqlite."""
    experiments_df = load_experiment_rankings(results_jsonl_path=results_jsonl_path)
    runs_df = load_paper_run_statuses(sqlite_path=sqlite_path)
    live_status = load_live_status(sqlite_path=sqlite_path)
    return build_dashboard_html(
        output_path=output_path,
        experiments_df=experiments_df,
        runs_df=runs_df,
        live_positions_df=live_status.positions_df,
        live_trades_df=live_status.today_trades_df,
        live_risk_df=live_status.risk_status_df,
        live_errors_df=live_status.recent_errors_df,
        reports_dir=reports_dir,
    )


__all__ = [
    "build_dashboard_html",
    "build_dashboard_site",
    "LiveStatusData",
    "load_experiment_rankings",
    "load_live_status",
    "load_paper_run_statuses",
]
