from __future__ import annotations

from pathlib import Path

import pandas as pd
from stratcheck.report import ReportBuilder, generate_performance_plots


def test_report_builder_golden_demo_html_contains_key_fields(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    asset_dir = report_dir / "assets"

    timestamps = pd.date_range("2024-01-01", periods=90, freq="D", tz="UTC")
    equity_curve = pd.Series(
        [100_000.0 + index * 95.0 for index in range(90)],
        index=timestamps,
    )
    plot_paths = generate_performance_plots(
        equity_curve=equity_curve,
        output_dir=asset_dir,
        prefix="demo",
    )

    overall_metrics = {
        "cagr": 0.12,
        "annual_volatility": 0.18,
        "sharpe": 0.67,
        "max_drawdown": -0.21,
        "turnover": 1.45,
        "win_rate": 0.56,
    }
    window_metrics = pd.DataFrame(
        [
            {
                "window_index": 0,
                "window_start": timestamps[0],
                "window_end": timestamps[29],
                "sharpe": 0.25,
                "max_drawdown": -0.10,
            },
            {
                "window_index": 1,
                "window_start": timestamps[30],
                "window_end": timestamps[59],
                "sharpe": -0.31,
                "max_drawdown": -0.21,
            },
        ]
    )
    config = {
        "report_name": "demo",
        "window_size": "6M",
        "step_size": "3M",
        "commission_bps": 5,
    }

    builder = ReportBuilder(output_dir=report_dir)
    html_path = builder.build(
        overall_metrics=overall_metrics,
        window_metrics_df=window_metrics,
        plot_paths=plot_paths,
        robustness_summary={
            "bootstrap_samples": 300,
            "sharpe_ci_low": 0.12,
            "sharpe_ci_high": 0.48,
        },
        sweep_results_df=pd.DataFrame(
            [
                {
                    "short_window": 10,
                    "long_window": 40,
                    "sharpe": 0.42,
                    "status": "ok",
                    "error": "",
                },
                {
                    "short_window": 20,
                    "long_window": 50,
                    "sharpe": 0.67,
                    "status": "ok",
                    "error": "",
                },
            ]
        ),
        sensitivity_metrics_df=pd.DataFrame(
            [
                {
                    "cost_assumption": "comm=0.0bps; slip=0.0bps; spread=0.0bps",
                    "commission_bps": 0.0,
                    "slippage_bps": 0.0,
                    "spread_bps": 0.0,
                    "total_cost_bps": 0.0,
                    "sharpe": 0.78,
                    "cagr": 0.15,
                },
                {
                    "cost_assumption": "comm=5.0bps; slip=5.0bps; spread=0.0bps",
                    "commission_bps": 5.0,
                    "slippage_bps": 5.0,
                    "spread_bps": 0.0,
                    "total_cost_bps": 10.0,
                    "sharpe": 0.64,
                    "cagr": 0.12,
                },
            ]
        ),
        sensitivity_plot_paths=[plot_paths[0]],
        regime_scorecard_df=pd.DataFrame(
            [
                {
                    "regime": "bull_calm",
                    "bars_count": 40,
                    "mean_return": 0.0012,
                    "total_return": 0.08,
                    "max_drawdown": -0.03,
                    "win_rate": 0.60,
                    "trade_count": 7,
                },
                {
                    "regime": "bear_volatile",
                    "bars_count": 30,
                    "mean_return": -0.0008,
                    "total_return": -0.04,
                    "max_drawdown": -0.09,
                    "win_rate": 0.42,
                    "trade_count": 5,
                },
            ]
        ),
        execution_quality_df=pd.DataFrame(
            [
                {
                    "orders_total": 12,
                    "filled_orders": 10,
                    "canceled_orders": 2,
                    "cancel_rate": 0.166667,
                    "partially_filled_orders": 3,
                    "partial_fill_ratio": 0.25,
                    "avg_slippage_bps": 4.5,
                    "median_slippage_bps": 4.2,
                    "avg_latency_seconds": 86400.0,
                    "median_latency_seconds": 86400.0,
                    "avg_latency_bars": 1.0,
                    "median_latency_bars": 1.0,
                }
            ]
        ),
        tuning_best_params={
            "short_window": 10,
            "long_window": 40,
            "objective_name": "worst_window_sharpe_minus_drawdown_penalty",
            "best_score": 0.41,
            "tuning_method": "grid",
        },
        tuning_trials_df=pd.DataFrame(
            [
                {
                    "trial_index": 0,
                    "short_window": 10,
                    "long_window": 40,
                    "objective_score": 0.41,
                    "worst_window_sharpe": 0.50,
                    "worst_window_drawdown": -0.045,
                    "status": "ok",
                    "error": "",
                },
                {
                    "trial_index": 1,
                    "short_window": 20,
                    "long_window": 50,
                    "objective_score": 0.31,
                    "worst_window_sharpe": 0.39,
                    "worst_window_drawdown": -0.040,
                    "status": "ok",
                    "error": "",
                },
            ]
        ),
        tuning_plot_paths=[plot_paths[1]],
        risk_flags=[
            {"check": "Walk-Forward Stability", "level": "red", "message": "Unstable windows."},
            {"check": "Hit-Rate vs Random", "level": "yellow", "message": "Weak edge."},
            {"check": "Autocorrelation", "level": "green", "message": "Looks fine."},
        ],
        validation_summary=[
            {
                "strategy": "MovingAverageCrossStrategy",
                "baseline": "vectorized_moving_average_cross",
                "status": "pass",
                "compared_points": 90,
                "tolerance_abs": 1e-6,
                "max_abs_error": 0.0,
                "mean_abs_error": 0.0,
                "rmse": 0.0,
                "message": "max_abs_error=0.0000000000, tolerance=0.0000010000, points=90",
            }
        ],
        corporate_actions_summary=[
            {
                "date": "2024-03-15",
                "type": "split",
                "details": "split ratio=2",
            },
            {
                "date": "2024-06-20",
                "type": "dividend",
                "details": "cash dividend=0.8",
            },
        ],
        universe_summary_df=pd.DataFrame(
            [
                {
                    "date": "2024-01-01",
                    "universe_size": 2,
                    "added": "AAPL,MSFT",
                    "removed": "",
                },
                {
                    "date": "2024-01-10",
                    "universe_size": 2,
                    "added": "NVDA",
                    "removed": "MSFT",
                },
            ]
        ),
        reproduce_command="python -m stratcheck run --config config.toml",
        full_config={
            "symbol": "QQQ",
            "window_size": "6M",
            "strategy": "stratcheck.core.strategy:MovingAverageCrossStrategy",
        },
        config=config,
    )

    assert html_path == report_dir / "demo.html"
    assert html_path.exists()
    content = html_path.read_text(encoding="utf-8")
    assert "Max Drawdown" in content
    assert "Window Metrics" in content
    assert "Summary" in content
    assert "CAGR" in content
    assert "Sharpe" in content
    assert "Turnover" in content
    assert "Corporate Actions" in content
    assert "split ratio=2" in content
    assert "Universe Dynamics" in content
    assert "Universe Size" in content
    assert "NVDA" in content
    assert "Bootstrap Sharpe CI" in content
    assert "Parameter Sweep" in content
    assert "Cost/Slippage Sensitivity" in content
    assert "Metrics by Cost Assumption" in content
    assert "comm=5.0bps; slip=5.0bps; spread=0.0bps" in content
    assert "Regime Scorecard" in content
    assert "bull_calm" in content
    assert "Execution Quality" in content
    assert "Avg Slippage (bps)" in content
    assert "Partial Fill Ratio" in content
    assert "Parameter Tuning" in content
    assert "Best Parameters" in content
    assert "Objective Name" in content
    assert "Risk Flags" in content
    assert "Backtest Validation" in content
    assert "Max Abs Error" in content
    assert "vectorized_moving_average_cross" in content
    assert "flag-red" in content
    assert "Sharpe CI Low" in content
    assert "window-row-worst" in content
    assert "Worst window highlighted" in content
    assert "Reproducibility" in content
    assert "Reproduce Command" in content
    assert "Show Full Config" in content
    assert "python -m stratcheck run --config config.toml" in content
    assert "&quot;symbol&quot;: &quot;QQQ&quot;" in content
    assert "assets/demo_equity.png" in content
