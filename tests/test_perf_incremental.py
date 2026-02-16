from __future__ import annotations

import pandas as pd
import pytest
from stratcheck.core.metrics import compute_metrics
from stratcheck.core.strategy import Fill
from stratcheck.perf import IncrementalPlotState, prepare_incremental_plot_series
from stratcheck.report.plots import generate_performance_plots


def test_compute_metrics_incremental_matches_full_metrics() -> None:
    timestamps = pd.date_range("2024-01-01", periods=12, freq="D", tz="UTC")
    equity_curve = pd.Series(
        [
            100_000.0,
            100_400.0,
            100_250.0,
            100_900.0,
            101_300.0,
            100_950.0,
            101_800.0,
            101_650.0,
            102_200.0,
            102_050.0,
            102_900.0,
            103_100.0,
        ],
        index=timestamps,
        dtype=float,
    )
    trades = [
        Fill(side="buy", qty=1.0, price=100.0, timestamp=timestamps[1], fee=0.1),
        Fill(side="sell", qty=1.0, price=103.0, timestamp=timestamps[4], fee=0.1),
        Fill(side="buy", qty=1.0, price=102.0, timestamp=timestamps[7], fee=0.1),
        Fill(side="sell", qty=1.0, price=105.0, timestamp=timestamps[10], fee=0.1),
    ]

    baseline_metrics = compute_metrics(equity_curve=equity_curve, trades=trades, bars_freq="1d")
    _, incremental_state = compute_metrics(
        equity_curve=equity_curve.iloc[:6],
        trades=trades[:2],
        bars_freq="1d",
        return_incremental_state=True,
    )
    incremental_metrics, incremental_state = compute_metrics(
        equity_curve=equity_curve,
        trades=trades,
        bars_freq="1d",
        incremental_state=incremental_state,
        return_incremental_state=True,
    )

    for metric_key, metric_value in baseline_metrics.items():
        assert incremental_metrics[metric_key] == pytest.approx(metric_value, rel=1e-12, abs=1e-12)
    assert incremental_state.processed_equity_points == len(equity_curve)
    assert incremental_state.processed_trades == len(trades)


def test_incremental_plot_series_matches_full_series() -> None:
    timestamps = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    equity_curve = pd.Series(
        [100.0, 101.0, 100.5, 102.0, 103.0, 102.5, 104.0, 103.7, 105.0, 106.0],
        index=timestamps,
        dtype=float,
    )

    state = IncrementalPlotState()
    drawdown_first, returns_first, state = prepare_incremental_plot_series(
        equity_curve=equity_curve.iloc[:5], state=state
    )
    drawdown_full, returns_full, state = prepare_incremental_plot_series(
        equity_curve=equity_curve, state=state
    )

    expected_drawdown = equity_curve / equity_curve.cummax() - 1.0
    expected_returns = equity_curve.pct_change().dropna()

    assert len(drawdown_first) == 5
    assert len(returns_first) == 4
    pd.testing.assert_series_equal(
        drawdown_full.astype(float),
        expected_drawdown.astype(float),
        check_freq=False,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        returns_full.astype(float),
        expected_returns.astype(float),
        check_freq=False,
        check_names=False,
    )
    assert state.processed_equity_points == len(equity_curve)


def test_generate_performance_plots_supports_incremental_state(tmp_path) -> None:
    timestamps = pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC")
    equity_curve = pd.Series(
        [100_000.0 + index * 50.0 for index in range(30)],
        index=timestamps,
        dtype=float,
    )

    _, plot_state = generate_performance_plots(
        equity_curve=equity_curve.iloc[:15],
        output_dir=tmp_path / "assets",
        prefix="inc",
        return_incremental_state=True,
    )
    paths, plot_state = generate_performance_plots(
        equity_curve=equity_curve,
        output_dir=tmp_path / "assets",
        prefix="inc",
        incremental_state=plot_state,
        return_incremental_state=True,
    )

    assert len(paths) == 3
    assert plot_state.processed_equity_points == len(equity_curve)
    assert len(plot_state.drawdown_values) == len(equity_curve)
