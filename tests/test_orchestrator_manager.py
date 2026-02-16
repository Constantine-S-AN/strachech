from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest
from stratcheck.orchestrator import MultiRunnerOrchestrator, RunnerResourceLimits, RunnerTask


def test_orchestrator_runs_multiple_tasks_with_isolated_namespaces(tmp_path: Path) -> None:
    bars = _build_bars()
    orchestrator = MultiRunnerOrchestrator(
        output_dir=tmp_path / "orchestrator",
        max_concurrent_runners=2,
        process_start_method="spawn",
        poll_interval_seconds=0.01,
    )

    tasks = [
        RunnerTask(
            task_id="runner_a",
            namespace="book_alpha",
            strategy_reference="stratcheck.strategies.baselines:BuyAndHoldStrategy",
            strategy_params={"target_position_qty": 1.0},
            symbol="AAPL",
            bars=bars,
            connector_params={
                "initial_cash": 10_000.0,
                "max_fill_ratio_per_step": 1.0,
                "max_volume_share": 1.0,
            },
        ),
        RunnerTask(
            task_id="runner_b",
            namespace="book_beta",
            strategy_reference="stratcheck.strategies.baselines:BuyAndHoldStrategy",
            strategy_params={"target_position_qty": 2.0},
            symbol="MSFT",
            bars=bars,
            connector_params={
                "initial_cash": 10_000.0,
                "max_fill_ratio_per_step": 1.0,
                "max_volume_share": 1.0,
            },
        ),
    ]

    results = orchestrator.run_tasks(tasks)
    assert len(results) == 2
    first_result, second_result = results

    assert first_result.task_id == "runner_a"
    assert second_result.task_id == "runner_b"
    assert first_result.status == "completed"
    assert second_result.status == "completed"
    assert first_result.runner_status == "completed"
    assert second_result.runner_status == "completed"

    assert first_result.sqlite_path != second_result.sqlite_path
    assert first_result.log_path != second_result.log_path
    assert first_result.sqlite_path.parent.name == "book_alpha"
    assert second_result.sqlite_path.parent.name == "book_beta"

    for result in results:
        assert result.sqlite_path.exists()
        assert result.log_path.exists()
        assert result.metrics_prom_path.exists()
        assert result.metrics_csv_path.exists()
        _assert_sqlite_has_run(result.sqlite_path)


def test_orchestrator_applies_default_runtime_limit(tmp_path: Path) -> None:
    bars = _build_bars(periods=30)
    orchestrator = MultiRunnerOrchestrator(
        output_dir=tmp_path / "orchestrator",
        max_concurrent_runners=1,
        process_start_method="spawn",
        poll_interval_seconds=0.01,
        default_resource_limits=RunnerResourceLimits(max_runtime_seconds=0.01),
    )
    task = RunnerTask(
        task_id="runner_timeout",
        strategy_reference="stratcheck.strategies.baselines:BuyAndHoldStrategy",
        strategy_params={"target_position_qty": 1.0},
        symbol="AAPL",
        bars=bars,
        connector_params={
            "initial_cash": 10_000.0,
            "max_fill_ratio_per_step": 1.0,
            "max_volume_share": 1.0,
        },
    )

    results = orchestrator.run_tasks([task])
    assert len(results) == 1
    result = results[0]
    assert result.task_id == "runner_timeout"
    assert result.status == "timeout"
    assert result.error_type == "TimeoutError"
    assert result.error_message is not None
    assert "max_runtime_seconds" in result.error_message


def test_orchestrator_rejects_namespace_collision(tmp_path: Path) -> None:
    bars = _build_bars()
    orchestrator = MultiRunnerOrchestrator(output_dir=tmp_path / "orchestrator")
    tasks = [
        RunnerTask(
            task_id="runner_x",
            namespace="risk-book/A",
            strategy_reference="stratcheck.strategies.baselines:BuyAndHoldStrategy",
            strategy_params={"target_position_qty": 1.0},
            symbol="AAPL",
            bars=bars,
        ),
        RunnerTask(
            task_id="runner_y",
            namespace="risk-book_A",
            strategy_reference="stratcheck.strategies.baselines:BuyAndHoldStrategy",
            strategy_params={"target_position_qty": 1.0},
            symbol="MSFT",
            bars=bars,
        ),
    ]

    with pytest.raises(ValueError, match="namespace values must be unique"):
        orchestrator.run_tasks(tasks)


def _build_bars(periods: int = 8) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=periods, freq="h", tz="UTC")
    prices = [100.0 + index for index in range(periods)]
    return pd.DataFrame(
        {
            "open": prices,
            "high": [value + 1.0 for value in prices],
            "low": [value - 1.0 for value in prices],
            "close": prices,
            "volume": [1_000.0 for _ in prices],
        },
        index=timestamps,
    )


def _assert_sqlite_has_run(sqlite_path: Path) -> None:
    connection = sqlite3.connect(sqlite_path)
    try:
        row = connection.execute("SELECT COUNT(*) FROM runs").fetchone()
    finally:
        connection.close()
    assert row is not None
    assert int(row[0]) >= 1
