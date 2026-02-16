from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from stratcheck.core.experiments import ExperimentRunner


def test_experiment_runner_builds_index_and_records_failures(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_sample_csv(data_dir=data_dir, symbol="BTCUSDT")

    configs_dir = tmp_path / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    _write_success_config(configs_dir=configs_dir, data_dir=data_dir)
    _write_failure_config(configs_dir=configs_dir, data_dir=data_dir)

    reports_dir = tmp_path / "reports"
    runner = ExperimentRunner(configs_dir=configs_dir, output_dir=reports_dir)

    summary_frame, index_path = runner.run_all()

    assert index_path == reports_dir / "index.html"
    assert index_path.exists()
    assert len(summary_frame) == 2
    assert set(summary_frame["status"]) == {"success", "failed"}

    success_row = summary_frame[summary_frame["status"] == "success"].iloc[0]
    report_path = reports_dir / str(success_row["report_path"])
    assert report_path.exists()

    failed_row = summary_frame[summary_frame["status"] == "failed"].iloc[0]
    assert "Strategy class not found" in str(failed_row["error"])

    index_html = index_path.read_text(encoding="utf-8")
    assert "Experiment Summary" in index_html
    assert "Sort by:" in index_html
    assert "cost_assumption" in index_html
    assert str(success_row["report_path"]) in index_html

    results_path = reports_dir / "results.jsonl"
    assert results_path.exists()
    lines = results_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    payloads = [json.loads(line) for line in lines]
    assert any(payload["status"] == "success" for payload in payloads)
    assert any(payload["status"] == "failed" for payload in payloads)


def _write_success_config(configs_dir: Path, data_dir: Path) -> None:
    config_text = f"""
symbol = "BTCUSDT"
data_path = "{data_dir.as_posix()}"
strategy = "stratcheck.core.strategy:MovingAverageCrossStrategy"
initial_cash = 100000
timeframe = "1d"
bars_freq = "1d"
report_name = "valid_strategy"

[cost_model]
commission_bps = 5
slippage_bps = 3

[windows]
window_size = "7D"
step_size = "7D"

[strategy_params]
short_window = 2
long_window = 4
target_position_qty = 1.0
"""
    (configs_dir / "valid.toml").write_text(config_text.strip() + "\n", encoding="utf-8")


def _write_failure_config(configs_dir: Path, data_dir: Path) -> None:
    config_text = f"""
symbol = "BTCUSDT"
data_path = "{data_dir.as_posix()}"
strategy = "stratcheck.core.strategy:MissingStrategy"
initial_cash = 100000
timeframe = "1d"
bars_freq = "1d"
report_name = "broken_strategy"

[cost_model]
commission_bps = 5
slippage_bps = 3

[windows]
window_size = "7D"
step_size = "7D"
"""
    (configs_dir / "broken.toml").write_text(config_text.strip() + "\n", encoding="utf-8")


def _write_sample_csv(data_dir: Path, symbol: str) -> None:
    timestamps = pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC")
    close_prices = np.linspace(100.0, 130.0, 30)
    bars = pd.DataFrame(
        {
            "timestamp": timestamps.astype(str),
            "open": close_prices - 0.2,
            "high": close_prices + 1.0,
            "low": close_prices - 1.0,
            "close": close_prices,
            "volume": np.full(30, 1_000),
        }
    )
    bars.to_csv(data_dir / f"{symbol}.csv", index=False)
