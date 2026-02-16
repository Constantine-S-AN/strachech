from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from stratcheck.core.experiments import ExperimentRunner


def test_experiment_runner_parallel_matches_sequential_results(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_sample_csv(data_dir=data_dir, symbol="BTCUSDT")

    configs_dir = tmp_path / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    _write_config(configs_dir=configs_dir, name="valid_a", data_dir=data_dir, short_window=2)
    _write_config(configs_dir=configs_dir, name="valid_b", data_dir=data_dir, short_window=3)
    _write_invalid_config(configs_dir=configs_dir, name="broken", data_dir=data_dir)

    sequential_reports = tmp_path / "reports_seq"
    parallel_reports = tmp_path / "reports_par"

    sequential_runner = ExperimentRunner(configs_dir=configs_dir, output_dir=sequential_reports)
    sequential_frame, _ = sequential_runner.run_all()

    parallel_runner = ExperimentRunner(
        configs_dir=configs_dir,
        output_dir=parallel_reports,
        parallel=True,
        max_workers=2,
    )
    parallel_frame, _ = parallel_runner.run_all()

    assert set(sequential_frame["experiment"]) == set(parallel_frame["experiment"])
    assert set(sequential_frame["status"]) == set(parallel_frame["status"])
    assert {"success", "failed"} == set(parallel_frame["status"])
    assert (parallel_reports / "results.jsonl").exists()


def _write_config(
    configs_dir: Path,
    name: str,
    data_dir: Path,
    short_window: int,
) -> None:
    config_text = f"""
symbol = "BTCUSDT"
data_path = "{data_dir.as_posix()}"
strategy = "stratcheck.core.strategy:MovingAverageCrossStrategy"
initial_cash = 100000
timeframe = "1d"
bars_freq = "1d"
report_name = "{name}"

[cost_model]
commission_bps = 5
slippage_bps = 3

[windows]
window_size = "7D"
step_size = "7D"

[strategy_params]
short_window = {short_window}
long_window = 6
target_position_qty = 1.0
"""
    (configs_dir / f"{name}.toml").write_text(config_text.strip() + "\n", encoding="utf-8")


def _write_invalid_config(
    configs_dir: Path,
    name: str,
    data_dir: Path,
) -> None:
    config_text = f"""
symbol = "BTCUSDT"
data_path = "{data_dir.as_posix()}"
strategy = "stratcheck.core.strategy:MissingStrategy"
initial_cash = 100000
timeframe = "1d"
bars_freq = "1d"
report_name = "{name}"

[cost_model]
commission_bps = 5
slippage_bps = 3

[windows]
window_size = "7D"
step_size = "7D"
"""
    (configs_dir / f"{name}.toml").write_text(config_text.strip() + "\n", encoding="utf-8")


def _write_sample_csv(data_dir: Path, symbol: str) -> None:
    timestamps = pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC")
    close_prices = np.linspace(100.0, 130.0, len(timestamps))
    bars = pd.DataFrame(
        {
            "timestamp": timestamps.astype(str),
            "open": close_prices - 0.2,
            "high": close_prices + 1.0,
            "low": close_prices - 1.0,
            "close": close_prices,
            "volume": np.full(len(timestamps), 1_000),
        }
    )
    bars.to_csv(data_dir / f"{symbol}.csv", index=False)
