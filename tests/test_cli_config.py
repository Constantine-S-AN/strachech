from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from stratcheck.cli import run_healthcheck_with_config, run_with_config


def test_run_with_config_generates_report(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path=tmp_path, report_name="run_demo")
    report_path = run_with_config(config_path=config_path)

    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "Overall Metrics" in content
    assert "Config" in content
    assert "Reproducibility" in content
    assert "python -m stratcheck run --config" in content
    assert "Show Full Config" in content
    assert "Backtest Validation" in content
    assert "Cost/Slippage Sensitivity" in content
    assert "Regime Scorecard" in content


def test_run_healthcheck_with_config_generates_report_and_summary(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path=tmp_path, report_name="healthcheck_demo")
    report_path, summary_json_path = run_healthcheck_with_config(config_path=config_path)

    assert report_path.exists()
    assert summary_json_path.exists()

    payload = json.loads(summary_json_path.read_text(encoding="utf-8"))
    assert "windows" in payload
    assert len(payload["windows"]) > 0

    content = report_path.read_text(encoding="utf-8")
    assert "Reproducibility" in content
    assert "python -m stratcheck healthcheck --config" in content
    assert "Backtest Validation" in content
    assert "Cost/Slippage Sensitivity" in content
    assert "Regime Scorecard" in content


def test_run_with_config_tuning_outputs_best_params_section(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path=tmp_path, report_name="run_tuning_demo", enable_tuning=True
    )
    report_path = run_with_config(config_path=config_path)

    content = report_path.read_text(encoding="utf-8")
    assert "Parameter Tuning" in content
    assert "Best Parameters" in content
    assert "worst_window_sharpe_minus_drawdown_penalty" in content


def _write_config(tmp_path: Path, report_name: str, enable_tuning: bool = False) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_sample_csv(data_dir=data_dir, symbol="BTCUSDT")

    report_dir = tmp_path / "reports"
    config_path = tmp_path / "config.toml"
    config_text = f"""
symbol = "BTCUSDT"
data_path = "{data_dir.as_posix()}"
strategy = "stratcheck.core.strategy:MovingAverageCrossStrategy"
initial_cash = 100000
timeframe = "1d"
report_name = "{report_name}"
report_dir = "{report_dir.as_posix()}"

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
    if enable_tuning:
        config_text += """

[tuning]
enabled = true
method = "grid"
n_iter = 4
drawdown_penalty = 1.5
window_size = "10D"
step_size = "5D"

[tuning.search_space]
short_window = [2, 3]
long_window = [4, 6]
"""
    config_path.write_text(config_text.strip() + "\n", encoding="utf-8")
    return config_path


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
