from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from stratcheck.cli import main, run_with_config


def test_run_with_config_persists_snapshot_files(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path=tmp_path, report_name="bundle_snapshot_demo")
    report_path = run_with_config(config_path=config_path)
    report_dir = report_path.parent

    runs_dir = report_dir / "runs"
    run_dirs = sorted(path for path in runs_dir.iterdir() if path.is_dir())
    assert len(run_dirs) == 1

    run_dir = run_dirs[0]
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "config.toml").exists()
    assert (run_dir / "data_hash.json").exists()
    assert (run_dir / "bars.csv").exists()
    assert (run_dir / "signals.csv").exists()
    assert (run_dir / "trades.csv").exists()
    assert (run_dir / "report.html").exists()

    hash_payload = json.loads((run_dir / "data_hash.json").read_text(encoding="utf-8"))
    expected_hash = str(hash_payload["bars_csv_sha256"])
    actual_hash = _sha256_file(run_dir / "bars.csv")
    assert actual_hash == expected_hash


def test_cli_bundle_and_reproduce_roundtrip(tmp_path: Path, capsys) -> None:
    config_path = _write_config(tmp_path=tmp_path, report_name="bundle_roundtrip_demo")
    report_path = run_with_config(config_path=config_path)
    report_dir = report_path.parent

    runs_dir = report_dir / "runs"
    run_dirs = sorted(path for path in runs_dir.iterdir() if path.is_dir())
    assert run_dirs
    run_id = run_dirs[0].name

    bundle_path = report_dir / "bundles" / f"{run_id}.zip"
    exit_code = main(
        [
            "bundle",
            "--run-id",
            run_id,
            "--runs-dir",
            str(runs_dir),
            "--output",
            str(bundle_path),
        ]
    )
    bundle_output = capsys.readouterr().out

    assert exit_code == 0
    assert "Bundle generated:" in bundle_output
    assert bundle_path.exists()

    with ZipFile(bundle_path) as bundle_zip:
        archive_files = set(bundle_zip.namelist())
    assert f"{run_id}/manifest.json" in archive_files
    assert f"{run_id}/data_hash.json" in archive_files
    assert f"{run_id}/signals.csv" in archive_files
    assert f"{run_id}/trades.csv" in archive_files
    assert f"{run_id}/report.html" in archive_files

    reproduce_dir = tmp_path / "reproduced"
    exit_code = main(["reproduce", str(bundle_path), "--output-dir", str(reproduce_dir)])
    reproduce_output = capsys.readouterr().out

    assert exit_code == 0
    assert f"Reproduced run: {run_id}" in reproduce_output

    reproduced_report = reproduce_dir / run_id / "report.html"
    assert reproduced_report.exists()
    html_content = reproduced_report.read_text(encoding="utf-8")
    assert "Overall Metrics" in html_content


def _write_config(tmp_path: Path, report_name: str) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_sample_csv(data_dir=data_dir, symbol="BTCUSDT")

    report_dir = tmp_path / "reports"
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"""
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
""".strip()
        + "\n",
        encoding="utf-8",
    )
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


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
