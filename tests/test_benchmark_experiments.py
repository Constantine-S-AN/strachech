from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_benchmark_experiments_script_runs_small_workload(tmp_path: Path) -> None:
    workspace = tmp_path / "bench"
    command = [
        sys.executable,
        "scripts/benchmark_experiments.py",
        "--workspace",
        str(workspace),
        "--configs",
        "4",
        "--bars",
        "80",
        "--workers",
        "2",
    ]
    result = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parent.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["config_count"] == 4
    assert payload["bars_count"] == 80
    assert payload["workers"] == 2

    summary_path = workspace / "benchmark_summary.json"
    assert summary_path.exists()
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_payload["config_count"] == 4
    assert summary_payload["workers"] == 2
