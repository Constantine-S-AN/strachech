from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_make_demo_assets_generates_compact_csv(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_csv = tmp_path / "QQQ.csv"

    subprocess.run(
        [
            sys.executable,
            "scripts/make_demo_assets.py",
            "--output",
            str(output_csv),
            "--periods",
            "45",
            "--seed",
            "3",
        ],
        check=True,
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    assert output_csv.exists()
    assert output_csv.stat().st_size < 100_000

    bars = pd.read_csv(output_csv)
    assert len(bars) == 45
    assert list(bars.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
