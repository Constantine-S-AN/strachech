from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_post_summary_script_outputs_overall_and_worst_window_metrics(tmp_path: Path) -> None:
    summary_path = tmp_path / "healthcheck_summary.json"
    summary_payload = {
        "windows": [
            {
                "window_index": 0,
                "window_start": "2024-01-01T00:00:00+00:00",
                "window_end": "2024-06-30T00:00:00+00:00",
                "cagr": 0.10,
                "sharpe": 1.2,
                "max_drawdown": -0.08,
            },
            {
                "window_index": 1,
                "window_start": "2024-07-01T00:00:00+00:00",
                "window_end": "2024-12-31T00:00:00+00:00",
                "cagr": -0.05,
                "sharpe": -0.3,
                "max_drawdown": -0.25,
            },
        ]
    }
    summary_path.write_text(json.dumps(summary_payload), encoding="utf-8")

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "post_summary.py"
    result = subprocess.run(
        [sys.executable, str(script_path), str(summary_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    output = result.stdout
    assert "## Healthcheck Summary" in output
    assert "### Overall" in output
    assert "- Windows: 2" in output
    assert "Worst Max Drawdown" in output
    assert "| Max Drawdown | 1 |" in output
    assert "| Sharpe | 1 |" in output
