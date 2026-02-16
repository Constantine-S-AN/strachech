from __future__ import annotations

from pathlib import Path

from stratcheck.cli import run_demo


def test_run_demo_generates_html_report(tmp_path: Path) -> None:
    report_path = tmp_path / "demo.html"
    output_path = run_demo(output_path=report_path, periods=200, seed=9)

    assert output_path == report_path
    assert output_path.exists()

    content = output_path.read_text(encoding="utf-8")
    assert "<html" in content.lower()
    assert "Stratcheck Report" in content
    assert "Summary" in content
    assert "Bootstrap Sharpe CI" in content
    assert "Parameter Sweep" in content
    assert "Cost/Slippage Sensitivity" in content
    assert "Regime Scorecard" in content
    assert "Risk Flags" in content
    assert "Reproducibility" in content
    assert "Show Full Config" in content
    assert "python -m stratcheck demo" in content
