from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from stratcheck.audit import RunAuditStore
from stratcheck.cli import main


def test_dashboard_command_builds_html_without_errors(tmp_path: Path, capsys) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "alpha.html").write_text("<html><body>alpha</body></html>", encoding="utf-8")

    results_path = reports_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as results_file:
        results_file.write(
            json.dumps(
                {
                    "experiment": "alpha",
                    "status": "success",
                    "sharpe": 1.1,
                    "cagr": 0.12,
                    "max_drawdown": -0.2,
                    "total_return": 0.3,
                    "report_path": "alpha.html",
                    "error": "",
                }
            )
            + "\n"
        )

    sqlite_path = reports_dir / "paper.sqlite"
    run_id = _create_sample_run(sqlite_path)

    output_path = reports_dir / "dashboard.html"
    exit_code = main(
        [
            "dashboard",
            "--results-jsonl",
            str(results_path),
            "--db",
            str(sqlite_path),
            "--output",
            str(output_path),
            "--reports-dir",
            str(reports_dir),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Dashboard generated:" in captured.out
    assert output_path.exists()

    html_content = output_path.read_text(encoding="utf-8")
    assert "Leaderboard" in html_content
    assert "Experiment Ranking" in html_content
    assert "Live Status" in html_content
    assert "Current Positions" in html_content
    assert "Paper Run Status" in html_content
    assert run_id in html_content
    assert "alpha.html" in html_content


def _create_sample_run(sqlite_path: Path) -> str:
    audit_store = RunAuditStore(sqlite_path=sqlite_path)
    connection = audit_store.connect()
    try:
        audit_store.ensure_schema(connection)
        run_id = audit_store.create_run(
            connection=connection,
            mode="paper",
            symbol="AAPL",
            strategy="tests:DummyStrategy",
            initial_cash=10_000.0,
            config={"source": "dashboard_cli"},
            note="ok",
        )
        event_time = pd.Timestamp("2024-01-02T00:00:00Z")
        audit_store.record_snapshot(
            connection=connection,
            run_id=run_id,
            timestamp=event_time,
            bar_index=0,
            cash=10_000.0,
            position_qty=0.0,
            equity=10_050.0,
            close_price=100.5,
            note="bar_close",
        )
        audit_store.finalize_run(
            connection=connection,
            run_id=run_id,
            status="completed",
            note="ok",
        )
        return run_id
    finally:
        connection.close()
