from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from stratcheck.audit import RunAuditStore
from stratcheck.dashboard import (
    load_experiment_rankings,
    load_live_status,
    load_paper_run_statuses,
)


def test_load_experiment_rankings_sorts_by_sharpe(tmp_path: Path) -> None:
    rows = [
        {
            "experiment": "alpha",
            "status": "success",
            "sharpe": 0.5,
            "cagr": 0.08,
            "max_drawdown": -0.12,
            "worst_window_sharpe": -0.30,
            "worst_window_drawdown": -0.18,
            "total_return": 0.15,
            "report_path": "alpha.html",
            "error": "",
        },
        {
            "experiment": "beta",
            "status": "success",
            "sharpe": 1.2,
            "cagr": 0.14,
            "max_drawdown": -0.10,
            "worst_window_sharpe": 0.10,
            "worst_window_drawdown": -0.11,
            "total_return": 0.22,
            "report_path": "beta.html",
            "error": "",
        },
        {
            "experiment": "broken",
            "status": "failed",
            "sharpe": None,
            "cagr": None,
            "max_drawdown": None,
            "worst_window_sharpe": None,
            "worst_window_drawdown": None,
            "total_return": None,
            "report_path": "",
            "error": "RuntimeError",
        },
    ]

    results_path = tmp_path / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as results_file:
        for row in rows:
            results_file.write(json.dumps(row) + "\n")

    frame = load_experiment_rankings(results_jsonl_path=results_path)

    assert frame["experiment"].tolist() == ["beta", "alpha", "broken"]
    assert int(frame.iloc[0]["rank"]) == 1
    assert int(frame.iloc[1]["rank"]) == 2
    assert frame.iloc[2]["status"] == "failed"
    assert pd.isna(frame.iloc[2]["rank"])
    assert float(frame.iloc[0]["worst_window_sharpe"]) == 0.10


def test_load_paper_run_statuses_returns_runtime_metrics(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "paper.sqlite"
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
            config={"source": "unit_test"},
            note="dashboard_query",
        )
        time_1 = pd.Timestamp("2024-01-01T00:00:00Z")
        time_2 = pd.Timestamp("2024-01-02T00:00:00Z")
        time_3 = pd.Timestamp("2024-01-03T00:00:00Z")

        audit_store.record_snapshot(
            connection=connection,
            run_id=run_id,
            timestamp=time_1,
            bar_index=0,
            cash=10_000.0,
            position_qty=0.0,
            equity=10_000.0,
            close_price=100.0,
            note="bar_close",
        )
        audit_store.record_snapshot(
            connection=connection,
            run_id=run_id,
            timestamp=time_2,
            bar_index=1,
            cash=8_900.0,
            position_qty=10.0,
            equity=9_800.0,
            close_price=90.0,
            note="bar_close",
        )
        audit_store.record_snapshot(
            connection=connection,
            run_id=run_id,
            timestamp=time_3,
            bar_index=2,
            cash=9_100.0,
            position_qty=10.0,
            equity=9_900.0,
            close_price=80.0,
            note="bar_close",
        )

        for order_index in range(3):
            audit_store.record_order(
                connection=connection,
                run_id=run_id,
                timestamp=time_2,
                order_id=f"ord-{order_index}",
                symbol="AAPL",
                side="buy",
                qty=1.0,
                order_type="market",
                status="filled",
                filled_qty=1.0,
                remaining_qty=0.0,
                avg_fill_price=100.0,
                note="order_matched",
            )
        audit_store.record_risk_event(
            connection=connection,
            run_id=run_id,
            timestamp=time_2,
            rule_name="max_position",
            action="block",
            reason="max_abs_position_qty_breached",
            details={"projected_position_qty": 3.0},
        )
        audit_store.record_risk_event(
            connection=connection,
            run_id=run_id,
            timestamp=time_3,
            rule_name="max_drawdown",
            action="halt",
            reason="max_drawdown_exceeded",
            details={"drawdown": 0.21},
        )
        audit_store.finalize_run(
            connection=connection,
            run_id=run_id,
            status="completed",
            note="ok",
        )
    finally:
        connection.close()

    frame = load_paper_run_statuses(sqlite_path=sqlite_path)
    assert len(frame) == 1
    record = frame.iloc[0]
    assert record["run_id"] == run_id
    assert record["status"] == "completed"
    assert record["latest_equity"] == 9_900.0
    assert record["pnl"] == -100.0
    assert record["drawdown"] == 0.02
    assert record["order_rate"] > 0.0
    assert record["error_rate"] == 0.0
    assert int(record["risk_rule_hits"]) == 2
    assert int(record["risk_halt_hits"]) == 1
    assert int(record["risk_block_hits"]) == 1


def test_load_live_status_returns_positions_trades_risk_and_errors(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "live.sqlite"
    log_path = tmp_path / "runner.log"
    log_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2024-01-02T09:31:00Z",
                        "level": "info",
                        "event": "heartbeat",
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2024-01-02T09:32:00Z",
                        "level": "error",
                        "event": "runner_error",
                        "message": "connector disconnected",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    audit_store = RunAuditStore(sqlite_path=sqlite_path)
    connection = audit_store.connect()
    try:
        audit_store.ensure_schema(connection)
        run_ok = audit_store.create_run(
            connection=connection,
            mode="paper",
            symbol="AAPL",
            strategy="tests:DemoStrategy",
            initial_cash=10_000.0,
            config={"log_path": str(log_path)},
            note="live_status_test",
        )
        run_killed = audit_store.create_run(
            connection=connection,
            mode="paper",
            symbol="MSFT",
            strategy="tests:DemoStrategy",
            initial_cash=12_000.0,
            config={},
            note="live_status_test",
        )

        t_2 = pd.Timestamp("2024-01-02T09:31:00Z")
        t_3 = pd.Timestamp("2024-01-02T09:32:00Z")

        audit_store.record_snapshot(
            connection=connection,
            run_id=run_ok,
            timestamp=t_2,
            bar_index=1,
            cash=9_700.0,
            position_qty=3.0,
            equity=10_030.0,
            close_price=110.0,
            note="bar_close",
        )
        audit_store.record_fill(
            connection=connection,
            run_id=run_ok,
            timestamp=t_2,
            order_id="ord-1",
            symbol="AAPL",
            side="buy",
            qty=2.0,
            price=110.0,
            reason="order_matched",
        )
        audit_store.record_risk_event(
            connection=connection,
            run_id=run_ok,
            timestamp=t_2,
            rule_name="max_position",
            action="block",
            reason="max_abs_position_qty_breached",
            details={"projected_position_qty": 3.0},
        )

        audit_store.record_risk_event(
            connection=connection,
            run_id=run_killed,
            timestamp=t_3,
            rule_name="max_drawdown",
            action="halt",
            reason="max_drawdown_exceeded",
            details={"drawdown": 0.30},
        )
        audit_store.finalize_run(
            connection=connection,
            run_id=run_ok,
            status="completed",
            note="ok",
        )
        audit_store.finalize_run(
            connection=connection,
            run_id=run_killed,
            status="killed",
            note="consecutive_errors_exceeded",
        )
    finally:
        connection.close()

    live_status = load_live_status(
        sqlite_path=sqlite_path,
        now_timestamp=pd.Timestamp("2024-01-02T12:00:00Z"),
    )

    assert not live_status.positions_df.empty
    position_rows = live_status.positions_df.set_index("run_id")
    assert float(position_rows.loc[run_ok, "position_qty"]) == 3.0
    assert float(position_rows.loc[run_ok, "position_notional"]) == 330.0

    trades_rows = live_status.today_trades_df.set_index("run_id")
    assert int(trades_rows.loc[run_ok, "trades_today"]) == 1
    assert float(trades_rows.loc[run_ok, "filled_qty_today"]) == 2.0

    risk_rows = live_status.risk_status_df.set_index("run_id")
    assert risk_rows.loc[run_ok, "risk_state"] == "warning"
    assert risk_rows.loc[run_killed, "risk_state"] == "critical"

    assert not live_status.recent_errors_df.empty
    assert "runner_log" in live_status.recent_errors_df["source"].tolist()
    assert "risk_events" in live_status.recent_errors_df["source"].tolist()
