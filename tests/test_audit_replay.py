from __future__ import annotations

from pathlib import Path

import pandas as pd
from stratcheck.audit import RunAuditStore
from stratcheck.cli import main, replay_run


def test_audit_store_write_and_query_consistency(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "audit.sqlite"
    audit_store = RunAuditStore(sqlite_path=sqlite_path)
    connection = audit_store.connect()
    try:
        audit_store.ensure_schema(connection)
        run_id = audit_store.create_run(
            connection=connection,
            mode="paper",
            symbol="QQQ",
            strategy="tests:DummyStrategy",
            initial_cash=100_000.0,
            config={"seed": 7, "window": 20},
            note="unit_test",
        )

        event_time = pd.Timestamp("2024-01-02T00:00:00Z")
        audit_store.record_signal(
            connection=connection,
            run_id=run_id,
            timestamp=event_time,
            bar_index=1,
            signal_index=0,
            side="buy",
            qty=1.0,
            order_type="market",
            limit_price=None,
            reason="strategy_signal",
        )
        audit_store.record_order(
            connection=connection,
            run_id=run_id,
            timestamp=event_time,
            order_id="ord-1",
            symbol="QQQ",
            side="buy",
            qty=1.0,
            order_type="market",
            status="new",
            filled_qty=0.0,
            remaining_qty=1.0,
            avg_fill_price=0.0,
            note="order_accepted",
        )
        audit_store.record_fill(
            connection=connection,
            run_id=run_id,
            timestamp=event_time,
            order_id="ord-1",
            symbol="QQQ",
            side="buy",
            qty=1.0,
            price=401.25,
            reason="order_matched",
        )
        audit_store.record_snapshot(
            connection=connection,
            run_id=run_id,
            timestamp=event_time,
            bar_index=1,
            cash=99_598.75,
            position_qty=1.0,
            equity=100_000.00,
            close_price=401.25,
            note="bar_close",
        )
        audit_store.finalize_run(
            connection=connection,
            run_id=run_id,
            status="completed",
            note="ok",
        )
    finally:
        connection.close()

    run_record = audit_store.get_run(run_id)
    assert run_record.run_id == run_id
    assert run_record.symbol == "QQQ"
    assert run_record.status == "completed"

    counts = audit_store.get_counts(run_id)
    assert counts == {
        "configs": 1,
        "signals": 1,
        "orders": 1,
        "fills": 1,
        "snapshots": 1,
    }


def test_replay_output_contains_key_fields(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "audit.sqlite"
    run_id = _create_sample_run(sqlite_path=sqlite_path)

    timeline_text = replay_run(run_id=run_id, sqlite_path=sqlite_path)

    assert f"Run ID: {run_id}" in timeline_text
    assert "Config:" in timeline_text
    assert "Timeline:" in timeline_text
    assert "SIGNAL" in timeline_text
    assert "ORDER" in timeline_text
    assert "FILL" in timeline_text
    assert "SNAPSHOT" in timeline_text
    assert "reason=strategy_signal" in timeline_text
    assert "order_id=ord-1" in timeline_text


def test_cli_replay_command_prints_timeline(tmp_path: Path, capsys) -> None:
    sqlite_path = tmp_path / "audit.sqlite"
    run_id = _create_sample_run(sqlite_path=sqlite_path)

    exit_code = main(
        [
            "replay",
            "--run-id",
            run_id,
            "--db",
            str(sqlite_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert f"Run ID: {run_id}" in captured.out
    assert "Timeline:" in captured.out
    assert "FILL" in captured.out


def _create_sample_run(sqlite_path: Path) -> str:
    audit_store = RunAuditStore(sqlite_path=sqlite_path)
    connection = audit_store.connect()
    try:
        audit_store.ensure_schema(connection)
        run_id = audit_store.create_run(
            connection=connection,
            mode="paper",
            symbol="QQQ",
            strategy="tests:DummyStrategy",
            initial_cash=100_000.0,
            config={"seed": 11},
            note="replay_test",
        )
        event_time = pd.Timestamp("2024-01-03T00:00:00Z")
        audit_store.record_signal(
            connection=connection,
            run_id=run_id,
            timestamp=event_time,
            bar_index=2,
            signal_index=0,
            side="buy",
            qty=2.0,
            order_type="market",
            limit_price=None,
            reason="strategy_signal",
        )
        audit_store.record_order(
            connection=connection,
            run_id=run_id,
            timestamp=event_time,
            order_id="ord-1",
            symbol="QQQ",
            side="buy",
            qty=2.0,
            order_type="market",
            status="partially_filled",
            filled_qty=1.0,
            remaining_qty=1.0,
            avg_fill_price=400.0,
            note="order_matched",
        )
        audit_store.record_fill(
            connection=connection,
            run_id=run_id,
            timestamp=event_time,
            order_id="ord-1",
            symbol="QQQ",
            side="buy",
            qty=1.0,
            price=400.0,
            reason="order_matched",
        )
        audit_store.record_snapshot(
            connection=connection,
            run_id=run_id,
            timestamp=event_time,
            bar_index=2,
            cash=99_600.0,
            position_qty=1.0,
            equity=100_001.0,
            close_price=401.0,
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
