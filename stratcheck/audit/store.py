"""SQLite run-audit store for reproducible timeline replay."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


@dataclass(slots=True)
class RunRecord:
    """Run metadata row."""

    run_id: str
    mode: str
    symbol: str
    strategy: str
    status: str
    started_at: str
    finished_at: str | None
    initial_cash: float
    note: str


class RunAuditStore:
    """Persist and replay run timelines from sqlite."""

    def __init__(self, sqlite_path: str | Path) -> None:
        self.sqlite_path = Path(sqlite_path)

    def connect(self) -> sqlite3.Connection:
        """Open sqlite connection."""
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.sqlite_path)
        connection.row_factory = sqlite3.Row
        return connection

    def ensure_schema(self, connection: sqlite3.Connection) -> None:
        """Create audit schema when absent."""
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    mode TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT NULL,
                    initial_cash REAL NOT NULL,
                    note TEXT NOT NULL DEFAULT ''
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS configs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs (run_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    bar_index INTEGER NOT NULL,
                    signal_index INTEGER NOT NULL,
                    side TEXT NOT NULL,
                    qty REAL NOT NULL,
                    order_type TEXT NOT NULL,
                    limit_price REAL NULL,
                    reason TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs (run_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    order_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    qty REAL NOT NULL,
                    order_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    filled_qty REAL NOT NULL,
                    remaining_qty REAL NOT NULL,
                    avg_fill_price REAL NOT NULL,
                    note TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs (run_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS fills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    order_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    qty REAL NOT NULL,
                    price REAL NOT NULL,
                    notional REAL NOT NULL,
                    reason TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs (run_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    bar_index INTEGER NOT NULL,
                    cash REAL NOT NULL,
                    position_qty REAL NOT NULL,
                    equity REAL NOT NULL,
                    close_price REAL NOT NULL,
                    note TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs (run_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS risk_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    rule_name TEXT NOT NULL,
                    action TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs (run_id)
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status)")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_signals_run_time ON signals(run_id, timestamp, id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_orders_run_time ON orders(run_id, timestamp, id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_fills_run_time ON fills(run_id, timestamp, id)"
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_snapshots_run_time
                ON snapshots(run_id, timestamp, id)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_risk_events_run_time
                ON risk_events(run_id, timestamp, id)
                """
            )

    def create_run(
        self,
        connection: sqlite3.Connection,
        *,
        mode: str,
        symbol: str,
        strategy: str,
        initial_cash: float,
        config: Mapping[str, Any],
        note: str = "",
    ) -> str:
        """Insert run + config and return run_id."""
        run_id = uuid.uuid4().hex[:12]
        started_at = _utc_now().isoformat()
        config_json = json.dumps(dict(config), ensure_ascii=False, sort_keys=True, default=str)

        with connection:
            connection.execute(
                """
                INSERT INTO runs (
                    run_id,
                    mode,
                    symbol,
                    strategy,
                    status,
                    started_at,
                    finished_at,
                    initial_cash,
                    note
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    str(mode),
                    str(symbol).upper(),
                    str(strategy),
                    "running",
                    started_at,
                    None,
                    float(initial_cash),
                    str(note),
                ),
            )
            connection.execute(
                """
                INSERT INTO configs (
                    run_id,
                    config_json,
                    created_at
                )
                VALUES (?, ?, ?)
                """,
                (
                    run_id,
                    config_json,
                    started_at,
                ),
            )
        return run_id

    def finalize_run(
        self,
        connection: sqlite3.Connection,
        *,
        run_id: str,
        status: str,
        note: str = "",
    ) -> None:
        """Update run terminal status."""
        with connection:
            connection.execute(
                """
                UPDATE runs
                SET status = ?,
                    finished_at = ?,
                    note = CASE
                        WHEN note = '' THEN ?
                        WHEN ? = '' THEN note
                        ELSE note || '; ' || ?
                    END
                WHERE run_id = ?
                """,
                (
                    str(status),
                    _utc_now().isoformat(),
                    str(note),
                    str(note),
                    str(note),
                    run_id,
                ),
            )

    def record_signal(
        self,
        connection: sqlite3.Connection,
        *,
        run_id: str,
        timestamp: pd.Timestamp,
        bar_index: int,
        signal_index: int,
        side: str,
        qty: float,
        order_type: str,
        limit_price: float | None,
        reason: str,
    ) -> None:
        """Insert strategy signal row."""
        with connection:
            connection.execute(
                """
                INSERT INTO signals (
                    run_id,
                    timestamp,
                    bar_index,
                    signal_index,
                    side,
                    qty,
                    order_type,
                    limit_price,
                    reason
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    _normalize_time(timestamp).isoformat(),
                    int(bar_index),
                    int(signal_index),
                    str(side),
                    float(qty),
                    str(order_type),
                    None if limit_price is None else float(limit_price),
                    str(reason),
                ),
            )

    def record_order(
        self,
        connection: sqlite3.Connection,
        *,
        run_id: str,
        timestamp: pd.Timestamp,
        order_id: str,
        symbol: str,
        side: str,
        qty: float,
        order_type: str,
        status: str,
        filled_qty: float,
        remaining_qty: float,
        avg_fill_price: float,
        note: str,
    ) -> None:
        """Insert order lifecycle event row."""
        with connection:
            connection.execute(
                """
                INSERT INTO orders (
                    run_id,
                    timestamp,
                    order_id,
                    symbol,
                    side,
                    qty,
                    order_type,
                    status,
                    filled_qty,
                    remaining_qty,
                    avg_fill_price,
                    note
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    _normalize_time(timestamp).isoformat(),
                    str(order_id),
                    str(symbol).upper(),
                    str(side),
                    float(qty),
                    str(order_type),
                    str(status),
                    float(filled_qty),
                    float(remaining_qty),
                    float(avg_fill_price),
                    str(note),
                ),
            )

    def record_fill(
        self,
        connection: sqlite3.Connection,
        *,
        run_id: str,
        timestamp: pd.Timestamp,
        order_id: str,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        reason: str,
    ) -> None:
        """Insert fill row."""
        with connection:
            connection.execute(
                """
                INSERT INTO fills (
                    run_id,
                    timestamp,
                    order_id,
                    symbol,
                    side,
                    qty,
                    price,
                    notional,
                    reason
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    _normalize_time(timestamp).isoformat(),
                    str(order_id),
                    str(symbol).upper(),
                    str(side),
                    float(qty),
                    float(price),
                    float(qty * price),
                    str(reason),
                ),
            )

    def record_snapshot(
        self,
        connection: sqlite3.Connection,
        *,
        run_id: str,
        timestamp: pd.Timestamp,
        bar_index: int,
        cash: float,
        position_qty: float,
        equity: float,
        close_price: float,
        note: str,
    ) -> None:
        """Insert state snapshot row."""
        with connection:
            connection.execute(
                """
                INSERT INTO snapshots (
                    run_id,
                    timestamp,
                    bar_index,
                    cash,
                    position_qty,
                    equity,
                    close_price,
                    note
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    _normalize_time(timestamp).isoformat(),
                    int(bar_index),
                    float(cash),
                    float(position_qty),
                    float(equity),
                    float(close_price),
                    str(note),
                ),
            )

    def record_risk_event(
        self,
        connection: sqlite3.Connection,
        *,
        run_id: str,
        timestamp: pd.Timestamp,
        rule_name: str,
        action: str,
        reason: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """Insert one risk-rule hit row."""
        details_payload = json.dumps(
            dict(details or {}),
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        with connection:
            connection.execute(
                """
                INSERT INTO risk_events (
                    run_id,
                    timestamp,
                    rule_name,
                    action,
                    reason,
                    details_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    _normalize_time(timestamp).isoformat(),
                    str(rule_name),
                    str(action),
                    str(reason),
                    details_payload,
                ),
            )

    def get_run(self, run_id: str) -> RunRecord:
        """Fetch single run row."""
        connection = self.connect()
        try:
            self.ensure_schema(connection)
            row = connection.execute(
                """
                SELECT
                    run_id,
                    mode,
                    symbol,
                    strategy,
                    status,
                    started_at,
                    finished_at,
                    initial_cash,
                    note
                FROM runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        finally:
            connection.close()

        if row is None:
            msg = f"Run not found: {run_id}"
            raise ValueError(msg)
        return RunRecord(
            run_id=str(row["run_id"]),
            mode=str(row["mode"]),
            symbol=str(row["symbol"]),
            strategy=str(row["strategy"]),
            status=str(row["status"]),
            started_at=str(row["started_at"]),
            finished_at=None if row["finished_at"] is None else str(row["finished_at"]),
            initial_cash=float(row["initial_cash"]),
            note=str(row["note"]),
        )

    def get_counts(self, run_id: str) -> dict[str, int]:
        """Return row counts for audit tables by run_id."""
        connection = self.connect()
        try:
            self.ensure_schema(connection)
            tables = ["configs", "signals", "orders", "fills", "snapshots"]
            counts: dict[str, int] = {}
            for table_name in tables:
                count_value = connection.execute(
                    f"SELECT COUNT(*) FROM {table_name} WHERE run_id = ?",
                    (run_id,),
                ).fetchone()[0]
                counts[table_name] = int(count_value)
            return counts
        finally:
            connection.close()

    def render_replay(self, run_id: str) -> str:
        """Render a human-readable timeline for a run."""
        run = self.get_run(run_id)
        connection = self.connect()
        try:
            self.ensure_schema(connection)
            config_row = connection.execute(
                """
                SELECT config_json
                FROM configs
                WHERE run_id = ?
                ORDER BY id ASC
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()

            timeline_rows = _load_timeline_rows(connection=connection, run_id=run_id)
        finally:
            connection.close()

        config_text = "{}" if config_row is None else str(config_row["config_json"])
        lines = [
            f"Run ID: {run.run_id}",
            f"Mode: {run.mode}",
            f"Symbol: {run.symbol}",
            f"Strategy: {run.strategy}",
            f"Status: {run.status}",
            f"Started: {run.started_at}",
            f"Finished: {run.finished_at or ''}",
            f"Initial Cash: {run.initial_cash:.2f}",
            f"Run Note: {run.note}",
            f"Config: {config_text}",
            "",
            "Timeline:",
        ]
        if not timeline_rows:
            lines.append("- (empty)")
            return "\n".join(lines)

        for row in timeline_rows:
            row_type = row["row_type"]
            timestamp = row["timestamp"]
            if row_type == "signal":
                lines.append(
                    " - "
                    f"{timestamp} SIGNAL "
                    f"bar={row['bar_index']} idx={row['signal_index']} "
                    f"side={row['side']} qty={row['qty']:.6f} "
                    f"type={row['order_type']} limit={_format_optional(row['limit_price'])} "
                    f"reason={row['reason']}"
                )
                continue
            if row_type == "order":
                lines.append(
                    " - "
                    f"{timestamp} ORDER "
                    f"order_id={row['order_id']} symbol={row['symbol']} "
                    f"side={row['side']} qty={row['qty']:.6f} type={row['order_type']} "
                    f"status={row['status']} filled={row['filled_qty']:.6f} "
                    f"remaining={row['remaining_qty']:.6f} "
                    f"avg_price={row['avg_fill_price']:.6f} note={row['note']}"
                )
                continue
            if row_type == "fill":
                lines.append(
                    " - "
                    f"{timestamp} FILL "
                    f"order_id={row['order_id']} symbol={row['symbol']} "
                    f"side={row['side']} qty={row['qty']:.6f} "
                    f"price={row['price']:.6f} notional={row['notional']:.6f} "
                    f"reason={row['reason']}"
                )
                continue
            if row_type == "risk":
                lines.append(
                    " - "
                    f"{timestamp} RISK "
                    f"rule={row['rule_name']} action={row['action']} "
                    f"reason={row['reason']} details={row['details_json']}"
                )
                continue
            lines.append(
                " - "
                f"{timestamp} SNAPSHOT "
                f"bar={row['bar_index']} cash={row['cash']:.6f} "
                f"position={row['position_qty']:.6f} equity={row['equity']:.6f} "
                f"close={row['close_price']:.6f} note={row['note']}"
            )

        return "\n".join(lines)


def _load_timeline_rows(
    connection: sqlite3.Connection,
    run_id: str,
) -> list[dict[str, Any]]:
    signal_rows = connection.execute(
        """
        SELECT
            'signal' AS row_type,
            timestamp,
            id AS sequence_id,
            bar_index,
            signal_index,
            side,
            qty,
            order_type,
            limit_price,
            reason
        FROM signals
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchall()
    order_rows = connection.execute(
        """
        SELECT
            'order' AS row_type,
            timestamp,
            id AS sequence_id,
            order_id,
            symbol,
            side,
            qty,
            order_type,
            status,
            filled_qty,
            remaining_qty,
            avg_fill_price,
            note
        FROM orders
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchall()
    fill_rows = connection.execute(
        """
        SELECT
            'fill' AS row_type,
            timestamp,
            id AS sequence_id,
            order_id,
            symbol,
            side,
            qty,
            price,
            notional,
            reason
        FROM fills
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchall()
    snapshot_rows = connection.execute(
        """
        SELECT
            'snapshot' AS row_type,
            timestamp,
            id AS sequence_id,
            bar_index,
            cash,
            position_qty,
            equity,
            close_price,
            note
        FROM snapshots
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchall()
    risk_rows = connection.execute(
        """
        SELECT
            'risk' AS row_type,
            timestamp,
            id AS sequence_id,
            rule_name,
            action,
            reason,
            details_json
        FROM risk_events
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchall()

    merged_rows: list[dict[str, Any]] = []
    for rows in [signal_rows, order_rows, fill_rows, risk_rows, snapshot_rows]:
        for row in rows:
            merged_rows.append(dict(row))

    type_priority = {
        "signal": 10,
        "order": 20,
        "fill": 30,
        "risk": 35,
        "snapshot": 40,
    }
    merged_rows.sort(
        key=lambda row: (
            _normalize_time(row["timestamp"]).value,
            type_priority.get(str(row["row_type"]), 99),
            int(row["sequence_id"]),
        )
    )
    return merged_rows


def _normalize_time(value: pd.Timestamp | str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _format_optional(value: Any) -> str:
    if value is None:
        return "None"
    return str(value)
