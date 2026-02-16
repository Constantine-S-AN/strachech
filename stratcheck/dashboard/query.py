"""Data query helpers for static dashboard pages."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_EXPERIMENT_COLUMNS = [
    "rank",
    "experiment",
    "status",
    "sharpe",
    "max_drawdown",
    "worst_window_sharpe",
    "worst_window_drawdown",
    "cagr",
    "total_return",
    "report_path",
    "error",
]
_RUN_COLUMNS = [
    "run_id",
    "status",
    "symbol",
    "strategy",
    "started_at",
    "finished_at",
    "latest_equity",
    "pnl",
    "drawdown",
    "order_rate",
    "error_rate",
    "risk_rule_hits",
    "risk_halt_hits",
    "risk_block_hits",
    "kill_reason",
]
_LIVE_POSITION_COLUMNS = [
    "run_id",
    "status",
    "symbol",
    "strategy",
    "position_qty",
    "close_price",
    "position_notional",
    "equity",
    "updated_at",
]
_LIVE_TRADES_COLUMNS = [
    "run_id",
    "status",
    "symbol",
    "strategy",
    "trades_today",
    "filled_qty_today",
    "notional_today",
    "last_trade_at",
]
_LIVE_RISK_COLUMNS = [
    "run_id",
    "status",
    "symbol",
    "strategy",
    "risk_state",
    "risk_halt_hits",
    "risk_block_hits",
    "last_risk_reason",
    "last_risk_action",
    "last_risk_at",
    "kill_reason",
]
_LIVE_ERROR_COLUMNS = [
    "timestamp",
    "run_id",
    "symbol",
    "source",
    "severity",
    "event",
    "message",
]


@dataclass(slots=True)
class LiveStatusData:
    """Live-status tables used by dashboard live panel."""

    positions_df: pd.DataFrame
    today_trades_df: pd.DataFrame
    risk_status_df: pd.DataFrame
    recent_errors_df: pd.DataFrame


def load_experiment_rankings(
    results_jsonl_path: str | Path = "reports/results.jsonl",
) -> pd.DataFrame:
    """Load experiment ranking table from results JSONL."""
    path = Path(results_jsonl_path)
    if not path.exists():
        return pd.DataFrame(columns=_EXPERIMENT_COLUMNS)

    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as results_file:
        for raw_line in results_file:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            records.append(dict(payload))

    if not records:
        return pd.DataFrame(columns=_EXPERIMENT_COLUMNS)

    frame = pd.DataFrame(records)
    metric_columns = [
        "sharpe",
        "max_drawdown",
        "worst_window_sharpe",
        "worst_window_drawdown",
        "cagr",
        "total_return",
    ]
    for metric_key in metric_columns:
        if metric_key in frame.columns:
            frame[metric_key] = pd.to_numeric(frame[metric_key], errors="coerce")
        else:
            frame[metric_key] = float("nan")

    if "error" not in frame.columns:
        frame["error"] = ""
    if "report_path" not in frame.columns:
        frame["report_path"] = ""
    if "status" not in frame.columns:
        frame["status"] = "unknown"
    if "experiment" not in frame.columns:
        frame["experiment"] = ""

    success_mask = frame["status"] == "success"
    success_rank = (
        frame.loc[success_mask, "sharpe"]
        .rank(method="min", ascending=False, na_option="bottom")
        .astype("Int64")
    )
    frame["rank"] = pd.Series([pd.NA] * len(frame), dtype="Int64")
    frame.loc[success_mask, "rank"] = success_rank

    frame["status_sort"] = frame["status"].map({"success": 0, "failed": 1}).fillna(2)
    frame = frame.sort_values(
        by=["status_sort", "rank", "sharpe", "experiment"],
        ascending=[True, True, False, True],
        na_position="last",
    ).reset_index(drop=True)
    frame = frame.drop(columns=["status_sort"])
    return frame.reindex(columns=_EXPERIMENT_COLUMNS, fill_value="")


def load_paper_run_statuses(
    sqlite_path: str | Path = "reports/paper_trading.sqlite",
) -> pd.DataFrame:
    """Load paper run states and runtime metrics from sqlite audit tables."""
    path = Path(sqlite_path)
    if not path.exists():
        return pd.DataFrame(columns=_RUN_COLUMNS)

    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        if not _has_table(connection, "runs"):
            return pd.DataFrame(columns=_RUN_COLUMNS)

        run_rows = connection.execute(
            """
            SELECT run_id, status, symbol, strategy, started_at, finished_at, initial_cash, note
            FROM runs
            ORDER BY started_at DESC
            """
        ).fetchall()
        if not run_rows:
            return pd.DataFrame(columns=_RUN_COLUMNS)

        records: list[dict[str, object]] = []
        for row in run_rows:
            run_id = str(row["run_id"])
            initial_cash = float(row["initial_cash"])
            snapshots = _load_snapshots(connection=connection, run_id=run_id)
            latest_equity = initial_cash if snapshots.empty else float(snapshots["equity"].iloc[-1])
            pnl_value = latest_equity - initial_cash
            drawdown_value = (
                _max_drawdown_from_equity(snapshots["equity"]) if not snapshots.empty else 0.0
            )

            orders_count = _count_rows(connection=connection, table_name="orders", run_id=run_id)
            error_events = 1 if str(row["status"]) in {"failed", "killed"} else 0
            blocked_events = _count_rows_with_note(
                connection=connection,
                run_id=run_id,
                note_keyword="breached",
            )
            risk_rule_hits = _count_rows(
                connection=connection,
                table_name="risk_events",
                run_id=run_id,
            )
            risk_halt_hits = _count_risk_events_by_action(
                connection=connection,
                run_id=run_id,
                action="halt",
            )
            risk_block_hits = _count_risk_events_by_action(
                connection=connection,
                run_id=run_id,
                action="block",
            )
            total_errors = error_events + blocked_events

            elapsed_seconds = _elapsed_seconds(
                started_at=str(row["started_at"]),
                finished_at=None if row["finished_at"] is None else str(row["finished_at"]),
            )
            order_rate = float(orders_count / elapsed_seconds) if elapsed_seconds > 0 else 0.0
            error_rate = float(total_errors / elapsed_seconds) if elapsed_seconds > 0 else 0.0

            records.append(
                {
                    "run_id": run_id,
                    "status": str(row["status"]),
                    "symbol": str(row["symbol"]),
                    "strategy": str(row["strategy"]),
                    "started_at": str(row["started_at"]),
                    "finished_at": "" if row["finished_at"] is None else str(row["finished_at"]),
                    "latest_equity": float(latest_equity),
                    "pnl": float(pnl_value),
                    "drawdown": float(drawdown_value),
                    "order_rate": float(order_rate),
                    "error_rate": float(error_rate),
                    "risk_rule_hits": int(risk_rule_hits),
                    "risk_halt_hits": int(risk_halt_hits),
                    "risk_block_hits": int(risk_block_hits),
                    "kill_reason": str(row["note"]),
                }
            )

        frame = pd.DataFrame(records)
        return frame.reindex(columns=_RUN_COLUMNS, fill_value="")
    finally:
        connection.close()


def load_live_status(
    sqlite_path: str | Path = "reports/paper_trading.sqlite",
    *,
    now_timestamp: pd.Timestamp | str | None = None,
    max_runs: int = 10,
    recent_error_limit: int = 20,
) -> LiveStatusData:
    """Load live-status dashboard tables from sqlite audit + runner logs."""
    path = Path(sqlite_path)
    if not path.exists():
        return _empty_live_status_data()
    if max_runs < 1:
        msg = "max_runs must be >= 1."
        raise ValueError(msg)
    if recent_error_limit < 1:
        msg = "recent_error_limit must be >= 1."
        raise ValueError(msg)

    utc_now = _normalize_timestamp(now_timestamp)
    day_start = utc_now.floor("D")
    day_end = day_start + pd.Timedelta(days=1)

    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        live_runs = _load_live_runs(connection=connection, max_runs=max_runs)
        if live_runs.empty:
            return _empty_live_status_data()

        positions_df = _load_live_positions(connection=connection, live_runs=live_runs)
        today_trades_df = _load_today_trades(
            connection=connection,
            live_runs=live_runs,
            day_start=day_start,
            day_end=day_end,
        )
        risk_status_df = _load_risk_status(connection=connection, live_runs=live_runs)
        recent_errors_df = _load_recent_errors(
            connection=connection,
            live_runs=live_runs,
            recent_error_limit=recent_error_limit,
        )

        return LiveStatusData(
            positions_df=positions_df,
            today_trades_df=today_trades_df,
            risk_status_df=risk_status_df,
            recent_errors_df=recent_errors_df,
        )
    finally:
        connection.close()


def _empty_live_status_data() -> LiveStatusData:
    return LiveStatusData(
        positions_df=pd.DataFrame(columns=_LIVE_POSITION_COLUMNS),
        today_trades_df=pd.DataFrame(columns=_LIVE_TRADES_COLUMNS),
        risk_status_df=pd.DataFrame(columns=_LIVE_RISK_COLUMNS),
        recent_errors_df=pd.DataFrame(columns=_LIVE_ERROR_COLUMNS),
    )


def _load_live_runs(connection: sqlite3.Connection, max_runs: int) -> pd.DataFrame:
    if not _has_table(connection, "runs"):
        return pd.DataFrame(
            columns=["run_id", "status", "symbol", "strategy", "started_at", "finished_at", "note"]
        )

    running_rows = connection.execute(
        """
        SELECT run_id, status, symbol, strategy, started_at, finished_at, note
        FROM runs
        WHERE LOWER(status) = 'running'
        ORDER BY started_at DESC
        LIMIT ?
        """,
        (int(max_runs),),
    ).fetchall()

    selected_rows = running_rows
    if not selected_rows:
        selected_rows = connection.execute(
            """
            SELECT run_id, status, symbol, strategy, started_at, finished_at, note
            FROM runs
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (int(max_runs),),
        ).fetchall()
    if not selected_rows:
        return pd.DataFrame(
            columns=["run_id", "status", "symbol", "strategy", "started_at", "finished_at", "note"]
        )
    return pd.DataFrame([dict(row) for row in selected_rows])


def _load_live_positions(connection: sqlite3.Connection, live_runs: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    has_snapshots = _has_table(connection, "snapshots")

    for _, run_row in live_runs.iterrows():
        run_id = str(run_row["run_id"])
        snapshot_row = None
        if has_snapshots:
            snapshot_row = connection.execute(
                """
                SELECT timestamp, position_qty, close_price, equity
                FROM snapshots
                WHERE run_id = ?
                ORDER BY timestamp DESC, id DESC
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()

        if snapshot_row is None:
            position_qty = 0.0
            close_price: float | None = None
            equity: float | None = None
            updated_at = ""
        else:
            position_qty = float(snapshot_row["position_qty"])
            close_price = float(snapshot_row["close_price"])
            equity = float(snapshot_row["equity"])
            updated_at = str(snapshot_row["timestamp"])

        position_notional = (
            None if close_price is None else float(position_qty * float(close_price))
        )
        records.append(
            {
                "run_id": run_id,
                "status": str(run_row["status"]),
                "symbol": str(run_row["symbol"]),
                "strategy": str(run_row["strategy"]),
                "position_qty": float(position_qty),
                "close_price": close_price,
                "position_notional": position_notional,
                "equity": equity,
                "updated_at": updated_at,
            }
        )

    if not records:
        return pd.DataFrame(columns=_LIVE_POSITION_COLUMNS)
    frame = pd.DataFrame(records)
    return frame.reindex(columns=_LIVE_POSITION_COLUMNS, fill_value="")


def _load_today_trades(
    connection: sqlite3.Connection,
    live_runs: pd.DataFrame,
    *,
    day_start: pd.Timestamp,
    day_end: pd.Timestamp,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    has_fills = _has_table(connection, "fills")

    for _, run_row in live_runs.iterrows():
        run_id = str(run_row["run_id"])
        trades_today = 0
        filled_qty_today = 0.0
        notional_today = 0.0
        last_trade_at = ""

        if has_fills:
            agg_row = connection.execute(
                """
                SELECT
                    COUNT(*) AS trades_today,
                    COALESCE(SUM(qty), 0.0) AS filled_qty_today,
                    COALESCE(SUM(notional), 0.0) AS notional_today,
                    MAX(timestamp) AS last_trade_at
                FROM fills
                WHERE run_id = ?
                  AND timestamp >= ?
                  AND timestamp < ?
                """,
                (
                    run_id,
                    day_start.isoformat(),
                    day_end.isoformat(),
                ),
            ).fetchone()
            if agg_row is not None:
                trades_today = int(agg_row["trades_today"])
                filled_qty_today = float(agg_row["filled_qty_today"])
                notional_today = float(agg_row["notional_today"])
                last_trade_at = (
                    "" if agg_row["last_trade_at"] is None else str(agg_row["last_trade_at"])
                )

        records.append(
            {
                "run_id": run_id,
                "status": str(run_row["status"]),
                "symbol": str(run_row["symbol"]),
                "strategy": str(run_row["strategy"]),
                "trades_today": int(trades_today),
                "filled_qty_today": float(filled_qty_today),
                "notional_today": float(notional_today),
                "last_trade_at": last_trade_at,
            }
        )

    if not records:
        return pd.DataFrame(columns=_LIVE_TRADES_COLUMNS)
    frame = pd.DataFrame(records)
    return frame.reindex(columns=_LIVE_TRADES_COLUMNS, fill_value="")


def _load_risk_status(connection: sqlite3.Connection, live_runs: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    has_risk_events = _has_table(connection, "risk_events")

    for _, run_row in live_runs.iterrows():
        run_id = str(run_row["run_id"])
        run_status = str(run_row["status"]).lower()
        risk_halt_hits = 0
        risk_block_hits = 0
        last_risk_reason = ""
        last_risk_action = ""
        last_risk_at = ""

        if has_risk_events:
            risk_halt_hits = _count_risk_events_by_action(
                connection=connection,
                run_id=run_id,
                action="halt",
            )
            risk_block_hits = _count_risk_events_by_action(
                connection=connection,
                run_id=run_id,
                action="block",
            )
            latest_risk = connection.execute(
                """
                SELECT timestamp, action, reason
                FROM risk_events
                WHERE run_id = ?
                ORDER BY timestamp DESC, id DESC
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()
            if latest_risk is not None:
                last_risk_reason = str(latest_risk["reason"])
                last_risk_action = str(latest_risk["action"])
                last_risk_at = str(latest_risk["timestamp"])

        if run_status in {"failed", "killed"} or risk_halt_hits > 0:
            risk_state = "critical"
        elif risk_block_hits > 0:
            risk_state = "warning"
        else:
            risk_state = "ok"

        records.append(
            {
                "run_id": run_id,
                "status": str(run_row["status"]),
                "symbol": str(run_row["symbol"]),
                "strategy": str(run_row["strategy"]),
                "risk_state": risk_state,
                "risk_halt_hits": int(risk_halt_hits),
                "risk_block_hits": int(risk_block_hits),
                "last_risk_reason": last_risk_reason,
                "last_risk_action": last_risk_action,
                "last_risk_at": last_risk_at,
                "kill_reason": str(run_row["note"]),
            }
        )

    if not records:
        return pd.DataFrame(columns=_LIVE_RISK_COLUMNS)
    frame = pd.DataFrame(records)
    return frame.reindex(columns=_LIVE_RISK_COLUMNS, fill_value="")


def _load_recent_errors(
    connection: sqlite3.Connection,
    live_runs: pd.DataFrame,
    *,
    recent_error_limit: int,
) -> pd.DataFrame:
    run_ids = [str(value) for value in live_runs["run_id"].tolist()]
    if not run_ids:
        return pd.DataFrame(columns=_LIVE_ERROR_COLUMNS)

    entries: list[dict[str, object]] = []
    run_meta = {
        str(row["run_id"]): {"symbol": str(row["symbol"])} for _, row in live_runs.iterrows()
    }

    for _, run_row in live_runs.iterrows():
        status_text = str(run_row["status"]).lower()
        if status_text in {"failed", "killed"}:
            entries.append(
                {
                    "timestamp": str(run_row["finished_at"] or run_row["started_at"]),
                    "run_id": str(run_row["run_id"]),
                    "symbol": str(run_row["symbol"]),
                    "source": "runs",
                    "severity": "error",
                    "event": "run_terminal",
                    "message": str(run_row["note"]),
                }
            )

    if _has_table(connection, "risk_events"):
        placeholders = ",".join(["?"] * len(run_ids))
        query = (
            "SELECT run_id, timestamp, action, reason, rule_name "
            "FROM risk_events "
            f"WHERE run_id IN ({placeholders}) "
            "AND (LOWER(action) = 'halt' OR LOWER(reason) LIKE '%error%') "
            "ORDER BY timestamp DESC, id DESC"
        )
        risk_rows = connection.execute(query, run_ids).fetchall()
        for risk_row in risk_rows:
            run_id = str(risk_row["run_id"])
            symbol_text = run_meta.get(run_id, {}).get("symbol", "")
            entries.append(
                {
                    "timestamp": str(risk_row["timestamp"]),
                    "run_id": run_id,
                    "symbol": symbol_text,
                    "source": "risk_events",
                    "severity": "error",
                    "event": str(risk_row["rule_name"]),
                    "message": str(risk_row["reason"]),
                }
            )

    entries.extend(
        _load_recent_log_errors(
            connection=connection,
            run_ids=run_ids,
            run_meta=run_meta,
            per_run_limit=recent_error_limit,
        )
    )
    if not entries:
        return pd.DataFrame(columns=_LIVE_ERROR_COLUMNS)

    entries.sort(key=lambda item: _normalize_timestamp(item["timestamp"]).value, reverse=True)
    limited_entries = entries[: int(recent_error_limit)]
    frame = pd.DataFrame(limited_entries)
    return frame.reindex(columns=_LIVE_ERROR_COLUMNS, fill_value="")


def _load_recent_log_errors(
    connection: sqlite3.Connection,
    *,
    run_ids: list[str],
    run_meta: dict[str, dict[str, str]],
    per_run_limit: int,
) -> list[dict[str, object]]:
    if not _has_table(connection, "configs"):
        return []

    entries: list[dict[str, object]] = []
    for run_id in run_ids:
        config_row = connection.execute(
            """
            SELECT config_json
            FROM configs
            WHERE run_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (run_id,),
        ).fetchone()
        if config_row is None:
            continue

        try:
            config_payload = json.loads(str(config_row["config_json"]))
        except json.JSONDecodeError:
            continue
        log_path_value = config_payload.get("log_path")
        if log_path_value is None:
            continue

        log_path = Path(str(log_path_value))
        if not log_path.exists():
            continue

        raw_lines = log_path.read_text(encoding="utf-8").splitlines()
        found_count = 0
        for raw_line in reversed(raw_lines):
            if found_count >= per_run_limit:
                break
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            level = str(payload.get("level", "")).lower()
            if level not in {"error", "critical"}:
                continue

            message_text = (
                str(payload.get("message"))
                if payload.get("message") is not None
                else str(payload.get("reason", ""))
            )
            entries.append(
                {
                    "timestamp": str(payload.get("timestamp", "")),
                    "run_id": run_id,
                    "symbol": run_meta.get(run_id, {}).get("symbol", ""),
                    "source": "runner_log",
                    "severity": level or "error",
                    "event": str(payload.get("event", "log_error")),
                    "message": message_text,
                }
            )
            found_count += 1

    return entries


def _has_table(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table' AND name=?
        """,
        (table_name,),
    ).fetchone()
    return row is not None


def _load_snapshots(connection: sqlite3.Connection, run_id: str) -> pd.DataFrame:
    if not _has_table(connection, "snapshots"):
        return pd.DataFrame(columns=["timestamp", "equity"])
    rows = connection.execute(
        """
        SELECT timestamp, equity
        FROM snapshots
        WHERE run_id = ?
        ORDER BY timestamp ASC, id ASC
        """,
        (run_id,),
    ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["timestamp", "equity"])
    frame = pd.DataFrame([dict(row) for row in rows])
    frame["equity"] = pd.to_numeric(frame["equity"], errors="coerce")
    frame = frame.dropna(subset=["equity"]).reset_index(drop=True)
    return frame


def _count_rows(connection: sqlite3.Connection, table_name: str, run_id: str) -> int:
    if not _has_table(connection, table_name):
        return 0
    value = connection.execute(
        f"SELECT COUNT(*) FROM {table_name} WHERE run_id = ?",
        (run_id,),
    ).fetchone()[0]
    return int(value)


def _count_rows_with_note(
    connection: sqlite3.Connection,
    run_id: str,
    note_keyword: str,
) -> int:
    if not _has_table(connection, "orders"):
        return 0
    value = connection.execute(
        """
        SELECT COUNT(*)
        FROM orders
        WHERE run_id = ?
          AND LOWER(note) LIKE ?
        """,
        (run_id, f"%{note_keyword.lower()}%"),
    ).fetchone()[0]
    return int(value)


def _count_risk_events_by_action(
    connection: sqlite3.Connection,
    run_id: str,
    action: str,
) -> int:
    if not _has_table(connection, "risk_events"):
        return 0
    value = connection.execute(
        """
        SELECT COUNT(*)
        FROM risk_events
        WHERE run_id = ?
          AND LOWER(action) = ?
        """,
        (run_id, str(action).lower()),
    ).fetchone()[0]
    return int(value)


def _max_drawdown_from_equity(equity_series: pd.Series) -> float:
    if equity_series.empty:
        return 0.0
    running_peak = equity_series.cummax()
    drawdown_series = (running_peak - equity_series) / running_peak.replace(0.0, float("nan"))
    drawdown_value = float(drawdown_series.max())
    if pd.isna(drawdown_value):
        return 0.0
    return drawdown_value


def _elapsed_seconds(started_at: str, finished_at: str | None) -> float:
    start_value = _normalize_timestamp(started_at)
    stop_value = _normalize_timestamp(finished_at)
    elapsed = (stop_value - start_value).total_seconds()
    return max(float(elapsed), 1e-9)


def _normalize_timestamp(value: pd.Timestamp | str | None) -> pd.Timestamp:
    if value is None or value == "":
        return pd.Timestamp.now(tz="UTC")
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")
