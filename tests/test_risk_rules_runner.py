from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
from stratcheck.connectors import LivePaperRunner, PaperBrokerConnector
from stratcheck.core.strategy import OrderIntent, PortfolioState
from stratcheck.dashboard import load_paper_run_statuses


def test_runner_max_daily_trades_rule_records_audit_and_dashboard(tmp_path: Path) -> None:
    connector = PaperBrokerConnector(
        initial_cash=10_000.0,
        max_fill_ratio_per_step=1.0,
        max_volume_share=1.0,
    )
    runner = LivePaperRunner(
        connector=connector,
        sqlite_path=tmp_path / "daily_limit.sqlite",
        symbol="AAPL",
        max_daily_trades=1,
    )

    class BuyEveryBarStrategy:
        def generate_orders(
            self,
            bars: pd.DataFrame,
            portfolio_state: PortfolioState,
        ) -> list[OrderIntent]:
            del portfolio_state
            return [OrderIntent(side="buy", qty=1.0, market=True)]

    timestamps = pd.date_range("2024-01-01 09:30:00", periods=4, freq="h", tz="UTC")
    bars = _build_bars(
        timestamps=timestamps,
        open_prices=[100.0, 100.0, 100.0, 100.0],
        close_prices=[100.0, 100.0, 100.0, 100.0],
    )

    result = runner.run(strategy=BuyEveryBarStrategy(), bars=bars)
    assert result.status == "completed"
    assert result.orders_placed == 1

    connection = sqlite3.connect(result.sqlite_path)
    try:
        risk_rows = connection.execute(
            """
            SELECT reason, action, rule_name
            FROM risk_events
            WHERE run_id = ?
            ORDER BY id ASC
            """,
            (result.run_id,),
        ).fetchall()
    finally:
        connection.close()

    assert len(risk_rows) >= 2
    assert {row[0] for row in risk_rows} == {"max_daily_trades_exceeded"}
    assert {row[1] for row in risk_rows} == {"block"}
    assert {row[2] for row in risk_rows} == {"max_daily_trades"}

    status_frame = load_paper_run_statuses(sqlite_path=result.sqlite_path)
    assert len(status_frame) == 1
    status_row = status_frame.iloc[0]
    assert int(status_row["risk_rule_hits"]) >= 2
    assert int(status_row["risk_block_hits"]) >= 2
    assert int(status_row["risk_halt_hits"]) == 0


def test_runner_data_anomaly_rule_halts_and_records_hit(tmp_path: Path) -> None:
    connector = PaperBrokerConnector(
        initial_cash=10_000.0,
        max_fill_ratio_per_step=1.0,
        max_volume_share=1.0,
    )
    runner = LivePaperRunner(
        connector=connector,
        sqlite_path=tmp_path / "abnormal_data.sqlite",
        symbol="AAPL",
    )

    class NoopStrategy:
        def generate_orders(
            self,
            bars: pd.DataFrame,
            portfolio_state: PortfolioState,
        ) -> list[OrderIntent]:
            del bars, portfolio_state
            return []

    timestamps = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    bars = _build_bars(
        timestamps=timestamps,
        open_prices=[100.0, 101.0, 102.0],
        close_prices=[100.0, float("nan"), 102.0],
    )

    result = runner.run(strategy=NoopStrategy(), bars=bars)
    assert result.status == "killed"
    assert result.kill_reason == "abnormal_data_detected"

    connection = sqlite3.connect(result.sqlite_path)
    try:
        risk_rows = connection.execute(
            """
            SELECT reason, action, rule_name
            FROM risk_events
            WHERE run_id = ?
            ORDER BY id ASC
            """,
            (result.run_id,),
        ).fetchall()
    finally:
        connection.close()

    assert len(risk_rows) >= 1
    assert risk_rows[0][0] == "abnormal_data_detected"
    assert risk_rows[0][1] == "halt"
    assert risk_rows[0][2] == "data_anomaly"

    status_frame = load_paper_run_statuses(sqlite_path=result.sqlite_path)
    assert len(status_frame) == 1
    status_row = status_frame.iloc[0]
    assert int(status_row["risk_halt_hits"]) >= 1
    assert "abnormal_data_detected" in str(status_row["kill_reason"])


def _build_bars(
    timestamps: pd.DatetimeIndex,
    open_prices: list[float],
    close_prices: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": open_prices,
            "high": [value + 1.0 for value in open_prices],
            "low": [value - 1.0 for value in open_prices],
            "close": close_prices,
            "volume": [1_000.0 for _ in open_prices],
        },
        index=timestamps,
    )
