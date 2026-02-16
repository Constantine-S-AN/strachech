from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
from stratcheck.audit import RunAuditStore
from stratcheck.connectors import LivePaperRunner, PaperBrokerConnector
from stratcheck.core.strategy import OrderIntent, PortfolioState


def test_paper_order_lifecycle_new_to_partial_to_filled() -> None:
    connector = PaperBrokerConnector(
        initial_cash=10_000.0,
        max_fill_ratio_per_step=0.5,
        max_volume_share=1.0,
    )

    placed_order = connector.place(symbol="AAPL", side="buy", qty=10.0, market=True)
    updates_round1 = list(connector.stream_updates())

    assert placed_order.status == "new"
    assert [update.status for update in updates_round1] == ["new"]

    connector.step_market(
        symbol="AAPL",
        bar=pd.Series({"open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "volume": 1000}),
        timestamp=pd.Timestamp("2024-01-02", tz="UTC"),
    )
    updates_round2 = list(connector.stream_updates())
    assert [update.status for update in updates_round2] == ["partially_filled"]
    assert updates_round2[0].fill_qty == 5.0
    assert updates_round2[0].filled_qty == 5.0

    connector.step_market(
        symbol="AAPL",
        bar=pd.Series({"open": 101.0, "high": 103.0, "low": 100.0, "close": 102.0, "volume": 1000}),
        timestamp=pd.Timestamp("2024-01-03", tz="UTC"),
    )
    updates_round3 = list(connector.stream_updates())
    assert [update.status for update in updates_round3] == ["filled"]
    assert updates_round3[0].fill_qty == 5.0
    assert updates_round3[0].remaining_qty == 0.0

    orders = connector.get_orders()
    assert len(orders) == 1
    assert orders[0].status == "filled"
    assert orders[0].filled_qty == 10.0

    positions = connector.get_positions()
    assert positions["AAPL"].qty == 10.0


def test_paper_order_lifecycle_new_to_partial_to_canceled() -> None:
    connector = PaperBrokerConnector(
        initial_cash=10_000.0,
        max_fill_ratio_per_step=0.5,
        max_volume_share=1.0,
    )

    placed_order = connector.place(
        symbol="AAPL",
        side="buy",
        qty=10.0,
        market=False,
        limit_price=100.0,
    )
    updates_round1 = list(connector.stream_updates())
    assert [update.status for update in updates_round1] == ["new"]

    connector.step_market(
        symbol="AAPL",
        bar=pd.Series({"open": 101.0, "high": 102.0, "low": 95.0, "close": 100.0, "volume": 1000}),
        timestamp=pd.Timestamp("2024-01-02", tz="UTC"),
    )
    updates_round2 = list(connector.stream_updates())
    assert [update.status for update in updates_round2] == ["partially_filled"]
    assert updates_round2[0].filled_qty == 5.0

    canceled = connector.cancel(order_id=placed_order.order_id)
    updates_round3 = list(connector.stream_updates())
    assert canceled.status == "canceled"
    assert [update.status for update in updates_round3] == ["canceled"]
    assert updates_round3[0].filled_qty == 5.0
    assert updates_round3[0].remaining_qty == 5.0

    connector.step_market(
        symbol="AAPL",
        bar=pd.Series({"open": 99.0, "high": 100.0, "low": 90.0, "close": 95.0, "volume": 1000}),
        timestamp=pd.Timestamp("2024-01-03", tz="UTC"),
    )
    assert list(connector.stream_updates()) == []

    orders = connector.get_orders()
    assert orders[0].status == "canceled"
    assert orders[0].filled_qty == 5.0


def test_live_paper_runner_persists_order_updates_to_sqlite(tmp_path: Path) -> None:
    connector = PaperBrokerConnector(
        initial_cash=10_000.0,
        max_fill_ratio_per_step=0.5,
        max_volume_share=1.0,
    )
    runner = LivePaperRunner(
        connector=connector,
        sqlite_path=tmp_path / "paper.sqlite",
        symbol="AAPL",
    )

    timestamps = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    bars = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [1000.0, 1000.0, 1000.0, 1000.0],
        },
        index=timestamps,
    )

    class OneShotBuyStrategy:
        def generate_orders(
            self,
            bars: pd.DataFrame,
            portfolio_state: PortfolioState,
        ) -> list[OrderIntent]:
            del portfolio_state
            if len(bars) == 1:
                return [OrderIntent(side="buy", qty=10.0, market=True)]
            return []

    result = runner.run(strategy=OneShotBuyStrategy(), bars=bars)
    assert result.run_id
    assert result.status == "completed"
    assert result.kill_reason is None
    assert result.orders_placed == 1
    assert result.updates_written >= 3
    assert result.sqlite_path.exists()
    assert result.log_path.exists()
    assert result.metrics_prom_path.exists()
    assert result.metrics_csv_path.exists()

    connection = sqlite3.connect(result.sqlite_path)
    try:
        order_statuses = [
            row[0]
            for row in connection.execute(
                "SELECT status FROM paper_order_updates ORDER BY id ASC"
            ).fetchall()
        ]
        assert order_statuses[:3] == ["new", "partially_filled", "filled"]

        final_order_status = connection.execute(
            "SELECT status FROM paper_orders ORDER BY order_id ASC LIMIT 1"
        ).fetchone()
        assert final_order_status is not None
        assert final_order_status[0] == "filled"
    finally:
        connection.close()

    audit_store = RunAuditStore(result.sqlite_path)
    run_record = audit_store.get_run(result.run_id)
    assert run_record.status == "completed"
    assert run_record.symbol == "AAPL"

    counts = audit_store.get_counts(result.run_id)
    assert counts["configs"] == 1
    assert counts["signals"] >= 1
    assert counts["orders"] >= 3
    assert counts["fills"] >= 2
    assert counts["snapshots"] == len(bars)
