from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
from stratcheck.audit import RunAuditStore
from stratcheck.connectors import LivePaperRunner, PaperBrokerConnector
from stratcheck.core.strategy import OrderIntent, PortfolioState
from stratcheck.ops import KillSwitchConfig


def test_kill_switch_triggers_on_max_drawdown(tmp_path: Path) -> None:
    connector = PaperBrokerConnector(
        initial_cash=10_000.0,
        max_fill_ratio_per_step=1.0,
        max_volume_share=1.0,
    )
    runner = LivePaperRunner(
        connector=connector,
        sqlite_path=tmp_path / "drawdown.sqlite",
        symbol="AAPL",
        kill_switch_config=KillSwitchConfig(
            max_consecutive_errors=3,
            max_drawdown=0.20,
            max_data_gap_steps=2,
        ),
    )

    class BuyOnceStrategy:
        def generate_orders(
            self,
            bars: pd.DataFrame,
            portfolio_state: PortfolioState,
        ) -> list[OrderIntent]:
            del portfolio_state
            if len(bars) == 1:
                return [OrderIntent(side="buy", qty=100.0, market=True)]
            return []

    bars = _build_bars(
        timestamps=pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC"),
        open_prices=[100.0, 100.0, 60.0, 55.0, 50.0],
        close_prices=[100.0, 100.0, 60.0, 55.0, 50.0],
    )

    result = runner.run(strategy=BuyOnceStrategy(), bars=bars)

    assert result.status == "killed"
    assert result.kill_reason == "max_drawdown_exceeded"
    assert result.kill_triggered is True
    assert result.processed_bars < len(bars)

    audit_store = RunAuditStore(result.sqlite_path)
    run_record = audit_store.get_run(result.run_id)
    assert run_record.status == "killed"
    assert "max_drawdown_exceeded" in run_record.note


def test_kill_switch_triggers_on_consecutive_errors(tmp_path: Path) -> None:
    connector = PaperBrokerConnector(
        initial_cash=10_000.0,
        max_fill_ratio_per_step=1.0,
        max_volume_share=1.0,
    )
    runner = LivePaperRunner(
        connector=connector,
        sqlite_path=tmp_path / "errors.sqlite",
        symbol="AAPL",
        kill_switch_config=KillSwitchConfig(
            max_consecutive_errors=2,
            max_drawdown=0.90,
            max_data_gap_steps=2,
        ),
    )

    class AlwaysErrorStrategy:
        def generate_orders(
            self,
            bars: pd.DataFrame,
            portfolio_state: PortfolioState,
        ) -> list[OrderIntent]:
            del bars, portfolio_state
            msg = "strategy boom"
            raise RuntimeError(msg)

    bars = _build_bars(
        timestamps=pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC"),
        open_prices=[100.0, 101.0, 102.0, 103.0, 104.0],
        close_prices=[100.0, 101.0, 102.0, 103.0, 104.0],
    )

    result = runner.run(strategy=AlwaysErrorStrategy(), bars=bars)

    assert result.status == "killed"
    assert result.kill_reason == "consecutive_errors_exceeded"
    assert result.processed_bars == 2

    with result.metrics_csv_path.open("r", encoding="utf-8") as metrics_file:
        metrics_rows = list(csv.DictReader(metrics_file))
    assert len(metrics_rows) == 1
    assert float(metrics_rows[0]["error_rate"]) > 0.0


def test_kill_switch_triggers_on_data_interruption(tmp_path: Path) -> None:
    connector = PaperBrokerConnector(
        initial_cash=10_000.0,
        max_fill_ratio_per_step=1.0,
        max_volume_share=1.0,
    )
    runner = LivePaperRunner(
        connector=connector,
        sqlite_path=tmp_path / "data_gap.sqlite",
        symbol="AAPL",
        kill_switch_config=KillSwitchConfig(
            max_consecutive_errors=3,
            max_drawdown=0.90,
            max_data_gap_steps=2,
        ),
    )

    class NoopStrategy:
        def generate_orders(
            self,
            bars: pd.DataFrame,
            portfolio_state: PortfolioState,
        ) -> list[OrderIntent]:
            del bars, portfolio_state
            return []

    bars = _build_bars(
        timestamps=pd.to_datetime(
            ["2024-01-01", "2024-01-02", "2024-01-10", "2024-01-11"],
            utc=True,
        ),
        open_prices=[100.0, 100.0, 100.0, 100.0],
        close_prices=[100.0, 100.0, 100.0, 100.0],
    )

    result = runner.run(strategy=NoopStrategy(), bars=bars)

    assert result.status == "killed"
    assert result.kill_reason == "data_interruption_detected"
    assert result.processed_bars == 2


def _build_bars(
    timestamps: pd.DatetimeIndex,
    open_prices: list[float],
    close_prices: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": open_prices,
            "high": [price + 1.0 for price in open_prices],
            "low": [price - 1.0 for price in open_prices],
            "close": close_prices,
            "volume": [1_000.0 for _ in open_prices],
        },
        index=timestamps,
    )
