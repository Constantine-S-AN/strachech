from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from stratcheck.core.universe import CSVUniverseProvider, run_dynamic_universe_backtest


def test_csv_universe_provider_switches_symbols_by_date(tmp_path: Path) -> None:
    csv_path = tmp_path / "universe_events.csv"
    _write_csv(
        csv_path,
        """
        date,symbol,active
        2024-01-01,AAPL,1
        2024-01-01,MSFT,1
        2024-01-05,MSFT,0
        2024-01-05,NVDA,1
        2024-01-10,AAPL,0
        """,
    )

    provider = CSVUniverseProvider(csv_path=csv_path)

    assert provider.get_universe("2023-12-31") == []
    assert provider.get_universe("2024-01-01") == ["AAPL", "MSFT"]
    assert provider.get_universe("2024-01-04") == ["AAPL", "MSFT"]
    assert provider.get_universe(pd.Timestamp("2024-01-05", tz="UTC")) == ["AAPL", "NVDA"]
    assert provider.get_universe("2024-01-10") == ["NVDA"]


def test_csv_universe_provider_history_has_size_and_rotation_fields(tmp_path: Path) -> None:
    csv_path = tmp_path / "universe_snapshots.csv"
    _write_csv(
        csv_path,
        """
        date,symbols
        2024-01-01,"AAPL,MSFT"
        2024-01-10,"AAPL,NVDA"
        2024-01-20,"AAPL,NVDA,AMZN"
        """,
    )

    provider = CSVUniverseProvider(csv_path=csv_path)
    history = provider.get_universe_history(start="2024-01-01", end="2024-01-20")

    assert list(history.columns) == ["date", "universe_size", "added", "removed", "symbols"]
    assert history["universe_size"].tolist() == [2, 2, 3]
    assert history["added"].tolist() == ["AAPL,MSFT", "NVDA", "AMZN"]
    assert history["removed"].tolist() == ["", "MSFT", ""]
    assert history["symbols"].tolist() == [
        ["AAPL", "MSFT"],
        ["AAPL", "NVDA"],
        ["AAPL", "AMZN", "NVDA"],
    ]

    filtered = provider.get_universe_history(start="2024-01-10", end="2024-01-10")
    assert len(filtered) == 1
    assert filtered.iloc[0]["added"] == "NVDA"
    assert filtered.iloc[0]["removed"] == "MSFT"


def test_dynamic_universe_backtest_updates_holdings_over_time(tmp_path: Path) -> None:
    csv_path = tmp_path / "universe_events.csv"
    _write_csv(
        csv_path,
        """
        date,symbol,active
        2024-01-01,AAPL,1
        2024-01-01,MSFT,1
        2024-01-03,MSFT,0
        2024-01-03,NVDA,1
        """,
    )
    provider = CSVUniverseProvider(csv_path=csv_path)

    timestamps = pd.to_datetime(
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        utc=True,
    )
    close_prices = pd.DataFrame(
        {
            "AAPL": [100.0, 110.0, 120.0, 132.0],
            "MSFT": [100.0, 90.0, 80.0, 70.0],
            "NVDA": [100.0, 100.0, 120.0, 108.0],
        },
        index=timestamps,
    )

    result = run_dynamic_universe_backtest(
        close_prices=close_prices,
        universe_provider=provider,
        initial_cash=100_000.0,
    )

    expected_second_period_return = (((120.0 / 110.0) - 1.0) + ((80.0 / 90.0) - 1.0)) / 2.0
    expected_equity_day3 = 100_000.0 * (1.0 + expected_second_period_return)

    assert result.returns.iloc[0] == pytest.approx(0.0)
    assert result.returns.iloc[1] == pytest.approx(0.0)
    assert result.returns.iloc[2] == pytest.approx(expected_second_period_return)
    assert result.returns.iloc[3] == pytest.approx(0.0)

    assert result.equity_curve.iloc[0] == pytest.approx(100_000.0)
    assert result.equity_curve.iloc[2] == pytest.approx(expected_equity_day3)
    assert result.equity_curve.iloc[3] == pytest.approx(expected_equity_day3)

    assert result.universe_history["universe_size"].tolist() == [2, 2]
    assert result.universe_history["added"].tolist() == ["AAPL,MSFT", "NVDA"]
    assert result.universe_history["removed"].tolist() == ["", "MSFT"]


def _write_csv(path: Path, content: str) -> None:
    lines = [line.strip() for line in content.strip().splitlines()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
