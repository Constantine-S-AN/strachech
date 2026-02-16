from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from stratcheck.core.calendar import MarketCalendar, normalize_bars_freq, periods_per_year


def test_is_trading_day_handles_weekends() -> None:
    market_calendar = MarketCalendar()

    assert market_calendar.is_trading_day(pd.Timestamp("2024-03-01", tz="America/New_York"))
    assert not market_calendar.is_trading_day(pd.Timestamp("2024-03-02", tz="America/New_York"))
    assert not market_calendar.is_trading_day(pd.Timestamp("2024-03-03", tz="America/New_York"))


def test_next_trading_day_skips_weekend() -> None:
    market_calendar = MarketCalendar()
    next_day = market_calendar.next_trading_day(
        pd.Timestamp("2024-03-01", tz="America/New_York"),
    )
    assert next_day == pd.Timestamp("2024-03-04", tz="America/New_York")


@pytest.mark.xfail(reason="TODO: Add US market holiday support.")
def test_us_market_holiday_todo_marker() -> None:
    market_calendar = MarketCalendar()
    assert not market_calendar.is_trading_day(pd.Timestamp("2024-01-01", tz="America/New_York"))


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("1d", "1D"),
        ("D", "1D"),
        ("1H", "1H"),
        ("hour", "1H"),
        ("1m", "1Min"),
        ("minute", "1Min"),
    ],
)
def test_normalize_bars_freq(raw_value: str, expected: str) -> None:
    assert normalize_bars_freq(raw_value) == expected


def test_periods_per_year_matches_trading_assumptions() -> None:
    assert periods_per_year("1D") == pytest.approx(252.0)
    assert periods_per_year("1H") == pytest.approx(252.0 * 6.5)
    assert periods_per_year("1Min") == pytest.approx(252.0 * 390.0)


def test_split_rolling_windows_uses_trading_days() -> None:
    market_calendar = MarketCalendar()

    timestamps = pd.date_range("2024-01-01 21:00:00", periods=10, freq="D", tz="UTC")
    close_prices = np.linspace(100.0, 109.0, 10)
    bars = pd.DataFrame(
        {
            "open": close_prices,
            "high": close_prices + 1.0,
            "low": close_prices - 1.0,
            "close": close_prices,
            "volume": np.full(10, 1_000),
        },
        index=timestamps,
    )

    windows = list(
        market_calendar.split_rolling_windows(
            bars=bars,
            window_size="3D",
            step_size="2D",
            bars_freq="1D",
        ),
    )

    market_index = bars.index.tz_convert("America/New_York")
    filtered_index = bars.index[market_index.weekday < 5]
    assert windows[0].bars.index.tolist() == filtered_index[:3].tolist()
    assert windows[1].bars.index.tolist() == filtered_index[2:5].tolist()
    for window in windows:
        assert all(ts.weekday() < 5 for ts in window.bars.index)
