"""Market calendar and frequency normalization utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from pandas.tseries.frequencies import to_offset

BarsFrequency = str
TimeInput = str | datetime | pd.Timestamp


@dataclass(slots=True)
class CalendarWindow:
    """A rolling window slice over market bars."""

    window_index: int
    start: pd.Timestamp
    end_exclusive: pd.Timestamp
    bars: pd.DataFrame


class MarketCalendar:
    """US equities market calendar (weekends handled; holidays TODO)."""

    def __init__(self, timezone: str = "America/New_York") -> None:
        self.timezone = ZoneInfo(timezone)

    def is_trading_day(self, value: TimeInput) -> bool:
        """Return True for weekday trading sessions.

        TODO:
        - Add US market holiday support.
        """
        timestamp = self._to_market_timezone(value)
        return timestamp.weekday() < 5

    def next_trading_day(self, value: TimeInput) -> pd.Timestamp:
        """Return the next trading day strictly after the input date."""
        timestamp = self._to_market_timezone(value).normalize()
        candidate = timestamp + pd.Timedelta(days=1)
        while not self.is_trading_day(candidate):
            candidate += pd.Timedelta(days=1)
        return candidate

    def filter_bars(self, bars: pd.DataFrame, bars_freq: str) -> pd.DataFrame:
        """Filter bars to market sessions based on frequency."""
        canonical_freq = normalize_bars_freq(bars_freq)
        if not isinstance(bars.index, pd.DatetimeIndex):
            msg = "bars must use a DatetimeIndex."
            raise ValueError(msg)

        market_index = _to_market_index(bars.index, self.timezone)
        keep_mask = pd.Series(market_index.weekday < 5, index=bars.index)

        if canonical_freq == "1D":
            median_spacing = _median_spacing(bars.index)
            if median_spacing is None or median_spacing <= pd.Timedelta(days=1):
                return bars.loc[keep_mask.to_numpy()].copy()
            return bars.copy()

        minutes_from_midnight = market_index.hour * 60 + market_index.minute
        session_mask = (minutes_from_midnight >= 570) & (minutes_from_midnight < 960)
        keep_mask = keep_mask & pd.Series(session_mask, index=bars.index)
        return bars.loc[keep_mask.to_numpy()].copy()

    def split_rolling_windows(
        self,
        bars: pd.DataFrame,
        window_size: str,
        step_size: str,
        bars_freq: str,
    ):
        """Split bars into rolling windows by trading periods or offsets."""
        canonical_freq = normalize_bars_freq(bars_freq)
        trading_bars = self.filter_bars(bars=bars, bars_freq=canonical_freq)

        window_periods = _parse_trading_periods(window_size, canonical_freq)
        step_periods = _parse_trading_periods(step_size, canonical_freq)
        if window_periods is not None and step_periods is not None:
            yield from self._split_by_bar_count(
                bars=trading_bars,
                window_periods=window_periods,
                step_periods=step_periods,
            )
            return

        yield from self._split_by_offset(
            bars=trading_bars,
            window_size=window_size,
            step_size=step_size,
        )

    def _split_by_bar_count(
        self,
        bars: pd.DataFrame,
        window_periods: int,
        step_periods: int,
    ):
        if window_periods <= 0 or step_periods <= 0:
            msg = "window_size and step_size must be positive."
            raise ValueError(msg)

        window_index = 0
        start_index = 0
        total_bars = len(bars)
        while start_index < total_bars:
            end_index = start_index + window_periods
            window_bars = bars.iloc[start_index:end_index]
            if not window_bars.empty:
                if len(window_bars) > 1:
                    step_delta = window_bars.index[-1] - window_bars.index[-2]
                else:
                    step_delta = pd.Timedelta(days=1)
                end_exclusive = window_bars.index[-1] + step_delta
                yield CalendarWindow(
                    window_index=window_index,
                    start=window_bars.index[0],
                    end_exclusive=end_exclusive,
                    bars=window_bars,
                )
                window_index += 1
            start_index += step_periods

    def _split_by_offset(
        self,
        bars: pd.DataFrame,
        window_size: str,
        step_size: str,
    ):
        if bars.empty:
            return

        window_offset = _parse_offset(window_size)
        step_offset = _parse_offset(step_size)

        series_start = bars.index[0]
        series_end = bars.index[-1]

        current_start = series_start
        window_index = 0
        while current_start <= series_end:
            current_end_exclusive = current_start + window_offset
            window_bars = bars[(bars.index >= current_start) & (bars.index < current_end_exclusive)]
            if not window_bars.empty:
                yield CalendarWindow(
                    window_index=window_index,
                    start=current_start,
                    end_exclusive=current_end_exclusive,
                    bars=window_bars,
                )
                window_index += 1

            next_start = current_start + step_offset
            if next_start <= current_start:
                msg = "step_size must move time forward."
                raise ValueError(msg)
            current_start = next_start

    def _to_market_timezone(self, value: TimeInput) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        return timestamp.tz_convert(self.timezone)


def normalize_bars_freq(value: str) -> BarsFrequency:
    """Normalize bars frequency into one of: 1D, 1H, 1Min."""
    normalized = value.strip().lower()
    alias_map = {
        "1d": "1D",
        "d": "1D",
        "day": "1D",
        "daily": "1D",
        "1h": "1H",
        "h": "1H",
        "hour": "1H",
        "hourly": "1H",
        "1min": "1Min",
        "min": "1Min",
        "mins": "1Min",
        "minute": "1Min",
        "minutes": "1Min",
        "1m": "1Min",
        "1t": "1Min",
        "t": "1Min",
    }
    if normalized not in alias_map:
        msg = f"Unsupported bars frequency: {value}. Use 1D/1H/1Min."
        raise ValueError(msg)
    return alias_map[normalized]


def periods_per_year(bars_freq: str) -> float:
    """Return annualization periods for normalized bars frequency."""
    canonical_freq = normalize_bars_freq(bars_freq)
    if canonical_freq == "1D":
        return 252.0
    if canonical_freq == "1H":
        return 252.0 * 6.5
    return 252.0 * 390.0


def _parse_trading_periods(value: str, bars_freq: str) -> int | None:
    text = value.strip()
    match = re.fullmatch(r"(\d+)\s*([A-Za-z]+)?", text)
    if match is None:
        return None

    count = int(match.group(1))
    if count <= 0:
        msg = "window_size and step_size must be positive."
        raise ValueError(msg)

    raw_unit = (match.group(2) or "").lower()
    if raw_unit == "":
        return count

    allowed_units = {
        "1D": {"d", "day", "days"},
        "1H": {"h", "hour", "hours"},
        "1Min": {"m", "min", "mins", "minute", "minutes", "t"},
    }
    canonical_freq = normalize_bars_freq(bars_freq)
    if raw_unit in allowed_units[canonical_freq]:
        return count
    return None


def _parse_offset(value: str):
    normalized = value.strip()
    month_match = re.fullmatch(r"(\d+)M", normalized)
    if month_match:
        normalized = f"{month_match.group(1)}ME"

    try:
        offset = to_offset(normalized)
    except ValueError as error:
        msg = f"Invalid offset expression: {value}"
        raise ValueError(msg) from error

    probe_time = pd.Timestamp("2024-01-01", tz="UTC")
    if probe_time + offset <= probe_time:
        msg = f"Offset must move time forward: {value}"
        raise ValueError(msg)
    return offset


def _to_market_index(index: pd.DatetimeIndex, market_timezone: ZoneInfo) -> pd.DatetimeIndex:
    if index.tz is None:
        timestamp_index = index.tz_localize("UTC")
    else:
        timestamp_index = index
    return timestamp_index.tz_convert(market_timezone)


def _median_spacing(index: pd.DatetimeIndex) -> pd.Timedelta | None:
    if len(index) < 2:
        return None
    sorted_index = index.sort_values()
    differences = sorted_index[1:] - sorted_index[:-1]
    non_zero = differences[differences > pd.Timedelta(0)]
    if len(non_zero) == 0:
        return None
    return non_zero[int(len(non_zero) / 2)]
