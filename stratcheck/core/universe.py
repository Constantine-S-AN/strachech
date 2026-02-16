"""Historical universe provider and dynamic-universe portfolio backtest helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd

TimeInput = str | datetime | pd.Timestamp | None
_DEFAULT_TIMEZONE = "UTC"


@runtime_checkable
class UniverseProvider(Protocol):
    """Historical universe provider interface."""

    def get_universe(self, as_of: TimeInput) -> list[str]:
        """Return active symbols at `as_of`."""

    def get_universe_history(
        self,
        start: TimeInput = None,
        end: TimeInput = None,
    ) -> pd.DataFrame:
        """Return history rows with universe size and add/remove lists."""


@dataclass(slots=True)
class UniverseBacktestResult:
    """Result of dynamic-universe portfolio backtest."""

    equity_curve: pd.Series
    returns: pd.Series
    universe_history: pd.DataFrame


class CSVUniverseProvider:
    """CSV-backed historical universe provider.

    Supported CSV formats:
    1) Event format:
       - columns: `date`, `symbol`, `active`
       - each row toggles symbol membership from that date onward
    2) Snapshot format:
       - columns: `date`, `symbols`
       - `symbols` can be comma/semicolon/pipe separated list
    """

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            msg = f"Universe CSV file not found: {self.csv_path}"
            raise FileNotFoundError(msg)

        raw_frame = pd.read_csv(self.csv_path)
        self._events = _parse_universe_events(raw_frame)
        self._history_frame = _build_universe_history(self._events)

    def get_universe(self, as_of: TimeInput) -> list[str]:
        """Return sorted active symbols at given timestamp."""
        timestamp = _parse_time(as_of)
        if timestamp is None:
            msg = "`as_of` cannot be None."
            raise ValueError(msg)

        history = self._history_frame[self._history_frame["date"] <= timestamp]
        if history.empty:
            return []
        symbols_value = history.iloc[-1]["symbols"]
        if not isinstance(symbols_value, list):
            return []
        return sorted(str(symbol).upper() for symbol in symbols_value)

    def get_universe_history(
        self,
        start: TimeInput = None,
        end: TimeInput = None,
    ) -> pd.DataFrame:
        """Return universe history with size and add/remove symbols."""
        start_time = _parse_time(start)
        end_time = _parse_time(end)
        history = self._history_frame.copy()
        if start_time is not None:
            history = history[history["date"] >= start_time]
        if end_time is not None:
            history = history[history["date"] <= end_time]
        return history.reset_index(drop=True)


def run_dynamic_universe_backtest(
    close_prices: pd.DataFrame,
    universe_provider: UniverseProvider,
    initial_cash: float = 100_000.0,
) -> UniverseBacktestResult:
    """Run equal-weight portfolio backtest with date-dependent universe.

    Assumptions:
    - Rebalance at each timestamp close into equal weights of current universe.
    - Return from `t-1` to `t` is computed using universe active at `t-1`.
    """
    if initial_cash <= 0:
        msg = "initial_cash must be positive."
        raise ValueError(msg)
    if close_prices.empty:
        msg = "close_prices cannot be empty."
        raise ValueError(msg)
    if not isinstance(close_prices.index, pd.DatetimeIndex):
        msg = "close_prices must use DatetimeIndex."
        raise ValueError(msg)

    prices = close_prices.sort_index().copy()
    prices = prices[~prices.index.duplicated(keep="last")]
    prices.columns = [str(column).upper() for column in prices.columns]
    for column_name in prices.columns:
        prices[column_name] = pd.to_numeric(prices[column_name], errors="coerce")

    index = prices.index
    equity_values = [float(initial_cash)]
    return_values = [0.0]

    for row_index in range(1, len(index)):
        previous_time = index[row_index - 1]
        current_time = index[row_index]

        previous_universe = universe_provider.get_universe(previous_time)
        period_return = _calculate_equal_weight_return(
            prices=prices,
            previous_time=previous_time,
            current_time=current_time,
            symbols=previous_universe,
        )
        current_equity = equity_values[-1] * (1.0 + period_return)
        equity_values.append(float(current_equity))
        return_values.append(float(period_return))

    equity_curve = pd.Series(
        data=equity_values,
        index=index,
        name="equity",
        dtype=float,
    )
    returns = pd.Series(
        data=return_values,
        index=index,
        name="returns",
        dtype=float,
    )
    universe_history = universe_provider.get_universe_history(
        start=index[0],
        end=index[-1],
    )
    return UniverseBacktestResult(
        equity_curve=equity_curve,
        returns=returns,
        universe_history=universe_history,
    )


def _parse_universe_events(raw_frame: pd.DataFrame) -> pd.DataFrame:
    if raw_frame.empty:
        msg = "Universe CSV cannot be empty."
        raise ValueError(msg)

    frame = raw_frame.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]

    if {"date", "symbol", "active"}.issubset(frame.columns):
        return _parse_event_rows(frame)
    if {"date", "symbols"}.issubset(frame.columns):
        return _parse_snapshot_rows(frame)

    msg = "Universe CSV must contain either columns `date,symbol,active` or `date,symbols`."
    raise ValueError(msg)


def _parse_event_rows(frame: pd.DataFrame) -> pd.DataFrame:
    events: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        timestamp = _parse_time(row.get("date"))
        if timestamp is None:
            msg = "Universe row date cannot be empty."
            raise ValueError(msg)
        symbol_text = str(row.get("symbol", "")).strip().upper()
        if not symbol_text:
            msg = "Universe row symbol cannot be empty."
            raise ValueError(msg)
        events.append(
            {
                "date": timestamp,
                "symbol": symbol_text,
                "active": _parse_active_flag(row.get("active")),
            }
        )

    events_frame = pd.DataFrame(events)
    events_frame = events_frame.sort_values(["date", "symbol"]).reset_index(drop=True)
    return events_frame


def _parse_snapshot_rows(frame: pd.DataFrame) -> pd.DataFrame:
    snapshot_frame = frame.copy()
    snapshot_frame["date"] = snapshot_frame["date"].map(_parse_time)
    if snapshot_frame["date"].isna().any():
        msg = "Universe snapshot date contains invalid value."
        raise ValueError(msg)
    snapshot_frame = snapshot_frame.sort_values("date").reset_index(drop=True)

    previous_symbols: set[str] = set()
    events: list[dict[str, Any]] = []
    for row in snapshot_frame.to_dict(orient="records"):
        current_symbols = set(_parse_symbols_list(row.get("symbols")))
        added_symbols = sorted(current_symbols - previous_symbols)
        removed_symbols = sorted(previous_symbols - current_symbols)
        for symbol in added_symbols:
            events.append({"date": row["date"], "symbol": symbol, "active": True})
        for symbol in removed_symbols:
            events.append({"date": row["date"], "symbol": symbol, "active": False})
        previous_symbols = current_symbols

    if not events:
        msg = "Universe snapshot rows produced no events."
        raise ValueError(msg)
    return pd.DataFrame(events).sort_values(["date", "symbol"]).reset_index(drop=True)


def _build_universe_history(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["date", "universe_size", "added", "removed", "symbols"])

    ordered = events.sort_values(["date", "symbol"]).reset_index(drop=True)
    active_symbols: set[str] = set()
    rows: list[dict[str, Any]] = []
    for event_time, group in ordered.groupby("date", sort=True):
        added_symbols: list[str] = []
        removed_symbols: list[str] = []
        for row in group.to_dict(orient="records"):
            symbol = str(row["symbol"]).upper()
            is_active = bool(row["active"])
            if is_active and symbol not in active_symbols:
                active_symbols.add(symbol)
                added_symbols.append(symbol)
            elif not is_active and symbol in active_symbols:
                active_symbols.remove(symbol)
                removed_symbols.append(symbol)

        sorted_symbols = sorted(active_symbols)
        rows.append(
            {
                "date": event_time,
                "universe_size": int(len(sorted_symbols)),
                "added": ",".join(sorted(added_symbols)),
                "removed": ",".join(sorted(removed_symbols)),
                "symbols": sorted_symbols,
            }
        )

    history = pd.DataFrame(rows)
    history = history.sort_values("date").reset_index(drop=True)
    return history


def _calculate_equal_weight_return(
    prices: pd.DataFrame,
    previous_time: pd.Timestamp,
    current_time: pd.Timestamp,
    symbols: list[str],
) -> float:
    if not symbols:
        return 0.0

    valid_returns: list[float] = []
    for symbol in symbols:
        if symbol not in prices.columns:
            continue
        previous_price = prices.at[previous_time, symbol]
        current_price = prices.at[current_time, symbol]
        if pd.isna(previous_price) or pd.isna(current_price):
            continue
        previous_price_value = float(previous_price)
        current_price_value = float(current_price)
        if previous_price_value <= 0:
            continue
        valid_returns.append((current_price_value / previous_price_value) - 1.0)

    if not valid_returns:
        return 0.0
    return float(sum(valid_returns) / len(valid_returns))


def _parse_active_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "add", "in"}:
        return True
    if text in {"0", "false", "no", "n", "remove", "out"}:
        return False
    msg = f"Invalid active flag value: {value!r}"
    raise ValueError(msg)


def _parse_symbols_list(raw_value: Any) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, float) and pd.isna(raw_value):
        return []
    text = str(raw_value).strip()
    if not text:
        return []
    for separator in ["|", ";", " "]:
        text = text.replace(separator, ",")
    symbols = [token.strip().upper() for token in text.split(",") if token.strip()]
    return sorted(set(symbols))


def _parse_time(value: TimeInput) -> pd.Timestamp | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(_DEFAULT_TIMEZONE)
    return timestamp.tz_convert(_DEFAULT_TIMEZONE)
