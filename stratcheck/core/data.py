"""Market data generation and provider utilities."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from stratcheck.core.corporate_actions import (
    CorporateAction,
    apply_corporate_actions_to_bars,
    filter_corporate_actions,
    load_corporate_actions_file,
    summarize_corporate_actions,
)

TimeInput = str | datetime | pd.Timestamp | None


class BarsSchema:
    """Canonical bars schema used by all data providers."""

    COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
    DATETIME_COLUMNS: tuple[str, ...] = ("timestamp", "datetime", "date", "time")
    INDEX_NAME = "timestamp"
    DEFAULT_TIMEZONE = "UTC"

    @classmethod
    def normalize(
        cls,
        bars: pd.DataFrame,
    ) -> pd.DataFrame:
        """Normalize bars to required columns and UTC datetime index."""
        normalized_bars = bars.copy()
        normalized_bars.columns = [
            str(column).strip().lower() for column in normalized_bars.columns
        ]

        datetime_index = cls._extract_datetime_index(bars=normalized_bars)
        normalized_bars.index = datetime_index
        normalized_bars.index.name = cls.INDEX_NAME

        missing_columns = [
            column for column in cls.COLUMNS if column not in normalized_bars.columns
        ]
        if missing_columns:
            columns_text = ", ".join(missing_columns)
            msg = f"Bars data is missing required columns: {columns_text}"
            raise ValueError(msg)

        normalized_bars = normalized_bars.loc[:, list(cls.COLUMNS)]
        for column in cls.COLUMNS:
            normalized_bars[column] = pd.to_numeric(normalized_bars[column], errors="coerce")

        if normalized_bars[list(cls.COLUMNS)].isna().any().any():
            msg = "Bars data contains invalid OHLCV values."
            raise ValueError(msg)

        normalized_bars = normalized_bars[~normalized_bars.index.duplicated(keep="last")]
        normalized_bars = normalized_bars.sort_index()
        return normalized_bars

    @classmethod
    def slice_range(
        cls,
        bars: pd.DataFrame,
        start: TimeInput = None,
        end: TimeInput = None,
    ) -> pd.DataFrame:
        """Slice bars with inclusive start/end bounds."""
        start_time = cls.parse_time(start)
        end_time = cls.parse_time(end)

        filtered_bars = bars
        if start_time is not None:
            filtered_bars = filtered_bars[filtered_bars.index >= start_time]
        if end_time is not None:
            filtered_bars = filtered_bars[filtered_bars.index <= end_time]
        return filtered_bars

    @classmethod
    def parse_time(cls, value: TimeInput) -> pd.Timestamp | None:
        """Convert user input to UTC-aware pandas Timestamp."""
        if value is None:
            return None

        parsed_time = pd.Timestamp(value)
        if parsed_time.tzinfo is None:
            return parsed_time.tz_localize(cls.DEFAULT_TIMEZONE)
        return parsed_time.tz_convert(cls.DEFAULT_TIMEZONE)

    @classmethod
    def _extract_datetime_index(
        cls,
        bars: pd.DataFrame,
    ) -> pd.DatetimeIndex:
        if isinstance(bars.index, pd.DatetimeIndex):
            datetime_values = bars.index
        else:
            datetime_column = cls._find_datetime_column(bars)
            if datetime_column is None:
                msg = (
                    "Bars data must provide a DatetimeIndex or one of these "
                    f"datetime columns: {', '.join(cls.DATETIME_COLUMNS)}"
                )
                raise ValueError(msg)
            datetime_values = bars.pop(datetime_column)

        parsed_values = pd.to_datetime(datetime_values, errors="coerce", format="mixed", utc=True)
        datetime_index = pd.DatetimeIndex(parsed_values)

        if datetime_index.isna().any():
            msg = "Bars data contains invalid datetime values."
            raise ValueError(msg)
        return datetime_index.tz_convert(cls.DEFAULT_TIMEZONE)

    @classmethod
    def _find_datetime_column(cls, bars: pd.DataFrame) -> str | None:
        for column in cls.DATETIME_COLUMNS:
            if column in bars.columns:
                return column
        return None


@runtime_checkable
class DataProvider(Protocol):
    """Abstract data provider interface."""

    def get_bars(
        self,
        symbol: str,
        start: TimeInput = None,
        end: TimeInput = None,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        """Fetch bars for a symbol within a time range."""


class CSVDataProvider:
    """Read market bars from `{data_dir}/{symbol}.csv`."""

    def __init__(
        self,
        data_dir: str | Path,
        actions_dir: str | Path | None = None,
        apply_corporate_actions: bool = True,
        use_parquet_cache: bool = False,
        cache_dir: str | Path | None = None,
        cache_index_path: str | Path | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.actions_dir = (
            Path(actions_dir) if actions_dir is not None else self.data_dir / "actions"
        )
        self.apply_corporate_actions = bool(apply_corporate_actions)
        self.use_parquet_cache = bool(use_parquet_cache)
        self.cache_dir = (
            Path(cache_dir) if cache_dir is not None else self.data_dir / "cache" / "parquet"
        )
        self.cache_index_path = (
            Path(cache_index_path)
            if cache_index_path is not None
            else self.cache_dir / "index.json"
        )
        self._actions_cache: dict[str, list[CorporateAction]] = {}
        self._cache_meta: dict[str, dict[str, object]] = {}

    def get_bars(
        self,
        symbol: str,
        start: TimeInput = None,
        end: TimeInput = None,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        normalized_timeframe = timeframe.lower()
        if normalized_timeframe not in {"1d", "d", "daily", "1day"}:
            msg = "CSVDataProvider currently supports daily bars only."
            raise ValueError(msg)

        raw_bars = self._load_raw_bars(symbol=symbol)
        standardized_bars = BarsSchema.normalize(raw_bars)
        if self.apply_corporate_actions:
            actions = self.get_corporate_actions(symbol=symbol)
            standardized_bars = apply_corporate_actions_to_bars(
                bars=standardized_bars,
                actions=actions,
            )
        return BarsSchema.slice_range(
            bars=standardized_bars,
            start=start,
            end=end,
        )

    def get_cache_meta(self, symbol: str) -> dict[str, object] | None:
        """Return latest parquet-cache metadata for a symbol."""
        return self._cache_meta.get(symbol.strip())

    def get_corporate_actions(
        self,
        symbol: str,
        start: TimeInput = None,
        end: TimeInput = None,
    ) -> list[CorporateAction]:
        """Load corporate actions from `actions/<symbol>.csv`."""
        symbol_text = symbol.strip()
        if symbol_text not in self._actions_cache:
            actions_path = self.actions_dir / f"{symbol_text}.csv"
            self._actions_cache[symbol_text] = load_corporate_actions_file(actions_path)
        all_actions = self._actions_cache[symbol_text]
        return filter_corporate_actions(
            actions=all_actions,
            start=start,
            end=end,
        )

    def get_corporate_actions_summary(
        self,
        symbol: str,
        start: TimeInput = None,
        end: TimeInput = None,
    ) -> list[dict[str, str]]:
        """Summarize action rows for report display."""
        actions = self.get_corporate_actions(
            symbol=symbol,
            start=start,
            end=end,
        )
        return summarize_corporate_actions(actions)

    def _load_raw_bars(self, symbol: str) -> pd.DataFrame:
        symbol_text = symbol.strip()
        direct_parquet_path = self.data_dir / f"{symbol_text}.parquet"
        if direct_parquet_path.exists():
            self._cache_meta[symbol_text] = {
                "cache_used": True,
                "cache_hit": True,
                "status": "direct_parquet",
                "cache_file": str(direct_parquet_path),
            }
            return pd.read_parquet(direct_parquet_path)

        csv_path = self.data_dir / f"{symbol_text}.csv"
        if not csv_path.exists():
            msg = f"CSV file not found: {csv_path}"
            raise FileNotFoundError(msg)

        if not self.use_parquet_cache:
            self._cache_meta[symbol_text] = {
                "cache_used": False,
                "cache_hit": False,
                "status": "csv",
                "source_file": str(csv_path),
            }
            return pd.read_csv(csv_path)

        from stratcheck.perf import load_or_build_parquet_cache

        raw_bars, cache_meta = load_or_build_parquet_cache(
            symbol=symbol_text,
            csv_path=csv_path,
            cache_dir=self.cache_dir,
            index_path=self.cache_index_path,
        )
        self._cache_meta[symbol_text] = cache_meta
        return raw_bars


def generate_random_ohlcv(periods: int = 400, seed: int = 42) -> pd.DataFrame:
    """Generate random OHLCV data for demo and tests."""
    if periods < 2:
        msg = "periods must be at least 2."
        raise ValueError(msg)

    random_state = np.random.default_rng(seed)
    timestamps = pd.date_range(end=pd.Timestamp.utcnow().floor("D"), periods=periods, freq="D")

    log_returns = random_state.normal(loc=0.0006, scale=0.02, size=periods)
    close_prices = 100.0 * np.exp(np.cumsum(log_returns))

    open_noise = random_state.normal(loc=0.0, scale=0.003, size=periods)
    previous_close = np.concatenate(([close_prices[0]], close_prices[:-1]))
    open_prices = previous_close * (1.0 + open_noise)

    high_spread = random_state.uniform(0.001, 0.02, size=periods)
    low_spread = random_state.uniform(0.001, 0.02, size=periods)
    high_prices = np.maximum(open_prices, close_prices) * (1.0 + high_spread)
    low_prices = np.minimum(open_prices, close_prices) * (1.0 - low_spread)
    low_prices = np.maximum(low_prices, 0.01)

    volumes = random_state.integers(500_000, 5_000_000, size=periods)

    market_data = pd.DataFrame(
        {
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volumes,
        },
        index=timestamps,
    )
    return BarsSchema.normalize(market_data)
