"""Stooq CSV provider with local cache support."""

from __future__ import annotations

from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd

from stratcheck.core.data import BarsSchema, TimeInput


class StooqCSVProvider:
    """Load daily bars from Stooq CSV endpoint with on-disk caching.

    Price adjustment note:
    - `adjust_prices=True` is currently not supported and will raise
      `NotImplementedError`.
    - Returned bars are the raw values from the source CSV.
    """

    def __init__(
        self,
        cache_dir: str | Path = "data/cache",
        auto_download: bool = True,
        adjust_prices: bool = False,
        default_market: str = "us",
        request_timeout_sec: float = 20.0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.auto_download = auto_download
        self.adjust_prices = adjust_prices
        self.default_market = default_market.lower().strip()
        self.request_timeout_sec = request_timeout_sec
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_bars(
        self,
        symbol: str,
        start: TimeInput = None,
        end: TimeInput = None,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        """Fetch daily bars for a symbol from cache or Stooq."""
        interval_code = _normalize_timeframe(timeframe)
        if self.adjust_prices:
            msg = (
                "Price adjustment for split/dividend is not supported yet. Use adjust_prices=False."
            )
            raise NotImplementedError(msg)

        source_symbol = self._normalize_symbol(symbol)
        cache_path = self._cache_path(source_symbol=source_symbol, interval_code=interval_code)

        if not cache_path.exists():
            if not self.auto_download:
                msg = f"Cached file not found and auto_download is disabled: {cache_path}"
                raise FileNotFoundError(msg)
            csv_text = self._download_csv_text(
                source_symbol=source_symbol,
                interval_code=interval_code,
            )
            cache_path.write_text(csv_text, encoding="utf-8")

        raw_bars = pd.read_csv(cache_path)
        normalized_bars = _normalize_stooq_columns(raw_bars)
        standardized_bars = BarsSchema.normalize(normalized_bars)
        return BarsSchema.slice_range(
            bars=standardized_bars,
            start=start,
            end=end,
        )

    def _normalize_symbol(self, symbol: str) -> str:
        raw_symbol = symbol.strip().lower()
        if not raw_symbol:
            msg = "symbol cannot be empty."
            raise ValueError(msg)
        if "." in raw_symbol:
            return raw_symbol
        return f"{raw_symbol}.{self.default_market}"

    def _cache_path(self, source_symbol: str, interval_code: str) -> Path:
        safe_symbol = source_symbol.replace(".", "_")
        filename = f"{safe_symbol}_{interval_code}.csv"
        return self.cache_dir / filename

    def _download_csv_text(self, source_symbol: str, interval_code: str) -> str:
        url = f"https://stooq.com/q/d/l/?s={source_symbol}&i={interval_code}"
        request = Request(url=url, headers={"User-Agent": "stratcheck/0.1"})

        try:
            with urlopen(request, timeout=self.request_timeout_sec) as response:
                csv_text = response.read().decode("utf-8")
        except URLError as error:
            msg = f"Failed to download bars for {source_symbol} from Stooq."
            raise RuntimeError(msg) from error

        if not csv_text.strip():
            msg = f"Received empty CSV response for {source_symbol}."
            raise RuntimeError(msg)
        if "No data" in csv_text:
            msg = f"No data returned by Stooq for {source_symbol}."
            raise RuntimeError(msg)
        return csv_text


def _normalize_timeframe(timeframe: str) -> str:
    normalized = timeframe.lower().strip()
    if normalized in {"1d", "d", "daily", "1day"}:
        return "d"
    msg = "StooqCSVProvider currently supports daily timeframe only."
    raise ValueError(msg)


def _normalize_stooq_columns(raw_bars: pd.DataFrame) -> pd.DataFrame:
    renamed_bars = raw_bars.rename(
        columns={
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    return renamed_bars
