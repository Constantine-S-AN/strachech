"""Strategy template base class with signal recording and logging."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from stratcheck.core.strategy import OrderIntent, PortfolioState


@dataclass(slots=True)
class StrategySignal:
    """Normalized signal object emitted by StrategyTemplate subclasses."""

    side: Literal["buy", "sell"]
    qty: float
    market: bool = True
    limit_price: float | None = None
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.side not in {"buy", "sell"}:
            msg = "side must be 'buy' or 'sell'."
            raise ValueError(msg)
        if self.qty <= 0:
            msg = "qty must be positive."
            raise ValueError(msg)
        if not self.market and self.limit_price is None:
            msg = "limit_price is required when market=False."
            raise ValueError(msg)


@dataclass(slots=True)
class SignalRecord:
    """Recorded signal row for diagnostics and audit."""

    timestamp: pd.Timestamp
    strategy: str
    side: str
    qty: float
    market: bool
    limit_price: float | None
    reason: str
    metadata: dict[str, Any]


class StrategyTemplate(ABC):
    """Base class to speed up writing strategy plugins.

    Subclasses implement `build_signals(...)` and return `list[StrategySignal]`.
    The template handles:
    - input sanity checks
    - conversion to `OrderIntent`
    - signal history capture
    - structured logging per emitted signal
    """

    def __init__(
        self,
        strategy_name: str | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.strategy_name = strategy_name or type(self).__name__
        self.logger = logger or logging.getLogger(self.strategy_name)
        self._signal_records: list[SignalRecord] = []

    def signal(
        self,
        side: Literal["buy", "sell"],
        qty: float,
        reason: str = "",
        *,
        market: bool = True,
        limit_price: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StrategySignal:
        """Convenience helper to build one normalized signal object."""
        return StrategySignal(
            side=side,
            qty=float(qty),
            market=bool(market),
            limit_price=limit_price,
            reason=str(reason),
            metadata=dict(metadata or {}),
        )

    def generate_orders(
        self,
        bars: pd.DataFrame,
        portfolio_state: PortfolioState,
    ) -> list[OrderIntent]:
        """Template method used by backtest engine."""
        self._validate_inputs(bars=bars, portfolio_state=portfolio_state)
        if bars.empty:
            return []

        signals = self.build_signals(
            bars=bars,
            portfolio_state=portfolio_state,
        )
        if not signals:
            return []

        timestamp = _resolve_timestamp(bars.index[-1])
        orders: list[OrderIntent] = []
        for signal_item in signals:
            order = OrderIntent(
                side=signal_item.side,
                qty=float(signal_item.qty),
                limit_price=signal_item.limit_price,
                market=bool(signal_item.market),
            )
            orders.append(order)
            self._record_signal(timestamp=timestamp, signal_item=signal_item)
        return orders

    @abstractmethod
    def build_signals(
        self,
        bars: pd.DataFrame,
        portfolio_state: PortfolioState,
    ) -> list[StrategySignal]:
        """Generate signal objects from bars and current portfolio state."""

    def get_signal_records(self) -> list[SignalRecord]:
        """Return a copy of all emitted signal records."""
        return list(self._signal_records)

    def get_signal_frame(self) -> pd.DataFrame:
        """Return emitted signal records as a DataFrame."""
        if not self._signal_records:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "strategy",
                    "side",
                    "qty",
                    "market",
                    "limit_price",
                    "reason",
                    "metadata",
                ]
            )
        frame = pd.DataFrame(
            [
                {
                    "timestamp": row.timestamp,
                    "strategy": row.strategy,
                    "side": row.side,
                    "qty": row.qty,
                    "market": row.market,
                    "limit_price": row.limit_price,
                    "reason": row.reason,
                    "metadata": row.metadata,
                }
                for row in self._signal_records
            ]
        )
        return frame.sort_values("timestamp").reset_index(drop=True)

    def clear_signal_records(self) -> None:
        """Clear previously recorded signals."""
        self._signal_records.clear()

    def _record_signal(
        self,
        timestamp: pd.Timestamp,
        signal_item: StrategySignal,
    ) -> None:
        record = SignalRecord(
            timestamp=timestamp,
            strategy=self.strategy_name,
            side=signal_item.side,
            qty=float(signal_item.qty),
            market=bool(signal_item.market),
            limit_price=None if signal_item.limit_price is None else float(signal_item.limit_price),
            reason=str(signal_item.reason),
            metadata=dict(signal_item.metadata),
        )
        self._signal_records.append(record)
        self.logger.info(
            "signal_emitted strategy=%s time=%s side=%s qty=%.6f reason=%s",
            self.strategy_name,
            timestamp.isoformat(),
            record.side,
            record.qty,
            record.reason,
        )

    def _validate_inputs(
        self,
        bars: pd.DataFrame,
        portfolio_state: PortfolioState,
    ) -> None:
        if not isinstance(bars, pd.DataFrame):
            msg = "bars must be a pandas DataFrame."
            raise TypeError(msg)
        if not isinstance(portfolio_state, PortfolioState):
            msg = "portfolio_state must be PortfolioState."
            raise TypeError(msg)
        if "close" not in bars.columns:
            msg = "bars must include a 'close' column."
            raise ValueError(msg)
        if not isinstance(bars.index, pd.DatetimeIndex):
            msg = "bars must use DatetimeIndex."
            raise ValueError(msg)


def _resolve_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")
