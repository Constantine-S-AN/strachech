"""Rebalance planning utilities from current positions to target weights."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import pandas as pd


@dataclass(slots=True)
class TradeInstruction:
    """Trade instruction required to reach target portfolio weights."""

    symbol: str
    side: Literal["buy", "sell"]
    quantity: float
    price: float
    current_weight: float
    target_weight: float
    trade_notional: float


class RebalancePlanner:
    """Generate rebalance trades from current positions and target weights."""

    def __init__(
        self,
        min_trade_notional: float = 0.0,
        min_trade_quantity: float = 0.0,
    ) -> None:
        if min_trade_notional < 0 or min_trade_quantity < 0:
            msg = "min_trade_notional and min_trade_quantity must be non-negative."
            raise ValueError(msg)
        self.min_trade_notional = float(min_trade_notional)
        self.min_trade_quantity = float(min_trade_quantity)

    def plan_rebalance(
        self,
        current_positions: pd.Series | Mapping[str, float],
        target_weights: pd.Series | Mapping[str, float],
        prices: pd.Series | Mapping[str, float],
        portfolio_value: float | None = None,
    ) -> list[TradeInstruction]:
        """Return buy/sell instructions to move from current to target weights."""
        position_series = _to_numeric_series(current_positions, name="current_positions")
        target_series = _to_numeric_series(target_weights, name="target_weights")
        price_series = _to_numeric_series(prices, name="prices")

        all_symbols = sorted(set(position_series.index) | set(target_series.index))
        position_aligned = position_series.reindex(all_symbols, fill_value=0.0)
        target_aligned = target_series.reindex(all_symbols, fill_value=0.0)
        price_aligned = price_series.reindex(all_symbols)

        if price_aligned.isna().any():
            missing = sorted(price_aligned[price_aligned.isna()].index.tolist())
            msg = f"Missing prices for symbols: {missing}"
            raise ValueError(msg)

        if (price_aligned <= 0).any():
            bad_symbols = sorted(price_aligned[price_aligned <= 0].index.tolist())
            msg = f"Prices must be positive for symbols: {bad_symbols}"
            raise ValueError(msg)

        if portfolio_value is None:
            portfolio_value = float((position_aligned * price_aligned).sum())
        if portfolio_value <= 0:
            msg = "portfolio_value must be positive."
            raise ValueError(msg)

        instructions: list[TradeInstruction] = []
        for symbol in all_symbols:
            price = float(price_aligned.loc[symbol])
            current_notional = float(position_aligned.loc[symbol] * price)
            target_notional = float(target_aligned.loc[symbol] * portfolio_value)
            trade_notional = target_notional - current_notional

            if abs(trade_notional) < self.min_trade_notional:
                continue

            trade_quantity = trade_notional / price
            if abs(trade_quantity) < self.min_trade_quantity:
                continue

            side: Literal["buy", "sell"] = "buy" if trade_quantity > 0 else "sell"
            instructions.append(
                TradeInstruction(
                    symbol=symbol,
                    side=side,
                    quantity=float(abs(trade_quantity)),
                    price=price,
                    current_weight=float(current_notional / portfolio_value),
                    target_weight=float(target_aligned.loc[symbol]),
                    trade_notional=float(abs(trade_notional)),
                )
            )

        return instructions


def _to_numeric_series(values: pd.Series | Mapping[str, float], name: str) -> pd.Series:
    if isinstance(values, pd.Series):
        series = values.copy()
    else:
        series = pd.Series(dict(values), dtype=float)

    series.index = [str(symbol).upper() for symbol in series.index]
    series = pd.to_numeric(series, errors="coerce")
    if series.isna().any():
        msg = f"{name} contains non-numeric values."
        raise ValueError(msg)
    return series.astype(float)
