"""Baseline strategy implementations used for comparison."""

from __future__ import annotations

import math

import pandas as pd

from stratcheck.core.calendar import periods_per_year
from stratcheck.core.strategy import OrderIntent, PortfolioState


class BuyAndHoldStrategy:
    """Buy and hold a fixed quantity of the asset."""

    def __init__(
        self,
        target_position_qty: float = 1.0,
        rebalance_tolerance: float = 1e-9,
    ) -> None:
        if target_position_qty <= 0:
            msg = "target_position_qty must be positive."
            raise ValueError(msg)
        if rebalance_tolerance < 0:
            msg = "rebalance_tolerance must be non-negative."
            raise ValueError(msg)

        self.target_position_qty = float(target_position_qty)
        self.rebalance_tolerance = float(rebalance_tolerance)

    def generate_orders(
        self,
        bars: pd.DataFrame,
        portfolio_state: PortfolioState,
    ) -> list[OrderIntent]:
        """Adjust holdings to target quantity."""
        if bars.empty:
            return []

        quantity_gap = self.target_position_qty - float(portfolio_state.position_qty)
        if abs(quantity_gap) <= self.rebalance_tolerance:
            return []
        if quantity_gap > 0:
            return [OrderIntent(side="buy", qty=float(quantity_gap), market=True)]
        return [OrderIntent(side="sell", qty=float(abs(quantity_gap)), market=True)]


class VolatilityTargetStrategy:
    """Long-only strategy that scales position by realized volatility."""

    def __init__(
        self,
        target_volatility: float = 0.15,
        lookback: int = 20,
        base_position_qty: float = 1.0,
        max_leverage: float = 2.0,
        bars_freq: str = "1d",
        min_volatility: float = 1e-6,
        rebalance_threshold: float = 0.05,
    ) -> None:
        if target_volatility <= 0:
            msg = "target_volatility must be positive."
            raise ValueError(msg)
        if lookback < 2:
            msg = "lookback must be at least 2."
            raise ValueError(msg)
        if base_position_qty <= 0:
            msg = "base_position_qty must be positive."
            raise ValueError(msg)
        if max_leverage <= 0:
            msg = "max_leverage must be positive."
            raise ValueError(msg)
        if min_volatility <= 0:
            msg = "min_volatility must be positive."
            raise ValueError(msg)
        if rebalance_threshold < 0:
            msg = "rebalance_threshold must be non-negative."
            raise ValueError(msg)

        self.target_volatility = float(target_volatility)
        self.lookback = int(lookback)
        self.base_position_qty = float(base_position_qty)
        self.max_leverage = float(max_leverage)
        self.bars_freq = bars_freq
        self.min_volatility = float(min_volatility)
        self.rebalance_threshold = float(rebalance_threshold)

    def generate_orders(
        self,
        bars: pd.DataFrame,
        portfolio_state: PortfolioState,
    ) -> list[OrderIntent]:
        """Target quantity = base_qty * clipped(target_vol / realized_vol)."""
        if "close" not in bars.columns:
            msg = "bars must include a 'close' column."
            raise ValueError(msg)
        if len(bars) < self.lookback + 1:
            return []

        close_prices = bars["close"].astype(float)
        returns = close_prices.pct_change().dropna()
        if len(returns) < self.lookback:
            return []

        window_returns = returns.iloc[-self.lookback :]
        annualized_volatility = float(
            window_returns.std(ddof=0) * math.sqrt(periods_per_year(self.bars_freq))
        )
        if not math.isfinite(annualized_volatility):
            return []

        if annualized_volatility <= self.min_volatility:
            target_leverage = self.max_leverage
        else:
            target_leverage = min(self.max_leverage, self.target_volatility / annualized_volatility)

        target_qty = self.base_position_qty * max(target_leverage, 0.0)
        quantity_gap = target_qty - float(portfolio_state.position_qty)
        if abs(quantity_gap) < self.rebalance_threshold:
            return []
        if quantity_gap > 0:
            return [OrderIntent(side="buy", qty=float(quantity_gap), market=True)]
        return [OrderIntent(side="sell", qty=float(abs(quantity_gap)), market=True)]


class MeanReversionZScoreStrategy:
    """Long-only mean-reversion strategy using rolling z-score on close."""

    def __init__(
        self,
        lookback: int = 20,
        entry_z: float = 1.5,
        exit_z: float = 0.25,
        target_position_qty: float = 1.0,
    ) -> None:
        if lookback < 2:
            msg = "lookback must be at least 2."
            raise ValueError(msg)
        if entry_z <= 0:
            msg = "entry_z must be positive."
            raise ValueError(msg)
        if exit_z < 0:
            msg = "exit_z must be non-negative."
            raise ValueError(msg)
        if exit_z > entry_z:
            msg = "exit_z should be <= entry_z."
            raise ValueError(msg)
        if target_position_qty <= 0:
            msg = "target_position_qty must be positive."
            raise ValueError(msg)

        self.lookback = int(lookback)
        self.entry_z = float(entry_z)
        self.exit_z = float(exit_z)
        self.target_position_qty = float(target_position_qty)

    def generate_orders(
        self,
        bars: pd.DataFrame,
        portfolio_state: PortfolioState,
    ) -> list[OrderIntent]:
        """Buy on large negative z-score and exit when price mean-reverts."""
        if "close" not in bars.columns:
            msg = "bars must include a 'close' column."
            raise ValueError(msg)
        if len(bars) < self.lookback:
            return []

        close_prices = bars["close"].astype(float)
        rolling_mean = close_prices.rolling(window=self.lookback, min_periods=self.lookback).mean()
        rolling_std = close_prices.rolling(window=self.lookback, min_periods=self.lookback).std(
            ddof=0
        )

        mean_value = rolling_mean.iloc[-1]
        std_value = rolling_std.iloc[-1]
        current_price = close_prices.iloc[-1]
        if pd.isna(mean_value) or pd.isna(std_value) or std_value <= 0:
            return []

        z_score = float((current_price - mean_value) / std_value)
        current_position = float(portfolio_state.position_qty)

        if current_position <= 0 and z_score <= -self.entry_z:
            return [OrderIntent(side="buy", qty=float(self.target_position_qty), market=True)]

        if current_position > 0 and z_score >= -self.exit_z:
            return [OrderIntent(side="sell", qty=float(current_position), market=True)]

        return []
