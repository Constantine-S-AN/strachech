"""Backtest engine and result containers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from stratcheck.core.metrics import compute_metrics
from stratcheck.core.strategy import (
    Fill,
    OrderIntent,
    PortfolioState,
    Strategy,
)
from stratcheck.quality import run_pre_backtest_guards


@dataclass(slots=True)
class FillQuote:
    """Quoted execution outcome from a cost model."""

    fill_price: float
    fee: float


@runtime_checkable
class CostModel(Protocol):
    """Pluggable trading cost model interface."""

    def quote_fill(
        self,
        side: Literal["buy", "sell"],
        qty: float,
        reference_price: float,
        bar: pd.Series,
    ) -> FillQuote:
        """Quote fill price + explicit fee for a trade."""

    def describe(self) -> dict[str, str | float | bool]:
        """Serialize model configuration for reports."""


@dataclass(slots=True)
class FixedBpsCostModel:
    """Commission + slippage in basis points (legacy model)."""

    commission_bps: float = 0.0
    slippage_bps: float = 0.0

    def __post_init__(self) -> None:
        if self.commission_bps < 0 or self.slippage_bps < 0:
            msg = "commission_bps and slippage_bps must be non-negative."
            raise ValueError(msg)

    def quote_fill(
        self,
        side: Literal["buy", "sell"],
        qty: float,
        reference_price: float,
        bar: pd.Series,
    ) -> FillQuote:
        del bar
        slippage_rate = self.slippage_bps / 10_000.0
        if side == "buy":
            fill_price = reference_price * (1.0 + slippage_rate)
        else:
            fill_price = reference_price * (1.0 - slippage_rate)
        fill_price = max(fill_price, 0.0)

        fee = qty * fill_price * (self.commission_bps / 10_000.0)
        return FillQuote(fill_price=float(fill_price), fee=float(fee))

    def describe(self) -> dict[str, str | float | bool]:
        return {
            "cost_model_type": "fixed_bps",
            "commission_bps": float(self.commission_bps),
            "slippage_bps": float(self.slippage_bps),
        }


@dataclass(slots=True)
class SpreadCostModel:
    """Spread model using high-low range or fixed spread basis points."""

    commission_bps: float = 0.0
    spread_bps: float = 0.0
    use_bar_range: bool = True
    range_multiplier: float = 0.25

    def __post_init__(self) -> None:
        if self.commission_bps < 0 or self.spread_bps < 0:
            msg = "commission_bps and spread_bps must be non-negative."
            raise ValueError(msg)
        if self.range_multiplier < 0:
            msg = "range_multiplier must be non-negative."
            raise ValueError(msg)

    def quote_fill(
        self,
        side: Literal["buy", "sell"],
        qty: float,
        reference_price: float,
        bar: pd.Series,
    ) -> FillQuote:
        spread_value = self._estimate_spread(reference_price=reference_price, bar=bar)
        half_spread = spread_value / 2.0
        if side == "buy":
            fill_price = reference_price + half_spread
        else:
            fill_price = reference_price - half_spread
        fill_price = max(fill_price, 0.0)

        fee = qty * fill_price * (self.commission_bps / 10_000.0)
        return FillQuote(fill_price=float(fill_price), fee=float(fee))

    def describe(self) -> dict[str, str | float | bool]:
        return {
            "cost_model_type": "spread",
            "commission_bps": float(self.commission_bps),
            "spread_bps": float(self.spread_bps),
            "use_bar_range": bool(self.use_bar_range),
            "range_multiplier": float(self.range_multiplier),
        }

    def _estimate_spread(self, reference_price: float, bar: pd.Series) -> float:
        if self.use_bar_range:
            high_price = _as_float_or_default(bar.get("high"), default=reference_price)
            low_price = _as_float_or_default(bar.get("low"), default=reference_price)
            range_value = max(high_price - low_price, 0.0)
            if range_value > 0:
                return range_value * self.range_multiplier
        return reference_price * (self.spread_bps / 10_000.0)


@dataclass(slots=True)
class MarketImpactToyModel:
    """Toy market-impact model: larger participation => larger slippage."""

    commission_bps: float = 0.0
    base_slippage_bps: float = 0.0
    impact_factor: float = 0.1

    def __post_init__(self) -> None:
        if self.commission_bps < 0 or self.base_slippage_bps < 0 or self.impact_factor < 0:
            msg = "commission_bps, base_slippage_bps, impact_factor must be non-negative."
            raise ValueError(msg)

    def quote_fill(
        self,
        side: Literal["buy", "sell"],
        qty: float,
        reference_price: float,
        bar: pd.Series,
    ) -> FillQuote:
        volume_value = _as_float_or_default(bar.get("volume"), default=1.0)
        available_volume = max(volume_value, 1.0)
        participation_rate = qty / available_volume
        impact_bps = self.base_slippage_bps + self.impact_factor * participation_rate * 10_000.0
        impact_rate = impact_bps / 10_000.0

        if side == "buy":
            fill_price = reference_price * (1.0 + impact_rate)
        else:
            fill_price = reference_price * (1.0 - impact_rate)
        fill_price = max(fill_price, 0.0)

        fee = qty * fill_price * (self.commission_bps / 10_000.0)
        return FillQuote(fill_price=float(fill_price), fee=float(fee))

    def describe(self) -> dict[str, str | float | bool]:
        return {
            "cost_model_type": "market_impact",
            "commission_bps": float(self.commission_bps),
            "base_slippage_bps": float(self.base_slippage_bps),
            "impact_factor": float(self.impact_factor),
        }


def build_cost_model(config: Mapping[str, object] | None) -> CostModel:
    """Build a cost model from a config mapping."""
    if config is None:
        return FixedBpsCostModel()

    model_type = str(config.get("type", "fixed_bps")).strip().lower()
    if model_type in {"fixed_bps", "fixed", "bps"}:
        return FixedBpsCostModel(
            commission_bps=_as_float_or_default(config.get("commission_bps"), default=0.0),
            slippage_bps=_as_float_or_default(config.get("slippage_bps"), default=0.0),
        )
    if model_type in {"spread", "spread_model"}:
        return SpreadCostModel(
            commission_bps=_as_float_or_default(config.get("commission_bps"), default=0.0),
            spread_bps=_as_float_or_default(config.get("spread_bps"), default=0.0),
            use_bar_range=_as_bool_or_default(config.get("use_bar_range"), default=True),
            range_multiplier=_as_float_or_default(config.get("range_multiplier"), default=0.25),
        )
    if model_type in {"market_impact", "impact", "market_impact_toy"}:
        return MarketImpactToyModel(
            commission_bps=_as_float_or_default(config.get("commission_bps"), default=0.0),
            base_slippage_bps=_as_float_or_default(config.get("base_slippage_bps"), default=0.0),
            impact_factor=_as_float_or_default(config.get("impact_factor"), default=0.1),
        )

    msg = f"Unsupported cost model type: {model_type}"
    raise ValueError(msg)


def _as_float_or_default(value: object, default: float) -> float:
    if value is None:
        return float(default)
    if isinstance(value, (int, float)):
        return float(value)
    msg = f"Expected numeric value, got: {value!r}"
    raise ValueError(msg)


def _as_bool_or_default(value: object, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    msg = f"Expected boolean value, got: {value!r}"
    raise ValueError(msg)


@dataclass(slots=True)
class OrderRecord:
    """Stored order intent with lifecycle details."""

    created_at: pd.Timestamp
    side: Literal["buy", "sell"]
    qty: float
    limit_price: float | None
    market: bool
    filled: bool = False
    fill_time: pd.Timestamp | None = None
    fill_price: float | None = None


@dataclass(slots=True)
class BacktestResult:
    """Container for strategy backtest outputs."""

    market_data: pd.DataFrame
    positions: pd.Series
    strategy_returns: pd.Series
    benchmark_returns: pd.Series
    equity_curve: pd.Series
    benchmark_curve: pd.Series
    drawdown_curve: pd.Series
    metrics: dict[str, float | int]
    orders: list[OrderRecord] = field(default_factory=list)
    trades: list[Fill] = field(default_factory=list)


class BacktestEngine:
    """Simplified backtest engine.

    Execution assumption:
    - Strategy signals are generated at each bar close.
    - Market orders are filled at the next bar open.
    """

    def run(
        self,
        strategy: Strategy,
        bars: pd.DataFrame,
        initial_cash: float,
        cost_model: CostModel | None = None,
    ) -> BacktestResult:
        """Run backtest and return portfolio trajectory and executions."""
        if initial_cash <= 0:
            msg = "initial_cash must be positive."
            raise ValueError(msg)

        normalized_bars = _prepare_bars(bars)
        run_pre_backtest_guards(
            strategy=strategy,
            bars=normalized_bars,
            execution_assumption="signal_on_close_fill_next_open",
        )
        model: CostModel = cost_model or FixedBpsCostModel()

        cash = float(initial_cash)
        position_qty = 0.0
        pending_orders: list[OrderRecord] = []
        all_orders: list[OrderRecord] = []
        trades: list[Fill] = []
        equity_points: list[tuple[pd.Timestamp, float]] = []
        position_points: list[tuple[pd.Timestamp, float]] = []

        bar_count = len(normalized_bars)
        for bar_index, (timestamp, bar_row) in enumerate(normalized_bars.iterrows()):
            open_price = float(bar_row["open"])
            close_price = float(bar_row["close"])

            if pending_orders:
                cash, position_qty = self._fill_pending_orders(
                    pending_orders=pending_orders,
                    timestamp=timestamp,
                    bar_row=bar_row,
                    reference_price=open_price,
                    cash=cash,
                    position_qty=position_qty,
                    cost_model=model,
                    trades=trades,
                )
                pending_orders = []

            equity = cash + position_qty * close_price
            equity_points.append((timestamp, equity))
            position_points.append((timestamp, position_qty))

            if bar_index >= bar_count - 1:
                continue

            portfolio_state = PortfolioState(
                cash=cash,
                position_qty=position_qty,
                equity=equity,
            )
            intents = strategy.generate_orders(
                bars=normalized_bars.iloc[: bar_index + 1],
                portfolio_state=portfolio_state,
            )
            new_orders = self._to_order_records(intents=intents, timestamp=timestamp)
            all_orders.extend(new_orders)
            pending_orders.extend(new_orders)

        equity_curve = pd.Series(
            data=[value for _, value in equity_points],
            index=pd.Index([stamp for stamp, _ in equity_points], name=normalized_bars.index.name),
            name="equity",
            dtype=float,
        )
        positions = pd.Series(
            data=[value for _, value in position_points],
            index=pd.Index(
                [stamp for stamp, _ in position_points],
                name=normalized_bars.index.name,
            ),
            name="position_qty",
            dtype=float,
        )
        strategy_returns = equity_curve.pct_change().fillna(0.0)
        benchmark_returns = normalized_bars["close"].astype(float).pct_change().fillna(0.0)
        benchmark_curve = initial_cash * (1.0 + benchmark_returns).cumprod()
        drawdown_curve = equity_curve / equity_curve.cummax() - 1.0
        metrics = compute_metrics(
            equity_curve=equity_curve,
            trades=trades,
            bars_freq="1d",
        )
        metrics["trade_entries"] = int(sum(1 for trade in trades if trade.side == "buy"))

        return BacktestResult(
            market_data=normalized_bars,
            positions=positions,
            strategy_returns=strategy_returns,
            benchmark_returns=benchmark_returns,
            equity_curve=equity_curve,
            benchmark_curve=benchmark_curve,
            drawdown_curve=drawdown_curve,
            metrics=metrics,
            orders=all_orders,
            trades=trades,
        )

    def _to_order_records(
        self,
        intents: list[OrderIntent],
        timestamp: pd.Timestamp,
    ) -> list[OrderRecord]:
        records: list[OrderRecord] = []
        for intent in intents:
            if not intent.market:
                msg = "BacktestEngine simplified mode supports market orders only."
                raise ValueError(msg)
            records.append(
                OrderRecord(
                    created_at=timestamp,
                    side=intent.side,
                    qty=float(intent.qty),
                    limit_price=intent.limit_price,
                    market=intent.market,
                )
            )
        return records

    def _fill_pending_orders(
        self,
        pending_orders: list[OrderRecord],
        timestamp: pd.Timestamp,
        bar_row: pd.Series,
        reference_price: float,
        cash: float,
        position_qty: float,
        cost_model: CostModel,
        trades: list[Fill],
    ) -> tuple[float, float]:
        for order in pending_orders:
            if order.side == "buy":
                requested_qty = float(order.qty)
                filled_qty = self._resolve_affordable_buy_qty(
                    requested_qty=requested_qty,
                    cash=cash,
                    reference_price=reference_price,
                    bar_row=bar_row,
                    cost_model=cost_model,
                )
            else:
                filled_qty = min(order.qty, position_qty)

            if filled_qty <= 0:
                continue

            quote = cost_model.quote_fill(
                side=order.side,
                qty=filled_qty,
                reference_price=reference_price,
                bar=bar_row,
            )
            fill_price = float(quote.fill_price)
            fee = float(quote.fee)
            notional = filled_qty * fill_price

            if order.side == "buy":
                cash -= notional + fee
                position_qty += filled_qty
            else:
                cash += notional - fee
                position_qty -= filled_qty

            order.filled = True
            order.fill_time = timestamp
            order.fill_price = float(fill_price)

            if order.side == "buy":
                implicit_slippage = filled_qty * max(fill_price - reference_price, 0.0)
            else:
                implicit_slippage = filled_qty * max(reference_price - fill_price, 0.0)
            trades.append(
                Fill(
                    side=order.side,
                    qty=float(filled_qty),
                    price=float(fill_price),
                    timestamp=timestamp,
                    fee=fee,
                    cost=float(fee + implicit_slippage),
                )
            )

        return cash, position_qty

    def _resolve_affordable_buy_qty(
        self,
        requested_qty: float,
        cash: float,
        reference_price: float,
        bar_row: pd.Series,
        cost_model: CostModel,
    ) -> float:
        if requested_qty <= 0 or cash <= 0:
            return 0.0

        full_quote = cost_model.quote_fill(
            side="buy",
            qty=requested_qty,
            reference_price=reference_price,
            bar=bar_row,
        )
        full_required_cash = requested_qty * full_quote.fill_price + full_quote.fee
        if full_required_cash <= cash:
            return requested_qty

        lower_qty = 0.0
        upper_qty = requested_qty
        for _ in range(40):
            middle_qty = (lower_qty + upper_qty) / 2.0
            if middle_qty <= 0:
                break
            quote = cost_model.quote_fill(
                side="buy",
                qty=middle_qty,
                reference_price=reference_price,
                bar=bar_row,
            )
            required_cash = middle_qty * quote.fill_price + quote.fee
            if required_cash <= cash:
                lower_qty = middle_qty
            else:
                upper_qty = middle_qty
        return lower_qty


def run_moving_average_backtest(
    market_data: pd.DataFrame,
    fast_window: int = 20,
    slow_window: int = 50,
) -> BacktestResult:
    """Run a simple moving-average crossover backtest."""
    if fast_window >= slow_window:
        msg = "fast_window must be smaller than slow_window."
        raise ValueError(msg)
    if "close" not in market_data.columns:
        msg = "market_data must include a 'close' column."
        raise ValueError(msg)

    close_prices = market_data["close"].astype(float)
    fast_average = close_prices.rolling(window=fast_window, min_periods=fast_window).mean()
    slow_average = close_prices.rolling(window=slow_window, min_periods=slow_window).mean()

    trend_signal = (fast_average > slow_average).astype(float).fillna(0.0)
    positions = trend_signal.shift(1).fillna(0.0)

    benchmark_returns = close_prices.pct_change().fillna(0.0)
    strategy_returns = positions * benchmark_returns

    equity_curve = (1.0 + strategy_returns).cumprod()
    benchmark_curve = (1.0 + benchmark_returns).cumprod()
    running_peak = equity_curve.cummax()
    drawdown_curve = equity_curve / running_peak - 1.0

    metrics = _calculate_legacy_metrics(
        strategy_returns=strategy_returns,
        equity_curve=equity_curve,
        drawdown_curve=drawdown_curve,
        positions=positions,
    )

    enriched_market_data = market_data.copy()
    enriched_market_data["fast_average"] = fast_average
    enriched_market_data["slow_average"] = slow_average

    return BacktestResult(
        market_data=enriched_market_data,
        positions=positions,
        strategy_returns=strategy_returns,
        benchmark_returns=benchmark_returns,
        equity_curve=equity_curve,
        benchmark_curve=benchmark_curve,
        drawdown_curve=drawdown_curve,
        metrics=metrics,
    )


def _prepare_bars(bars: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(bars.index, pd.DatetimeIndex):
        msg = "bars must use a DatetimeIndex."
        raise ValueError(msg)

    required_columns = {"open", "close"}
    missing_columns = required_columns.difference(bars.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        msg = f"bars is missing required columns: {missing_text}"
        raise ValueError(msg)

    normalized_bars = bars.sort_index().copy()
    normalized_bars = normalized_bars[~normalized_bars.index.duplicated(keep="last")]
    return normalized_bars


def _calculate_legacy_metrics(
    strategy_returns: pd.Series,
    equity_curve: pd.Series,
    drawdown_curve: pd.Series,
    positions: pd.Series,
) -> dict[str, float | int]:
    total_points = len(strategy_returns)
    periods_per_year = 252

    if total_points == 0:
        return {
            "total_return": 0.0,
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "trade_entries": 0,
        }

    total_return = float(equity_curve.iloc[-1] - 1.0)
    annual_return = float((equity_curve.iloc[-1] ** (periods_per_year / total_points)) - 1.0)
    annual_volatility = float(strategy_returns.std(ddof=0) * np.sqrt(periods_per_year))
    sharpe_ratio = float(annual_return / annual_volatility) if annual_volatility > 0 else 0.0
    max_drawdown = float(drawdown_curve.min())
    win_rate = float((strategy_returns > 0.0).mean())
    trade_entries = int((positions.diff().fillna(positions) > 0.0).sum())

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "trade_entries": trade_entries,
    }
