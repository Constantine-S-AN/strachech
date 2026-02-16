"""Risk-rule DSL and evaluation helpers for live/paper execution loops."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import pandas as pd

RuleAction = Literal["halt", "block"]


@dataclass(slots=True)
class RuleContext:
    """Snapshot passed into rule evaluation."""

    timestamp: pd.Timestamp
    bar_index: int
    bar: pd.Series | None = None
    previous_timestamp: pd.Timestamp | None = None
    expected_interval: pd.Timedelta | None = None
    previous_close: float | None = None
    equity: float | None = None
    peak_equity: float | None = None
    current_position_qty: float | None = None
    projected_position_qty: float | None = None
    order_side: str | None = None
    order_qty: float | None = None
    daily_trade_count: int | None = None


@dataclass(slots=True)
class RuleHit:
    """One rule hit emitted by evaluation."""

    rule_name: str
    action: RuleAction
    reason: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


class RiskRule(Protocol):
    """Protocol for rule objects."""

    name: str

    def evaluate(self, context: RuleContext) -> RuleHit | None:
        """Evaluate one context and return a hit when triggered."""

    def describe(self) -> dict[str, Any]:
        """Serialize rule configuration for run metadata."""


@dataclass(slots=True)
class MaxDrawdownRule:
    """Halt when drawdown breaches threshold."""

    max_drawdown: float
    name: str = "max_drawdown"
    action: RuleAction = "halt"

    def __post_init__(self) -> None:
        if not (0 <= self.max_drawdown < 1):
            msg = "max_drawdown must be in [0, 1)."
            raise ValueError(msg)

    def evaluate(self, context: RuleContext) -> RuleHit | None:
        if context.equity is None or context.peak_equity is None:
            return None
        peak_value = float(context.peak_equity)
        equity_value = float(context.equity)
        if peak_value <= 0:
            return None
        drawdown_ratio = (peak_value - equity_value) / peak_value
        if drawdown_ratio < self.max_drawdown:
            return None
        return RuleHit(
            rule_name=self.name,
            action=self.action,
            reason="max_drawdown_exceeded",
            message=(
                f"drawdown {drawdown_ratio:.6f} exceeded threshold {float(self.max_drawdown):.6f}"
            ),
            details={
                "drawdown": float(drawdown_ratio),
                "max_drawdown": float(self.max_drawdown),
                "equity": float(equity_value),
                "peak_equity": float(peak_value),
            },
        )

    def describe(self) -> dict[str, Any]:
        return {
            "rule": self.name,
            "action": self.action,
            "max_drawdown": float(self.max_drawdown),
        }


@dataclass(slots=True)
class MaxPositionRule:
    """Block orders when projected position exceeds absolute cap."""

    max_abs_position_qty: float
    name: str = "max_position"
    action: RuleAction = "block"

    def __post_init__(self) -> None:
        if self.max_abs_position_qty <= 0:
            msg = "max_abs_position_qty must be positive."
            raise ValueError(msg)

    def evaluate(self, context: RuleContext) -> RuleHit | None:
        if context.projected_position_qty is None:
            return None
        projected_qty = float(context.projected_position_qty)
        if abs(projected_qty) <= self.max_abs_position_qty:
            return None
        return RuleHit(
            rule_name=self.name,
            action=self.action,
            reason="max_abs_position_qty_breached",
            message=(
                f"projected position {projected_qty:.6f} exceeds "
                f"max_abs_position_qty {float(self.max_abs_position_qty):.6f}"
            ),
            details={
                "projected_position_qty": float(projected_qty),
                "max_abs_position_qty": float(self.max_abs_position_qty),
                "order_side": str(context.order_side or ""),
                "order_qty": (None if context.order_qty is None else float(context.order_qty)),
            },
        )

    def describe(self) -> dict[str, Any]:
        return {
            "rule": self.name,
            "action": self.action,
            "max_abs_position_qty": float(self.max_abs_position_qty),
        }


@dataclass(slots=True)
class MaxDailyTradesRule:
    """Block new orders after daily trade-count cap is reached."""

    max_trades_per_day: int
    name: str = "max_daily_trades"
    action: RuleAction = "block"

    def __post_init__(self) -> None:
        if self.max_trades_per_day < 1:
            msg = "max_trades_per_day must be >= 1."
            raise ValueError(msg)

    def evaluate(self, context: RuleContext) -> RuleHit | None:
        if context.daily_trade_count is None:
            return None
        used_count = int(context.daily_trade_count)
        if used_count < self.max_trades_per_day:
            return None
        return RuleHit(
            rule_name=self.name,
            action=self.action,
            reason="max_daily_trades_exceeded",
            message=(f"daily trade limit reached ({used_count} >= {int(self.max_trades_per_day)})"),
            details={
                "daily_trade_count": int(used_count),
                "max_trades_per_day": int(self.max_trades_per_day),
            },
        )

    def describe(self) -> dict[str, Any]:
        return {
            "rule": self.name,
            "action": self.action,
            "max_trades_per_day": int(self.max_trades_per_day),
        }


@dataclass(slots=True)
class DataAnomalyHaltRule:
    """Halt on abnormal data: data gaps, invalid OHLCV values, or extreme jumps."""

    max_data_gap_steps: int = 2
    max_abs_return: float | None = None
    name: str = "data_anomaly"
    action: RuleAction = "halt"

    def __post_init__(self) -> None:
        if self.max_data_gap_steps < 1:
            msg = "max_data_gap_steps must be >= 1."
            raise ValueError(msg)
        if self.max_abs_return is not None and self.max_abs_return <= 0:
            msg = "max_abs_return must be positive when provided."
            raise ValueError(msg)

    def evaluate(self, context: RuleContext) -> RuleHit | None:
        gap_hit = self._check_gap(context=context)
        if gap_hit is not None:
            return gap_hit

        bar_hit = self._check_bar_values(context=context)
        if bar_hit is not None:
            return bar_hit

        return_hit = self._check_return_jump(context=context)
        if return_hit is not None:
            return return_hit
        return None

    def _check_gap(self, context: RuleContext) -> RuleHit | None:
        if context.previous_timestamp is None or context.expected_interval is None:
            return None
        if context.expected_interval <= pd.Timedelta(0):
            return None
        observed_gap = context.timestamp - context.previous_timestamp
        allowed_gap = context.expected_interval * int(self.max_data_gap_steps)
        if observed_gap <= allowed_gap:
            return None
        return RuleHit(
            rule_name=self.name,
            action=self.action,
            reason="data_interruption_detected",
            message=(f"observed gap {observed_gap} exceeded allowed gap {allowed_gap}"),
            details={
                "observed_gap_seconds": float(observed_gap.total_seconds()),
                "allowed_gap_seconds": float(allowed_gap.total_seconds()),
                "max_data_gap_steps": int(self.max_data_gap_steps),
            },
        )

    def _check_bar_values(self, context: RuleContext) -> RuleHit | None:
        if context.bar is None:
            return None

        required_fields = ["open", "high", "low", "close", "volume"]
        parsed_values: dict[str, float] = {}
        for field_name in required_fields:
            if field_name not in context.bar:
                return RuleHit(
                    rule_name=self.name,
                    action=self.action,
                    reason="abnormal_data_detected",
                    message=f"missing field in bar: {field_name}",
                    details={"missing_field": field_name},
                )
            raw_value = context.bar[field_name]
            try:
                numeric_value = float(raw_value)
            except (TypeError, ValueError):
                return RuleHit(
                    rule_name=self.name,
                    action=self.action,
                    reason="abnormal_data_detected",
                    message=f"non-numeric field in bar: {field_name}",
                    details={"field": field_name, "value": str(raw_value)},
                )
            if not math.isfinite(numeric_value):
                return RuleHit(
                    rule_name=self.name,
                    action=self.action,
                    reason="abnormal_data_detected",
                    message=f"non-finite field in bar: {field_name}",
                    details={"field": field_name, "value": float(numeric_value)},
                )
            parsed_values[field_name] = numeric_value

        if parsed_values["high"] < parsed_values["low"]:
            return RuleHit(
                rule_name=self.name,
                action=self.action,
                reason="abnormal_data_detected",
                message="bar high is below bar low",
                details={
                    "high": float(parsed_values["high"]),
                    "low": float(parsed_values["low"]),
                },
            )
        if (
            min(
                parsed_values["open"],
                parsed_values["high"],
                parsed_values["low"],
                parsed_values["close"],
            )
            <= 0
        ):
            return RuleHit(
                rule_name=self.name,
                action=self.action,
                reason="abnormal_data_detected",
                message="bar contains non-positive price",
                details={
                    "open": float(parsed_values["open"]),
                    "high": float(parsed_values["high"]),
                    "low": float(parsed_values["low"]),
                    "close": float(parsed_values["close"]),
                },
            )
        if parsed_values["volume"] < 0:
            return RuleHit(
                rule_name=self.name,
                action=self.action,
                reason="abnormal_data_detected",
                message="bar contains negative volume",
                details={"volume": float(parsed_values["volume"])},
            )
        return None

    def _check_return_jump(self, context: RuleContext) -> RuleHit | None:
        if self.max_abs_return is None:
            return None
        if context.bar is None or context.previous_close is None:
            return None
        previous_close = float(context.previous_close)
        if previous_close <= 0:
            return None

        if "close" not in context.bar:
            return None
        current_close = float(context.bar["close"])
        return_ratio = abs((current_close - previous_close) / previous_close)
        if return_ratio <= float(self.max_abs_return):
            return None
        return RuleHit(
            rule_name=self.name,
            action=self.action,
            reason="abnormal_return_detected",
            message=(
                f"absolute return {return_ratio:.6f} exceeded "
                f"threshold {float(self.max_abs_return):.6f}"
            ),
            details={
                "abs_return": float(return_ratio),
                "max_abs_return": float(self.max_abs_return),
                "previous_close": float(previous_close),
                "current_close": float(current_close),
            },
        )

    def describe(self) -> dict[str, Any]:
        return {
            "rule": self.name,
            "action": self.action,
            "max_data_gap_steps": int(self.max_data_gap_steps),
            "max_abs_return": (None if self.max_abs_return is None else float(self.max_abs_return)),
        }


class RuleBook:
    """Collection-style evaluator for rule objects."""

    def __init__(self, rules: list[RiskRule] | None = None) -> None:
        self.rules: list[RiskRule] = list(rules or [])

    def evaluate(
        self,
        context: RuleContext,
        actions: set[RuleAction] | None = None,
    ) -> list[RuleHit]:
        hits: list[RuleHit] = []
        for rule in self.rules:
            hit = rule.evaluate(context)
            if hit is None:
                continue
            if actions is not None and hit.action not in actions:
                continue
            hits.append(hit)
        return hits

    def describe(self) -> list[dict[str, Any]]:
        descriptions: list[dict[str, Any]] = []
        for rule in self.rules:
            descriptions.append(rule.describe())
        return descriptions
