"""Backtest quality guards for lookahead and data leakage checks."""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from stratcheck.core.strategy import Strategy


@dataclass(slots=True)
class GuardFlag:
    """Single guard flag item rendered in reports."""

    check: str
    level: str
    message: str


class GuardViolationError(ValueError):
    """Raised when one or more hard-fail quality guards are triggered."""

    def __init__(self, message: str, flags: list[GuardFlag]) -> None:
        super().__init__(message)
        self.flags = flags


class QualityGuard(Protocol):
    """Protocol for quality guards executed before backtests."""

    def validate(
        self,
        strategy: Strategy,
        bars: pd.DataFrame,
        execution_assumption: str,
    ) -> list[GuardFlag]:
        """Validate strategy + bars and return guard flags."""


@dataclass(slots=True)
class LookaheadGuard:
    """Detect obvious future-data usage patterns in strategy logic."""

    check_name: str = "LookaheadGuard"

    def validate(
        self,
        strategy: Strategy,
        bars: pd.DataFrame,
        execution_assumption: str,
    ) -> list[GuardFlag]:
        del bars
        flags: list[GuardFlag] = []
        source_text = _strategy_source_text(strategy)
        if not source_text:
            return flags

        if re.search(r"\.shift\(\s*-\d+\s*\)", source_text):
            flags.append(
                GuardFlag(
                    check=self.check_name,
                    level="red",
                    message=(
                        "Detected negative shift usage (e.g. shift(-1)); "
                        "this is a likely lookahead leak."
                    ),
                )
            )

        if re.search(r"\bnp\.roll\([^)]*,\s*-\d+", source_text):
            flags.append(
                GuardFlag(
                    check=self.check_name,
                    level="red",
                    message=(
                        "Detected np.roll with negative offset; this can reference future values."
                    ),
                )
            )

        assumes_close_fill = bool(getattr(strategy, "assume_fill_on_close", False))
        if assumes_close_fill and execution_assumption == "signal_on_close_fill_next_open":
            flags.append(
                GuardFlag(
                    check=self.check_name,
                    level="red",
                    message=(
                        "Strategy assumes close-bar execution but engine fills next-open; "
                        "timing mismatch can cause leakage."
                    ),
                )
            )

        return flags


@dataclass(slots=True)
class DataLeakGuard:
    """Validate feature timestamp alignment and suspicious forward-looking feature names."""

    check_name: str = "DataLeakGuard"

    def validate(
        self,
        strategy: Strategy,
        bars: pd.DataFrame,
        execution_assumption: str,
    ) -> list[GuardFlag]:
        del strategy, execution_assumption
        flags: list[GuardFlag] = []

        if not isinstance(bars.index, pd.DatetimeIndex):
            return flags

        index_utc = _to_utc_index(bars.index)

        timestamp_columns = [
            column_name
            for column_name in bars.columns
            if column_name.lower().endswith(("_ts", "_timestamp"))
        ]
        for column_name in timestamp_columns:
            parsed_timestamps = pd.to_datetime(bars[column_name], utc=True, errors="coerce")
            if parsed_timestamps.isna().all():
                continue
            leak_mask = parsed_timestamps > index_utc
            if bool(leak_mask.any()):
                first_position = int(leak_mask.to_numpy().argmax())
                first_bar_time = bars.index[first_position]
                first_feature_time = parsed_timestamps.iloc[first_position]
                flags.append(
                    GuardFlag(
                        check=self.check_name,
                        level="red",
                        message=(
                            f"Feature timestamp column '{column_name}' leaks future data: "
                            f"feature_time={first_feature_time} > bar_time={first_bar_time}."
                        ),
                    )
                )

        suspicious_pattern = re.compile(r"(?:^|_)(future|next|lead|forward|tplus)(?:_|$)")
        base_columns = {"open", "high", "low", "close", "volume"}
        for column_name in bars.columns:
            lowered = column_name.lower()
            if lowered in base_columns:
                continue
            if suspicious_pattern.search(lowered) and bars[column_name].notna().any():
                flags.append(
                    GuardFlag(
                        check=self.check_name,
                        level="red",
                        message=(
                            f"Suspicious forward-looking feature column detected: '{column_name}'."
                        ),
                    )
                )

        return flags


def run_pre_backtest_guards(
    strategy: Strategy,
    bars: pd.DataFrame,
    execution_assumption: str = "signal_on_close_fill_next_open",
    guards: list[QualityGuard] | None = None,
) -> list[GuardFlag]:
    """Run quality guards before each backtest; raise on red-level violations."""
    guard_list = guards or [LookaheadGuard(), DataLeakGuard()]
    all_flags: list[GuardFlag] = []
    for guard in guard_list:
        all_flags.extend(
            guard.validate(
                strategy=strategy,
                bars=bars,
                execution_assumption=execution_assumption,
            )
        )

    blocking_flags = [flag for flag in all_flags if flag.level.lower() == "red"]
    if blocking_flags:
        details = "; ".join(f"[{flag.check}] {flag.message}" for flag in blocking_flags)
        raise GuardViolationError(
            message=f"Quality guard violation: {details}", flags=blocking_flags
        )
    return all_flags


def _strategy_source_text(strategy: Strategy) -> str:
    generate_orders = getattr(strategy, "generate_orders", None)
    if generate_orders is None:
        return ""
    try:
        return inspect.getsource(generate_orders)
    except (OSError, TypeError):
        return ""


def _to_utc_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index.tz is None:
        return index.tz_localize("UTC")
    return index.tz_convert("UTC")
