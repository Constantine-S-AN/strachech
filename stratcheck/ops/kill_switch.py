"""Kill-switch guardrails for long-running paper/live loops."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class KillSwitchConfig:
    """Kill-switch thresholds."""

    max_consecutive_errors: int = 3
    max_drawdown: float = 0.30
    max_data_gap_steps: int = 2

    def __post_init__(self) -> None:
        if self.max_consecutive_errors < 1:
            msg = "max_consecutive_errors must be >= 1."
            raise ValueError(msg)
        if not (0 <= self.max_drawdown < 1):
            msg = "max_drawdown must be in [0, 1)."
            raise ValueError(msg)
        if self.max_data_gap_steps < 1:
            msg = "max_data_gap_steps must be >= 1."
            raise ValueError(msg)


@dataclass(slots=True)
class KillSwitchState:
    """Current kill-switch state."""

    consecutive_errors: int = 0
    peak_equity: float | None = None
    kill_triggered: bool = False
    kill_reason: str | None = None


class KillSwitch:
    """Evaluate kill conditions: errors, drawdown, and data interruption."""

    def __init__(self, config: KillSwitchConfig | None = None) -> None:
        self.config = config or KillSwitchConfig()
        self.state = KillSwitchState()

    def on_error(self) -> str | None:
        """Record one processing error and return reason when threshold breaches."""
        self.state.consecutive_errors += 1
        if self.state.consecutive_errors >= self.config.max_consecutive_errors:
            return self._trigger("consecutive_errors_exceeded")
        return None

    def on_success(self) -> None:
        """Reset consecutive error counter after a successful cycle."""
        self.state.consecutive_errors = 0

    def check_drawdown(self, equity: float) -> str | None:
        """Check drawdown threshold against running equity peak."""
        equity_value = float(equity)
        if self.state.peak_equity is None:
            self.state.peak_equity = equity_value
            return None

        self.state.peak_equity = max(self.state.peak_equity, equity_value)
        if self.state.peak_equity <= 0:
            return None
        drawdown_ratio = (self.state.peak_equity - equity_value) / self.state.peak_equity
        if drawdown_ratio >= self.config.max_drawdown:
            return self._trigger("max_drawdown_exceeded")
        return None

    def check_data_interruption(
        self,
        *,
        previous_timestamp: pd.Timestamp | None,
        current_timestamp: pd.Timestamp,
        expected_interval: pd.Timedelta | None,
    ) -> str | None:
        """Check data continuity by timestamp gap against expected interval."""
        if previous_timestamp is None or expected_interval is None:
            return None
        if expected_interval <= pd.Timedelta(0):
            return None
        observed_gap = current_timestamp - previous_timestamp
        allowed_gap = expected_interval * self.config.max_data_gap_steps
        if observed_gap > allowed_gap:
            return self._trigger("data_interruption_detected")
        return None

    def _trigger(self, reason: str) -> str:
        self.state.kill_triggered = True
        self.state.kill_reason = reason
        return reason
