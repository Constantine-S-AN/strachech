from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from stratcheck.cli import run_with_config


def test_run_with_config_guard_violation_writes_risk_flag_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_leaky_strategy_module(tmp_path=tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_sample_csv(data_dir=data_dir, symbol="BTCUSDT")

    report_dir = tmp_path / "reports"
    report_name = "guard_failure_demo"
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"""
symbol = "BTCUSDT"
data_path = "{data_dir.as_posix()}"
strategy = "leaky_strategy_module:LeakyStrategy"
initial_cash = 100000
timeframe = "1d"
report_name = "{report_name}"
report_dir = "{report_dir.as_posix()}"

[cost_model]
commission_bps = 5
slippage_bps = 3

[windows]
window_size = "7D"
step_size = "7D"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError) as error_info:
        run_with_config(config_path=config_path)

    assert "Guard report generated:" in str(error_info.value)
    report_path = report_dir / f"{report_name}.html"
    assert report_path.exists()

    content = report_path.read_text(encoding="utf-8")
    assert "Risk Flags" in content
    assert "LookaheadGuard" in content


def _write_sample_csv(data_dir: Path, symbol: str) -> None:
    timestamps = pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC")
    close_prices = pd.Series([100.0 + index for index in range(30)], dtype=float)
    bars = pd.DataFrame(
        {
            "timestamp": timestamps.astype(str),
            "open": close_prices - 0.2,
            "high": close_prices + 1.0,
            "low": close_prices - 1.0,
            "close": close_prices,
            "volume": 1_000,
        }
    )
    bars.to_csv(data_dir / f"{symbol}.csv", index=False)


def _write_leaky_strategy_module(tmp_path: Path) -> None:
    module_path = tmp_path / "leaky_strategy_module.py"
    module_path.write_text(
        """
from __future__ import annotations

import pandas as pd

from stratcheck.core.strategy import OrderIntent, PortfolioState


class LeakyStrategy:
    def generate_orders(
        self,
        bars: pd.DataFrame,
        portfolio_state: PortfolioState,
    ) -> list[OrderIntent]:
        del portfolio_state
        if len(bars) < 3:
            return []

        leaked_future_close = bars["close"].shift(-1).iloc[-2]
        current_close = float(bars["close"].iloc[-1])
        if float(leaked_future_close) > current_close:
            return [OrderIntent(side="buy", qty=1.0, market=True)]
        return []
""".strip()
        + "\n",
        encoding="utf-8",
    )
