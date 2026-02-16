from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from stratcheck.core.healthcheck import run_healthcheck
from stratcheck.core.strategy import PortfolioState


class NoopStrategy:
    def generate_orders(
        self,
        bars: pd.DataFrame,
        portfolio_state: PortfolioState,
    ):
        del bars, portfolio_state
        return []


def test_run_healthcheck_splits_24_months_into_4_windows(tmp_path: Path) -> None:
    timestamps = pd.date_range("2023-01-31", periods=24, freq="ME", tz="UTC")
    close_prices = np.linspace(100.0, 123.0, 24)
    bars = pd.DataFrame(
        {
            "open": close_prices,
            "high": close_prices + 1.0,
            "low": close_prices - 1.0,
            "close": close_prices,
            "volume": np.full(24, 1_000),
        },
        index=timestamps,
    )

    summary_frame, json_path = run_healthcheck(
        strategy=NoopStrategy(),
        bars=bars,
        window_size="6M",
        step_size="6M",
        output_json_path=tmp_path / "healthcheck_summary.json",
    )

    assert len(summary_frame) == 4
    expected_window_starts = [timestamps[0], timestamps[6], timestamps[12], timestamps[18]]
    expected_window_ends = [timestamps[5], timestamps[11], timestamps[17], timestamps[23]]
    assert summary_frame["window_start"].tolist() == expected_window_starts
    assert summary_frame["window_end"].tolist() == expected_window_ends
    assert summary_frame["bars_count"].tolist() == [6, 6, 6, 6]

    assert json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(payload["windows"]) == 4
