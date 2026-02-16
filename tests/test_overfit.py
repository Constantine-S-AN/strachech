from __future__ import annotations

import numpy as np
import pandas as pd
from stratcheck.core.overfit import evaluate_overfit_risk


def test_evaluate_overfit_risk_returns_summary_and_flags() -> None:
    returns = pd.Series([0.01, 0.02, -0.01, 0.015, -0.005, 0.01], dtype=float)
    windows = pd.DataFrame(
        {
            "window_index": [0, 1, 2],
            "sharpe": [0.8, 0.7, 0.9],
        }
    )

    summary, flags = evaluate_overfit_risk(returns=returns, window_metrics_df=windows)

    assert {"autocorr_lag1", "hit_rate", "stability_score", "sharpe_variance"}.issubset(
        summary.keys()
    )
    assert len(flags) == 3
    assert {flag.check for flag in flags} == {
        "Autocorrelation",
        "Hit-Rate vs Random",
        "Walk-Forward Stability",
    }


def test_evaluate_overfit_risk_flags_red_when_windows_unstable() -> None:
    returns = pd.Series(np.repeat([0.01, -0.01], 20), dtype=float)
    windows = pd.DataFrame(
        {
            "window_index": [0, 1, 2, 3],
            "sharpe": [1.2, -1.0, 1.1, -0.9],
        }
    )

    _, flags = evaluate_overfit_risk(returns=returns, window_metrics_df=windows)
    flag_by_check = {flag.check: flag for flag in flags}
    assert flag_by_check["Walk-Forward Stability"].level == "red"
