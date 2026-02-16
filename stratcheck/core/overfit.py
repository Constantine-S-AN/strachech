"""Simple overfit-risk diagnostics and risk flags."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class RiskFlag:
    """Single risk flag item displayed in reports."""

    check: str
    level: str
    message: str


def evaluate_overfit_risk(
    returns: pd.Series,
    window_metrics_df: pd.DataFrame,
    worst_window_sharpe_threshold: float = -0.5,
) -> tuple[dict[str, float], list[RiskFlag]]:
    """Compute overfit diagnostics and convert them to red/yellow/green flags."""
    returns_series = pd.Series(returns, dtype=float).dropna()
    autocorr_lag1, hit_rate, hit_rate_zscore = _white_noise_metrics(returns_series)
    sharpe_variance, worst_window_sharpe, stability_score = _walk_forward_metrics(
        window_metrics_df=window_metrics_df,
    )

    risk_flags = [
        _autocorr_flag(autocorr_lag1),
        _hit_rate_flag(hit_rate, hit_rate_zscore),
        _walk_forward_flag(
            sharpe_variance=sharpe_variance,
            worst_window_sharpe=worst_window_sharpe,
            worst_window_sharpe_threshold=worst_window_sharpe_threshold,
        ),
    ]

    red_count = float(sum(1 for flag in risk_flags if flag.level == "red"))
    yellow_count = float(sum(1 for flag in risk_flags if flag.level == "yellow"))
    green_count = float(sum(1 for flag in risk_flags if flag.level == "green"))

    summary = {
        "autocorr_lag1": autocorr_lag1,
        "hit_rate": hit_rate,
        "hit_rate_zscore": hit_rate_zscore,
        "sharpe_variance": sharpe_variance,
        "worst_window_sharpe": worst_window_sharpe,
        "stability_score": stability_score,
        "risk_red_count": red_count,
        "risk_yellow_count": yellow_count,
        "risk_green_count": green_count,
    }
    return summary, risk_flags


def _white_noise_metrics(returns_series: pd.Series) -> tuple[float, float, float]:
    if len(returns_series) < 2:
        return 0.0, 0.5, 0.0

    autocorr_lag1 = float(returns_series.autocorr(lag=1))
    hit_rate = float((returns_series > 0.0).mean())
    sample_size = float(len(returns_series))
    standard_error = np.sqrt(0.25 / sample_size)
    hit_rate_zscore = float((hit_rate - 0.5) / standard_error) if standard_error > 0 else 0.0
    return autocorr_lag1, hit_rate, hit_rate_zscore


def _walk_forward_metrics(window_metrics_df: pd.DataFrame) -> tuple[float, float, float]:
    if "sharpe" in window_metrics_df.columns:
        sharpe_series = pd.to_numeric(window_metrics_df["sharpe"], errors="coerce").dropna()
    else:
        sharpe_series = pd.Series(dtype=float)

    if sharpe_series.empty:
        return 0.0, 0.0, 50.0

    sharpe_variance = float(sharpe_series.var(ddof=0))
    worst_window_sharpe = float(sharpe_series.min())

    variance_penalty = min(50.0, sharpe_variance * 40.0)
    worst_penalty = 0.0 if worst_window_sharpe >= 0 else min(50.0, abs(worst_window_sharpe) * 40.0)
    stability_score = float(max(0.0, 100.0 - variance_penalty - worst_penalty))
    return sharpe_variance, worst_window_sharpe, stability_score


def _autocorr_flag(autocorr_lag1: float) -> RiskFlag:
    absolute_value = abs(autocorr_lag1)
    if absolute_value >= 0.30:
        level = "red"
        message = f"Lag-1 autocorr={autocorr_lag1:.3f} is high."
    elif absolute_value >= 0.15:
        level = "yellow"
        message = f"Lag-1 autocorr={autocorr_lag1:.3f} is moderate."
    else:
        level = "green"
        message = f"Lag-1 autocorr={autocorr_lag1:.3f} is near white noise."
    return RiskFlag(check="Autocorrelation", level=level, message=message)


def _hit_rate_flag(hit_rate: float, hit_rate_zscore: float) -> RiskFlag:
    absolute_zscore = abs(hit_rate_zscore)
    hit_rate_text = f"{hit_rate:.3f}"
    zscore_text = f"{hit_rate_zscore:.2f}"
    if hit_rate_zscore <= -1.50:
        level = "red"
        message = (
            f"Hit-rate={hit_rate_text} is significantly below random baseline (z={zscore_text})."
        )
    elif absolute_zscore < 0.50:
        level = "red"
        message = f"Hit-rate={hit_rate_text} is close to random baseline (z={zscore_text})."
    elif hit_rate_zscore < 1.50:
        level = "yellow"
        message = (
            f"Hit-rate={hit_rate_text} is only weakly above random baseline (z={zscore_text})."
        )
    else:
        level = "green"
        message = f"Hit-rate={hit_rate_text} deviates from random baseline (z={zscore_text})."
    return RiskFlag(check="Hit-Rate vs Random", level=level, message=message)


def _walk_forward_flag(
    sharpe_variance: float,
    worst_window_sharpe: float,
    worst_window_sharpe_threshold: float,
) -> RiskFlag:
    variance_text = f"{sharpe_variance:.3f}"
    worst_text = f"{worst_window_sharpe:.3f}"
    if worst_window_sharpe < worst_window_sharpe_threshold or sharpe_variance > 1.0:
        level = "red"
        message = f"Window stability is poor (var={variance_text}, worst sharpe={worst_text})."
    elif worst_window_sharpe < 0.0 or sharpe_variance > 0.4:
        level = "yellow"
        message = f"Window stability is mixed (var={variance_text}, worst sharpe={worst_text})."
    else:
        level = "green"
        message = (
            f"Window stability looks consistent (var={variance_text}, worst sharpe={worst_text})."
        )
    return RiskFlag(check="Walk-Forward Stability", level=level, message=message)
