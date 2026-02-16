"""Regime analysis based on volatility and trend strength."""

from __future__ import annotations

import numpy as np
import pandas as pd


def classify_market_regimes(
    bars: pd.DataFrame,
    vol_window: int = 20,
    trend_window: int = 20,
    trend_threshold: float = 0.02,
) -> pd.DataFrame:
    """Split bars into market regimes using rolling volatility and trend."""
    if "close" not in bars.columns:
        msg = "bars must include a 'close' column."
        raise ValueError(msg)
    if not isinstance(bars.index, pd.DatetimeIndex):
        msg = "bars must use DatetimeIndex."
        raise ValueError(msg)
    if vol_window < 2 or trend_window < 2:
        msg = "vol_window and trend_window must be at least 2."
        raise ValueError(msg)

    close_prices = bars["close"].astype(float)
    period_returns = close_prices.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    volatility = period_returns.rolling(window=vol_window, min_periods=2).std()
    trend_strength = close_prices.pct_change(periods=trend_window).fillna(0.0)

    volatility = volatility.fillna(float(volatility.median()) if volatility.notna().any() else 0.0)
    vol_median = float(volatility.median()) if len(volatility) > 0 else 0.0

    regime_frame = pd.DataFrame(
        {
            "close": close_prices,
            "return": period_returns,
            "volatility": volatility,
            "trend": trend_strength,
        },
        index=bars.index,
    )
    regime_frame["regime"] = regime_frame.apply(
        lambda row: _classify_row(
            trend=float(row["trend"]),
            volatility=float(row["volatility"]),
            vol_median=vol_median,
            trend_threshold=float(trend_threshold),
        ),
        axis=1,
    )
    return regime_frame


def compute_regime_scorecard(
    bars: pd.DataFrame,
    equity_curve: pd.Series,
    trades: list[object],
    vol_window: int = 20,
    trend_window: int = 20,
    trend_threshold: float = 0.02,
) -> pd.DataFrame:
    """Compute return/drawdown/win-rate scorecard per regime."""
    regime_frame = classify_market_regimes(
        bars=bars,
        vol_window=vol_window,
        trend_window=trend_window,
        trend_threshold=trend_threshold,
    )
    strategy_returns = (
        equity_curve.astype(float).pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )
    aligned_returns = strategy_returns.reindex(regime_frame.index).fillna(0.0)

    trade_counts = _trade_count_by_regime(
        trades=trades,
        regime_labels=regime_frame["regime"],
    )

    records: list[dict[str, float | str | int]] = []
    for regime_name, subset in regime_frame.groupby("regime", sort=True):
        subset_returns = aligned_returns.loc[subset.index]
        if subset_returns.empty:
            continue

        cumulative_curve = (1.0 + subset_returns).cumprod()
        running_peak = cumulative_curve.cummax()
        drawdown_curve = cumulative_curve / running_peak - 1.0

        records.append(
            {
                "regime": str(regime_name),
                "bars_count": int(len(subset_returns)),
                "mean_return": float(subset_returns.mean()),
                "total_return": float(cumulative_curve.iloc[-1] - 1.0),
                "max_drawdown": float(drawdown_curve.min()),
                "win_rate": float((subset_returns > 0).mean()),
                "trade_count": int(trade_counts.get(str(regime_name), 0)),
            }
        )

    if not records:
        return pd.DataFrame(
            columns=[
                "regime",
                "bars_count",
                "mean_return",
                "total_return",
                "max_drawdown",
                "win_rate",
                "trade_count",
            ]
        )

    scorecard = pd.DataFrame(records)
    scorecard = scorecard.sort_values(by=["bars_count", "regime"], ascending=[False, True])
    return scorecard.reset_index(drop=True)


def _classify_row(
    trend: float,
    volatility: float,
    vol_median: float,
    trend_threshold: float,
) -> str:
    if trend >= trend_threshold and volatility <= vol_median:
        return "bull_calm"
    if trend >= trend_threshold and volatility > vol_median:
        return "bull_volatile"
    if trend <= -trend_threshold and volatility <= vol_median:
        return "bear_calm"
    if trend <= -trend_threshold and volatility > vol_median:
        return "bear_volatile"
    return "sideways"


def _trade_count_by_regime(
    trades: list[object],
    regime_labels: pd.Series,
) -> dict[str, int]:
    if not trades:
        return {}

    counts: dict[str, int] = {}
    for fill in trades:
        fill_timestamp = getattr(fill, "timestamp", None)
        if fill_timestamp is None:
            continue
        timestamp = pd.Timestamp(fill_timestamp)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        if timestamp not in regime_labels.index:
            continue
        regime_name = str(regime_labels.loc[timestamp])
        counts[regime_name] = counts.get(regime_name, 0) + 1
    return counts
