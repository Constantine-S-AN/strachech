"""Baseline strategy library for stratcheck."""

from stratcheck.strategies.baselines import (
    BuyAndHoldStrategy,
    MeanReversionZScoreStrategy,
    VolatilityTargetStrategy,
)

__all__ = [
    "BuyAndHoldStrategy",
    "VolatilityTargetStrategy",
    "MeanReversionZScoreStrategy",
]
