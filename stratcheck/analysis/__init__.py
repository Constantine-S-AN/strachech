"""Analysis helpers for sensitivity, regimes, and execution quality."""

from stratcheck.analysis.execution_quality import compute_execution_quality_scorecard
from stratcheck.analysis.regimes import classify_market_regimes, compute_regime_scorecard
from stratcheck.analysis.sensitivity import plot_cost_sensitivity, run_cost_sensitivity_scan

__all__ = [
    "compute_execution_quality_scorecard",
    "classify_market_regimes",
    "compute_regime_scorecard",
    "plot_cost_sensitivity",
    "run_cost_sensitivity_scan",
]
