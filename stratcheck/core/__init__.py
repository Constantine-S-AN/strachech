"""Core backtest components."""

from stratcheck.core.backtest import (
    BacktestEngine,
    BacktestResult,
    CostModel,
    FixedBpsCostModel,
    MarketImpactToyModel,
    OrderRecord,
    SpreadCostModel,
    build_cost_model,
    run_moving_average_backtest,
)
from stratcheck.core.bundle import (
    ReproduceResult,
    RunSnapshot,
    build_run_snapshot,
    bundle_snapshot,
    detect_source_data_path,
    reproduce_snapshot,
)
from stratcheck.core.calendar import (
    CalendarWindow,
    MarketCalendar,
    normalize_bars_freq,
    periods_per_year,
)
from stratcheck.core.corporate_actions import (
    CorporateAction,
    CorporateActionType,
    apply_corporate_actions_to_bars,
    filter_corporate_actions,
    load_corporate_actions_file,
    parse_corporate_actions,
    summarize_corporate_actions,
)
from stratcheck.core.data import BarsSchema, CSVDataProvider, DataProvider, generate_random_ohlcv
from stratcheck.core.data_sources import StooqCSVProvider
from stratcheck.core.experiments import ExperimentRunner
from stratcheck.core.healthcheck import run_healthcheck
from stratcheck.core.metrics import compute_metrics
from stratcheck.core.overfit import RiskFlag, evaluate_overfit_risk
from stratcheck.core.robustness import bootstrap_sharpe_ci, parameter_sweep
from stratcheck.core.strategy import (
    Fill,
    MovingAverageCrossStrategy,
    OrderIntent,
    PortfolioState,
    Strategy,
)
from stratcheck.core.universe import (
    CSVUniverseProvider,
    UniverseBacktestResult,
    UniverseProvider,
    run_dynamic_universe_backtest,
)

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "BarsSchema",
    "CalendarWindow",
    "CorporateAction",
    "CorporateActionType",
    "CSVDataProvider",
    "CSVUniverseProvider",
    "CostModel",
    "DataProvider",
    "ExperimentRunner",
    "Fill",
    "FixedBpsCostModel",
    "MarketCalendar",
    "MarketImpactToyModel",
    "MovingAverageCrossStrategy",
    "OrderRecord",
    "OrderIntent",
    "PortfolioState",
    "ReproduceResult",
    "RiskFlag",
    "RunSnapshot",
    "SpreadCostModel",
    "StooqCSVProvider",
    "Strategy",
    "UniverseBacktestResult",
    "UniverseProvider",
    "apply_corporate_actions_to_bars",
    "bootstrap_sharpe_ci",
    "build_run_snapshot",
    "build_cost_model",
    "bundle_snapshot",
    "compute_metrics",
    "detect_source_data_path",
    "evaluate_overfit_risk",
    "filter_corporate_actions",
    "generate_random_ohlcv",
    "load_corporate_actions_file",
    "normalize_bars_freq",
    "parse_corporate_actions",
    "parameter_sweep",
    "periods_per_year",
    "reproduce_snapshot",
    "run_healthcheck",
    "run_dynamic_universe_backtest",
    "run_moving_average_backtest",
    "summarize_corporate_actions",
]
