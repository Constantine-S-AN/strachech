"""Performance utilities: parquet cache, incremental updates, and parallel helpers."""

from stratcheck.perf.incremental import (
    IncrementalMetricsState,
    IncrementalPlotState,
    compute_metrics_incremental,
    prepare_incremental_plot_series,
)
from stratcheck.perf.parquet import load_or_build_parquet_cache, parquet_engine_available

__all__ = [
    "IncrementalMetricsState",
    "IncrementalPlotState",
    "compute_metrics_incremental",
    "load_or_build_parquet_cache",
    "parquet_engine_available",
    "prepare_incremental_plot_series",
]
