# Changelog

All notable changes to this project are documented in this file.

## [0.1.0] - 2026-02-16

### Added

- Initial `stratcheck` package with CLI commands:
  - `python -m stratcheck demo`
  - `python -m stratcheck run --config <path>`
  - `python -m stratcheck healthcheck --config <path>`
- Backtest engine with market order execution, trade logs, equity curve output, and pluggable cost models:
  - `FixedBpsCostModel`
  - `SpreadCostModel`
  - `MarketImpactToyModel`
- Data layer:
  - `DataProvider` + `CSVDataProvider`
  - `StooqCSVProvider` with local cache
- Strategy interface and built-in strategies:
  - `MovingAverageCrossStrategy`
  - `BuyAndHoldStrategy`
  - `VolatilityTargetStrategy`
  - `MeanReversionZScoreStrategy`
- Metrics and healthcheck modules:
  - CAGR, annual volatility, Sharpe, max drawdown, turnover, win rate, average trade PnL
  - Rolling-window walk-forward summary and JSON export
- Robustness analysis:
  - Bootstrap Sharpe confidence interval
  - Parameter sweep output
- Overfit risk flags with report rendering
- Reporting:
  - PNG charts (equity, drawdown, returns histogram)
  - HTML report with summary cards, worst-window highlight, and reproducibility section
- Batch experiment runner:
  - Run all configs in a directory
  - Generate `reports/index.html` and `reports/results.jsonl`
- CI:
  - Ruff lint/format check
  - Pytest
  - Demo report generation and report artifact upload
  - Healthcheck markdown summary to GitHub Job Summary
- Utility scripts:
  - `scripts/post_summary.py`
  - `scripts/make_demo_assets.py`
- Release engineering:
  - Packaging metadata improved for wheel/sdist publishing
  - Added `Dockerfile`, `docker-compose.yml`, and `.dockerignore`
  - Added docs pages (`Quickstart`, `QQQ 轮动策略教程`)
  - Added `examples/qqq_rotation` reproducible sample project
  - Added CI tag release job to build distribution and run demo
- Performance upgrades:
  - Parquet cache support with cache index metadata (`CSVDataProvider` + `stratcheck.perf.parquet`)
  - Incremental metrics and plot-series support (`stratcheck.perf.incremental`)
  - Process-parallel experiment execution via `ProcessPoolExecutor`
  - Added `scripts/benchmark_experiments.py` for 100-config runtime benchmarking
