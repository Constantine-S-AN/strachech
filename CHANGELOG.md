# Changelog

All notable changes to this project are documented in this file.

## [0.3.0] - Planned

### Milestone

- 执行引擎增强：扩展执行算法（如 VWAP/POV）与更细粒度执行成本分析
- 风险体系增强：组合级风险预算、压力测试与熔断联动
- 编排可靠性：runner 资源配额策略、恢复与重放能力增强
- 平台化能力：dashboard 历史对比与筛选增强，发布工件规范化

## [0.2.0] - Planned

### Milestone

- `RealPaperConnector`：REST 下单/撤单、WebSocket 更新、速率限制、重连、幂等下单
- `Execution Quality`：滑点、成交延迟、撤单率、部分成交占比指标与报告面板
- 风险规则 DSL：最大回撤、最大仓位、最大日交易次数、异常数据停机
- 告警通道：console + email/telegram/webhook（后两者先占位）
- Orchestrator：单进程管理多 runner，独立日志与数据库命名空间
- Dashboard：Leaderboard 多指标排序 + Live 状态页
- Secrets 与交付：环境变量/本地加密存储、CI 脱敏、Docker 与 Make 命令

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
