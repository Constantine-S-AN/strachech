# Tutorial: QQQ 低频轮动（Regime + Sensitivity）

本教程演示一个低频 QQQ 轮动流程，并重点查看报告中的两块内容：

- `Regime Scorecard`
- `Cost/Slippage Sensitivity`

目标是让你在“策略逻辑可跑通”之外，也能快速判断策略在不同市场状态和不同成本假设下是否稳健。

## 1) 安装依赖

```bash
python -m pip install -e ".[dev]"
```

## 2) 准备数据

低频窗口通常更长，建议使用更长序列：

```bash
python scripts/make_demo_assets.py --output data/QQQ.csv --periods 900 --seed 7
```

## 3) 复制并调整配置

先基于已有示例复制一份低频版本：

```bash
cp examples/qqq_rotation/config.toml configs/examples/qqq_low_frequency_rotation.toml
```

把 `configs/examples/qqq_low_frequency_rotation.toml` 关键字段改成：

```toml
symbol = "QQQ"
data_path = "../../data"
strategy = "stratcheck.core.strategy:MovingAverageCrossStrategy"
initial_cash = 100000
timeframe = "1d"
bars_freq = "1d"
report_name = "qqq_low_frequency_rotation"
report_dir = "../../reports"

[cost_model]
type = "fixed_bps"
commission_bps = 2
slippage_bps = 1

[windows]
window_size = "12M"
step_size = "6M"

[strategy_params]
short_window = 30
long_window = 120
target_position_qty = 1.0
```

这组参数会明显降低交易频率，更接近“低频轮动”。

## 4) 运行报告

```bash
python -m stratcheck run --config configs/examples/qqq_low_frequency_rotation.toml
```

输出：

- `reports/qqq_low_frequency_rotation.html`
- `reports/assets/qqq_low_frequency_rotation_cost_sensitivity.png`

## 5) 解读 Regime Scorecard

打开 `reports/qqq_low_frequency_rotation.html`，找到 `Regime Scorecard` 面板，重点看：

- `regime`: 市场状态标签（趋势/波动组合）
- `bars_count`: 该状态样本数（样本太少要谨慎）
- `cagr` / `max_drawdown` / `win_rate`: 各状态下收益与风险
- `trade_count`: 该状态下触发交易次数

一个实用结论规则：只在单一 regime 好看、其余 regime 明显失效时，优先认为策略过拟合风险偏高。

## 6) 解读 Cost/Slippage Sensitivity

同一份报告中查看 `Cost/Slippage Sensitivity` 面板：

- `commission_bps` / `slippage_bps` / `spread_bps`: 成本假设网格
- `total_cost_bps`: 总成本
- `sharpe` / `cagr`: 成本上升后的指标退化程度

默认会扫描一个小网格（如 0 和 5 bps 组合）。你要关注的是“指标下降是否平滑”，而不是只看零成本那一格。

## 7) 运行 walk-forward 体检

```bash
python -m stratcheck healthcheck --config configs/examples/qqq_low_frequency_rotation.toml
```

输出：

- `reports/healthcheck_summary.json`

建议把 `healthcheck` 的最差窗口指标和 `Regime Scorecard` 结合起来看，判断是否存在“某些市场状态 + 某些时间窗口”双重脆弱。

## 复现清单

- 数据：`data/QQQ.csv`
- 配置：`configs/examples/qqq_low_frequency_rotation.toml`
- 报告：`reports/qqq_low_frequency_rotation.html`
- 体检摘要：`reports/healthcheck_summary.json`
