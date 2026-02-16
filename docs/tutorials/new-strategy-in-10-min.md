# Tutorial: 10 分钟写一个新策略并跑出报告

这个教程目标是：从 0 开始，10 分钟内写出一个可运行的策略插件，并生成 HTML 报告。

## 1) 生成策略模板

```bash
python -m stratcheck create-strategy MyStrategy
```

会生成两个文件：

- `stratcheck/strategies/my_strategy.py`
- `configs/examples/my_strategy.toml`

## 2) 实现策略逻辑

打开 `stratcheck/strategies/my_strategy.py`，你只需要改 `build_signals(...)`。

模板已经提供：

- `StrategyTemplate`：统一输入输出
- `self.signal(...)`：快速构造信号
- 自动日志与 `signals` 记录（`get_signal_frame()` 可直接查看）

## 3) 跑报告

```bash
python -m stratcheck run --config configs/examples/my_strategy.toml
```

输出：

- `reports/my_strategy.html`
- `reports/assets/*.png`

## 4) 常见迭代

1. 调整参数：修改 `configs/examples/my_strategy.toml` 里的 `[strategy_params]`
2. 做稳健性：添加 `[windows]` 滚动窗口并运行 `healthcheck`
3. 调优：打开 `[tuning]` 配置做 `grid` 或 `random` 搜索

## 5) 可选：覆盖已有模板

如果你要覆盖同名模板：

```bash
python -m stratcheck create-strategy MyStrategy --force
```
