# QQQ 轮动策略从 0 到报告

这个教程用一个“QQQ / 现金”轮动思路演示完整流程：

- 当短均线上穿长均线时持有 QQQ
- 当短均线下穿长均线时回到现金

策略类使用 `MovingAverageCrossStrategy`，通过订单意图实现仓位切换。

## 步骤 1：安装依赖

```bash
python -m pip install -e ".[dev]"
```

## 步骤 2：准备可复现数据

```bash
python scripts/make_demo_assets.py --output data/QQQ.csv --periods 180 --seed 7
```

## 步骤 3：准备配置

直接使用仓库内示例：

```bash
cat examples/qqq_rotation/config.toml
```

关键参数：

- `short_window = 15`
- `long_window = 50`
- `target_position_qty = 1.0`

## 步骤 4：运行单次回测报告

```bash
python -m stratcheck run --config examples/qqq_rotation/config.toml
```

输出：

- `reports/qqq_rotation_tutorial.html`
- `reports/assets/qqq_rotation_tutorial_*.png`

## 步骤 5：运行 walk-forward 体检

```bash
python -m stratcheck healthcheck --config examples/qqq_rotation/config.toml
```

输出：

- `reports/healthcheck_summary.json`
- 报告中包含窗口级指标表与最差窗口高亮

## 步骤 6：可选容器运行

```bash
docker compose run --rm runner
```

默认会生成实验报告与 paper sqlite，可继续执行：

```bash
docker compose run --rm dashboard
```

## 复现检查清单

- 数据文件：`data/QQQ.csv`
- 配置文件：`examples/qqq_rotation/config.toml`
- 回测报告：`reports/qqq_rotation_tutorial.html`
- 体检摘要：`reports/healthcheck_summary.json`
