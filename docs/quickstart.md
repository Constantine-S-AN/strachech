# Quickstart

## 1. 安装

开发模式（本地改代码即时生效）：

```bash
python -m pip install -e ".[dev]"
```

发布版本（PyPI）：

```bash
pip install stratcheck
```

## 2. 生成示例数据

```bash
python scripts/make_demo_assets.py --output data/QQQ.csv --periods 120 --seed 7
```

## 3. 运行示例策略

```bash
python -m stratcheck run --config configs/examples/buy_and_hold.toml
```

输出：

- `reports/qqq_buy_and_hold.html`
- `reports/assets/*.png`

## 4. 运行健康体检

```bash
python -m stratcheck healthcheck --config configs/examples/volatility_target.toml
```

输出：

- `reports/qqq_volatility_target.html`
- `reports/healthcheck_summary.json`

## 5. 查看实验面板

```bash
python -m stratcheck dashboard \
  --results-jsonl reports/results.jsonl \
  --db reports/paper_trading.sqlite \
  --output reports/dashboard.html \
  --reports-dir reports
```

输出：

- `reports/dashboard.html`
