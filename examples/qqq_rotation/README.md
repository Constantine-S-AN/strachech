# QQQ Rotation Example

这个示例项目展示如何在 `stratcheck` 中复现一个最小轮动策略流程。

## 运行

```bash
python scripts/make_demo_assets.py --output data/QQQ.csv --periods 180 --seed 7
python -m stratcheck run --config examples/qqq_rotation/config.toml
python -m stratcheck healthcheck --config examples/qqq_rotation/config.toml
```

## 输出

- `reports/qqq_rotation_tutorial.html`
- `reports/healthcheck_summary.json`
