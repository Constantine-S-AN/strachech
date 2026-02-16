# Stratcheck 文档

`stratcheck` 是一个策略体检与报告工具，面向“可复现”的量化研究流程。

## 文档导航

- [Quickstart](quickstart.md)
- [Secrets 安全配置指南](security-secrets.md)
- [Portfolio Showcase Guide](showcase.md)
- [Tutorial: QQQ 轮动策略从 0 到报告](tutorials/qqq-rotation-from-zero.md)
- [Tutorial: QQQ 低频轮动（Regime + Sensitivity）](tutorials/qqq-low-frequency-rotation-regime-sensitivity.md)
- [Tutorial: 执行算法（TWAP）+ Execution Quality 报告](tutorials/twap-execution-quality.md)
- [Tutorial: 10 分钟写一个新策略并跑出报告](tutorials/new-strategy-in-10-min.md)
- [常见坑：幸存者偏差、未来函数、成本假设](common-pitfalls.md)
- [Release Milestones（v0.2.0 / v0.3.0）](release-milestones.md)

## 你能得到什么

- 一条命令生成策略报告（HTML + 图表）
- Walk-forward 稳健性体检（窗口汇总 + JSON）
- 批量实验、Dashboard、审计回放
- 可用于 CI 的 release-ready 工程骨架
