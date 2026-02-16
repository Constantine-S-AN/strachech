# Interview Pack: Stratcheck

## 30-Second Elevator Pitch

### English
Stratcheck is a delivery-first quant research toolkit that turns strategy experiments into auditable artifacts. It combines config-driven runs, walk-forward health checks, execution-quality diagnostics, and static report publishing, so reviewers can verify both performance and process integrity from one link.

### 中文
Stratcheck 是一个“交付优先”的量化研究工具链：把策略实验变成可审计的交付物。它通过配置驱动运行、walk-forward 体检、执行质量诊断和静态页面发布，让面试官或团队在一个链接里同时看到结果与过程可信度。

## 2-Minute Expansion

### 1) Trustworthy (可信)
- 结果不只看收益，还包含风险、执行质量、窗口最差表现等指标。
- 支持审计回放（audit replay），可以从 `run_id` 追溯关键事件链路。

### 2) Reproducible (可复现)
- 统一以 config 驱动，降低“本地环境特例”导致的偏差。
- 支持 `bundle` / `reproduce`，可跨机器复现同一报告结论。

### 3) Observable (可观测)
- 输出标准化 report/dashboard，覆盖实验排行、运行状态与关键图表。
- 在 CI 中自动生成和上传产物，便于审查和追踪版本变化。

### 4) Extensible (可扩展)
- 策略、成本模型、数据源、分析模块都可插拔。
- 能从单策略报告演进到多策略实验管理与运营视角展示。

## Common Follow-up Questions

### Q1: How do you handle cost assumptions?
- Treat costs as first-class config inputs (commission/slippage/spread/impact models).
- Always run sensitivity checks to avoid relying on one optimistic setting.
- In interviews, highlight that decision quality depends on stability under cost perturbation.

### Q2: How do you avoid look-ahead bias (future leakage)?
- Enforce time-ordered data processing and signal generation.
- Separate feature/signal timestamps from execution timestamps.
- Use walk-forward windows and replay checks to catch suspiciously perfect behavior.

### Q3: How do you evaluate robustness and overfitting risk?
- Use walk-forward metrics (especially worst-window performance) as gating criteria.
- Combine bootstrap confidence intervals and parameter-sweep diagnostics.
- Prefer strategies that are slightly weaker but stable over those that are brittle at one setting.
