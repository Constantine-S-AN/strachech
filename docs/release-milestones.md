# Release Milestones

本页用于对外同步 `v0.2.0` 与 `v0.3.0` 的里程碑范围、验收标准，以及公开 demo 的复现方式。

## v0.2.0 Milestone

目标：完成可复现研究到 paper 运行的一体化闭环，并提供可审计的运行与风险控制能力。

范围：

- 连接器与执行：
  - `RealPaperConnector`（REST 下单/撤单 + WebSocket 订单更新）
  - 速率限制、断线重连、幂等下单（`client_order_id`）
  - 默认仅允许 paper 环境（配置开关显式控制）
- 执行质量：
  - `execution_quality` 指标（滑点、成交延迟、撤单率、部分成交占比）
  - 报告新增 `Execution Quality` 面板
- 风险规则：
  - 规则对象 DSL
  - 最大回撤、最大仓位、最大日交易次数、异常数据停机
  - 风险命中审计与 dashboard 展示
- 运维告警：
  - `console` + `email/telegram/webhook`（后两者可占位）
  - `kill switch` / 数据中断 / 订单卡住 / 回撤超阈值
- 编排与看板：
  - 多 runner 进程编排
  - 每个 runner 独立日志与数据库命名空间
  - dashboard `Leaderboard` 与 `Live Status`
- Secrets 与交付：
  - 环境变量 + 本地加密存储（可选）
  - CI 禁止打印敏感信息
  - `Dockerfile`、`docker-compose`、`make demo`、`make dashboard`

验收标准（DoD）：

- 主流程命令可运行：`demo` / `run` / `healthcheck` / `dashboard`
- 关键模块测试通过：连接器、风险、告警、编排、secrets
- 文档包含教程、常见坑、安全配置与复现说明

## v0.3.0 Milestone

目标：在 v0.2.0 的基础上，提升生产可用性、执行深度与跨策略运行规模。

范围：

- 执行引擎增强：
  - 扩展算法（如 VWAP/POV）
  - 更细粒度交易成本与成交路径统计
- 风险与组合层：
  - 组合级风险预算
  - 压力场景和熔断联动
- 编排与可靠性：
  - runner 资源配额可配置化（CPU/内存/并发）
  - 异常恢复策略与重放工具增强
- 平台化能力：
  - dashboard 指标筛选和历史对比增强
  - 发布工件与版本化产物管理规范化

验收标准（DoD）：

- 在多策略并行下保持稳定运行
- 回放、审计、告警链路闭环验证
- 发布文档、治理文档和 demo 资产完整

## Public Demo

### Demo Screenshot

![Release Demo Report](images/release-demo-report.png)

截图文件：`docs/images/release-demo-report.png`

### Reproduction Commands

1. 生成可复现数据：

```bash
python scripts/make_demo_assets.py --output data/QQQ.csv --periods 240 --seed 7
```

2. 运行 demo 报告：

```bash
python -m stratcheck demo --output reports/release_demo.html --periods 240 --seed 7
```

3. 生成 dashboard：

```bash
python -m stratcheck dashboard \
  --results-jsonl reports/results.jsonl \
  --db reports/paper_trading.sqlite \
  --output reports/dashboard.html \
  --reports-dir reports
```

4. 使用容器复现（可选）：

```bash
docker compose run --rm runner
docker compose run --rm dashboard
```

## Open-Source Governance

- License: `LICENSE`
- Contribution guide: `CONTRIBUTING.md`
- Code owners: `.github/CODEOWNERS`
- Security policy: `SECURITY.md`
