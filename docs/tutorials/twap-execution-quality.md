# Tutorial: 执行算法（TWAP）+ Execution Quality 报告

本教程目标：

1. 用 `create-strategy` 快速生成策略模板
2. 改成一个简单可运行的 TWAP 切片执行策略
3. 在报告中读取 `Execution Quality` 指标（滑点、延迟、撤单率、部分成交占比）

## 1) 生成策略模板

```bash
python -m stratcheck create-strategy QQQTwapStrategy
```

生成文件：

- `stratcheck/strategies/qqq_twap_strategy.py`
- `configs/examples/qqq_twap_strategy.toml`

## 2) 填入 TWAP 策略代码

将 `stratcheck/strategies/qqq_twap_strategy.py` 替换为：

```python
"""TWAP execution example strategy for QQQ."""

from __future__ import annotations

import pandas as pd

from stratcheck.core.strategy import PortfolioState
from stratcheck.sdk import StrategySignal, StrategyTemplate


class QQQTwapStrategy(StrategyTemplate):
    """MA direction + TWAP slicing for position transitions."""

    def __init__(
        self,
        short_window: int = 20,
        long_window: int = 80,
        target_position_qty: float = 1.0,
        twap_slices: int = 4,
        rebalance_tolerance: float = 1e-6,
    ) -> None:
        super().__init__(strategy_name="QQQTwapStrategy")
        if short_window < 2:
            msg = "short_window must be >= 2."
            raise ValueError(msg)
        if long_window <= short_window:
            msg = "long_window must be greater than short_window."
            raise ValueError(msg)
        if target_position_qty <= 0:
            msg = "target_position_qty must be positive."
            raise ValueError(msg)
        if twap_slices < 1:
            msg = "twap_slices must be >= 1."
            raise ValueError(msg)

        self.short_window = int(short_window)
        self.long_window = int(long_window)
        self.target_position_qty = float(target_position_qty)
        self.twap_slices = int(twap_slices)
        self.rebalance_tolerance = float(rebalance_tolerance)

        self._active_side: str | None = None
        self._remaining_qty = 0.0
        self._remaining_slices = 0
        self._last_target_qty = 0.0

    def build_signals(
        self,
        bars: pd.DataFrame,
        portfolio_state: PortfolioState,
    ) -> list[StrategySignal]:
        if len(bars) < self.long_window + 1:
            return []

        close_prices = bars["close"].astype(float)
        short_average = close_prices.rolling(
            window=self.short_window,
            min_periods=self.short_window,
        ).mean().iloc[-1]
        long_average = close_prices.rolling(
            window=self.long_window,
            min_periods=self.long_window,
        ).mean().iloc[-1]
        if pd.isna(short_average) or pd.isna(long_average):
            return []

        desired_qty = self.target_position_qty if short_average > long_average else 0.0
        current_qty = float(portfolio_state.position_qty)
        quantity_gap = desired_qty - current_qty

        schedule_done = (
            self._remaining_slices <= 0 or self._remaining_qty <= self.rebalance_tolerance
        )
        target_changed = abs(desired_qty - self._last_target_qty) > self.rebalance_tolerance
        if abs(quantity_gap) > self.rebalance_tolerance and (schedule_done or target_changed):
            self._active_side = "buy" if quantity_gap > 0 else "sell"
            self._remaining_qty = abs(quantity_gap)
            self._remaining_slices = self.twap_slices
            self._last_target_qty = desired_qty

        if (
            self._active_side is None
            or self._remaining_slices <= 0
            or self._remaining_qty <= self.rebalance_tolerance
        ):
            return []

        slice_qty = self._remaining_qty / float(self._remaining_slices)
        if slice_qty <= self.rebalance_tolerance:
            return []

        self._remaining_qty = max(self._remaining_qty - slice_qty, 0.0)
        self._remaining_slices -= 1
        executed_slice = self.twap_slices - self._remaining_slices

        return [
            self.signal(
                side=self._active_side,
                qty=float(slice_qty),
                reason=f"twap slice {executed_slice}/{self.twap_slices}",
                metadata={
                    "desired_qty": float(desired_qty),
                    "current_qty": float(current_qty),
                    "remaining_qty": float(self._remaining_qty),
                },
            )
        ]
```

## 3) 准备数据与配置

生成数据：

```bash
python scripts/make_demo_assets.py --output data/QQQ.csv --periods 600 --seed 11
```

将 `configs/examples/qqq_twap_strategy.toml` 改成：

```toml
symbol = "QQQ"
data_path = "../../data"
strategy = "stratcheck.strategies.qqq_twap_strategy:QQQTwapStrategy"
initial_cash = 100000
timeframe = "1d"
bars_freq = "1d"
report_name = "qqq_twap_execution"
report_dir = "../../reports"

[cost_model]
type = "fixed_bps"
commission_bps = 2
slippage_bps = 1

[windows]
window_size = "6M"
step_size = "3M"

[strategy_params]
short_window = 20
long_window = 80
target_position_qty = 1.0
twap_slices = 4
```

## 4) 运行并生成报告

```bash
python -m stratcheck run --config configs/examples/qqq_twap_strategy.toml
```

输出：

- `reports/qqq_twap_execution.html`
- `reports/assets/*.png`

## 5) 查看 Execution Quality 面板

打开 `reports/qqq_twap_execution.html`，查看 `Execution Quality`：

- `Avg/Median Slippage (bps)`：`signal_price` 对比 `fill_price` 的偏差
- `Avg/Median Latency Seconds` 与 `Avg/Median Latency Bars`：下单到成交延迟
- `Cancel Rate`：撤单比例
- `Partial Fill Ratio`：部分成交占比

同时还会有 `orders_total`、`filled_orders`、`canceled_orders` 等总览字段，便于你快速判断执行健康度。

## 6) 建议的对照实验

将 `twap_slices` 改成 `1` 再跑一次，比较两份报告的 `Execution Quality` 指标，判断切片执行是否改善了滑点与延迟。

备注：在简化回测环境下，撤单率和部分成交占比可能接近 0；在 paper/真实连接器和更复杂订单类型下，这两项通常更有信息量。
