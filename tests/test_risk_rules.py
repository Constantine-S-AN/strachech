from __future__ import annotations

import pandas as pd
from stratcheck.risk import (
    DataAnomalyHaltRule,
    MaxDailyTradesRule,
    MaxDrawdownRule,
    MaxPositionRule,
    RuleBook,
    RuleContext,
)


def test_rulebook_evaluates_halt_and_block_rules() -> None:
    context = RuleContext(
        timestamp=pd.Timestamp("2024-01-01T00:00:00Z"),
        bar_index=10,
        equity=80.0,
        peak_equity=100.0,
        projected_position_qty=3.0,
        order_side="buy",
        order_qty=1.0,
        daily_trade_count=4,
    )
    rule_book = RuleBook(
        rules=[
            MaxDrawdownRule(max_drawdown=0.15),
            MaxPositionRule(max_abs_position_qty=2.0),
            MaxDailyTradesRule(max_trades_per_day=3),
        ]
    )

    halt_hits = rule_book.evaluate(context=context, actions={"halt"})
    block_hits = rule_book.evaluate(context=context, actions={"block"})

    assert len(halt_hits) == 1
    assert halt_hits[0].reason == "max_drawdown_exceeded"
    assert len(block_hits) == 2
    assert {item.reason for item in block_hits} == {
        "max_abs_position_qty_breached",
        "max_daily_trades_exceeded",
    }


def test_data_anomaly_rule_detects_gap_and_bad_bar() -> None:
    rule = DataAnomalyHaltRule(max_data_gap_steps=2, max_abs_return=0.20)

    gap_hit = rule.evaluate(
        RuleContext(
            timestamp=pd.Timestamp("2024-01-10T00:00:00Z"),
            bar_index=5,
            previous_timestamp=pd.Timestamp("2024-01-01T00:00:00Z"),
            expected_interval=pd.Timedelta(days=1),
        )
    )
    assert gap_hit is not None
    assert gap_hit.reason == "data_interruption_detected"

    bar_hit = rule.evaluate(
        RuleContext(
            timestamp=pd.Timestamp("2024-01-02T00:00:00Z"),
            bar_index=1,
            bar=pd.Series(
                {"open": 100.0, "high": 101.0, "low": 99.0, "close": -1.0, "volume": 100.0}
            ),
        )
    )
    assert bar_hit is not None
    assert bar_hit.reason == "abnormal_data_detected"


def test_data_anomaly_rule_detects_abnormal_return_jump() -> None:
    rule = DataAnomalyHaltRule(max_data_gap_steps=2, max_abs_return=0.10)

    return_hit = rule.evaluate(
        RuleContext(
            timestamp=pd.Timestamp("2024-01-02T00:00:00Z"),
            bar_index=1,
            previous_close=100.0,
            bar=pd.Series(
                {"open": 100.0, "high": 125.0, "low": 99.0, "close": 120.0, "volume": 100.0}
            ),
        )
    )
    assert return_hit is not None
    assert return_hit.reason == "abnormal_return_detected"
