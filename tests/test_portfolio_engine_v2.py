from __future__ import annotations

import pandas as pd
import pytest
from stratcheck.portfolio import Optimizer, PortfolioConstraints, RebalancePlanner


def test_optimizer_enforces_asset_and_sector_constraints() -> None:
    optimizer = Optimizer()
    desired_weights = pd.Series({"AAPL": 0.7, "MSFT": 0.4, "XOM": 0.3})
    sectors = {"AAPL": "tech", "MSFT": "tech", "XOM": "energy"}
    constraints = PortfolioConstraints(
        max_weight_per_asset=0.5,
        max_sector_exposure={"tech": 0.6, "energy": 0.5},
        max_gross_exposure=1.0,
        max_net_exposure=1.0,
    )

    optimized_weights = optimizer.optimize(
        desired_weights=desired_weights,
        current_weights=None,
        sectors=sectors,
        constraints=constraints,
    )

    assert optimized_weights["AAPL"] == pytest.approx(1.0 / 3.0, rel=1e-6)
    assert optimized_weights["MSFT"] == pytest.approx(4.0 / 15.0, rel=1e-6)
    assert optimized_weights["XOM"] == pytest.approx(0.3, rel=1e-6)

    assert float(optimized_weights.max()) <= 0.5 + 1e-12
    tech_exposure = float(optimized_weights.loc[["AAPL", "MSFT"]].sum())
    energy_exposure = float(optimized_weights.loc[["XOM"]].sum())
    assert tech_exposure <= 0.6 + 1e-12
    assert energy_exposure <= 0.5 + 1e-12


def test_optimizer_enforces_max_turnover_limit() -> None:
    optimizer = Optimizer()
    current_weights = pd.Series({"AAPL": 0.5, "MSFT": 0.5})
    desired_weights = pd.Series({"AAPL": 1.0, "MSFT": 0.0})
    constraints = PortfolioConstraints(
        max_weight_per_asset=1.0,
        max_turnover=0.2,
        max_gross_exposure=1.0,
        max_net_exposure=1.0,
    )

    optimized_weights = optimizer.optimize(
        desired_weights=desired_weights,
        current_weights=current_weights,
        constraints=constraints,
    )
    turnover = optimizer.compute_turnover(
        current_weights=current_weights,
        target_weights=optimized_weights,
    )

    assert optimized_weights["AAPL"] == pytest.approx(0.7, rel=1e-6)
    assert optimized_weights["MSFT"] == pytest.approx(0.3, rel=1e-6)
    assert turnover == pytest.approx(0.2, rel=1e-6)


def test_rebalance_planner_creates_expected_trade_instructions() -> None:
    planner = RebalancePlanner()
    trades = planner.plan_rebalance(
        current_positions={"AAPL": 10.0, "MSFT": 0.0},
        target_weights={"AAPL": 0.25, "MSFT": 0.25},
        prices={"AAPL": 100.0, "MSFT": 100.0},
        portfolio_value=2_000.0,
    )

    assert len(trades) == 2

    aapl_trade = trades[0]
    assert aapl_trade.symbol == "AAPL"
    assert aapl_trade.side == "sell"
    assert aapl_trade.quantity == pytest.approx(5.0)
    assert aapl_trade.trade_notional == pytest.approx(500.0)

    msft_trade = trades[1]
    assert msft_trade.symbol == "MSFT"
    assert msft_trade.side == "buy"
    assert msft_trade.quantity == pytest.approx(5.0)
    assert msft_trade.trade_notional == pytest.approx(500.0)
