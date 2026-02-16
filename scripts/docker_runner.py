"""Container runner entrypoint for producing demo artifacts used by dashboard."""

from __future__ import annotations

from pathlib import Path

from stratcheck.connectors import LivePaperRunner, PaperBrokerConnector
from stratcheck.core import CSVDataProvider, ExperimentRunner
from stratcheck.strategies import BuyAndHoldStrategy


def main() -> int:
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    experiment_runner = ExperimentRunner(
        configs_dir=Path("configs/examples"),
        output_dir=reports_dir,
        parallel=False,
    )
    _, index_path = experiment_runner.run_all()

    bars = CSVDataProvider(data_dir=Path("data")).get_bars(symbol="QQQ")
    connector = PaperBrokerConnector(
        initial_cash=100_000.0,
        max_fill_ratio_per_step=1.0,
        max_volume_share=1.0,
    )
    runner = LivePaperRunner(
        connector=connector,
        sqlite_path=reports_dir / "paper_trading.sqlite",
        symbol="QQQ",
    )
    run_result = runner.run(
        strategy=BuyAndHoldStrategy(target_position_qty=1.0),
        bars=bars.iloc[:90],
    )

    print(f"Experiments index: {index_path}")
    print(f"Paper trading sqlite: {run_result.sqlite_path}")
    print(f"Paper run id: {run_result.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
