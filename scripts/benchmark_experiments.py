"""Benchmark experiment runtime with sequential vs process-parallel execution."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from stratcheck.core.experiments import ExperimentRunner


def run_benchmark(
    workspace: Path,
    config_count: int = 100,
    workers: int | None = None,
    bars_count: int = 240,
) -> dict[str, float | int]:
    """Generate N configs and benchmark sequential vs parallel experiment runs."""
    if config_count <= 0:
        msg = "config_count must be positive."
        raise ValueError(msg)
    if bars_count <= 20:
        msg = "bars_count must be greater than 20."
        raise ValueError(msg)
    if workers is not None and workers <= 0:
        msg = "workers must be a positive integer when provided."
        raise ValueError(msg)

    effective_workers = workers if workers is not None else max(1, os.cpu_count() or 1)

    workspace.mkdir(parents=True, exist_ok=True)
    data_dir = workspace / "data"
    configs_dir = workspace / "configs"
    sequential_reports_dir = workspace / "reports_seq"
    parallel_reports_dir = workspace / "reports_par"
    data_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    _write_sample_csv(data_dir=data_dir, symbol="BTCUSDT", bars_count=bars_count)
    _write_many_configs(configs_dir=configs_dir, data_dir=data_dir, config_count=config_count)

    sequential_runner = ExperimentRunner(
        configs_dir=configs_dir,
        output_dir=sequential_reports_dir,
        parallel=False,
    )
    started_at = time.perf_counter()
    sequential_runner.run_all()
    sequential_seconds = time.perf_counter() - started_at

    parallel_runner = ExperimentRunner(
        configs_dir=configs_dir,
        output_dir=parallel_reports_dir,
        parallel=True,
        max_workers=effective_workers,
    )
    started_at = time.perf_counter()
    parallel_runner.run_all()
    parallel_seconds = time.perf_counter() - started_at

    speedup = sequential_seconds / parallel_seconds if parallel_seconds > 0 else 0.0
    summary = {
        "config_count": int(config_count),
        "bars_count": int(bars_count),
        "workers": int(effective_workers),
        "sequential_seconds": float(sequential_seconds),
        "parallel_seconds": float(parallel_seconds),
        "speedup": float(speedup),
    }
    summary_path = workspace / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for benchmark script."""
    parser = argparse.ArgumentParser(description="Benchmark stratcheck experiment runtime.")
    parser.add_argument(
        "--workspace",
        default="reports/benchmarks/exp100",
        help="Benchmark workspace directory. Default: reports/benchmarks/exp100",
    )
    parser.add_argument(
        "--configs",
        type=int,
        default=100,
        help="Number of generated config files. Default: 100",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Max workers for process pool. Default: use executor default",
    )
    parser.add_argument(
        "--bars",
        type=int,
        default=240,
        help="Bar count in synthetic dataset. Default: 240",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    summary = run_benchmark(
        workspace=Path(args.workspace),
        config_count=args.configs,
        workers=args.workers,
        bars_count=args.bars,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _write_sample_csv(data_dir: Path, symbol: str, bars_count: int) -> None:
    timestamps = pd.date_range("2023-01-01", periods=bars_count, freq="D", tz="UTC")
    random_generator = np.random.default_rng(7)
    close_prices = 100.0 + np.cumsum(random_generator.normal(loc=0.08, scale=0.9, size=bars_count))
    bars = pd.DataFrame(
        {
            "timestamp": timestamps.astype(str),
            "open": close_prices - 0.25,
            "high": close_prices + 1.10,
            "low": close_prices - 1.05,
            "close": close_prices,
            "volume": np.full(bars_count, 1_000),
        }
    )
    bars.to_csv(data_dir / f"{symbol}.csv", index=False)


def _write_many_configs(configs_dir: Path, data_dir: Path, config_count: int) -> None:
    for index in range(config_count):
        short_window = 5 + (index % 10)
        long_window = 20 + (index % 20)
        if short_window >= long_window:
            long_window = short_window + 5

        config_text = f"""
symbol = "BTCUSDT"
data_path = "{data_dir.as_posix()}"
strategy = "stratcheck.core.strategy:MovingAverageCrossStrategy"
initial_cash = 100000
timeframe = "1d"
bars_freq = "1d"
report_name = "bench_{index:03d}"

[cost_model]
commission_bps = 2
slippage_bps = 1

[windows]
window_size = "60D"
step_size = "30D"

[performance]
use_parquet_cache = true

[strategy_params]
short_window = {short_window}
long_window = {long_window}
target_position_qty = 1.0
"""
        (configs_dir / f"bench_{index:03d}.toml").write_text(
            config_text.strip() + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    raise SystemExit(main())
