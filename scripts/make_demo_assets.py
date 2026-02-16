"""Generate a tiny, reproducible CSV dataset for local demo runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_demo_csv(
    output_csv: Path,
    periods: int = 90,
    seed: int = 7,
    start: str = "2024-01-02",
) -> Path:
    """Create a compact OHLCV CSV file for demo use."""
    if periods < 30:
        msg = "periods must be at least 30."
        raise ValueError(msg)

    random_generator = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start, periods=periods, freq="B", tz="UTC")

    returns = random_generator.normal(loc=0.0006, scale=0.011, size=periods)
    close_prices = 380.0 * np.exp(np.cumsum(returns))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0] * (1.0 - 0.001)

    intra_spread = random_generator.uniform(0.002, 0.012, size=periods)
    high_prices = np.maximum(open_prices, close_prices) * (1.0 + intra_spread / 2.0)
    low_prices = np.minimum(open_prices, close_prices) * (1.0 - intra_spread / 2.0)
    volumes = random_generator.integers(200_000, 1_200_000, size=periods)

    bars = pd.DataFrame(
        {
            "timestamp": timestamps.astype(str),
            "open": np.round(open_prices, 4),
            "high": np.round(high_prices, 4),
            "low": np.round(low_prices, 4),
            "close": np.round(close_prices, 4),
            "volume": volumes.astype(int),
        }
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    bars.to_csv(output_csv, index=False)
    return output_csv


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for demo asset generation."""
    parser = argparse.ArgumentParser(
        description="Generate a tiny reproducible OHLCV CSV for stratcheck demos.",
    )
    parser.add_argument(
        "--output",
        default="data/QQQ.csv",
        help="Output CSV path. Default: data/QQQ.csv",
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=90,
        help="Number of bars to generate. Default: 90",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducibility. Default: 7",
    )
    parser.add_argument(
        "--start",
        default="2024-01-02",
        help="Start date for business-day bars. Default: 2024-01-02",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    output_path = make_demo_csv(
        output_csv=Path(args.output),
        periods=args.periods,
        seed=args.seed,
        start=args.start,
    )
    file_size = output_path.stat().st_size
    print(f"Demo CSV generated: {output_path.resolve()} ({file_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
