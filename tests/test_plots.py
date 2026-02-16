from __future__ import annotations

from pathlib import Path

import pandas as pd
from stratcheck.report.plots import generate_performance_plots


def test_generate_performance_plots_writes_three_png_files(tmp_path: Path) -> None:
    timestamps = pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC")
    equity_values = pd.Series(
        [100_000.0 + index * 125.0 for index in range(60)],
        index=timestamps,
    )

    image_paths = generate_performance_plots(
        equity_curve=equity_values,
        returns=None,
        output_dir=tmp_path / "assets",
        prefix="test_run",
    )

    assert len(image_paths) == 3
    for image_path in image_paths:
        assert image_path.suffix == ".png"
        assert image_path.exists()
        assert image_path.stat().st_size > 0
