"""Run bundle snapshot, export, and reproduction helpers."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from stratcheck.core.backtest import OrderRecord
from stratcheck.core.strategy import Fill


@dataclass(slots=True)
class RunSnapshot:
    """Metadata for a persisted run snapshot on disk."""

    run_id: str
    run_dir: Path
    report_path: Path


@dataclass(slots=True)
class ReproduceResult:
    """Result of reproducing a report from a bundle archive."""

    run_id: str
    run_dir: Path
    report_path: Path


def build_run_snapshot(
    *,
    run_mode: Literal["run", "healthcheck", "demo"],
    report_path: Path,
    report_dir: Path,
    report_name: str,
    config_path: Path | None,
    raw_config: Mapping[str, Any] | None,
    symbol: str,
    strategy_reference: str,
    bars: pd.DataFrame,
    equity_curve: pd.Series,
    orders: Sequence[OrderRecord],
    trades: Sequence[Fill],
    overall_metrics: Mapping[str, float | int],
    asset_paths: Sequence[Path],
    window_metrics_df: pd.DataFrame | None = None,
    summary_json_path: Path | None = None,
    source_data_path: Path | None = None,
    runs_dir: Path | None = None,
    run_id: str | None = None,
) -> RunSnapshot:
    """Persist one run snapshot with config, report, and key intermediates."""
    run_identifier = run_id or generate_run_id()
    snapshots_dir = runs_dir or report_dir / "runs"
    run_snapshot_dir = snapshots_dir / run_identifier
    run_snapshot_dir.mkdir(parents=True, exist_ok=False)

    copied_report_path = run_snapshot_dir / "report.html"
    shutil.copy2(report_path, copied_report_path)
    _copy_assets(asset_paths=asset_paths, output_dir=run_snapshot_dir / "assets")

    bars_snapshot_path = run_snapshot_dir / "bars.csv"
    _write_bars_csv(bars=bars, output_path=bars_snapshot_path)

    equity_curve_path = run_snapshot_dir / "equity_curve.csv"
    _write_equity_curve_csv(equity_curve=equity_curve, output_path=equity_curve_path)

    signals_path = run_snapshot_dir / "signals.csv"
    _write_signals_csv(orders=orders, output_path=signals_path)

    trades_path = run_snapshot_dir / "trades.csv"
    _write_trades_csv(trades=trades, output_path=trades_path)

    if window_metrics_df is not None and not window_metrics_df.empty:
        window_metrics_df.to_csv(run_snapshot_dir / "window_metrics.csv", index=False)

    if summary_json_path is not None and summary_json_path.exists():
        shutil.copy2(summary_json_path, run_snapshot_dir / "healthcheck_summary.json")

    if config_path is not None and config_path.exists():
        shutil.copy2(config_path, run_snapshot_dir / "config.toml")

    if raw_config is not None:
        _write_json(
            path=run_snapshot_dir / "config.expanded.json",
            payload=dict(raw_config),
        )

    _write_json(path=run_snapshot_dir / "overall_metrics.json", payload=dict(overall_metrics))

    hashes_payload = {
        "bars_csv_sha256": _sha256_file(bars_snapshot_path),
        "bars_rows": int(len(bars)),
        "source_data_path": str(source_data_path) if source_data_path is not None else "",
    }
    if source_data_path is not None and source_data_path.exists() and source_data_path.is_file():
        hashes_payload["source_data_sha256"] = _sha256_file(source_data_path)
    _write_json(path=run_snapshot_dir / "data_hash.json", payload=hashes_payload)

    manifest_payload = {
        "bundle_version": 1,
        "run_id": run_identifier,
        "run_mode": run_mode,
        "created_at": datetime.now(UTC).isoformat(),
        "symbol": symbol,
        "strategy": strategy_reference,
        "report_name": report_name,
        "report_file": "report.html",
        "assets_dir": "assets",
        "files": {
            "config": "config.toml" if config_path is not None and config_path.exists() else "",
            "expanded_config": "config.expanded.json" if raw_config is not None else "",
            "bars": "bars.csv",
            "equity_curve": "equity_curve.csv",
            "signals": "signals.csv",
            "trades": "trades.csv",
            "metrics": "overall_metrics.json",
            "data_hash": "data_hash.json",
            "window_metrics": (
                "window_metrics.csv"
                if window_metrics_df is not None and not window_metrics_df.empty
                else ""
            ),
            "healthcheck_summary": (
                "healthcheck_summary.json"
                if summary_json_path is not None and summary_json_path.exists()
                else ""
            ),
        },
    }
    _write_json(path=run_snapshot_dir / "manifest.json", payload=manifest_payload)

    return RunSnapshot(
        run_id=run_identifier,
        run_dir=run_snapshot_dir,
        report_path=copied_report_path,
    )


def bundle_snapshot(
    *,
    run_id: str,
    runs_dir: Path,
    output_path: Path | None = None,
) -> Path:
    """Export one run snapshot directory into a zip archive."""
    run_snapshot_dir = runs_dir / run_id
    if not run_snapshot_dir.exists() or not run_snapshot_dir.is_dir():
        msg = f"Run snapshot not found: {run_snapshot_dir}"
        raise FileNotFoundError(msg)

    zip_path = output_path or (runs_dir.parent / "bundles" / f"{run_id}.zip")
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        file_paths = sorted(path for path in run_snapshot_dir.rglob("*") if path.is_file())
        for file_path in file_paths:
            archive_name = Path(run_id) / file_path.relative_to(run_snapshot_dir)
            zip_file.write(file_path, arcname=str(archive_name))

    return zip_path


def reproduce_snapshot(
    *,
    bundle_path: Path,
    output_dir: Path,
) -> ReproduceResult:
    """Reproduce a report by extracting bundle and verifying bundled data hash."""
    if not bundle_path.exists():
        msg = f"Bundle not found: {bundle_path}"
        raise FileNotFoundError(msg)

    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(bundle_path, mode="r") as zip_file:
        _assert_safe_archive(zip_file)
        top_level_dirs = _archive_top_level_dirs(zip_file)

        if len(top_level_dirs) == 1:
            run_id = next(iter(top_level_dirs))
            run_snapshot_dir = output_dir / run_id
            if run_snapshot_dir.exists():
                shutil.rmtree(run_snapshot_dir)
            zip_file.extractall(path=output_dir)
        else:
            run_id = bundle_path.stem
            run_snapshot_dir = output_dir / run_id
            if run_snapshot_dir.exists():
                shutil.rmtree(run_snapshot_dir)
            run_snapshot_dir.mkdir(parents=True, exist_ok=False)
            for zip_info in zip_file.infolist():
                member_name = zip_info.filename
                if not member_name or member_name.endswith("/"):
                    continue
                target_path = run_snapshot_dir / member_name
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with zip_file.open(zip_info, mode="r") as source_handle:
                    with target_path.open("wb") as target_handle:
                        shutil.copyfileobj(source_handle, target_handle)

    data_hash_path = run_snapshot_dir / "data_hash.json"
    bars_snapshot_path = run_snapshot_dir / "bars.csv"
    if data_hash_path.exists() and bars_snapshot_path.exists():
        hash_payload = json.loads(data_hash_path.read_text(encoding="utf-8"))
        expected_hash = str(hash_payload.get("bars_csv_sha256", "")).strip()
        if expected_hash:
            actual_hash = _sha256_file(bars_snapshot_path)
            if actual_hash != expected_hash:
                msg = f"Bundled bars hash mismatch: expected={expected_hash}, actual={actual_hash}"
                raise ValueError(msg)

    manifest_path = run_snapshot_dir / "manifest.json"
    report_file = "report.html"
    if manifest_path.exists():
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        report_file = str(manifest_payload.get("report_file", report_file))

    reproduced_report_path = run_snapshot_dir / report_file
    if not reproduced_report_path.exists():
        msg = f"Reproduced report not found: {reproduced_report_path}"
        raise FileNotFoundError(msg)

    return ReproduceResult(
        run_id=run_id,
        run_dir=run_snapshot_dir,
        report_path=reproduced_report_path,
    )


def detect_source_data_path(data_path: Path, symbol: str) -> Path | None:
    """Try to resolve source data file for one symbol from configured data path."""
    if data_path.is_file():
        return data_path

    if data_path.is_dir():
        for extension in (".parquet", ".csv"):
            candidate = data_path / f"{symbol}{extension}"
            if candidate.exists():
                return candidate
    return None


def generate_run_id() -> str:
    """Create a run identifier safe for filenames and CLI usage."""
    time_fragment = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    random_fragment = uuid.uuid4().hex[:8]
    return f"run_{time_fragment}_{random_fragment}"


def _copy_assets(asset_paths: Sequence[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied_names: set[str] = set()
    for asset_path in asset_paths:
        if not asset_path.exists() or not asset_path.is_file():
            continue
        if asset_path.name in copied_names:
            continue
        shutil.copy2(asset_path, output_dir / asset_path.name)
        copied_names.add(asset_path.name)


def _write_bars_csv(bars: pd.DataFrame, output_path: Path) -> None:
    bars_frame = bars.reset_index()
    first_column = str(bars_frame.columns[0])
    if first_column != "timestamp":
        bars_frame = bars_frame.rename(columns={first_column: "timestamp"})
    bars_frame.to_csv(output_path, index=False)


def _write_equity_curve_csv(equity_curve: pd.Series, output_path: Path) -> None:
    equity_frame = equity_curve.rename("equity").reset_index()
    first_column = str(equity_frame.columns[0])
    if first_column != "timestamp":
        equity_frame = equity_frame.rename(columns={first_column: "timestamp"})
    equity_frame.to_csv(output_path, index=False)


def _write_signals_csv(orders: Sequence[OrderRecord], output_path: Path) -> None:
    signal_rows: list[dict[str, Any]] = []
    for order in orders:
        signal_rows.append(
            {
                "created_at": order.created_at.isoformat(),
                "side": order.side,
                "qty": float(order.qty),
                "limit_price": ("" if order.limit_price is None else float(order.limit_price)),
                "market": bool(order.market),
                "filled": bool(order.filled),
                "fill_time": "" if order.fill_time is None else order.fill_time.isoformat(),
                "fill_price": "" if order.fill_price is None else float(order.fill_price),
            }
        )

    pd.DataFrame(signal_rows).to_csv(output_path, index=False)


def _write_trades_csv(trades: Sequence[Fill], output_path: Path) -> None:
    trade_rows: list[dict[str, Any]] = []
    for trade in trades:
        trade_rows.append(
            {
                "timestamp": trade.timestamp.isoformat(),
                "side": trade.side,
                "qty": float(trade.qty),
                "price": float(trade.price),
                "fee": float(trade.fee),
                "cost": float(trade.cost),
            }
        )

    pd.DataFrame(trade_rows).to_csv(output_path, index=False)


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _archive_top_level_dirs(zip_file: zipfile.ZipFile) -> set[str]:
    top_levels: set[str] = set()
    for name in zip_file.namelist():
        member = Path(name)
        if not member.parts:
            continue
        top_levels.add(member.parts[0])
    return {value for value in top_levels if value}


def _assert_safe_archive(zip_file: zipfile.ZipFile) -> None:
    for zip_info in zip_file.infolist():
        member_path = Path(zip_info.filename)
        if member_path.is_absolute() or ".." in member_path.parts:
            msg = f"Unsafe archive entry: {zip_info.filename}"
            raise ValueError(msg)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _json_default(value: object) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        item = value.item()
        if isinstance(item, object):
            return item
    msg = f"Unsupported JSON type: {type(value)!r}"
    raise TypeError(msg)
