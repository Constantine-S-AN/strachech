"""Portfolio optimizer with lightweight projection-style constraints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

import pandas as pd

SectorCaps: TypeAlias = float | Mapping[str, float] | None


@dataclass(slots=True)
class PortfolioConstraints:
    """Constraint set for portfolio target weights."""

    max_weight_per_asset: float = 1.0
    max_sector_exposure: SectorCaps = None
    max_turnover: float | None = None
    max_gross_exposure: float = 1.0
    max_net_exposure: float = 1.0
    allow_short: bool = False

    def __post_init__(self) -> None:
        if self.max_weight_per_asset <= 0:
            msg = "max_weight_per_asset must be positive."
            raise ValueError(msg)
        if self.max_turnover is not None and self.max_turnover < 0:
            msg = "max_turnover must be non-negative when provided."
            raise ValueError(msg)
        if self.max_gross_exposure <= 0:
            msg = "max_gross_exposure must be positive."
            raise ValueError(msg)
        if self.max_net_exposure <= 0:
            msg = "max_net_exposure must be positive."
            raise ValueError(msg)

        caps = _normalize_sector_caps(self.max_sector_exposure)
        for sector_name, cap_value in caps.items():
            if cap_value < 0:
                msg = f"Sector cap must be non-negative for sector={sector_name!r}."
                raise ValueError(msg)


class Optimizer:
    """Greedy projection optimizer for constrained target weights."""

    def optimize(
        self,
        desired_weights: pd.Series | Mapping[str, float],
        current_weights: pd.Series | Mapping[str, float] | None = None,
        sectors: Mapping[str, str] | None = None,
        constraints: PortfolioConstraints | None = None,
    ) -> pd.Series:
        """Project desired weights to satisfy exposure and turnover constraints."""
        applied_constraints = constraints or PortfolioConstraints()
        desired_series = _to_weight_series(desired_weights, name="desired_weights")
        if current_weights is None:
            current_series = _to_weight_series({}, name="current_weights")
        else:
            current_series = _to_weight_series(current_weights, name="current_weights")

        all_symbols = sorted(set(desired_series.index) | set(current_series.index))
        desired_aligned = desired_series.reindex(all_symbols, fill_value=0.0)
        current_aligned = current_series.reindex(all_symbols, fill_value=0.0)

        projected = desired_aligned.copy()
        projected = _apply_asset_caps(projected, applied_constraints)
        projected = _apply_sector_caps(
            projected,
            sectors=sectors or {},
            sector_caps=applied_constraints.max_sector_exposure,
            allow_short=applied_constraints.allow_short,
        )
        projected = _apply_exposure_limits(projected, applied_constraints)

        if applied_constraints.max_turnover is not None:
            projected = _apply_turnover_limit(
                current_weights=current_aligned,
                target_weights=projected,
                max_turnover=applied_constraints.max_turnover,
            )

        return projected

    @staticmethod
    def compute_turnover(
        current_weights: pd.Series | Mapping[str, float],
        target_weights: pd.Series | Mapping[str, float],
    ) -> float:
        """Compute one-way turnover = 0.5 * sum(abs(target-current))."""
        current_series = _to_weight_series(current_weights, name="current_weights")
        target_series = _to_weight_series(target_weights, name="target_weights")
        all_symbols = sorted(set(current_series.index) | set(target_series.index))
        current_aligned = current_series.reindex(all_symbols, fill_value=0.0)
        target_aligned = target_series.reindex(all_symbols, fill_value=0.0)
        turnover = 0.5 * float((target_aligned - current_aligned).abs().sum())
        return turnover


def _to_weight_series(
    values: pd.Series | Mapping[str, float],
    name: str,
) -> pd.Series:
    if isinstance(values, pd.Series):
        series = values.copy()
    else:
        series = pd.Series(dict(values), dtype=float)

    series.index = [str(symbol).upper() for symbol in series.index]
    series = pd.to_numeric(series, errors="coerce")
    if series.isna().any():
        msg = f"{name} contains non-numeric values."
        raise ValueError(msg)
    return series.astype(float)


def _normalize_sector_caps(sector_caps: SectorCaps) -> dict[str, float]:
    if sector_caps is None:
        return {}
    if isinstance(sector_caps, (int, float)):
        return {"*": float(sector_caps)}
    return {str(sector).upper(): float(cap_value) for sector, cap_value in sector_caps.items()}


def _apply_asset_caps(
    weights: pd.Series,
    constraints: PortfolioConstraints,
) -> pd.Series:
    capped = weights.copy()
    max_weight = constraints.max_weight_per_asset
    if constraints.allow_short:
        return capped.clip(lower=-max_weight, upper=max_weight)
    return capped.clip(lower=0.0, upper=max_weight)


def _apply_sector_caps(
    weights: pd.Series,
    sectors: Mapping[str, str],
    sector_caps: SectorCaps,
    allow_short: bool,
) -> pd.Series:
    normalized_caps = _normalize_sector_caps(sector_caps)
    if not normalized_caps:
        return weights

    adjusted = weights.copy()
    normalized_sectors = {
        str(symbol).upper(): str(sector).upper() for symbol, sector in sectors.items()
    }

    if "*" in normalized_caps:
        capped_sectors = {
            normalized_sectors.get(symbol, "__UNMAPPED__") for symbol in adjusted.index
        }
        sector_cap_map = {sector_name: normalized_caps["*"] for sector_name in capped_sectors}
    else:
        sector_cap_map = normalized_caps

    for sector_name, cap_value in sector_cap_map.items():
        sector_symbols = [
            symbol
            for symbol in adjusted.index
            if normalized_sectors.get(symbol, "__UNMAPPED__") == sector_name
        ]
        if not sector_symbols:
            continue

        exposure_series = adjusted.loc[sector_symbols]
        if allow_short:
            sector_exposure = float(exposure_series.abs().sum())
        else:
            sector_exposure = float(exposure_series.clip(lower=0.0).sum())
        if sector_exposure <= cap_value or sector_exposure == 0.0:
            continue
        scale_factor = cap_value / sector_exposure
        adjusted.loc[sector_symbols] = exposure_series * scale_factor

    return adjusted


def _apply_exposure_limits(
    weights: pd.Series,
    constraints: PortfolioConstraints,
) -> pd.Series:
    adjusted = weights.copy()
    if not constraints.allow_short:
        adjusted = adjusted.clip(lower=0.0)

    gross_exposure = float(adjusted.abs().sum())
    if gross_exposure > constraints.max_gross_exposure and gross_exposure > 0:
        adjusted = adjusted * (constraints.max_gross_exposure / gross_exposure)

    net_exposure = float(abs(adjusted.sum()))
    if net_exposure > constraints.max_net_exposure and net_exposure > 0:
        adjusted = adjusted * (constraints.max_net_exposure / net_exposure)

    return adjusted


def _apply_turnover_limit(
    current_weights: pd.Series,
    target_weights: pd.Series,
    max_turnover: float,
) -> pd.Series:
    turnover = Optimizer.compute_turnover(
        current_weights=current_weights,
        target_weights=target_weights,
    )
    if turnover <= max_turnover or turnover == 0.0:
        return target_weights

    interpolation_ratio = max_turnover / turnover
    limited = current_weights + interpolation_ratio * (target_weights - current_weights)
    return limited.astype(float)
