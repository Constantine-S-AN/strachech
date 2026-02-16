"""Corporate action models and adjustment utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

CorporateActionType = Literal["split", "dividend", "symbol_change", "delist"]
TimeInput = str | datetime | pd.Timestamp | None
_DEFAULT_TIMEZONE = "UTC"


@dataclass(slots=True)
class CorporateAction:
    """Corporate action event associated with a symbol."""

    timestamp: pd.Timestamp
    action_type: CorporateActionType
    value: float | None = None
    from_symbol: str | None = None
    to_symbol: str | None = None
    note: str | None = None


def load_corporate_actions_file(path: Path) -> list[CorporateAction]:
    """Load corporate actions from CSV file."""
    if not path.exists():
        return []

    raw_actions = pd.read_csv(path)
    return parse_corporate_actions(raw_actions)


def parse_corporate_actions(actions_frame: pd.DataFrame) -> list[CorporateAction]:
    """Parse and validate action rows."""
    if actions_frame.empty:
        return []

    normalized = actions_frame.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]

    timestamp_column = _first_existing_column(
        normalized,
        candidates=("timestamp", "date", "effective_date", "event_time"),
    )
    if timestamp_column is None:
        msg = "Corporate actions file must include a timestamp column."
        raise ValueError(msg)
    if "type" not in normalized.columns:
        msg = "Corporate actions file must include a `type` column."
        raise ValueError(msg)

    value_column = _first_existing_column(
        normalized,
        candidates=("value", "ratio", "amount"),
    )
    from_symbol_column = _first_existing_column(
        normalized,
        candidates=("from_symbol", "old_symbol"),
    )
    to_symbol_column = _first_existing_column(
        normalized,
        candidates=("to_symbol", "new_symbol"),
    )
    note_column = _first_existing_column(
        normalized,
        candidates=("note", "description"),
    )

    actions: list[CorporateAction] = []
    for row in normalized.to_dict(orient="records"):
        parsed_timestamp = _parse_time(row.get(timestamp_column))
        if parsed_timestamp is None:
            msg = "Corporate action timestamp cannot be empty."
            raise ValueError(msg)

        action_type = str(row.get("type", "")).strip().lower()
        if action_type not in {"split", "dividend", "symbol_change", "delist"}:
            msg = f"Unsupported corporate action type: {action_type!r}"
            raise ValueError(msg)

        raw_value = row.get(value_column) if value_column is not None else None
        value = _parse_action_value(raw_value, action_type=action_type)

        from_symbol = (
            _as_optional_text(row.get(from_symbol_column))
            if from_symbol_column is not None
            else None
        )
        to_symbol = (
            _as_optional_text(row.get(to_symbol_column)) if to_symbol_column is not None else None
        )
        note = _as_optional_text(row.get(note_column)) if note_column is not None else None

        actions.append(
            CorporateAction(
                timestamp=parsed_timestamp,
                action_type=action_type,
                value=value,
                from_symbol=from_symbol,
                to_symbol=to_symbol,
                note=note,
            )
        )

    actions.sort(key=lambda action: action.timestamp)
    return actions


def filter_corporate_actions(
    actions: list[CorporateAction],
    start: TimeInput = None,
    end: TimeInput = None,
) -> list[CorporateAction]:
    """Filter actions to inclusive date range."""
    start_time = _parse_time(start)
    end_time = _parse_time(end)

    filtered: list[CorporateAction] = []
    for action in actions:
        if start_time is not None and action.timestamp < start_time:
            continue
        if end_time is not None and action.timestamp > end_time:
            continue
        filtered.append(action)
    return filtered


def apply_corporate_actions_to_bars(
    bars: pd.DataFrame,
    actions: list[CorporateAction],
) -> pd.DataFrame:
    """Apply corporate actions to bars (split support enabled)."""
    adjusted_bars = bars.copy()
    for column_name in ["open", "high", "low", "close", "volume"]:
        if column_name in adjusted_bars.columns:
            adjusted_bars[column_name] = adjusted_bars[column_name].astype(float)
    if not actions:
        return adjusted_bars

    split_actions = [action for action in actions if action.action_type == "split"]
    if not split_actions:
        return adjusted_bars

    for action in sorted(split_actions, key=lambda item: item.timestamp):
        ratio = _split_ratio(action)
        historical_mask = adjusted_bars.index < action.timestamp
        if not historical_mask.any():
            continue
        for price_column in ["open", "high", "low", "close"]:
            adjusted_bars.loc[historical_mask, price_column] = (
                adjusted_bars.loc[historical_mask, price_column] / ratio
            )
        adjusted_bars.loc[historical_mask, "volume"] = (
            adjusted_bars.loc[historical_mask, "volume"] * ratio
        )

    return adjusted_bars


def summarize_corporate_actions(actions: list[CorporateAction]) -> list[dict[str, str]]:
    """Summarize actions for report rendering."""
    summary_rows: list[dict[str, str]] = []
    for action in actions:
        date_text = action.timestamp.strftime("%Y-%m-%d")
        details = _action_details_text(action)
        summary_rows.append(
            {
                "date": date_text,
                "type": action.action_type,
                "details": details,
            }
        )
    return summary_rows


def _action_details_text(action: CorporateAction) -> str:
    if action.action_type == "split":
        ratio = _split_ratio(action)
        detail = f"split ratio={ratio:g}"
    elif action.action_type == "dividend":
        if action.value is None:
            detail = "cash dividend"
        else:
            detail = f"cash dividend={action.value:g}"
    elif action.action_type == "symbol_change":
        old_text = action.from_symbol or "unknown"
        new_text = action.to_symbol or "unknown"
        detail = f"symbol {old_text} -> {new_text}"
    else:
        detail = "delisted"

    if action.note:
        return f"{detail}; {action.note}"
    return detail


def _split_ratio(action: CorporateAction) -> float:
    if action.value is None:
        msg = "Split action must include ratio value."
        raise ValueError(msg)
    ratio = float(action.value)
    if ratio <= 0:
        msg = f"Split ratio must be positive, got: {ratio!r}"
        raise ValueError(msg)
    return ratio


def _parse_action_value(raw_value: Any, action_type: str) -> float | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, float) and pd.isna(raw_value):
        return None

    text = str(raw_value).strip()
    if not text:
        return None
    if action_type == "split" and ":" in text:
        left_text, right_text = text.split(":", maxsplit=1)
        left = float(left_text)
        right = float(right_text)
        if right == 0:
            msg = f"Invalid split ratio value: {raw_value!r}"
            raise ValueError(msg)
        return left / right
    return float(text)


def _as_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


def _first_existing_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _parse_time(value: TimeInput) -> pd.Timestamp | None:
    if value is None:
        return None
    parsed_time = pd.Timestamp(value)
    if parsed_time.tzinfo is None:
        return parsed_time.tz_localize(_DEFAULT_TIMEZONE)
    return parsed_time.tz_convert(_DEFAULT_TIMEZONE)
