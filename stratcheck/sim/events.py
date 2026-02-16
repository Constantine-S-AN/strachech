"""Event types and queue for event-driven backtesting."""

from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Literal

import pandas as pd

TimeInForce = Literal["GTC", "DAY", "IOC"]


@dataclass(slots=True)
class Event:
    """Base event with timestamp."""

    timestamp: pd.Timestamp


@dataclass(slots=True)
class MarketDataEvent(Event):
    """New market bar event."""

    bar_index: int
    bar: pd.Series


@dataclass(slots=True)
class TimerEvent(Event):
    """Timer callback event (used for strategy evaluation at bar close)."""

    bar_index: int
    name: str = "bar_close"


@dataclass(slots=True)
class OrderEvent(Event):
    """Submitted order event."""

    order_id: int
    created_bar_index: int
    side: Literal["buy", "sell"]
    qty: float
    market: bool
    limit_price: float | None
    time_in_force: TimeInForce


@dataclass(slots=True)
class FillEvent(Event):
    """Executed fill event."""

    order_id: int
    side: Literal["buy", "sell"]
    qty: float
    fill_price: float
    fee: float
    reference_price: float


EventType = MarketDataEvent | TimerEvent | OrderEvent | FillEvent


def _event_priority(event: EventType) -> int:
    if isinstance(event, MarketDataEvent):
        return 10
    if isinstance(event, FillEvent):
        return 20
    if isinstance(event, TimerEvent):
        return 30
    return 40


@dataclass(order=True, slots=True)
class ScheduledEvent:
    """Internal heap wrapper with stable ordering."""

    timestamp: pd.Timestamp
    priority: int
    sequence: int
    event: EventType = field(compare=False)


class EventPriorityQueue:
    """Priority queue ordered by timestamp, event priority, then insertion order."""

    def __init__(self) -> None:
        self._heap: list[ScheduledEvent] = []
        self._sequence = 0

    def push(self, event: EventType) -> None:
        scheduled = ScheduledEvent(
            timestamp=event.timestamp,
            priority=_event_priority(event),
            sequence=self._sequence,
            event=event,
        )
        heappush(self._heap, scheduled)
        self._sequence += 1

    def pop(self) -> EventType:
        return heappop(self._heap).event

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)
