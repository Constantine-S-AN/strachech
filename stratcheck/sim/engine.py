"""Event-driven backtest engine v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from stratcheck.core.backtest import (
    BacktestResult,
    CostModel,
    FillQuote,
    FixedBpsCostModel,
    OrderRecord,
)
from stratcheck.core.metrics import compute_metrics
from stratcheck.core.strategy import Fill, PortfolioState, Strategy
from stratcheck.sim.events import (
    EventPriorityQueue,
    FillEvent,
    MarketDataEvent,
    OrderEvent,
    TimeInForce,
    TimerEvent,
)
from stratcheck.sim.fill_models import (
    BarFillModel,
    FillInstruction,
    FillModel,
    FillModelOrder,
    FillModelState,
)

_ORDER_EPSILON = 1e-12


@dataclass(slots=True)
class _SimOrder:
    order_id: int
    created_at: pd.Timestamp
    created_bar_index: int
    first_eligible_bar_index: int
    last_eligible_bar_index: int | None
    side: Literal["buy", "sell"]
    market: bool
    limit_price: float | None
    time_in_force: TimeInForce
    initial_qty: float
    remaining_qty: float
    filled_qty: float = 0.0
    filled_notional: float = 0.0
    last_fill_time: pd.Timestamp | None = None


@dataclass(slots=True)
class _RuntimeState:
    bars: pd.DataFrame
    cost_model: CostModel
    cash: float
    position_qty: float
    next_order_id: int
    orders_by_id: dict[int, _SimOrder]
    order_records_by_id: dict[int, OrderRecord]
    active_order_ids: list[int]
    trades: list[Fill]
    equity_points: list[tuple[pd.Timestamp, float]]
    position_points: list[tuple[pd.Timestamp, float]]


class BacktestEngineV2:
    """Event-driven backtest engine with limit, partial fill, and TIF support.

    Event order per timestamp:
    1. `MarketDataEvent`: match active orders on current bar.
    2. `FillEvent`: record fills produced by matching.
    3. `TimerEvent`: evaluate strategy at bar close and emit new orders.
    4. `OrderEvent`: accept orders into the book for future bars.

    Execution assumptions:
    - Strategy signals are evaluated at bar close (`TimerEvent`).
    - Market and limit orders become eligible from the next bar.
    - Market orders use next bar `open` as reference.
    - Limit orders fill when touched (`buy: low <= limit`, `sell: high >= limit`).
    - Per-bar quantity is capped by `max_volume_share * bar.volume`.
    - `DAY` and `IOC` orders expire after first eligible bar.
    """

    def __init__(
        self,
        max_volume_share: float = 1.0,
        default_time_in_force: TimeInForce = "GTC",
        fill_model: FillModel | None = None,
    ) -> None:
        if max_volume_share <= 0:
            msg = "max_volume_share must be positive."
            raise ValueError(msg)
        if max_volume_share > 1.0:
            msg = "max_volume_share must be <= 1.0."
            raise ValueError(msg)
        self.max_volume_share = float(max_volume_share)
        self.default_time_in_force = _normalize_time_in_force(default_time_in_force)
        self.fill_model: FillModel = fill_model or BarFillModel()

    def run(
        self,
        strategy: Strategy,
        bars: pd.DataFrame,
        initial_cash: float,
        cost_model: CostModel | None = None,
        bars_freq: str = "1d",
    ) -> BacktestResult:
        """Run event-driven backtest and return a `BacktestResult`."""
        if initial_cash <= 0:
            msg = "initial_cash must be positive."
            raise ValueError(msg)

        normalized_bars = _normalize_bars(bars)
        if normalized_bars.empty:
            msg = "bars must contain at least one row."
            raise ValueError(msg)

        model: CostModel = cost_model or FixedBpsCostModel()
        state = _RuntimeState(
            bars=normalized_bars,
            cost_model=model,
            cash=float(initial_cash),
            position_qty=0.0,
            next_order_id=1,
            orders_by_id={},
            order_records_by_id={},
            active_order_ids=[],
            trades=[],
            equity_points=[],
            position_points=[],
        )
        event_queue = EventPriorityQueue()
        for bar_index, (timestamp, bar_row) in enumerate(normalized_bars.iterrows()):
            event_queue.push(
                MarketDataEvent(
                    timestamp=timestamp,
                    bar_index=bar_index,
                    bar=bar_row,
                )
            )
            event_queue.push(
                TimerEvent(
                    timestamp=timestamp,
                    bar_index=bar_index,
                )
            )

        while event_queue:
            event = event_queue.pop()
            if isinstance(event, MarketDataEvent):
                self._handle_market_data_event(
                    event=event,
                    state=state,
                    event_queue=event_queue,
                )
                continue
            if isinstance(event, FillEvent):
                self._handle_fill_event(
                    event=event,
                    state=state,
                )
                continue
            if isinstance(event, TimerEvent):
                self._handle_timer_event(
                    event=event,
                    strategy=strategy,
                    state=state,
                    event_queue=event_queue,
                )
                continue
            self._handle_order_event(
                event=event,
                state=state,
            )

        return self._build_result(
            state=state,
            initial_cash=initial_cash,
            bars_freq=bars_freq,
        )

    def _handle_market_data_event(
        self,
        event: MarketDataEvent,
        state: _RuntimeState,
        event_queue: EventPriorityQueue,
    ) -> None:
        bar_index = event.bar_index
        bars_slice = state.bars.iloc[: bar_index + 1]

        active_ids_snapshot = list(state.active_order_ids)
        for order_id in active_ids_snapshot:
            order_state = state.orders_by_id[order_id]
            if not self._is_order_active_for_bar(order_state=order_state, bar_index=bar_index):
                self._deactivate_order(order_id=order_id, state=state)

        fill_orders = self._build_fill_orders(
            state=state,
            bar_index=bar_index,
        )
        if not fill_orders:
            return

        fill_model_state = FillModelState(
            bar_index=bar_index,
            timestamp=event.timestamp,
            cash=state.cash,
            position_qty=state.position_qty,
            max_volume_share=self.max_volume_share,
        )
        instructions = self.fill_model.simulate_fills(
            orders=fill_orders,
            bars_slice=bars_slice,
            state=fill_model_state,
        )
        instruction_by_order_id: dict[int, FillInstruction] = {}
        for instruction in instructions:
            if instruction.order_id in instruction_by_order_id:
                msg = f"Duplicate fill instruction for order_id={instruction.order_id}"
                raise ValueError(msg)
            instruction_by_order_id[instruction.order_id] = instruction

        provisional_cash = state.cash
        provisional_position = state.position_qty
        for order_id in list(state.active_order_ids):
            order_state = state.orders_by_id[order_id]
            raw_instruction = instruction_by_order_id.get(order_id)
            if raw_instruction is None:
                continue

            should_expire = bool(raw_instruction.expire)
            requested_qty = min(float(raw_instruction.qty), order_state.remaining_qty)
            if requested_qty > _ORDER_EPSILON:
                reference_price = raw_instruction.reference_price
                if reference_price is None:
                    msg = f"Missing reference_price for order_id={order_id}"
                    raise ValueError(msg)
                current_bar = bars_slice.iloc[-1]
                if order_state.side == "buy":
                    executable_qty = self._resolve_affordable_buy_qty(
                        requested_qty=requested_qty,
                        available_cash=provisional_cash,
                        reference_price=float(reference_price),
                        bar_row=current_bar,
                        cost_model=state.cost_model,
                    )
                else:
                    executable_qty = min(requested_qty, provisional_position)

                if executable_qty > _ORDER_EPSILON:
                    quote = state.cost_model.quote_fill(
                        side=order_state.side,
                        qty=executable_qty,
                        reference_price=float(reference_price),
                        bar=current_bar,
                    )
                    notional = executable_qty * float(quote.fill_price)
                    if order_state.side == "buy":
                        provisional_cash -= notional + float(quote.fee)
                        provisional_position += executable_qty
                    else:
                        provisional_cash += notional - float(quote.fee)
                        provisional_position -= executable_qty

                    order_state.remaining_qty = max(order_state.remaining_qty - executable_qty, 0.0)
                    order_state.filled_qty += executable_qty
                    order_state.filled_notional += executable_qty * float(quote.fill_price)
                    order_state.last_fill_time = event.timestamp
                    self._update_order_record_from_state(
                        order_state=order_state,
                        order_record=state.order_records_by_id[order_id],
                    )
                    event_queue.push(
                        FillEvent(
                            timestamp=event.timestamp,
                            order_id=order_id,
                            side=order_state.side,
                            qty=executable_qty,
                            fill_price=float(quote.fill_price),
                            fee=float(quote.fee),
                            reference_price=float(reference_price),
                        )
                    )

            if order_state.remaining_qty <= _ORDER_EPSILON or should_expire:
                self._deactivate_order(order_id=order_id, state=state)

        state.cash = provisional_cash
        state.position_qty = provisional_position

    def _handle_fill_event(
        self,
        event: FillEvent,
        state: _RuntimeState,
    ) -> None:
        implicit_slippage = event.qty * abs(event.fill_price - event.reference_price)
        state.trades.append(
            Fill(
                side=event.side,
                qty=float(event.qty),
                price=float(event.fill_price),
                timestamp=event.timestamp,
                fee=float(event.fee),
                cost=float(event.fee + implicit_slippage),
            )
        )
        order_state = state.orders_by_id[event.order_id]
        order_record = state.order_records_by_id[event.order_id]
        self._update_order_record_from_state(
            order_state=order_state,
            order_record=order_record,
        )

    def _handle_timer_event(
        self,
        event: TimerEvent,
        strategy: Strategy,
        state: _RuntimeState,
        event_queue: EventPriorityQueue,
    ) -> None:
        close_price = float(state.bars.iloc[event.bar_index]["close"])
        equity = state.cash + state.position_qty * close_price
        state.equity_points.append((event.timestamp, equity))
        state.position_points.append((event.timestamp, state.position_qty))

        if event.bar_index >= len(state.bars) - 1:
            return

        portfolio_state = PortfolioState(
            cash=state.cash,
            position_qty=state.position_qty,
            equity=equity,
        )
        bars_slice = state.bars.iloc[: event.bar_index + 1]
        intents = strategy.generate_orders(
            bars=bars_slice,
            portfolio_state=portfolio_state,
        )
        for intent in intents:
            time_in_force = _normalize_time_in_force(
                getattr(intent, "time_in_force", self.default_time_in_force),
                default=self.default_time_in_force,
            )
            event_queue.push(
                OrderEvent(
                    timestamp=event.timestamp,
                    order_id=state.next_order_id,
                    created_bar_index=event.bar_index,
                    side=intent.side,
                    qty=float(intent.qty),
                    market=bool(intent.market),
                    limit_price=intent.limit_price,
                    time_in_force=time_in_force,
                )
            )
            state.next_order_id += 1

    def _handle_order_event(
        self,
        event: OrderEvent,
        state: _RuntimeState,
    ) -> None:
        if event.qty <= 0:
            return
        if not event.market and event.limit_price is None:
            msg = "Limit orders must include `limit_price`."
            raise ValueError(msg)

        if event.time_in_force in {"DAY", "IOC"}:
            last_valid_bar_index = event.created_bar_index + 1
        else:
            last_valid_bar_index = None

        order_state = _SimOrder(
            order_id=event.order_id,
            created_at=event.timestamp,
            created_bar_index=event.created_bar_index,
            first_eligible_bar_index=event.created_bar_index + 1,
            last_eligible_bar_index=last_valid_bar_index,
            side=event.side,
            market=event.market,
            limit_price=event.limit_price,
            time_in_force=event.time_in_force,
            initial_qty=float(event.qty),
            remaining_qty=float(event.qty),
        )
        state.orders_by_id[event.order_id] = order_state
        state.order_records_by_id[event.order_id] = OrderRecord(
            created_at=event.timestamp,
            side=event.side,
            qty=float(event.qty),
            limit_price=event.limit_price,
            market=event.market,
        )

        if event.created_bar_index < len(state.bars) - 1:
            state.active_order_ids.append(event.order_id)

    def _build_fill_orders(
        self,
        state: _RuntimeState,
        bar_index: int,
    ) -> list[FillModelOrder]:
        fill_orders: list[FillModelOrder] = []
        for order_id in state.active_order_ids:
            order_state = state.orders_by_id[order_id]
            if not self._is_order_active_for_bar(order_state=order_state, bar_index=bar_index):
                continue
            fill_orders.append(
                FillModelOrder(
                    order_id=order_state.order_id,
                    side=order_state.side,
                    market=order_state.market,
                    limit_price=order_state.limit_price,
                    remaining_qty=order_state.remaining_qty,
                    first_eligible_bar_index=order_state.first_eligible_bar_index,
                    last_eligible_bar_index=order_state.last_eligible_bar_index,
                    time_in_force=order_state.time_in_force,
                )
            )
        return fill_orders

    def _build_result(
        self,
        state: _RuntimeState,
        initial_cash: float,
        bars_freq: str,
    ) -> BacktestResult:
        equity_curve = pd.Series(
            data=[value for _, value in state.equity_points],
            index=pd.Index(
                [timestamp for timestamp, _ in state.equity_points],
                name=state.bars.index.name,
            ),
            name="equity",
            dtype=float,
        )
        positions = pd.Series(
            data=[value for _, value in state.position_points],
            index=pd.Index(
                [timestamp for timestamp, _ in state.position_points],
                name=state.bars.index.name,
            ),
            name="position_qty",
            dtype=float,
        )
        strategy_returns = equity_curve.pct_change().fillna(0.0)
        benchmark_returns = state.bars["close"].astype(float).pct_change().fillna(0.0)
        benchmark_curve = initial_cash * (1.0 + benchmark_returns).cumprod()
        drawdown_curve = equity_curve / equity_curve.cummax() - 1.0
        metrics = compute_metrics(
            equity_curve=equity_curve,
            trades=state.trades,
            bars_freq=bars_freq,
        )
        metrics["trade_entries"] = int(sum(1 for trade in state.trades if trade.side == "buy"))

        ordered_ids = sorted(state.order_records_by_id.keys())
        order_records = [state.order_records_by_id[order_id] for order_id in ordered_ids]

        return BacktestResult(
            market_data=state.bars,
            positions=positions,
            strategy_returns=strategy_returns,
            benchmark_returns=benchmark_returns,
            equity_curve=equity_curve,
            benchmark_curve=benchmark_curve,
            drawdown_curve=drawdown_curve,
            metrics=metrics,
            orders=order_records,
            trades=state.trades,
        )

    def _is_order_active_for_bar(self, order_state: _SimOrder, bar_index: int) -> bool:
        if order_state.remaining_qty <= _ORDER_EPSILON:
            return False
        if order_state.last_eligible_bar_index is None:
            return True
        return bar_index <= order_state.last_eligible_bar_index

    def _deactivate_order(self, order_id: int, state: _RuntimeState) -> None:
        state.active_order_ids = [
            current_order_id
            for current_order_id in state.active_order_ids
            if current_order_id != order_id
        ]

    def _update_order_record_from_state(
        self,
        order_state: _SimOrder,
        order_record: OrderRecord,
    ) -> None:
        if order_state.filled_qty <= _ORDER_EPSILON:
            return
        average_fill_price = order_state.filled_notional / order_state.filled_qty
        order_record.fill_time = order_state.last_fill_time
        order_record.fill_price = float(average_fill_price)
        order_record.filled = order_state.remaining_qty <= _ORDER_EPSILON

    def _resolve_affordable_buy_qty(
        self,
        requested_qty: float,
        available_cash: float,
        reference_price: float,
        bar_row: pd.Series,
        cost_model: CostModel,
    ) -> float:
        if requested_qty <= _ORDER_EPSILON or available_cash <= _ORDER_EPSILON:
            return 0.0

        full_quote = cost_model.quote_fill(
            side="buy",
            qty=requested_qty,
            reference_price=reference_price,
            bar=bar_row,
        )
        required_cash = requested_qty * full_quote.fill_price + full_quote.fee
        if required_cash <= available_cash:
            return requested_qty

        return _search_affordable_qty(
            requested_qty=requested_qty,
            available_cash=available_cash,
            reference_price=reference_price,
            bar_row=bar_row,
            cost_model=cost_model,
        )


def _normalize_time_in_force(
    value: object,
    default: TimeInForce = "GTC",
) -> TimeInForce:
    if value is None:
        return default
    text = str(value).strip().upper()
    if text in {"GTC", "DAY", "IOC"}:
        return text  # type: ignore[return-value]
    msg = f"Unsupported time_in_force value: {value!r}. Use GTC/DAY/IOC."
    raise ValueError(msg)


def _normalize_bars(bars: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(bars.index, pd.DatetimeIndex):
        msg = "bars must use a DatetimeIndex."
        raise ValueError(msg)

    required_columns = {"open", "close"}
    missing = required_columns.difference(bars.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        msg = f"bars is missing required columns: {missing_text}"
        raise ValueError(msg)

    normalized = bars.sort_index().copy()
    normalized = normalized[~normalized.index.duplicated(keep="last")]
    if "high" not in normalized.columns:
        normalized["high"] = normalized[["open", "close"]].max(axis=1)
    if "low" not in normalized.columns:
        normalized["low"] = normalized[["open", "close"]].min(axis=1)
    if "volume" not in normalized.columns:
        normalized["volume"] = 1_000_000.0

    for column_name in ["open", "high", "low", "close", "volume"]:
        normalized[column_name] = normalized[column_name].astype(float)
    return normalized


def _search_affordable_qty(
    requested_qty: float,
    available_cash: float,
    reference_price: float,
    bar_row: pd.Series,
    cost_model: CostModel,
) -> float:
    lower_bound = 0.0
    upper_bound = requested_qty
    for _ in range(40):
        candidate_qty = (lower_bound + upper_bound) / 2.0
        if candidate_qty <= _ORDER_EPSILON:
            break
        quote: FillQuote = cost_model.quote_fill(
            side="buy",
            qty=candidate_qty,
            reference_price=reference_price,
            bar=bar_row,
        )
        required_cash = candidate_qty * quote.fill_price + quote.fee
        if required_cash <= available_cash:
            lower_bound = candidate_qty
        else:
            upper_bound = candidate_qty
    return lower_bound
