"""Live-like paper runner with audit, metrics, logs, and kill-switch support."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from stratcheck.audit import RunAuditStore
from stratcheck.connectors.base import BrokerOrder, BrokerPosition, OrderUpdate
from stratcheck.connectors.paper import PaperBrokerConnector
from stratcheck.core.strategy import PortfolioState, Strategy
from stratcheck.ops import JsonEventLogger, KillSwitch, KillSwitchConfig, RuntimeMetrics
from stratcheck.risk import (
    DataAnomalyHaltRule,
    MaxDailyTradesRule,
    MaxDrawdownRule,
    MaxPositionRule,
    RiskRule,
    RuleBook,
    RuleContext,
    RuleHit,
)


@dataclass(slots=True)
class LivePaperRunResult:
    """Summary of one paper run."""

    run_id: str
    status: str
    processed_bars: int
    orders_placed: int
    updates_written: int
    risk_events: int
    equity_curve: pd.Series
    sqlite_path: Path
    log_path: Path
    metrics_prom_path: Path
    metrics_csv_path: Path
    kill_reason: str | None = None

    @property
    def kill_triggered(self) -> bool:
        """Whether kill switch terminated this run."""
        return self.status == "killed"


class LivePaperRunner:
    """Run strategy in an event loop against a paper broker connector."""

    def __init__(
        self,
        connector: PaperBrokerConnector,
        sqlite_path: str | Path = "reports/paper_trading.sqlite",
        symbol: str = "DEMO",
        max_abs_position_qty: float | None = None,
        max_daily_trades: int | None = None,
        risk_rules: list[RiskRule] | None = None,
        kill_switch_config: KillSwitchConfig | None = None,
        log_path: str | Path | None = None,
        metrics_prom_path: str | Path | None = None,
        metrics_csv_path: str | Path | None = None,
    ) -> None:
        if max_abs_position_qty is not None and max_abs_position_qty <= 0:
            msg = "max_abs_position_qty must be positive when provided."
            raise ValueError(msg)
        if max_daily_trades is not None and max_daily_trades < 1:
            msg = "max_daily_trades must be >= 1 when provided."
            raise ValueError(msg)

        self.connector = connector
        self.sqlite_path = Path(sqlite_path)
        self.symbol = str(symbol).upper()
        self.max_abs_position_qty = max_abs_position_qty
        self.max_daily_trades = max_daily_trades
        self.audit_store = RunAuditStore(self.sqlite_path)

        output_dir = self.sqlite_path.parent
        if log_path is None:
            self.log_path = output_dir / "paper_runner.jsonl"
        else:
            self.log_path = Path(log_path)
        self.metrics_prom_path = (
            Path(metrics_prom_path)
            if metrics_prom_path is not None
            else output_dir / "paper_metrics.prom"
        )
        self.metrics_csv_path = (
            Path(metrics_csv_path)
            if metrics_csv_path is not None
            else output_dir / "paper_metrics.csv"
        )
        self.kill_switch_config = kill_switch_config or KillSwitchConfig()
        self.rule_book = RuleBook(
            rules=risk_rules if risk_rules is not None else self._default_risk_rules()
        )

    def run(self, strategy: Strategy, bars: pd.DataFrame) -> LivePaperRunResult:
        """Run event loop and persist order lifecycle updates into SQLite."""
        normalized_bars = self._normalize_bars(bars)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)

        logger = JsonEventLogger(path=self.log_path)
        metrics = RuntimeMetrics(initial_equity=float(self.connector.cash))
        kill_switch = KillSwitch(config=self.kill_switch_config)
        expected_interval = self._infer_expected_interval(normalized_bars.index)

        connection = self.audit_store.connect()
        run_id: str | None = None
        try:
            self.audit_store.ensure_schema(connection)
            self._ensure_schema(connection)
            run_id = self.audit_store.create_run(
                connection=connection,
                mode="paper",
                symbol=self.symbol,
                strategy=_strategy_label(strategy),
                initial_cash=float(self.connector.cash),
                config={
                    "symbol": self.symbol,
                    "max_abs_position_qty": self.max_abs_position_qty,
                    "max_daily_trades": self.max_daily_trades,
                    "max_fill_ratio_per_step": self.connector.max_fill_ratio_per_step,
                    "max_volume_share": self.connector.max_volume_share,
                    "allow_short": self.connector.allow_short,
                    "kill_switch": {
                        "max_consecutive_errors": self.kill_switch_config.max_consecutive_errors,
                        "max_drawdown": self.kill_switch_config.max_drawdown,
                        "max_data_gap_steps": self.kill_switch_config.max_data_gap_steps,
                    },
                    "risk_rules": self.rule_book.describe(),
                    "log_path": str(self.log_path),
                    "metrics_prom_path": str(self.metrics_prom_path),
                    "metrics_csv_path": str(self.metrics_csv_path),
                },
                note="live_paper_runner",
            )
            logger.emit(
                level="info",
                event="run_started",
                run_id=run_id,
                symbol=self.symbol,
                bars_count=len(normalized_bars),
                expected_interval=str(expected_interval) if expected_interval is not None else None,
            )

            orders_placed = 0
            updates_written = 0
            risk_events = 0
            processed_bars = 0
            kill_reason: str | None = None
            previous_timestamp: pd.Timestamp | None = None
            previous_close: float | None = None
            peak_equity = float(self.connector.cash)
            daily_trade_counts: dict[str, int] = {}
            equity_points: list[tuple[pd.Timestamp, float]] = []

            for bar_index, (timestamp, bar_row) in enumerate(normalized_bars.iterrows()):
                bar_rule_hit = self._first_rule_hit(
                    context=RuleContext(
                        timestamp=timestamp,
                        bar_index=bar_index,
                        bar=bar_row,
                        previous_timestamp=previous_timestamp,
                        expected_interval=expected_interval,
                        previous_close=previous_close,
                    ),
                    actions={"halt"},
                )
                if bar_rule_hit is not None:
                    kill_reason = bar_rule_hit.reason
                    risk_events += 1
                    metrics.record_error()
                    self._record_risk_hit(
                        connection=connection,
                        run_id=run_id,
                        timestamp=timestamp,
                        hit=bar_rule_hit,
                        side="system",
                        qty=0.0,
                    )
                    logger.emit(
                        level="error",
                        event="risk_rule_triggered",
                        run_id=run_id,
                        reason=bar_rule_hit.reason,
                        rule_name=bar_rule_hit.rule_name,
                        action=bar_rule_hit.action,
                        message=bar_rule_hit.message,
                        details=bar_rule_hit.details,
                        bar_index=bar_index,
                        timestamp=timestamp,
                    )
                    processed_bars = bar_index
                    break
                previous_timestamp = timestamp

                try:
                    self.connector.step_market(symbol=self.symbol, bar=bar_row, timestamp=timestamp)
                    updates_written += self._sync_order_updates(
                        connection=connection,
                        run_id=run_id,
                        logger=logger,
                    )

                    bar_slice = normalized_bars.iloc[: bar_index + 1]
                    close_price = float(bar_row["close"])
                    current_position = self._position_for_symbol()
                    equity_value = float(self.connector.cash + current_position.qty * close_price)
                    peak_equity = max(float(peak_equity), float(equity_value))
                    equity_points.append((timestamp, equity_value))
                    metrics.record_bar(equity=equity_value)

                    self.audit_store.record_snapshot(
                        connection=connection,
                        run_id=run_id,
                        timestamp=timestamp,
                        bar_index=bar_index,
                        cash=float(self.connector.cash),
                        position_qty=float(current_position.qty),
                        equity=equity_value,
                        close_price=close_price,
                        note="bar_close",
                    )

                    drawdown_rule_hit = self._first_rule_hit(
                        context=RuleContext(
                            timestamp=timestamp,
                            bar_index=bar_index,
                            equity=equity_value,
                            peak_equity=peak_equity,
                        ),
                        actions={"halt"},
                    )
                    if drawdown_rule_hit is not None:
                        kill_reason = drawdown_rule_hit.reason
                        risk_events += 1
                        self._record_risk_hit(
                            connection=connection,
                            run_id=run_id,
                            timestamp=timestamp,
                            hit=drawdown_rule_hit,
                            side="system",
                            qty=0.0,
                        )
                        logger.emit(
                            level="error",
                            event="risk_rule_triggered",
                            run_id=run_id,
                            reason=drawdown_rule_hit.reason,
                            rule_name=drawdown_rule_hit.rule_name,
                            action=drawdown_rule_hit.action,
                            message=drawdown_rule_hit.message,
                            details=drawdown_rule_hit.details,
                            bar_index=bar_index,
                            timestamp=timestamp,
                            equity=equity_value,
                        )
                        processed_bars = bar_index + 1
                        break

                    portfolio_state = PortfolioState(
                        cash=float(self.connector.cash),
                        position_qty=float(current_position.qty),
                        average_entry_price=float(current_position.average_price),
                        equity=equity_value,
                    )
                    try:
                        intents = strategy.generate_orders(
                            bars=bar_slice,
                            portfolio_state=portfolio_state,
                        )
                        kill_switch.on_success()
                    except Exception as error:
                        metrics.record_error()
                        error_reason = kill_switch.on_error()
                        logger.emit(
                            level="error",
                            event="strategy_error",
                            run_id=run_id,
                            error_type=type(error).__name__,
                            message=str(error),
                            bar_index=bar_index,
                            timestamp=timestamp,
                        )
                        if error_reason is not None:
                            kill_reason = error_reason
                            risk_events += 1
                            self._record_risk_hit(
                                connection=connection,
                                run_id=run_id,
                                timestamp=timestamp,
                                hit=RuleHit(
                                    rule_name="kill_switch",
                                    action="halt",
                                    reason=error_reason,
                                    message=error_reason,
                                    details={},
                                ),
                                side="system",
                                qty=0.0,
                            )
                            logger.emit(
                                level="error",
                                event="kill_switch_triggered",
                                run_id=run_id,
                                reason=error_reason,
                                bar_index=bar_index,
                                timestamp=timestamp,
                            )
                            processed_bars = bar_index + 1
                            break
                        processed_bars = bar_index + 1
                        continue

                    projected_position = float(current_position.qty)
                    halted_on_order_rule = False
                    for signal_index, intent in enumerate(intents):
                        signal_reason = str(getattr(intent, "reason", "strategy_signal"))
                        order_type = "market" if intent.market else "limit"
                        self.audit_store.record_signal(
                            connection=connection,
                            run_id=run_id,
                            timestamp=timestamp,
                            bar_index=bar_index,
                            signal_index=signal_index,
                            side=intent.side,
                            qty=float(intent.qty),
                            order_type=order_type,
                            limit_price=intent.limit_price,
                            reason=signal_reason,
                        )

                        day_key = self._day_key(timestamp)
                        used_daily_trades = int(daily_trade_counts.get(day_key, 0))
                        projected_change = intent.qty if intent.side == "buy" else -intent.qty
                        next_position = projected_position + projected_change

                        order_rule_context = RuleContext(
                            timestamp=timestamp,
                            bar_index=bar_index,
                            current_position_qty=projected_position,
                            projected_position_qty=next_position,
                            order_side=intent.side,
                            order_qty=float(intent.qty),
                            daily_trade_count=used_daily_trades,
                        )
                        order_halt_hit = self._first_rule_hit(
                            context=order_rule_context,
                            actions={"halt"},
                        )
                        if order_halt_hit is not None:
                            kill_reason = order_halt_hit.reason
                            risk_events += 1
                            self._record_risk_hit(
                                connection=connection,
                                run_id=run_id,
                                timestamp=timestamp,
                                hit=order_halt_hit,
                                side=intent.side,
                                qty=float(intent.qty),
                            )
                            logger.emit(
                                level="error",
                                event="risk_rule_triggered",
                                run_id=run_id,
                                reason=order_halt_hit.reason,
                                rule_name=order_halt_hit.rule_name,
                                action=order_halt_hit.action,
                                message=order_halt_hit.message,
                                details=order_halt_hit.details,
                                side=intent.side,
                                qty=float(intent.qty),
                                bar_index=bar_index,
                                timestamp=timestamp,
                            )
                            halted_on_order_rule = True
                            break

                        order_block_hit = self._first_rule_hit(
                            context=order_rule_context,
                            actions={"block"},
                        )
                        if order_block_hit is not None:
                            risk_events += 1
                            self._record_risk_hit(
                                connection=connection,
                                run_id=run_id,
                                timestamp=timestamp,
                                hit=order_block_hit,
                                side=intent.side,
                                qty=float(intent.qty),
                            )
                            self.audit_store.record_order(
                                connection=connection,
                                run_id=run_id,
                                timestamp=timestamp,
                                order_id=f"blocked-{bar_index}-{signal_index}",
                                symbol=self.symbol,
                                side=intent.side,
                                qty=float(intent.qty),
                                order_type=order_type,
                                status="canceled",
                                filled_qty=0.0,
                                remaining_qty=float(intent.qty),
                                avg_fill_price=0.0,
                                note=order_block_hit.reason,
                            )
                            logger.emit(
                                level="warning",
                                event="order_blocked",
                                run_id=run_id,
                                reason=order_block_hit.reason,
                                rule_name=order_block_hit.rule_name,
                                action=order_block_hit.action,
                                message=order_block_hit.message,
                                details=order_block_hit.details,
                                side=intent.side,
                                qty=float(intent.qty),
                                bar_index=bar_index,
                                timestamp=timestamp,
                            )
                            continue

                        self.connector.place(
                            symbol=self.symbol,
                            side=intent.side,
                            qty=float(intent.qty),
                            market=bool(intent.market),
                            limit_price=intent.limit_price,
                        )
                        projected_position = next_position
                        daily_trade_counts[day_key] = used_daily_trades + 1
                        orders_placed += 1
                        metrics.record_order()
                        logger.emit(
                            level="info",
                            event="order_submitted",
                            run_id=run_id,
                            side=intent.side,
                            qty=float(intent.qty),
                            order_type=order_type,
                            limit_price=intent.limit_price,
                            bar_index=bar_index,
                            timestamp=timestamp,
                        )

                    if halted_on_order_rule:
                        processed_bars = bar_index + 1
                        break

                    updates_written += self._sync_order_updates(
                        connection=connection,
                        run_id=run_id,
                        logger=logger,
                    )
                    processed_bars = bar_index + 1
                    previous_close = close_price
                except Exception as error:
                    metrics.record_error()
                    error_reason = kill_switch.on_error()
                    logger.emit(
                        level="error",
                        event="runner_error",
                        run_id=run_id,
                        error_type=type(error).__name__,
                        message=str(error),
                        bar_index=bar_index,
                        timestamp=timestamp,
                    )
                    if error_reason is not None:
                        kill_reason = error_reason
                        risk_events += 1
                        self._record_risk_hit(
                            connection=connection,
                            run_id=run_id,
                            timestamp=timestamp,
                            hit=RuleHit(
                                rule_name="kill_switch",
                                action="halt",
                                reason=error_reason,
                                message=error_reason,
                                details={},
                            ),
                            side="system",
                            qty=0.0,
                        )
                        logger.emit(
                            level="error",
                            event="kill_switch_triggered",
                            run_id=run_id,
                            reason=error_reason,
                            bar_index=bar_index,
                            timestamp=timestamp,
                        )
                        processed_bars = bar_index + 1
                        break
                    processed_bars = bar_index + 1
                    continue

            metrics.finalize()
            prom_path = metrics.export_prometheus(self.metrics_prom_path)
            csv_path = metrics.export_csv(self.metrics_csv_path)
            metric_snapshot = metrics.snapshot()

            status = "killed" if kill_reason is not None else "completed"
            note = kill_reason or "ok"
            self.audit_store.finalize_run(
                connection=connection,
                run_id=run_id,
                status=status,
                note=note,
            )

            logger.emit(
                level="info",
                event="run_finished",
                run_id=run_id,
                status=status,
                kill_reason=kill_reason,
                processed_bars=processed_bars,
                orders_placed=orders_placed,
                updates_written=updates_written,
                risk_events=risk_events,
                pnl=metric_snapshot.pnl,
                drawdown=metric_snapshot.drawdown,
                order_rate=metric_snapshot.order_rate,
                error_rate=metric_snapshot.error_rate,
                metrics_prom_path=prom_path,
                metrics_csv_path=csv_path,
            )

            equity_curve = pd.Series(
                [value for _, value in equity_points],
                index=pd.DatetimeIndex([time for time, _ in equity_points]),
                name="equity",
                dtype=float,
            )
            return LivePaperRunResult(
                run_id=run_id,
                status=status,
                processed_bars=processed_bars,
                orders_placed=orders_placed,
                updates_written=updates_written,
                risk_events=risk_events,
                equity_curve=equity_curve,
                sqlite_path=self.sqlite_path,
                log_path=self.log_path,
                metrics_prom_path=prom_path,
                metrics_csv_path=csv_path,
                kill_reason=kill_reason,
            )
        except Exception as error:
            metrics.record_error()
            metrics.finalize()
            metrics.export_prometheus(self.metrics_prom_path)
            metrics.export_csv(self.metrics_csv_path)

            if run_id is not None:
                self.audit_store.finalize_run(
                    connection=connection,
                    run_id=run_id,
                    status="failed",
                    note=f"{type(error).__name__}: {error}",
                )
            logger.emit(
                level="error",
                event="run_failed",
                run_id=run_id,
                error_type=type(error).__name__,
                message=str(error),
            )
            raise
        finally:
            connection.close()

    def _position_for_symbol(self) -> BrokerPosition:
        positions = self.connector.get_positions()
        return positions.get(self.symbol, BrokerPosition(symbol=self.symbol, qty=0.0))

    def _default_risk_rules(self) -> list[RiskRule]:
        rules: list[RiskRule] = [
            DataAnomalyHaltRule(
                max_data_gap_steps=self.kill_switch_config.max_data_gap_steps,
                max_abs_return=None,
            ),
            MaxDrawdownRule(max_drawdown=self.kill_switch_config.max_drawdown),
        ]
        if self.max_abs_position_qty is not None:
            rules.append(MaxPositionRule(max_abs_position_qty=self.max_abs_position_qty))
        if self.max_daily_trades is not None:
            rules.append(MaxDailyTradesRule(max_trades_per_day=self.max_daily_trades))
        return rules

    def _first_rule_hit(
        self,
        context: RuleContext,
        actions: set[str],
    ) -> RuleHit | None:
        hits = self.rule_book.evaluate(context=context, actions=set(actions))
        if not hits:
            return None
        return hits[0]

    def _record_risk_hit(
        self,
        connection: sqlite3.Connection,
        run_id: str,
        timestamp: pd.Timestamp,
        hit: RuleHit,
        side: str,
        qty: float,
    ) -> None:
        self.audit_store.record_risk_event(
            connection=connection,
            run_id=run_id,
            timestamp=timestamp,
            rule_name=hit.rule_name,
            action=hit.action,
            reason=hit.reason,
            details=hit.details,
        )
        self._insert_risk_event(
            connection=connection,
            run_id=run_id,
            timestamp=timestamp,
            side=side,
            qty=qty,
            reason=hit.reason,
            rule_name=hit.rule_name,
            action=hit.action,
            details=hit.details,
        )

    def _sync_order_updates(
        self,
        connection: sqlite3.Connection,
        run_id: str,
        logger: JsonEventLogger,
    ) -> int:
        updates = list(self.connector.stream_updates())
        if not updates:
            return 0

        orders_by_id = {order.order_id: order for order in self.connector.get_orders()}
        with connection:
            for update in updates:
                order = orders_by_id[update.order_id]
                self._upsert_order(connection=connection, order=order)
                self._insert_order_update(connection=connection, update=update)
                self.audit_store.record_order(
                    connection=connection,
                    run_id=run_id,
                    timestamp=update.timestamp,
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    qty=float(order.qty),
                    order_type=order.order_type,
                    status=update.status,
                    filled_qty=float(update.filled_qty),
                    remaining_qty=float(update.remaining_qty),
                    avg_fill_price=float(order.avg_fill_price),
                    note=update.note,
                )
                logger.emit(
                    level="info",
                    event="order_update",
                    run_id=run_id,
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    status=update.status,
                    fill_qty=float(update.fill_qty),
                    fill_price=update.fill_price,
                    filled_qty=float(update.filled_qty),
                    remaining_qty=float(update.remaining_qty),
                )
                if update.fill_qty > 0 and update.fill_price is not None:
                    self.audit_store.record_fill(
                        connection=connection,
                        run_id=run_id,
                        timestamp=update.timestamp,
                        order_id=order.order_id,
                        symbol=order.symbol,
                        side=order.side,
                        qty=float(update.fill_qty),
                        price=float(update.fill_price),
                        reason=update.note,
                    )
        return len(updates)

    def _ensure_schema(self, connection: sqlite3.Connection) -> None:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_orders (
                    order_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    qty REAL NOT NULL,
                    order_type TEXT NOT NULL,
                    limit_price REAL NULL,
                    status TEXT NOT NULL,
                    filled_qty REAL NOT NULL,
                    avg_fill_price REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    canceled_at TEXT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_order_updates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    status TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    filled_qty REAL NOT NULL,
                    remaining_qty REAL NOT NULL,
                    fill_qty REAL NOT NULL,
                    fill_price REAL NULL,
                    note TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_risk_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NULL,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    side TEXT NOT NULL,
                    qty REAL NOT NULL,
                    reason TEXT NOT NULL,
                    rule_name TEXT NOT NULL DEFAULT '',
                    action TEXT NOT NULL DEFAULT 'halt',
                    details_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
        self._ensure_paper_risk_event_columns(connection=connection)

    def _upsert_order(self, connection: sqlite3.Connection, order: BrokerOrder) -> None:
        connection.execute(
            """
            INSERT INTO paper_orders (
                order_id,
                symbol,
                side,
                qty,
                order_type,
                limit_price,
                status,
                filled_qty,
                avg_fill_price,
                created_at,
                updated_at,
                canceled_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(order_id) DO UPDATE SET
                status = excluded.status,
                filled_qty = excluded.filled_qty,
                avg_fill_price = excluded.avg_fill_price,
                updated_at = excluded.updated_at,
                canceled_at = excluded.canceled_at
            """,
            (
                order.order_id,
                order.symbol,
                order.side,
                float(order.qty),
                order.order_type,
                None if order.limit_price is None else float(order.limit_price),
                order.status,
                float(order.filled_qty),
                float(order.avg_fill_price),
                order.created_at.isoformat(),
                order.updated_at.isoformat(),
                None if order.canceled_at is None else order.canceled_at.isoformat(),
            ),
        )

    def _insert_order_update(self, connection: sqlite3.Connection, update: OrderUpdate) -> None:
        connection.execute(
            """
            INSERT INTO paper_order_updates (
                order_id,
                symbol,
                status,
                timestamp,
                filled_qty,
                remaining_qty,
                fill_qty,
                fill_price,
                note
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                update.order_id,
                update.symbol,
                update.status,
                update.timestamp.isoformat(),
                float(update.filled_qty),
                float(update.remaining_qty),
                float(update.fill_qty),
                None if update.fill_price is None else float(update.fill_price),
                update.note,
            ),
        )

    def _insert_risk_event(
        self,
        connection: sqlite3.Connection,
        run_id: str,
        timestamp: pd.Timestamp,
        side: str,
        qty: float,
        reason: str,
        rule_name: str,
        action: str,
        details: dict[str, object] | None,
    ) -> None:
        details_json = json.dumps(
            dict(details or {}),
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        with connection:
            connection.execute(
                """
                INSERT INTO paper_risk_events (
                    run_id,
                    symbol,
                    timestamp,
                    side,
                    qty,
                    reason,
                    rule_name,
                    action,
                    details_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    self.symbol,
                    timestamp.isoformat(),
                    side,
                    float(qty),
                    reason,
                    rule_name,
                    action,
                    details_json,
                ),
            )

    @staticmethod
    def _day_key(timestamp: pd.Timestamp) -> str:
        parsed = pd.Timestamp(timestamp)
        if parsed.tzinfo is None:
            parsed = parsed.tz_localize("UTC")
        else:
            parsed = parsed.tz_convert("UTC")
        return parsed.strftime("%Y-%m-%d")

    @staticmethod
    def _ensure_paper_risk_event_columns(connection: sqlite3.Connection) -> None:
        column_rows = connection.execute("PRAGMA table_info(paper_risk_events)").fetchall()
        existing_columns = {str(row[1]) for row in column_rows}
        updates: list[str] = []
        if "run_id" not in existing_columns:
            updates.append("ALTER TABLE paper_risk_events ADD COLUMN run_id TEXT NULL")
        if "rule_name" not in existing_columns:
            updates.append(
                "ALTER TABLE paper_risk_events ADD COLUMN rule_name TEXT NOT NULL DEFAULT ''"
            )
        if "action" not in existing_columns:
            updates.append(
                "ALTER TABLE paper_risk_events ADD COLUMN action TEXT NOT NULL DEFAULT 'halt'"
            )
        if "details_json" not in existing_columns:
            updates.append(
                "ALTER TABLE paper_risk_events ADD COLUMN details_json TEXT NOT NULL DEFAULT '{}'"
            )

        with connection:
            for statement in updates:
                connection.execute(statement)

    @staticmethod
    def _normalize_bars(bars: pd.DataFrame) -> pd.DataFrame:
        if bars.empty:
            msg = "bars cannot be empty."
            raise ValueError(msg)
        if not isinstance(bars.index, pd.DatetimeIndex):
            msg = "bars must use DatetimeIndex."
            raise ValueError(msg)
        required_columns = {"open", "high", "low", "close", "volume"}
        missing_columns = sorted(required_columns.difference(bars.columns))
        if missing_columns:
            msg = f"bars missing required columns: {missing_columns}"
            raise ValueError(msg)

        normalized = bars.sort_index().copy()
        normalized = normalized[~normalized.index.duplicated(keep="last")]
        return normalized

    @staticmethod
    def _infer_expected_interval(index: pd.DatetimeIndex) -> pd.Timedelta | None:
        if len(index) < 2:
            return None
        diff_series = pd.Series(index).diff().dropna()
        positive_diffs = diff_series[diff_series > pd.Timedelta(0)]
        if positive_diffs.empty:
            return None
        return pd.Timedelta(positive_diffs.median())


def _strategy_label(strategy: Strategy) -> str:
    strategy_type = type(strategy)
    return f"{strategy_type.__module__}:{strategy_type.__name__}"
