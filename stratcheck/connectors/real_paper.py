"""Real-broker paper connector with REST execution and optional WebSocket updates."""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Any, Protocol
from urllib import error as urllib_error
from urllib import request as urllib_request

import pandas as pd

from stratcheck.connectors.base import (
    BrokerOrder,
    BrokerPosition,
    OrderSide,
    OrderStatus,
    OrderType,
    OrderUpdate,
)

_EPSILON = 1e-12
_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _normalize_timestamp(value: pd.Timestamp | str | None) -> pd.Timestamp:
    if value is None:
        return _utc_now()
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        return parsed.tz_localize("UTC")
    return parsed.tz_convert("UTC")


def _as_float(value: Any, default: float | None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _first_present(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def _normalize_side(value: Any) -> OrderSide:
    normalized = str(value).strip().lower()
    if normalized not in {"buy", "sell"}:
        msg = f"Unsupported order side: {value!r}"
        raise ValueError(msg)
    return normalized  # type: ignore[return-value]


def _normalize_order_type(value: Any) -> OrderType:
    normalized = str(value).strip().lower()
    if normalized in {"market", "mkt"}:
        return "market"
    if normalized in {"limit", "lmt"}:
        return "limit"
    msg = f"Unsupported order type: {value!r}"
    raise ValueError(msg)


def _normalize_status(value: Any) -> OrderStatus:
    normalized = str(value).strip().lower()
    aliases: dict[str, OrderStatus] = {
        "new": "new",
        "accepted": "new",
        "open": "new",
        "pending": "new",
        "partially_filled": "partially_filled",
        "partiallyfilled": "partially_filled",
        "partial_fill": "partially_filled",
        "filled": "filled",
        "executed": "filled",
        "complete": "filled",
        "canceled": "canceled",
        "cancelled": "canceled",
    }
    if normalized not in aliases:
        msg = f"Unsupported order status: {value!r}"
        raise ValueError(msg)
    return aliases[normalized]


@dataclass(slots=True, frozen=True)
class _HttpResponse:
    status_code: int
    payload: Any
    headers: dict[str, str]


class HttpTransport(Protocol):
    """Protocol for injectable REST transport."""

    def __call__(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any] | None,
        timeout_seconds: float,
    ) -> _HttpResponse: ...


class WebSocketConnection(Protocol):
    """Protocol for websocket connection abstraction used by the background loop."""

    def recv(self) -> str | bytes: ...

    def send(self, payload: str) -> None: ...

    def close(self) -> None: ...


class WebSocketFactory(Protocol):
    """Factory protocol creating websocket connections."""

    def __call__(
        self,
        url: str,
        headers: dict[str, str],
        timeout_seconds: float,
    ) -> WebSocketConnection: ...


@dataclass(slots=True, frozen=True)
class _ClientOrderSpec:
    symbol: str
    side: OrderSide
    qty: float
    order_type: OrderType
    limit_price: float | None


class _FixedWindowRateLimiter:
    """Thread-safe fixed-window limiter for outbound REST calls."""

    def __init__(self, max_calls: int, window_seconds: float) -> None:
        if max_calls <= 0:
            msg = "max_calls must be positive."
            raise ValueError(msg)
        if window_seconds <= 0:
            msg = "window_seconds must be positive."
            raise ValueError(msg)

        self.max_calls = int(max_calls)
        self.window_seconds = float(window_seconds)
        self._window_start = time.monotonic()
        self._calls_in_window = 0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            sleep_seconds = 0.0
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._window_start
                if elapsed >= self.window_seconds:
                    self._window_start = now
                    self._calls_in_window = 0

                if self._calls_in_window < self.max_calls:
                    self._calls_in_window += 1
                    return

                sleep_seconds = max(self.window_seconds - elapsed, 0.0)

            time.sleep(max(sleep_seconds, 0.001))


def _default_http_transport(
    method: str,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any] | None,
    timeout_seconds: float,
) -> _HttpResponse:
    body: bytes | None = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")

    request_obj = urllib_request.Request(url=url, data=body, method=method.upper())
    for header_name, header_value in headers.items():
        request_obj.add_header(header_name, header_value)

    try:
        with urllib_request.urlopen(request_obj, timeout=timeout_seconds) as response:
            raw_body = response.read().decode("utf-8").strip()
            parsed_payload: Any = {}
            if raw_body:
                try:
                    parsed_payload = json.loads(raw_body)
                except json.JSONDecodeError:
                    parsed_payload = {"raw": raw_body}
            return _HttpResponse(
                status_code=int(response.status),
                payload=parsed_payload,
                headers={key.lower(): value for key, value in response.headers.items()},
            )
    except urllib_error.HTTPError as http_error:
        raw_body = http_error.read().decode("utf-8").strip()
        parsed_payload: Any = {}
        if raw_body:
            try:
                parsed_payload = json.loads(raw_body)
            except json.JSONDecodeError:
                parsed_payload = {"raw": raw_body}
        header_items = http_error.headers.items() if http_error.headers is not None else []
        return _HttpResponse(
            status_code=int(http_error.code),
            payload=parsed_payload,
            headers={key.lower(): value for key, value in header_items},
        )


def _default_websocket_factory(
    url: str,
    headers: dict[str, str],
    timeout_seconds: float,
) -> WebSocketConnection:
    try:
        import websocket  # type: ignore[import-not-found]
    except ImportError as error:
        msg = (
            "WebSocket updates require the `websocket-client` package when `websocket_url` is set."
        )
        raise RuntimeError(msg) from error

    header_lines = [
        f"{header_name}: {header_value}"
        for header_name, header_value in headers.items()
    ]
    return websocket.create_connection(url, header=header_lines, timeout=timeout_seconds)


class RealPaperConnector:
    """REST + WebSocket connector for broker paper trading with safety guardrails.

    Defaults are intentionally paper-safe:
    - `environment` defaults to `"paper"`.
    - non-paper environments require explicit `allow_live_environment=True`.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        environment: str = "paper",
        allow_live_environment: bool = False,
        orders_path: str = "/orders",
        positions_path: str = "/positions",
        rate_limit_max_calls: int = 30,
        rate_limit_window_seconds: float = 1.0,
        rest_timeout_seconds: float = 10.0,
        rest_retry_limit: int = 3,
        rest_retry_backoff_seconds: float = 0.4,
        websocket_url: str | None = None,
        websocket_enabled: bool = True,
        auto_start_websocket: bool = True,
        websocket_subscribe_message: dict[str, Any] | None = None,
        websocket_ping_message: dict[str, Any] | None = None,
        websocket_ping_interval_seconds: float = 20.0,
        websocket_timeout_seconds: float = 10.0,
        websocket_reconnect_min_seconds: float = 1.0,
        websocket_reconnect_max_seconds: float = 30.0,
        static_headers: dict[str, str] | None = None,
        http_transport: HttpTransport | None = None,
        websocket_factory: WebSocketFactory | None = None,
    ) -> None:
        if not str(base_url).strip():
            msg = "base_url must be non-empty."
            raise ValueError(msg)
        if rest_timeout_seconds <= 0:
            msg = "rest_timeout_seconds must be positive."
            raise ValueError(msg)
        if rest_retry_limit < 0:
            msg = "rest_retry_limit cannot be negative."
            raise ValueError(msg)
        if rest_retry_backoff_seconds <= 0:
            msg = "rest_retry_backoff_seconds must be positive."
            raise ValueError(msg)
        if websocket_ping_interval_seconds <= 0:
            msg = "websocket_ping_interval_seconds must be positive."
            raise ValueError(msg)
        if websocket_timeout_seconds <= 0:
            msg = "websocket_timeout_seconds must be positive."
            raise ValueError(msg)
        if websocket_reconnect_min_seconds <= 0:
            msg = "websocket_reconnect_min_seconds must be positive."
            raise ValueError(msg)
        if websocket_reconnect_max_seconds < websocket_reconnect_min_seconds:
            msg = "websocket_reconnect_max_seconds must be >= websocket_reconnect_min_seconds."
            raise ValueError(msg)

        normalized_environment = str(environment).strip().lower()
        if normalized_environment != "paper" and not allow_live_environment:
            msg = (
                "RealPaperConnector defaults to paper-only. "
                "Set allow_live_environment=True explicitly to use non-paper environments."
            )
            raise ValueError(msg)

        self.base_url = str(base_url).rstrip("/")
        self.api_key = api_key
        self.environment = normalized_environment
        self.allow_live_environment = bool(allow_live_environment)
        self.orders_path = str(orders_path)
        self.positions_path = str(positions_path)
        self.rest_timeout_seconds = float(rest_timeout_seconds)
        self.rest_retry_limit = int(rest_retry_limit)
        self.rest_retry_backoff_seconds = float(rest_retry_backoff_seconds)
        self.websocket_url = websocket_url
        self.websocket_enabled = bool(websocket_enabled)
        self.websocket_subscribe_message = websocket_subscribe_message
        self.websocket_ping_message = websocket_ping_message
        self.websocket_ping_interval_seconds = float(websocket_ping_interval_seconds)
        self.websocket_timeout_seconds = float(websocket_timeout_seconds)
        self.websocket_reconnect_min_seconds = float(websocket_reconnect_min_seconds)
        self.websocket_reconnect_max_seconds = float(websocket_reconnect_max_seconds)
        self.static_headers = dict(static_headers or {})

        self._http_transport = http_transport or _default_http_transport
        self._websocket_factory = websocket_factory or _default_websocket_factory
        self._rate_limiter = _FixedWindowRateLimiter(
            max_calls=rate_limit_max_calls,
            window_seconds=rate_limit_window_seconds,
        )

        self._state_lock = threading.RLock()
        self._orders_by_id: dict[str, BrokerOrder] = {}
        self._positions: dict[str, BrokerPosition] = {}
        self._client_order_specs: dict[str, _ClientOrderSpec] = {}
        self._client_order_to_order_id: dict[str, str] = {}
        self._pending_updates: deque[OrderUpdate] = deque()

        self._ws_thread: threading.Thread | None = None
        self._ws_stop_event = threading.Event()
        self._ws_connected_event = threading.Event()

        if auto_start_websocket and self.websocket_enabled and self.websocket_url:
            self.start_websocket()

    def __enter__(self) -> RealPaperConnector:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()

    def close(self) -> None:
        """Stop background websocket thread."""
        self.stop_websocket()

    def start_websocket(self) -> None:
        """Start websocket background consumer thread when configured."""
        if not self.websocket_enabled or not self.websocket_url:
            return
        with self._state_lock:
            if self._ws_thread is not None and self._ws_thread.is_alive():
                return
            self._ws_stop_event.clear()
            self._ws_thread = threading.Thread(
                target=self._run_websocket_loop,
                name="stratcheck-real-paper-ws",
                daemon=True,
            )
            self._ws_thread.start()

    def stop_websocket(self, join_timeout_seconds: float = 3.0) -> None:
        """Signal websocket loop to stop and wait briefly for thread exit."""
        self._ws_stop_event.set()
        thread = self._ws_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=join_timeout_seconds)
        self._ws_connected_event.clear()

    def wait_until_websocket_connected(self, timeout_seconds: float = 3.0) -> bool:
        """Wait until websocket connection is established."""
        return self._ws_connected_event.wait(timeout=timeout_seconds)

    @property
    def websocket_connected(self) -> bool:
        """Whether websocket loop is currently connected."""
        return self._ws_connected_event.is_set()

    def place(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        *,
        market: bool = True,
        limit_price: float | None = None,
        client_order_id: str | None = None,
    ) -> BrokerOrder:
        """Place order via REST with client_order_id idempotency."""
        normalized_symbol = str(symbol).upper()
        normalized_side = _normalize_side(side)
        normalized_qty = float(qty)
        normalized_type: OrderType = "market" if market else "limit"
        normalized_limit_price = None if limit_price is None else float(limit_price)

        if normalized_qty <= 0:
            msg = "qty must be positive."
            raise ValueError(msg)
        if normalized_type == "limit" and normalized_limit_price is None:
            msg = "limit_price is required for limit orders."
            raise ValueError(msg)

        resolved_client_order_id = str(client_order_id or uuid.uuid4()).strip()
        if not resolved_client_order_id:
            msg = "client_order_id cannot be blank."
            raise ValueError(msg)

        intended_order = _ClientOrderSpec(
            symbol=normalized_symbol,
            side=normalized_side,
            qty=normalized_qty,
            order_type=normalized_type,
            limit_price=normalized_limit_price,
        )
        with self._state_lock:
            previous_spec = self._client_order_specs.get(resolved_client_order_id)
            if previous_spec is not None:
                if previous_spec != intended_order:
                    msg = (
                        f"client_order_id={resolved_client_order_id!r} already exists with "
                        "different order parameters."
                    )
                    raise ValueError(msg)
                existing_order_id = self._client_order_to_order_id.get(resolved_client_order_id)
                if existing_order_id and existing_order_id in self._orders_by_id:
                    return replace(self._orders_by_id[existing_order_id])

        payload: dict[str, Any] = {
            "symbol": normalized_symbol,
            "side": normalized_side,
            "qty": normalized_qty,
            "order_type": normalized_type,
            "client_order_id": resolved_client_order_id,
        }
        if normalized_type == "limit":
            payload["limit_price"] = normalized_limit_price

        response = self._request_with_retries("POST", self.orders_path, payload=payload)
        if response.status_code not in {200, 201, 202, 409}:
            msg = f"Order placement failed with status={response.status_code}: {response.payload!r}"
            raise RuntimeError(msg)

        order_payload = self._extract_order_payload(response.payload)
        if order_payload is None:
            order_payload = {
                "order_id": resolved_client_order_id,
                "client_order_id": resolved_client_order_id,
                "symbol": normalized_symbol,
                "side": normalized_side,
                "qty": normalized_qty,
                "order_type": normalized_type,
                "limit_price": normalized_limit_price,
                "status": "new",
                "filled_qty": 0.0,
            }

        order, reported_fill_qty, reported_fill_price = self._order_from_wire(
            payload=order_payload,
            default_order_id=resolved_client_order_id,
        )
        synced_order = self._sync_order(
            order=order,
            note="order_accepted",
            reported_fill_qty=reported_fill_qty,
            reported_fill_price=reported_fill_price,
        )
        with self._state_lock:
            self._client_order_specs[resolved_client_order_id] = intended_order
            self._client_order_to_order_id[resolved_client_order_id] = synced_order.order_id
        return replace(synced_order)

    def cancel(self, order_id: str) -> BrokerOrder:
        """Cancel order via REST endpoint."""
        normalized_order_id = str(order_id).strip()
        if not normalized_order_id:
            msg = "order_id cannot be blank."
            raise ValueError(msg)

        response = self._request_with_retries(
            "DELETE",
            f"{self.orders_path.rstrip('/')}/{normalized_order_id}",
            payload=None,
        )
        if response.status_code not in {200, 202, 204}:
            msg = f"Order cancel failed with status={response.status_code}: {response.payload!r}"
            raise RuntimeError(msg)

        order_payload = self._extract_order_payload(response.payload)
        if order_payload is None:
            with self._state_lock:
                known_order = self._orders_by_id.get(normalized_order_id)
            if known_order is None:
                msg = f"Unknown order_id={normalized_order_id!r}"
                raise ValueError(msg)
            now = _utc_now()
            fallback = replace(
                known_order,
                status="canceled",
                canceled_at=now,
                updated_at=now,
            )
            synced = self._sync_order(order=fallback, note="order_canceled")
            return replace(synced)

        order, reported_fill_qty, reported_fill_price = self._order_from_wire(
            payload=order_payload,
            default_order_id=normalized_order_id,
        )
        synced = self._sync_order(
            order=order,
            note="order_canceled",
            reported_fill_qty=reported_fill_qty,
            reported_fill_price=reported_fill_price,
        )
        return replace(synced)

    def get_orders(self) -> list[BrokerOrder]:
        """Return local order snapshots updated by REST responses and websocket events."""
        with self._state_lock:
            ordered_orders = sorted(
                self._orders_by_id.values(),
                key=lambda order_snapshot: (
                    order_snapshot.created_at.value,
                    order_snapshot.order_id,
                ),
            )
            return [replace(order) for order in ordered_orders]

    def get_positions(self) -> dict[str, BrokerPosition]:
        """Return local position snapshots inferred from fills or refreshed from REST."""
        with self._state_lock:
            return {symbol: replace(position) for symbol, position in self._positions.items()}

    def stream_updates(self) -> Iterator[OrderUpdate]:
        """Yield pending updates once and clear update queue."""
        while True:
            with self._state_lock:
                if not self._pending_updates:
                    return
                update = self._pending_updates.popleft()
            yield replace(update)

    def refresh_orders(self) -> list[BrokerOrder]:
        """Pull current orders from broker REST and reconcile local cache."""
        response = self._request_with_retries("GET", self.orders_path, payload=None)
        if response.status_code not in {200, 206}:
            msg = f"Order refresh failed with status={response.status_code}: {response.payload!r}"
            raise RuntimeError(msg)

        for item in self._extract_order_payloads(response.payload):
            order, reported_fill_qty, reported_fill_price = self._order_from_wire(payload=item)
            self._sync_order(
                order=order,
                note="order_refresh",
                reported_fill_qty=reported_fill_qty,
                reported_fill_price=reported_fill_price,
            )
        return self.get_orders()

    def refresh_positions(self) -> dict[str, BrokerPosition]:
        """Pull current positions from broker REST and replace local position snapshots."""
        response = self._request_with_retries("GET", self.positions_path, payload=None)
        if response.status_code not in {200, 206}:
            msg = (
                "Position refresh failed with "
                f"status={response.status_code}: {response.payload!r}"
            )
            raise RuntimeError(msg)

        payload_list = self._extract_position_payloads(response.payload)
        next_positions: dict[str, BrokerPosition] = {}
        for item in payload_list:
            symbol_value = str(_first_present(item, ("symbol", "ticker")) or "").upper()
            if not symbol_value:
                continue
            qty_value = float(
                _as_float(
                    _first_present(item, ("qty", "quantity", "position_qty")),
                    0.0,
                )
                or 0.0
            )
            avg_value = float(
                _as_float(_first_present(item, ("average_price", "avg_price")), 0.0)
                or 0.0
            )
            if abs(qty_value) <= _EPSILON:
                continue
            next_positions[symbol_value] = BrokerPosition(
                symbol=symbol_value,
                qty=qty_value,
                average_price=avg_value,
            )

        with self._state_lock:
            self._positions = next_positions
            return {symbol: replace(position) for symbol, position in self._positions.items()}

    def _compose_url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-Stratcheck-Environment": self.environment,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.static_headers)
        return headers

    def _request_with_retries(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
    ) -> _HttpResponse:
        url = self._compose_url(path)
        response: _HttpResponse | None = None
        for attempt in range(self.rest_retry_limit + 1):
            self._rate_limiter.acquire()
            try:
                response = self._http_transport(
                    method=method,
                    url=url,
                    headers=self._build_headers(),
                    payload=payload,
                    timeout_seconds=self.rest_timeout_seconds,
                )
            except Exception as error:
                if attempt >= self.rest_retry_limit:
                    msg = f"REST {method} {url} failed after retries."
                    raise RuntimeError(msg) from error
                time.sleep(self._retry_delay(attempt))
                continue

            if response.status_code in _RETRY_STATUS_CODES and attempt < self.rest_retry_limit:
                retry_after_seconds = self._retry_after_seconds(response.headers)
                backoff_seconds = retry_after_seconds or self._retry_delay(attempt)
                time.sleep(backoff_seconds)
                continue
            return response

        if response is None:
            msg = f"REST {method} {url} failed without response."
            raise RuntimeError(msg)
        return response

    def _retry_delay(self, attempt: int) -> float:
        return self.rest_retry_backoff_seconds * (2**attempt)

    @staticmethod
    def _retry_after_seconds(headers: dict[str, str]) -> float | None:
        retry_after_value = headers.get("retry-after")
        if retry_after_value is None:
            return None
        parsed = _as_float(retry_after_value, default=None)
        if parsed is None or parsed <= 0:
            return None
        return float(parsed)

    def _run_websocket_loop(self) -> None:
        if not self.websocket_url:
            return

        reconnect_delay = self.websocket_reconnect_min_seconds
        while not self._ws_stop_event.is_set():
            connection: WebSocketConnection | None = None
            try:
                connection = self._websocket_factory(
                    url=self.websocket_url,
                    headers=self._build_headers(),
                    timeout_seconds=self.websocket_timeout_seconds,
                )
                self._ws_connected_event.set()
                reconnect_delay = self.websocket_reconnect_min_seconds

                if self.websocket_subscribe_message is not None:
                    connection.send(json.dumps(self.websocket_subscribe_message))

                last_ping_at = time.monotonic()
                while not self._ws_stop_event.is_set():
                    if (
                        self.websocket_ping_message is not None
                        and time.monotonic() - last_ping_at >= self.websocket_ping_interval_seconds
                    ):
                        connection.send(json.dumps(self.websocket_ping_message))
                        last_ping_at = time.monotonic()

                    try:
                        raw_message = connection.recv()
                    except Exception as receive_error:
                        if self._looks_like_timeout(receive_error):
                            continue
                        raise

                    if raw_message in {"", b"", None}:
                        msg = "WebSocket returned empty payload."
                        raise RuntimeError(msg)
                    self._consume_websocket_payload(raw_message)
            except Exception:
                self._ws_connected_event.clear()
                if connection is not None:
                    try:
                        connection.close()
                    except Exception:
                        pass
                if self._ws_stop_event.is_set():
                    return
                time.sleep(reconnect_delay)
                reconnect_delay = min(
                    reconnect_delay * 2.0,
                    self.websocket_reconnect_max_seconds,
                )
            else:
                self._ws_connected_event.clear()
                if connection is not None:
                    try:
                        connection.close()
                    except Exception:
                        pass

    @staticmethod
    def _looks_like_timeout(error: Exception) -> bool:
        if isinstance(error, TimeoutError):
            return True
        return "timed out" in str(error).lower()

    def _consume_websocket_payload(self, raw_message: str | bytes) -> None:
        message_text = (
            raw_message.decode("utf-8")
            if isinstance(raw_message, bytes)
            else str(raw_message)
        )
        stripped = message_text.strip()
        if not stripped:
            return
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return

        order_payload = self._extract_order_payload(payload)
        if order_payload is None:
            return

        order, reported_fill_qty, reported_fill_price = self._order_from_wire(payload=order_payload)
        synced = self._sync_order(
            order=order,
            note="websocket_update",
            reported_fill_qty=reported_fill_qty,
            reported_fill_price=reported_fill_price,
        )
        client_order_id = _first_present(order_payload, ("client_order_id", "clientOrderId"))
        if client_order_id is None:
            return
        normalized_client_order_id = str(client_order_id).strip()
        if not normalized_client_order_id:
            return

        spec = _ClientOrderSpec(
            symbol=synced.symbol,
            side=synced.side,
            qty=float(synced.qty),
            order_type=synced.order_type,
            limit_price=synced.limit_price,
        )
        with self._state_lock:
            self._client_order_specs.setdefault(normalized_client_order_id, spec)
            self._client_order_to_order_id[normalized_client_order_id] = synced.order_id

    def _sync_order(
        self,
        order: BrokerOrder,
        note: str,
        reported_fill_qty: float | None = None,
        reported_fill_price: float | None = None,
    ) -> BrokerOrder:
        with self._state_lock:
            previous = self._orders_by_id.get(order.order_id)
            merged_order = order
            if previous is not None:
                merged_order = replace(
                    order,
                    created_at=previous.created_at,
                    canceled_at=(
                        order.canceled_at
                        if order.status == "canceled"
                        else previous.canceled_at
                    ),
                )

            previous_filled = previous.filled_qty if previous is not None else 0.0
            delta_filled = max(float(merged_order.filled_qty - previous_filled), 0.0)

            emitted_fill_qty = (
                float(reported_fill_qty)
                if reported_fill_qty is not None and reported_fill_qty >= 0
                else delta_filled
            )
            emitted_fill_price = reported_fill_price
            if emitted_fill_qty > _EPSILON and emitted_fill_price is None:
                emitted_fill_price = (
                    float(merged_order.avg_fill_price)
                    if merged_order.avg_fill_price > 0
                    else None
                )

            self._orders_by_id[merged_order.order_id] = merged_order

            if delta_filled > _EPSILON:
                self._apply_fill_to_positions(
                    order=merged_order,
                    fill_qty=delta_filled,
                    fill_price=emitted_fill_price,
                )

            should_emit = previous is None
            if previous is not None:
                status_changed = merged_order.status != previous.status
                filled_changed = abs(merged_order.filled_qty - previous.filled_qty) > _EPSILON
                should_emit = status_changed or filled_changed or emitted_fill_qty > _EPSILON

            if should_emit:
                self._pending_updates.append(
                    OrderUpdate(
                        order_id=merged_order.order_id,
                        symbol=merged_order.symbol,
                        status=merged_order.status,
                        timestamp=merged_order.updated_at,
                        filled_qty=float(merged_order.filled_qty),
                        remaining_qty=float(merged_order.remaining_qty),
                        fill_qty=float(emitted_fill_qty),
                        fill_price=(
                            None if emitted_fill_price is None else float(emitted_fill_price)
                        ),
                        note=note,
                    )
                )

            return merged_order

    def _apply_fill_to_positions(
        self,
        order: BrokerOrder,
        fill_qty: float,
        fill_price: float | None,
    ) -> None:
        existing_position = self._positions.get(
            order.symbol,
            BrokerPosition(order.symbol, 0.0, 0.0),
        )
        effective_fill_price = (
            float(fill_price)
            if fill_price is not None
            else float(existing_position.average_price)
        )

        if order.side == "buy":
            next_qty = existing_position.qty + fill_qty
            if next_qty <= _EPSILON:
                self._positions.pop(order.symbol, None)
                return
            existing_value = existing_position.qty * existing_position.average_price
            new_value = fill_qty * effective_fill_price
            average_price = (existing_value + new_value) / next_qty
            self._positions[order.symbol] = BrokerPosition(order.symbol, next_qty, average_price)
            return

        next_qty = existing_position.qty - fill_qty
        if abs(next_qty) <= _EPSILON:
            self._positions.pop(order.symbol, None)
            return
        next_average = existing_position.average_price if next_qty > 0 else effective_fill_price
        self._positions[order.symbol] = BrokerPosition(order.symbol, next_qty, next_average)

    def _extract_order_payload(self, payload: Any) -> dict[str, Any] | None:
        if isinstance(payload, list):
            if not payload:
                return None
            first = payload[0]
            return first if isinstance(first, dict) else None
        if not isinstance(payload, dict):
            return None
        if "order" in payload and isinstance(payload["order"], dict):
            return payload["order"]
        if "data" in payload and isinstance(payload["data"], dict):
            nested = payload["data"]
            if "order" in nested and isinstance(nested["order"], dict):
                return nested["order"]
            if "order_id" in nested or "id" in nested:
                return nested
        if "order_id" in payload or "id" in payload:
            return payload
        return None

    def _extract_order_payloads(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            if "orders" in payload and isinstance(payload["orders"], list):
                return [item for item in payload["orders"] if isinstance(item, dict)]
            if "data" in payload and isinstance(payload["data"], list):
                return [item for item in payload["data"] if isinstance(item, dict)]
            one = self._extract_order_payload(payload)
            if one is not None:
                return [one]
        return []

    @staticmethod
    def _extract_position_payloads(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            if "positions" in payload and isinstance(payload["positions"], list):
                return [item for item in payload["positions"] if isinstance(item, dict)]
            if "data" in payload and isinstance(payload["data"], list):
                return [item for item in payload["data"] if isinstance(item, dict)]
            if "symbol" in payload:
                return [payload]
        return []

    def _order_from_wire(
        self,
        payload: dict[str, Any],
        default_order_id: str | None = None,
    ) -> tuple[BrokerOrder, float | None, float | None]:
        order_id_value = _first_present(payload, ("order_id", "id"))
        if order_id_value is None:
            order_id_value = default_order_id
        if order_id_value is None:
            msg = f"Broker order payload missing order_id: {payload!r}"
            raise ValueError(msg)

        symbol_value = str(_first_present(payload, ("symbol", "ticker")) or "").upper()
        if not symbol_value:
            msg = f"Broker order payload missing symbol: {payload!r}"
            raise ValueError(msg)

        side_value = _normalize_side(_first_present(payload, ("side",)))
        qty_value = _as_float(_first_present(payload, ("qty", "quantity", "size")), default=None)
        if qty_value is None:
            msg = f"Broker order payload missing qty: {payload!r}"
            raise ValueError(msg)

        order_type_raw = _first_present(payload, ("order_type", "type"))
        order_type_value = _normalize_order_type(order_type_raw or "market")
        limit_price_raw = _as_float(_first_present(payload, ("limit_price", "price")), default=None)
        limit_price_value = limit_price_raw if order_type_value == "limit" else None

        status_raw = _first_present(payload, ("status", "state"))
        status_value = _normalize_status(status_raw or "new")

        filled_qty_value = float(
            _as_float(
                _first_present(
                    payload,
                    ("filled_qty", "filled_quantity", "filled", "executed_qty"),
                ),
                default=0.0,
            )
            or 0.0
        )
        avg_fill_price_value = float(
            _as_float(
                _first_present(payload, ("avg_fill_price", "average_fill_price", "avg_price")),
                default=0.0,
            )
            or 0.0
        )

        created_timestamp = _normalize_timestamp(
            _first_present(payload, ("created_at", "submitted_at", "created_time"))
        )
        updated_timestamp = _normalize_timestamp(
            _first_present(payload, ("updated_at", "updated_time")) or created_timestamp
        )
        canceled_raw = _first_present(payload, ("canceled_at", "cancelled_at"))
        canceled_timestamp = (
            _normalize_timestamp(canceled_raw)
            if canceled_raw is not None
            else None
        )

        order = BrokerOrder(
            order_id=str(order_id_value),
            symbol=symbol_value,
            side=side_value,
            qty=float(qty_value),
            order_type=order_type_value,
            limit_price=limit_price_value,
            status=status_value,
            filled_qty=filled_qty_value,
            avg_fill_price=avg_fill_price_value,
            created_at=created_timestamp,
            updated_at=updated_timestamp,
            canceled_at=canceled_timestamp,
        )

        reported_fill_qty = _as_float(
            _first_present(payload, ("last_fill_qty", "fill_qty")),
            default=None,
        )
        reported_fill_price = _as_float(
            _first_present(payload, ("last_fill_price", "fill_price")),
            default=None,
        )
        return order, reported_fill_qty, reported_fill_price
