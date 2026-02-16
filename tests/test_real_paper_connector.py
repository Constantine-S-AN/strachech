from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import pytest

from stratcheck.connectors import RealPaperConnector


@dataclass(slots=True)
class FakeHttpResponse:
    status_code: int
    payload: Any
    headers: dict[str, str]


class FakeWebSocketConnection:
    def __init__(self, scripted_messages: list[str]) -> None:
        self._messages = list(scripted_messages)
        self.sent_payloads: list[str] = []
        self.closed = False

    def recv(self) -> str:
        if self._messages:
            message = self._messages.pop(0)
            if message == "__TIMEOUT__":
                raise TimeoutError("timed out")
            if message == "__CLOSE__":
                raise RuntimeError("socket closed")
            return message
        time.sleep(0.01)
        raise TimeoutError("timed out")

    def send(self, payload: str) -> None:
        self.sent_payloads.append(payload)

    def close(self) -> None:
        self.closed = True


def test_real_paper_connector_blocks_live_environment_by_default() -> None:
    with pytest.raises(ValueError, match="paper-only"):
        RealPaperConnector(
            base_url="https://broker.example",
            environment="live",
            auto_start_websocket=False,
        )

    connector = RealPaperConnector(
        base_url="https://broker.example",
        environment="live",
        allow_live_environment=True,
        auto_start_websocket=False,
    )
    try:
        assert connector.environment == "live"
    finally:
        connector.close()


def test_real_paper_connector_place_is_idempotent_by_client_order_id() -> None:
    call_records: list[tuple[str, str, dict[str, Any] | None]] = []

    def fake_http_transport(
        method: str,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any] | None,
        timeout_seconds: float,
    ) -> FakeHttpResponse:
        del headers, timeout_seconds
        call_records.append((method, url, payload))
        assert payload is not None
        return FakeHttpResponse(
            status_code=201,
            payload={
                "order": {
                    "order_id": "oid-1",
                    "client_order_id": payload["client_order_id"],
                    "symbol": payload["symbol"],
                    "side": payload["side"],
                    "qty": payload["qty"],
                    "order_type": payload["order_type"],
                    "status": "new",
                    "filled_qty": 0.0,
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z",
                }
            },
            headers={},
        )

    connector = RealPaperConnector(
        base_url="https://broker.example",
        websocket_enabled=False,
        auto_start_websocket=False,
        http_transport=fake_http_transport,
    )
    try:
        first = connector.place(
            symbol="AAPL",
            side="buy",
            qty=3.0,
            market=True,
            client_order_id="cid-1",
        )
        second = connector.place(
            symbol="AAPL",
            side="buy",
            qty=3.0,
            market=True,
            client_order_id="cid-1",
        )

        assert first.order_id == "oid-1"
        assert second.order_id == "oid-1"
        assert len(call_records) == 1

        updates = list(connector.stream_updates())
        assert [item.status for item in updates] == ["new"]

        with pytest.raises(ValueError, match="different order parameters"):
            connector.place(
                symbol="AAPL",
                side="buy",
                qty=5.0,
                market=True,
                client_order_id="cid-1",
            )
    finally:
        connector.close()


def test_real_paper_connector_rate_limit_throttles_rest_calls() -> None:
    call_times: list[float] = []

    def fake_http_transport(
        method: str,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any] | None,
        timeout_seconds: float,
    ) -> FakeHttpResponse:
        del method, url, headers, timeout_seconds
        call_times.append(time.monotonic())
        assert payload is not None
        order_index = len(call_times)
        return FakeHttpResponse(
            status_code=201,
            payload={
                "order": {
                    "order_id": f"oid-{order_index}",
                    "client_order_id": payload["client_order_id"],
                    "symbol": payload["symbol"],
                    "side": payload["side"],
                    "qty": payload["qty"],
                    "order_type": payload["order_type"],
                    "status": "new",
                    "filled_qty": 0.0,
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z",
                }
            },
            headers={},
        )

    connector = RealPaperConnector(
        base_url="https://broker.example",
        websocket_enabled=False,
        auto_start_websocket=False,
        rate_limit_max_calls=1,
        rate_limit_window_seconds=0.08,
        rest_retry_limit=0,
        http_transport=fake_http_transport,
    )
    try:
        connector.place(symbol="AAPL", side="buy", qty=1.0, client_order_id="cid-10")
        connector.place(symbol="AAPL", side="buy", qty=1.0, client_order_id="cid-11")
    finally:
        connector.close()

    assert len(call_times) == 2
    assert call_times[1] - call_times[0] >= 0.06


def test_real_paper_connector_websocket_reconnects_and_emits_updates() -> None:
    connection_scripts = [
        [
            json.dumps(
                {
                    "event": "order_update",
                    "order": {
                        "order_id": "oid-ws",
                        "client_order_id": "cid-ws",
                        "symbol": "AAPL",
                        "side": "buy",
                        "qty": 1.0,
                        "order_type": "market",
                        "status": "new",
                        "filled_qty": 0.0,
                        "avg_fill_price": 0.0,
                        "created_at": "2024-01-01T00:00:00Z",
                        "updated_at": "2024-01-01T00:00:00Z",
                    },
                }
            ),
            "__CLOSE__",
        ],
        [
            json.dumps(
                {
                    "type": "order",
                    "order": {
                        "order_id": "oid-ws",
                        "client_order_id": "cid-ws",
                        "symbol": "AAPL",
                        "side": "buy",
                        "qty": 1.0,
                        "order_type": "market",
                        "status": "filled",
                        "filled_qty": 1.0,
                        "avg_fill_price": 100.0,
                        "last_fill_qty": 1.0,
                        "last_fill_price": 100.0,
                        "updated_at": "2024-01-01T00:00:01Z",
                    },
                }
            ),
            "__TIMEOUT__",
        ],
    ]
    created_connections: list[FakeWebSocketConnection] = []

    def fake_websocket_factory(
        url: str,
        headers: dict[str, str],
        timeout_seconds: float,
    ) -> FakeWebSocketConnection:
        del url, headers, timeout_seconds
        index = len(created_connections)
        script = connection_scripts[index] if index < len(connection_scripts) else ["__TIMEOUT__"]
        connection = FakeWebSocketConnection(scripted_messages=script)
        created_connections.append(connection)
        return connection

    def unused_http_transport(
        method: str,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any] | None,
        timeout_seconds: float,
    ) -> FakeHttpResponse:
        del method, url, headers, payload, timeout_seconds
        raise AssertionError("REST transport should not be called in websocket-only test.")

    connector = RealPaperConnector(
        base_url="https://broker.example",
        websocket_url="wss://broker.example/orders",
        websocket_enabled=True,
        auto_start_websocket=True,
        websocket_reconnect_min_seconds=0.01,
        websocket_reconnect_max_seconds=0.05,
        websocket_timeout_seconds=0.01,
        websocket_subscribe_message={"op": "subscribe_orders"},
        http_transport=unused_http_transport,
        websocket_factory=fake_websocket_factory,
    )

    captured_updates = []
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            captured_updates.extend(connector.stream_updates())
            if any(item.status == "filled" for item in captured_updates):
                break
            time.sleep(0.02)
    finally:
        connector.close()

    assert len(created_connections) >= 2
    assert [item.status for item in captured_updates][:2] == ["new", "filled"]

    orders = connector.get_orders()
    assert len(orders) == 1
    assert orders[0].status == "filled"
    positions = connector.get_positions()
    assert positions["AAPL"].qty == 1.0
