"""Alert routing helpers with console output and placeholder delivery channels."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TextIO

import pandas as pd

AlertType = Literal[
    "kill_switch",
    "data_interruption",
    "order_stuck",
    "drawdown_threshold",
]
AlertSeverity = Literal["info", "warning", "error", "critical"]


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _normalize_timestamp(value: pd.Timestamp | str | None) -> pd.Timestamp:
    if value is None:
        return _utc_now()
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        return parsed.tz_localize("UTC")
    return parsed.tz_convert("UTC")


@dataclass(slots=True)
class AlertEvent:
    """Single alert payload emitted to one or more channels."""

    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    timestamp: pd.Timestamp = field(default_factory=_utc_now)
    run_id: str | None = None
    symbol: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize alert to JSON-safe dictionary."""
        payload: dict[str, Any] = {
            "timestamp": _normalize_timestamp(self.timestamp).isoformat(),
            "alert_type": self.alert_type,
            "severity": self.severity,
            "title": str(self.title),
            "message": str(self.message),
            "details": _to_jsonable(self.details),
        }
        if self.run_id is not None:
            payload["run_id"] = str(self.run_id)
        if self.symbol is not None:
            payload["symbol"] = str(self.symbol).upper()
        return payload


class AlertChannel(Protocol):
    """Protocol shared by alert delivery channels."""

    name: str

    def send(self, alert: AlertEvent) -> None:
        """Deliver one alert payload."""


class ConsoleAlertChannel:
    """Write alert lines to console stream."""

    name = "console"

    def __init__(self, stream: TextIO | None = None, json_mode: bool = False) -> None:
        self._stream = stream
        self.json_mode = bool(json_mode)

    def send(self, alert: AlertEvent) -> None:
        payload = alert.to_dict()
        if self.json_mode:
            line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        else:
            parts = [
                f"[{payload['timestamp']}]",
                f"[{payload['severity']}:{payload['alert_type']}]",
                f"{payload['title']}: {payload['message']}",
            ]
            if "run_id" in payload:
                parts.append(f"run_id={payload['run_id']}")
            if "symbol" in payload:
                parts.append(f"symbol={payload['symbol']}")
            line = " ".join(parts)
        destination = self._stream
        if destination is None:
            print(line)
            return
        destination.write(line + "\n")
        destination.flush()


class EmailAlertChannel:
    """Email alert placeholder channel.

    Current behavior stores alerts into `sent_alerts` only. SMTP delivery can be
    added later without changing router API.
    """

    name = "email"

    def __init__(
        self,
        recipients: list[str] | None = None,
        sender: str | None = None,
        smtp_host: str | None = None,
    ) -> None:
        self.recipients = [str(item) for item in (recipients or [])]
        self.sender = None if sender is None else str(sender)
        self.smtp_host = None if smtp_host is None else str(smtp_host)
        self.sent_alerts: list[AlertEvent] = []

    def send(self, alert: AlertEvent) -> None:
        self.sent_alerts.append(alert)


class TelegramAlertChannel:
    """Telegram alert placeholder channel.

    Current behavior stores alerts into `sent_alerts` only. Bot API delivery can be
    added later without changing router API.
    """

    name = "telegram"

    def __init__(self, chat_id: str | None = None, bot_token: str | None = None) -> None:
        self.chat_id = None if chat_id is None else str(chat_id)
        self.bot_token = None if bot_token is None else str(bot_token)
        self.sent_alerts: list[AlertEvent] = []

    def send(self, alert: AlertEvent) -> None:
        self.sent_alerts.append(alert)


class WebhookAlertChannel:
    """Webhook alert placeholder channel.

    Current behavior stores alerts into `sent_alerts` only. HTTP POST delivery can
    be added later without changing router API.
    """

    name = "webhook"

    def __init__(self, url: str | None = None, headers: dict[str, str] | None = None) -> None:
        self.url = None if url is None else str(url)
        self.headers = {str(key): str(value) for key, value in (headers or {}).items()}
        self.sent_alerts: list[AlertEvent] = []

    def send(self, alert: AlertEvent) -> None:
        self.sent_alerts.append(alert)


class AlertRouter:
    """Route alerts to configured channels."""

    def __init__(self, channels: list[AlertChannel] | None = None) -> None:
        if channels is None:
            self.channels: list[AlertChannel] = [ConsoleAlertChannel()]
        else:
            self.channels = list(channels)

    def add_channel(self, channel: AlertChannel) -> None:
        """Register one more delivery channel."""
        self.channels.append(channel)

    def emit(self, alert: AlertEvent) -> AlertEvent:
        """Publish one alert to all configured channels."""
        for channel in self.channels:
            channel.send(alert)
        return alert

    def alert_kill_switch(
        self,
        *,
        reason: str,
        run_id: str | None = None,
        symbol: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> AlertEvent:
        """Emit kill-switch alert."""
        payload_details = dict(details or {})
        payload_details.setdefault("reason", str(reason))
        return self.emit(
            AlertEvent(
                alert_type="kill_switch",
                severity="critical",
                title="Kill switch triggered",
                message=f"Kill switch reason: {reason}",
                run_id=run_id,
                symbol=symbol,
                details=payload_details,
            )
        )

    def alert_data_interruption(
        self,
        *,
        observed_gap: pd.Timedelta,
        allowed_gap: pd.Timedelta,
        run_id: str | None = None,
        symbol: str | None = None,
        current_timestamp: pd.Timestamp | str | None = None,
        previous_timestamp: pd.Timestamp | str | None = None,
        details: dict[str, Any] | None = None,
    ) -> AlertEvent:
        """Emit data interruption alert when market data continuity is broken."""
        payload_details = dict(details or {})
        payload_details.setdefault("observed_gap_seconds", float(observed_gap.total_seconds()))
        payload_details.setdefault("allowed_gap_seconds", float(allowed_gap.total_seconds()))
        if previous_timestamp is not None:
            payload_details.setdefault(
                "previous_timestamp",
                _normalize_timestamp(previous_timestamp).isoformat(),
            )
        if current_timestamp is not None:
            payload_details.setdefault(
                "current_timestamp",
                _normalize_timestamp(current_timestamp).isoformat(),
            )

        return self.emit(
            AlertEvent(
                alert_type="data_interruption",
                severity="critical",
                title="Data interruption detected",
                message=(
                    f"Observed market-data gap {observed_gap} exceeded allowed gap {allowed_gap}."
                ),
                run_id=run_id,
                symbol=symbol,
                details=payload_details,
            )
        )

    def alert_order_stuck(
        self,
        *,
        order_id: str,
        stuck_seconds: float,
        status: str,
        run_id: str | None = None,
        symbol: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> AlertEvent:
        """Emit order-stuck alert for slow or stale order lifecycle updates."""
        payload_details = dict(details or {})
        payload_details.setdefault("order_id", str(order_id))
        payload_details.setdefault("stuck_seconds", float(stuck_seconds))
        payload_details.setdefault("status", str(status))

        return self.emit(
            AlertEvent(
                alert_type="order_stuck",
                severity="error",
                title="Order appears stuck",
                message=(f"Order {order_id} stayed in {status} for {float(stuck_seconds):.1f}s."),
                run_id=run_id,
                symbol=symbol,
                details=payload_details,
            )
        )

    def alert_drawdown_threshold(
        self,
        *,
        drawdown: float,
        threshold: float,
        run_id: str | None = None,
        symbol: str | None = None,
        equity: float | None = None,
        peak_equity: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> AlertEvent:
        """Emit drawdown-threshold breach alert."""
        payload_details = dict(details or {})
        payload_details.setdefault("drawdown", float(drawdown))
        payload_details.setdefault("threshold", float(threshold))
        if equity is not None:
            payload_details.setdefault("equity", float(equity))
        if peak_equity is not None:
            payload_details.setdefault("peak_equity", float(peak_equity))

        return self.emit(
            AlertEvent(
                alert_type="drawdown_threshold",
                severity="critical",
                title="Drawdown threshold exceeded",
                message=(
                    f"Drawdown {float(drawdown):.6f} exceeded threshold {float(threshold):.6f}."
                ),
                run_id=run_id,
                symbol=symbol,
                details=payload_details,
            )
        )


def build_alert_router(
    *,
    enable_console: bool = True,
    enable_email: bool = False,
    enable_telegram: bool = False,
    enable_webhook: bool = False,
    email_recipients: list[str] | None = None,
    email_sender: str | None = None,
    email_smtp_host: str | None = None,
    telegram_chat_id: str | None = None,
    telegram_bot_token: str | None = None,
    webhook_url: str | None = None,
    webhook_headers: dict[str, str] | None = None,
) -> AlertRouter:
    """Construct an alert router from channel flags."""
    channels: list[AlertChannel] = []
    if enable_console:
        channels.append(ConsoleAlertChannel())
    if enable_email:
        channels.append(
            EmailAlertChannel(
                recipients=email_recipients,
                sender=email_sender,
                smtp_host=email_smtp_host,
            )
        )
    if enable_telegram:
        channels.append(
            TelegramAlertChannel(
                chat_id=telegram_chat_id,
                bot_token=telegram_bot_token,
            )
        )
    if enable_webhook:
        channels.append(
            WebhookAlertChannel(
                url=webhook_url,
                headers=webhook_headers,
            )
        )

    if not channels:
        channels.append(ConsoleAlertChannel())
    return AlertRouter(channels=channels)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return float(value.total_seconds())
    if isinstance(value, dict):
        return {str(key): _to_jsonable(raw) for key, raw in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = [
    "AlertChannel",
    "AlertEvent",
    "AlertRouter",
    "AlertSeverity",
    "AlertType",
    "ConsoleAlertChannel",
    "EmailAlertChannel",
    "TelegramAlertChannel",
    "WebhookAlertChannel",
    "build_alert_router",
]
