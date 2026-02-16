from __future__ import annotations

import io

import pandas as pd
from stratcheck.ops.alerts import (
    AlertRouter,
    ConsoleAlertChannel,
    EmailAlertChannel,
    TelegramAlertChannel,
    WebhookAlertChannel,
    build_alert_router,
)


def test_alert_router_emits_supported_alert_types() -> None:
    console_output = io.StringIO()
    console_channel = ConsoleAlertChannel(stream=console_output)
    email_channel = EmailAlertChannel(recipients=["ops@example.com"])
    telegram_channel = TelegramAlertChannel(chat_id="123")
    webhook_channel = WebhookAlertChannel(url="https://hooks.example.com/alerts")

    router = AlertRouter(
        channels=[
            console_channel,
            email_channel,
            telegram_channel,
            webhook_channel,
        ]
    )

    kill_event = router.alert_kill_switch(
        reason="consecutive_errors_exceeded",
        run_id="run-1",
        symbol="AAPL",
    )
    data_event = router.alert_data_interruption(
        observed_gap=pd.Timedelta(minutes=20),
        allowed_gap=pd.Timedelta(minutes=5),
        run_id="run-1",
        symbol="AAPL",
        previous_timestamp=pd.Timestamp("2024-01-01T09:30:00Z"),
        current_timestamp=pd.Timestamp("2024-01-01T09:50:00Z"),
    )
    stuck_event = router.alert_order_stuck(
        order_id="oid-1",
        stuck_seconds=91.0,
        status="new",
        run_id="run-1",
        symbol="AAPL",
    )
    drawdown_event = router.alert_drawdown_threshold(
        drawdown=0.33,
        threshold=0.30,
        run_id="run-1",
        symbol="AAPL",
        equity=6700.0,
        peak_equity=10000.0,
    )

    assert kill_event.alert_type == "kill_switch"
    assert data_event.alert_type == "data_interruption"
    assert stuck_event.alert_type == "order_stuck"
    assert drawdown_event.alert_type == "drawdown_threshold"

    assert len(email_channel.sent_alerts) == 4
    assert len(telegram_channel.sent_alerts) == 4
    assert len(webhook_channel.sent_alerts) == 4

    console_text = console_output.getvalue()
    assert "kill_switch" in console_text
    assert "data_interruption" in console_text
    assert "order_stuck" in console_text
    assert "drawdown_threshold" in console_text


def test_build_alert_router_flags_build_channels() -> None:
    router = build_alert_router(
        enable_console=False,
        enable_email=True,
        enable_telegram=True,
        enable_webhook=True,
        email_recipients=["risk@example.com"],
        telegram_chat_id="200",
        webhook_url="https://hooks.example.com/alpha",
    )

    assert len(router.channels) == 3
    channel_names = {channel.name for channel in router.channels}
    assert channel_names == {"email", "telegram", "webhook"}

    event = router.alert_kill_switch(reason="max_drawdown_exceeded")
    assert event.severity == "critical"
    assert event.details["reason"] == "max_drawdown_exceeded"


def test_build_alert_router_falls_back_to_console_when_all_disabled() -> None:
    router = build_alert_router(
        enable_console=False,
        enable_email=False,
        enable_telegram=False,
        enable_webhook=False,
    )

    assert len(router.channels) == 1
    assert router.channels[0].name == "console"
