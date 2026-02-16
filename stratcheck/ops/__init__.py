"""Operational utilities for logging, metrics, alerts, and kill switch controls."""

from stratcheck.ops.alerts import (
    AlertChannel,
    AlertEvent,
    AlertRouter,
    AlertSeverity,
    AlertType,
    ConsoleAlertChannel,
    EmailAlertChannel,
    TelegramAlertChannel,
    WebhookAlertChannel,
    build_alert_router,
)
from stratcheck.ops.kill_switch import KillSwitch, KillSwitchConfig, KillSwitchState
from stratcheck.ops.logging import JsonEventLogger
from stratcheck.ops.metrics import RuntimeMetrics, RuntimeMetricsSnapshot
from stratcheck.ops.secrets import (
    LocalEncryptedSecretStore,
    SecretManager,
    SecretValue,
    is_ci_environment,
    local_encryption_available,
    mask_secret,
    read_secret_env,
    sanitize_logging_payload,
)

__all__ = [
    "AlertChannel",
    "AlertEvent",
    "AlertRouter",
    "AlertSeverity",
    "AlertType",
    "ConsoleAlertChannel",
    "EmailAlertChannel",
    "JsonEventLogger",
    "KillSwitch",
    "KillSwitchConfig",
    "KillSwitchState",
    "LocalEncryptedSecretStore",
    "RuntimeMetrics",
    "RuntimeMetricsSnapshot",
    "SecretManager",
    "SecretValue",
    "TelegramAlertChannel",
    "WebhookAlertChannel",
    "build_alert_router",
    "is_ci_environment",
    "local_encryption_available",
    "mask_secret",
    "read_secret_env",
    "sanitize_logging_payload",
]
