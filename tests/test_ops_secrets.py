from __future__ import annotations

import json
from pathlib import Path

import pytest

from stratcheck.ops import JsonEventLogger
from stratcheck.ops.secrets import (
    LocalEncryptedSecretStore,
    SecretManager,
    SecretValue,
    local_encryption_available,
    mask_secret,
    read_secret_env,
    sanitize_logging_payload,
)


def test_read_secret_env_required_and_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRATCHECK_API_KEY", "abc-123")
    assert read_secret_env("STRATCHECK_API_KEY", required=True) == "abc-123"
    assert read_secret_env("STRATCHECK_MISSING", default="fallback") == "fallback"
    with pytest.raises(KeyError, match="Missing required environment secret"):
        read_secret_env("STRATCHECK_MISSING_REQUIRED", required=True)


def test_secret_manager_prefers_env_over_local(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("STRATCHECK_SECRETS_PASSWORD", "local-password")
    store = LocalEncryptedSecretStore(path=tmp_path / "secrets.enc")
    if local_encryption_available():
        store.set_secret("broker_api_key", "local-value")
    manager = SecretManager(env_prefix="STRATCHECK_", local_store=store)

    monkeypatch.setenv("STRATCHECK_BROKER_API_KEY", "env-value")
    secret = manager.get_secret("broker_api_key", required=True)
    assert secret is not None
    assert isinstance(secret, SecretValue)
    assert secret.reveal() == "env-value"
    assert secret.source == "env"


def test_mask_secret_in_ci_is_fully_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CI", "true")
    assert mask_secret("super-sensitive-value") == "[REDACTED]"

    secret = SecretValue(name="token", raw_value="abc123xyz", source="env")
    assert str(secret) == "[REDACTED]"
    assert "[REDACTED]" in repr(secret)


def test_sanitize_logging_payload_redacts_sensitive_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CI", "true")
    payload = sanitize_logging_payload(
        {
            "api_key": "plain-key",
            "nested": {"password": "p@ssword", "ok": 1},
            "token_value": SecretValue(name="token", raw_value="aaa", source="env"),
            "message": "safe",
        }
    )
    assert payload["api_key"] == "[REDACTED]"
    assert payload["nested"]["password"] == "[REDACTED]"
    assert payload["nested"]["ok"] == 1
    assert payload["token_value"] == "[REDACTED]"
    assert payload["message"] == "safe"


def test_json_logger_redacts_sensitive_fields_in_ci(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CI", "true")
    log_path = tmp_path / "runner.jsonl"
    logger = JsonEventLogger(path=log_path)
    logger.emit(
        level="info",
        event="connect",
        api_key="my-key",
        token="my-token",
        message="ok",
    )

    payload = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["api_key"] == "[REDACTED]"
    assert payload["token"] == "[REDACTED]"
    assert payload["message"] == "ok"


def test_local_encrypted_store_roundtrip(tmp_path: Path) -> None:
    if not local_encryption_available():
        pytest.skip("cryptography not installed")

    store = LocalEncryptedSecretStore(
        path=tmp_path / "secrets.enc",
        password="unit-test-password",
    )
    store.set_secret("paper_api_key", "value-001")
    store.set_secret("paper_api_secret", "value-002")

    assert store.get_secret("paper_api_key") == "value-001"
    assert store.get_secret("paper_api_secret") == "value-002"
    assert store.get_secret("not_found") is None
    assert store.list_keys() == ["paper_api_key", "paper_api_secret"]

    removed = store.delete_secret("paper_api_key")
    assert removed is True
    assert store.get_secret("paper_api_key") is None
