"""Secret management helpers: env loading, optional encrypted local store, and safe logging."""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

_SENSITIVE_KEY_TOKENS = (
    "secret",
    "token",
    "password",
    "passphrase",
    "api_key",
    "apikey",
    "access_key",
    "private_key",
)
_LOCAL_STORE_SCHEMA_VERSION = 1


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def is_ci_environment(env: Mapping[str, str] | None = None) -> bool:
    """Return whether current process is running in CI environment."""
    env_map = env if env is not None else os.environ
    ci_indicators = [
        env_map.get("CI", ""),
        env_map.get("GITHUB_ACTIONS", ""),
        env_map.get("GITLAB_CI", ""),
        env_map.get("BUILDKITE", ""),
        env_map.get("CIRCLECI", ""),
        env_map.get("JENKINS_URL", ""),
    ]
    for value in ci_indicators:
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized not in {"", "0", "false", "no", "off"}:
            return True
    return False


def mask_secret(
    value: str | None,
    *,
    visible_prefix: int = 2,
    visible_suffix: int = 2,
    force_full_redaction: bool = False,
) -> str:
    """Mask secret values for logs and UI."""
    if value is None:
        return "[MISSING]"
    text_value = str(value)
    if text_value == "":
        return "[EMPTY]"
    if force_full_redaction or is_ci_environment():
        return "[REDACTED]"

    prefix_length = max(int(visible_prefix), 0)
    suffix_length = max(int(visible_suffix), 0)
    if len(text_value) <= prefix_length + suffix_length:
        return "*" * len(text_value)
    return (
        text_value[:prefix_length]
        + ("*" * (len(text_value) - prefix_length - suffix_length))
        + text_value[-suffix_length:]
    )


@dataclass(slots=True, frozen=True)
class SecretValue:
    """Secret holder with safe string representation."""

    name: str
    raw_value: str
    source: str

    def reveal(self) -> str:
        """Return raw secret value for runtime use."""
        return self.raw_value

    def masked(self, *, force_full_redaction: bool = False) -> str:
        """Return masked representation safe for logs."""
        return mask_secret(self.raw_value, force_full_redaction=force_full_redaction)

    def __str__(self) -> str:
        return self.masked()

    def __repr__(self) -> str:
        return (
            "SecretValue("
            f"name={self.name!r}, "
            f"source={self.source!r}, "
            f"value={self.masked(force_full_redaction=True)!r}"
            ")"
        )


def read_secret_env(
    env_var: str,
    *,
    default: str | None = None,
    required: bool = False,
    allow_empty: bool = False,
) -> str | None:
    """Read one secret from environment with optional required check."""
    env_name = str(env_var).strip()
    if not env_name:
        msg = "env_var cannot be blank."
        raise ValueError(msg)

    value = os.environ.get(env_name)
    if value is None:
        if required and default is None:
            msg = f"Missing required environment secret: {env_name}"
            raise KeyError(msg)
        return default
    if value == "" and not allow_empty:
        if required and default is None:
            msg = f"Environment secret is empty: {env_name}"
            raise ValueError(msg)
        return default
    return value


def local_encryption_available() -> bool:
    """Return whether optional cryptography package is available."""
    try:
        from cryptography.fernet import Fernet  # noqa: F401
        from cryptography.hazmat.primitives import hashes  # noqa: F401
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass(slots=True)
class LocalEncryptedSecretStore:
    """Optional encrypted local secret store backed by JSON file."""

    path: Path
    password: str | None = None
    password_env_var: str = "STRATCHECK_SECRETS_PASSWORD"
    kdf_iterations: int = 120_000

    def __init__(
        self,
        path: str | Path,
        *,
        password: str | None = None,
        password_env_var: str = "STRATCHECK_SECRETS_PASSWORD",
        kdf_iterations: int = 120_000,
    ) -> None:
        if int(kdf_iterations) < 10_000:
            msg = "kdf_iterations must be >= 10000."
            raise ValueError(msg)
        self.path = Path(path)
        self.password = password
        self.password_env_var = str(password_env_var)
        self.kdf_iterations = int(kdf_iterations)

    def list_keys(self) -> list[str]:
        payload = self._load_store()
        secret_map = payload.get("secrets", {})
        if not isinstance(secret_map, dict):
            return []
        return sorted(str(key) for key in secret_map.keys())

    def get_secret(self, key: str) -> str | None:
        normalized_key = _normalize_secret_key(key)
        payload = self._load_store()
        secret_map = payload.get("secrets", {})
        if not isinstance(secret_map, dict):
            return None
        token = secret_map.get(normalized_key)
        if token is None:
            return None

        fernet = self._build_fernet(payload=payload)
        decrypted_bytes = fernet.decrypt(str(token).encode("utf-8"))
        return decrypted_bytes.decode("utf-8")

    def set_secret(self, key: str, value: str) -> None:
        normalized_key = _normalize_secret_key(key)
        normalized_value = str(value)
        payload = self._load_store()
        if not payload:
            payload = self._new_store_payload()

        secret_map = payload.get("secrets")
        if not isinstance(secret_map, dict):
            secret_map = {}
            payload["secrets"] = secret_map

        fernet = self._build_fernet(payload=payload)
        encrypted_token = fernet.encrypt(normalized_value.encode("utf-8")).decode("utf-8")
        secret_map[normalized_key] = encrypted_token

        payload["updated_at"] = _utc_now().isoformat()
        self._write_store(payload)

    def delete_secret(self, key: str) -> bool:
        normalized_key = _normalize_secret_key(key)
        payload = self._load_store()
        secret_map = payload.get("secrets", {})
        if not isinstance(secret_map, dict):
            return False
        if normalized_key not in secret_map:
            return False

        del secret_map[normalized_key]
        payload["updated_at"] = _utc_now().isoformat()
        self._write_store(payload)
        return True

    def _new_store_payload(self) -> dict[str, Any]:
        salt = os.urandom(16)
        return {
            "schema_version": _LOCAL_STORE_SCHEMA_VERSION,
            "kdf": {
                "name": "pbkdf2_sha256",
                "iterations": int(self.kdf_iterations),
                "salt_b64": base64.b64encode(salt).decode("ascii"),
            },
            "secrets": {},
            "updated_at": _utc_now().isoformat(),
        }

    def _load_store(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            msg = "Invalid secret store payload type."
            raise ValueError(msg)
        return payload

    def _write_store(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def _resolve_password(self) -> str:
        if self.password is not None and self.password != "":
            return str(self.password)
        env_password = os.environ.get(self.password_env_var)
        if env_password:
            return str(env_password)
        msg = (
            "Encrypted local secret store requires password. "
            f"Set password arg or env {self.password_env_var!r}."
        )
        raise ValueError(msg)

    def _build_fernet(self, payload: Mapping[str, Any]):
        if not local_encryption_available():
            msg = (
                "Local encrypted secret store requires optional dependency "
                "'cryptography'. Install with `pip install cryptography`."
            )
            raise RuntimeError(msg)

        from cryptography.fernet import Fernet
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

        kdf_payload = payload.get("kdf")
        if not isinstance(kdf_payload, dict):
            msg = "Secret store missing kdf metadata."
            raise ValueError(msg)
        if str(kdf_payload.get("name")) != "pbkdf2_sha256":
            msg = "Unsupported kdf type in secret store."
            raise ValueError(msg)

        iterations = int(kdf_payload.get("iterations", self.kdf_iterations))
        salt_b64 = kdf_payload.get("salt_b64")
        if salt_b64 is None:
            msg = "Secret store missing kdf salt."
            raise ValueError(msg)
        salt = base64.b64decode(str(salt_b64))

        password_bytes = self._resolve_password().encode("utf-8")
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=iterations,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
        return Fernet(key)


@dataclass(slots=True)
class SecretManager:
    """Unified secret resolver: env first, optional encrypted local store fallback."""

    env_prefix: str = ""
    local_store: LocalEncryptedSecretStore | None = None

    def __init__(
        self,
        *,
        env_prefix: str = "",
        local_store: LocalEncryptedSecretStore | None = None,
    ) -> None:
        self.env_prefix = str(env_prefix)
        self.local_store = local_store

    def env_var_name(self, name: str) -> str:
        normalized = _normalize_secret_key(name).upper()
        return f"{self.env_prefix}{normalized}"

    def get_secret(
        self,
        name: str,
        *,
        env_var: str | None = None,
        required: bool = False,
        default: str | None = None,
        allow_local_store: bool = True,
    ) -> SecretValue | None:
        """Resolve secret from env then optional local store."""
        normalized_name = _normalize_secret_key(name)
        resolved_env_var = env_var or self.env_var_name(normalized_name)
        env_value = read_secret_env(
            resolved_env_var,
            default=None,
            required=False,
            allow_empty=False,
        )
        if env_value is not None:
            return SecretValue(name=normalized_name, raw_value=env_value, source="env")

        if allow_local_store and self.local_store is not None:
            local_value = self.local_store.get_secret(normalized_name)
            if local_value is not None:
                return SecretValue(
                    name=normalized_name,
                    raw_value=local_value,
                    source="local_store",
                )

        if default is not None:
            return SecretValue(name=normalized_name, raw_value=default, source="default")

        if required:
            msg = (
                f"Missing required secret: {normalized_name}. "
                f"Checked env var {resolved_env_var!r}"
                + (" and local encrypted store." if self.local_store is not None else ".")
            )
            raise KeyError(msg)
        return None

    def set_local_secret(self, name: str, value: str) -> None:
        """Write one secret into local encrypted store."""
        if self.local_store is None:
            msg = "local_store is not configured."
            raise ValueError(msg)
        self.local_store.set_secret(name, value)


def sanitize_logging_payload(
    fields: Mapping[str, Any],
    *,
    force_full_redaction: bool | None = None,
) -> dict[str, Any]:
    """Redact secret-like fields for logging. CI defaults to full redaction."""
    if force_full_redaction is None:
        effective_full_redaction = is_ci_environment()
    else:
        effective_full_redaction = bool(force_full_redaction)

    return {
        str(key): _sanitize_value(
            key_name=str(key),
            value=value,
            force_full_redaction=effective_full_redaction,
        )
        for key, value in fields.items()
    }


def _sanitize_value(key_name: str, value: Any, force_full_redaction: bool) -> Any:
    if isinstance(value, SecretValue):
        return value.masked(force_full_redaction=force_full_redaction)

    if _looks_sensitive_key(key_name):
        return mask_secret(str(value), force_full_redaction=force_full_redaction)

    if isinstance(value, Mapping):
        return {
            str(nested_key): _sanitize_value(
                key_name=str(nested_key),
                value=nested_value,
                force_full_redaction=force_full_redaction,
            )
            for nested_key, nested_value in value.items()
        }

    if isinstance(value, list):
        return [
            _sanitize_value(
                key_name=key_name,
                value=item,
                force_full_redaction=force_full_redaction,
            )
            for item in value
        ]

    if isinstance(value, tuple):
        return tuple(
            _sanitize_value(
                key_name=key_name,
                value=item,
                force_full_redaction=force_full_redaction,
            )
            for item in value
        )

    return value


def _looks_sensitive_key(key_name: str) -> bool:
    normalized = str(key_name).strip().lower()
    if not normalized:
        return False
    return any(token in normalized for token in _SENSITIVE_KEY_TOKENS)


def _normalize_secret_key(value: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        msg = "secret key cannot be blank."
        raise ValueError(msg)
    return normalized


__all__ = [
    "LocalEncryptedSecretStore",
    "SecretManager",
    "SecretValue",
    "is_ci_environment",
    "local_encryption_available",
    "mask_secret",
    "read_secret_env",
    "sanitize_logging_payload",
]
