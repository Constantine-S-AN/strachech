# Secrets 安全配置指南

本文说明如何在 `stratcheck` 中安全读取与存放敏感信息
（API Key、Token、密码）。

## 1. 优先使用环境变量

推荐把 secrets 放在环境变量中，不写入 `config.toml`：

```bash
export STRATCHECK_BROKER_API_KEY="..."
export STRATCHECK_BROKER_API_SECRET="..."
```

Python 中读取：

```python
from stratcheck.ops import SecretManager

secret_manager = SecretManager(env_prefix="STRATCHECK_")
api_key = secret_manager.get_secret("broker_api_key", required=True)
api_secret = secret_manager.get_secret("broker_api_secret", required=True)

client = {
    "api_key": api_key.reveal(),
    "api_secret": api_secret.reveal(),
}
```

注意：
- `SecretValue` 默认 `str(...)` 是脱敏字符串，避免误打日志。
- 真正使用时调用 `secret.reveal()` 获取原值。

## 2. 可选：本地加密存储

如果你不想每次都导出环境变量，可启用本地加密存储（可选能力）：

1. 安装可选依赖：

```bash
pip install cryptography
```

2. 设置本地存储密码（不要提交到仓库）：

```bash
export STRATCHECK_SECRETS_PASSWORD="your-strong-password"
```

3. 写入与读取：

```python
from pathlib import Path
from stratcheck.ops import LocalEncryptedSecretStore, SecretManager

store = LocalEncryptedSecretStore(path=Path.home() / ".config/stratcheck/secrets.enc")
store.set_secret("broker_api_key", "...")
store.set_secret("broker_api_secret", "...")

secret_manager = SecretManager(env_prefix="STRATCHECK_", local_store=store)
api_key = secret_manager.get_secret("broker_api_key", required=True)
```

建议：
- 存储文件放在仓库外（如 `~/.config/stratcheck/secrets.enc`）。
- 不要把 `STRATCHECK_SECRETS_PASSWORD` 写入代码或提交到仓库。
- 存储文件权限建议仅当前用户可读写（模块会尝试设置 `0600`）。

## 3. CI 中严禁打印 secrets

`stratcheck.ops` 的日志路径默认会对敏感字段脱敏：
- 在 CI 环境（`CI=true` 等）中，敏感字段强制输出 `[REDACTED]`。
- 敏感键名包含：`secret`、`token`、`password`、`api_key` 等关键词。

建议在 CI 中同时遵守：
- 不在 shell 打开 `set -x`（避免命令回显 secrets）。
- 不把 secrets 写到 artifacts、调试输出、或失败日志。
- 仅通过 CI Secret Store 注入环境变量
  （GitHub Actions Secrets / GitLab CI Variables 等）。

## 4. config 放置建议

- `config.toml` 仅存放非敏感参数（symbol、窗口、成本参数等）。
- secrets 统一通过环境变量或本地加密存储加载。
- 如需模板，使用占位符并避免真实值：

```toml
[connectors.real_paper]
enabled = true
base_url = "https://broker.example/api"
environment = "paper"
allow_live_environment = false
# api_key / api_secret 不要写在 config.toml
```
