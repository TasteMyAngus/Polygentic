import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from context_server.config import ConfigurationError, Settings, load_credentials


def _write_credentials(tmp_path: Path, entries: list[dict[str, Any]]) -> Path:
    path = tmp_path / "credentials.json"
    path.write_text(
        json.dumps({"version": 1, "credentials": entries}),
        encoding="utf-8",
    )
    if os.name == "posix":
        path.chmod(0o600)
    return path


def test_load_credentials_keeps_only_provider_and_digest(tmp_path: Path) -> None:
    token = "random-local-test-token"
    path = _write_credentials(
        tmp_path,
        [{"provider": "codex", "token_sha256": hashlib.sha256(token.encode()).hexdigest()}],
    )

    credentials = load_credentials(path)

    assert len(credentials) == 1
    assert credentials[0].provider == "codex"
    assert credentials[0].token_sha256 == hashlib.sha256(token.encode()).hexdigest()


@pytest.mark.parametrize(
    "entries",
    [
        [],
        [{"provider": "Codex", "token_sha256": "0" * 64}],
        [{"provider": "codex", "token_sha256": "invalid"}],
        [
            {"provider": "codex", "token_sha256": "0" * 64},
            {"provider": "codex", "token_sha256": "1" * 64},
        ],
        [
            {"provider": "codex", "token_sha256": "0" * 64},
            {"provider": "claude", "token_sha256": "0" * 64},
        ],
    ],
)
def test_load_credentials_rejects_invalid_entries(
    tmp_path: Path, entries: list[dict[str, Any]]
) -> None:
    path = _write_credentials(tmp_path, entries)

    with pytest.raises(ConfigurationError):
        load_credentials(path)


@pytest.mark.skipif(os.name != "posix", reason="POSIX file permissions are unavailable")
def test_load_credentials_rejects_group_or_world_access(tmp_path: Path) -> None:
    path = _write_credentials(
        tmp_path,
        [{"provider": "codex", "token_sha256": "0" * 64}],
    )
    path.chmod(0o644)

    with pytest.raises(ConfigurationError, match="permissions"):
        load_credentials(path)


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.5", "localhost"])
def test_settings_reject_non_loopback_bind_hosts(
    monkeypatch: pytest.MonkeyPatch, host: str
) -> None:
    monkeypatch.setenv("CONTEXT_SERVER_CREDENTIALS_FILE", "not-read")
    monkeypatch.setenv("CONTEXT_SERVER_HOST", host)

    with pytest.raises(ConfigurationError, match="loopback"):
        Settings.from_environment()


@pytest.mark.parametrize(
    "origin",
    ["*", "https://example.com/path", "https://user@example.com", "https://example.com:bad"],
)
def test_origin_configuration_requires_exact_valid_origins(
    monkeypatch: pytest.MonkeyPatch, origin: str
) -> None:
    monkeypatch.setenv("CONTEXT_SERVER_CREDENTIALS_FILE", "not-read")
    monkeypatch.setenv("CONTEXT_SERVER_ALLOWED_ORIGINS", origin)
    with pytest.raises(ConfigurationError):
        Settings.from_environment()


@pytest.mark.skipif(os.name != "posix", reason="POSIX secret file permissions are required")
def test_local_oauth_loads_owner_only_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    credentials = _write_credentials(tmp_path, [{"provider": "codex", "token_sha256": "0" * 64}])
    secret = tmp_path / "secret"
    secret.write_text("test-only-secret", encoding="ascii")
    secret.chmod(0o600)
    monkeypatch.setenv("CONTEXT_SERVER_CREDENTIALS_FILE", str(credentials))
    monkeypatch.setenv("CONTEXT_SERVER_GITHUB_CLIENT_ID", "clientid")
    monkeypatch.setenv("CONTEXT_SERVER_GITHUB_CLIENT_SECRET_FILE", str(secret))
    monkeypatch.setenv("CONTEXT_SERVER_GITHUB_ADMIN_USER_ID", "42")
    monkeypatch.setenv("CONTEXT_SERVER_GITHUB_PROVIDER", "copilot")
    monkeypatch.setenv("CONTEXT_SERVER_OAUTH_BASE_URL", "https://mcp.example.com")

    settings = Settings.from_environment()
    assert settings.local_github_oauth is not None
    assert settings.local_github_oauth.client_secret == "test-only-secret"
    assert settings.local_github_oauth.provider == "copilot"
    assert settings.local_github_oauth.external_base_url == "https://mcp.example.com"
    assert "test-only-secret" not in repr(settings)

    monkeypatch.setenv("CONTEXT_SERVER_GITHUB_PROVIDER", "Invalid Provider")
    with pytest.raises(ConfigurationError, match="provider ID"):
        Settings.from_environment()
    monkeypatch.setenv("CONTEXT_SERVER_GITHUB_PROVIDER", "copilot")
    secret.chmod(0o644)
    with pytest.raises(ConfigurationError, match="owner-only"):
        Settings.from_environment()


@pytest.mark.parametrize(
    "base_url",
    [
        "http://mcp.example.com",
        "https://mcp.example.com/path",
        "https://user@mcp.example.com",
        "https://localhost",
        "https://mcp.example.com?query=1",
    ],
)
def test_oauth_base_url_requires_external_https_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, base_url: str
) -> None:
    credentials = _write_credentials(tmp_path, [{"provider": "codex", "token_sha256": "0" * 64}])
    secret = tmp_path / "secret"
    secret.write_text("test-only-secret", encoding="ascii")
    secret.chmod(0o600)
    monkeypatch.setenv("CONTEXT_SERVER_CREDENTIALS_FILE", str(credentials))
    monkeypatch.setenv("CONTEXT_SERVER_GITHUB_CLIENT_ID", "clientid")
    monkeypatch.setenv("CONTEXT_SERVER_GITHUB_CLIENT_SECRET_FILE", str(secret))
    monkeypatch.setenv("CONTEXT_SERVER_GITHUB_ADMIN_USER_ID", "42")
    monkeypatch.setenv("CONTEXT_SERVER_OAUTH_BASE_URL", base_url)

    with pytest.raises(ConfigurationError, match="OAuth base URL"):
        Settings.from_environment()


def test_local_oauth_rejects_partial_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    from context_server.config import _load_local_github_oauth

    monkeypatch.setenv("CONTEXT_SERVER_GITHUB_CLIENT_ID", "clientid")
    monkeypatch.delenv("CONTEXT_SERVER_GITHUB_CLIENT_SECRET_FILE", raising=False)
    monkeypatch.delenv("CONTEXT_SERVER_GITHUB_ADMIN_USER_ID", raising=False)
    monkeypatch.delenv("CONTEXT_SERVER_GITHUB_PROVIDER", raising=False)
    with pytest.raises(ConfigurationError, match="All local GitHub OAuth"):
        _load_local_github_oauth()
