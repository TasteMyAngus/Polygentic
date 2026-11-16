from __future__ import annotations

import ipaddress
import json
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

_PROVIDER_ID = re.compile(r"^[a-z][a-z0-9-]{1,31}$")
_SHA256 = re.compile(r"^[a-fA-F0-9]{64}$")
_HOST_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_MAX_CREDENTIAL_FILE_BYTES = 64 * 1024


class ConfigurationError(ValueError):
    """Raised when required server configuration is invalid."""


@dataclass(frozen=True)
class ProviderCredential:
    provider: str
    token_sha256: str


@dataclass(frozen=True)
class LocalGitHubOAuth:
    client_id: str
    client_secret: str = field(repr=False)
    admin_user_id: str
    provider: str = "claude"
    external_base_url: str | None = None


@dataclass(frozen=True)
class Settings:
    credentials: tuple[ProviderCredential, ...]
    host: str = "127.0.0.1"
    port: int = 8000
    allowed_hosts: tuple[str, ...] = ("localhost", "127.0.0.1", "::1")
    allowed_origins: tuple[str, ...] = ()
    local_github_oauth: LocalGitHubOAuth | None = None

    @classmethod
    def from_environment(cls) -> Settings:
        credential_file = os.environ.get("CONTEXT_SERVER_CREDENTIALS_FILE")
        if not credential_file:
            raise ConfigurationError(
                "CONTEXT_SERVER_CREDENTIALS_FILE must name the protected credential file"
            )

        host = os.environ.get("CONTEXT_SERVER_HOST", "127.0.0.1")
        try:
            address = ipaddress.ip_address(host)
        except ValueError as exc:
            raise ConfigurationError("CONTEXT_SERVER_HOST must be a loopback IP address") from exc
        if not address.is_loopback:
            raise ConfigurationError("CONTEXT_SERVER_HOST must be a loopback IP address")

        port_value = os.environ.get("CONTEXT_SERVER_PORT", "8000")
        try:
            port = int(port_value)
        except ValueError as exc:
            raise ConfigurationError("CONTEXT_SERVER_PORT must be an integer") from exc
        if not 1 <= port <= 65535:
            raise ConfigurationError("CONTEXT_SERVER_PORT must be between 1 and 65535")

        default_hosts = ("localhost", "127.0.0.1", "::1")
        allowed_hosts = _parse_hosts(os.environ.get("CONTEXT_SERVER_ALLOWED_HOSTS"), default_hosts)
        allowed_origins = _parse_origins(os.environ.get("CONTEXT_SERVER_ALLOWED_ORIGINS"))
        credentials = load_credentials(Path(credential_file))
        oauth = _load_local_github_oauth()
        if oauth is not None and host != "127.0.0.1":
            raise ConfigurationError("Local GitHub OAuth requires the 127.0.0.1 bind address")

        return cls(
            credentials=credentials,
            host=host,
            port=port,
            allowed_hosts=allowed_hosts,
            allowed_origins=allowed_origins,
            local_github_oauth=oauth,
        )


def _load_local_github_oauth() -> LocalGitHubOAuth | None:
    client_id = os.environ.get("CONTEXT_SERVER_GITHUB_CLIENT_ID")
    secret_file = os.environ.get("CONTEXT_SERVER_GITHUB_CLIENT_SECRET_FILE")
    admin_user_id = os.environ.get("CONTEXT_SERVER_GITHUB_ADMIN_USER_ID")
    provider = os.environ.get("CONTEXT_SERVER_GITHUB_PROVIDER")
    external_base_url = os.environ.get("CONTEXT_SERVER_OAUTH_BASE_URL")
    if all(
        value is None
        for value in (client_id, secret_file, admin_user_id, provider, external_base_url)
    ):
        return None
    if not all((client_id, secret_file, admin_user_id)):
        raise ConfigurationError("All local GitHub OAuth settings must be provided")
    assert client_id is not None and secret_file is not None and admin_user_id is not None
    if not client_id.isascii() or not client_id.isalnum():
        raise ConfigurationError("Invalid GitHub OAuth client ID")
    if not admin_user_id.isascii() or not admin_user_id.isdecimal():
        raise ConfigurationError("GitHub administrator ID must be numeric")
    if provider is not None and not _PROVIDER_ID.fullmatch(provider):
        raise ConfigurationError("Invalid GitHub OAuth provider ID")
    if external_base_url is not None:
        parsed = urlsplit(external_base_url)
        try:
            _ = parsed.port
        except ValueError as exc:
            raise ConfigurationError("Invalid OAuth base URL") from exc
        if (
            parsed.scheme != "https"
            or not parsed.netloc.isascii()
            or parsed.hostname is None
            or parsed.hostname.lower() == "localhost"
            or parsed.hostname.lower().endswith(".localhost")
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ConfigurationError("OAuth base URL must be an HTTPS origin")
        external_base_url = f"https://{parsed.netloc.lower()}"
    if os.name != "posix":
        raise ConfigurationError("Local GitHub OAuth requires a POSIX secret file")

    path = Path(secret_file)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as secret:
            file_stat = os.fstat(secret.fileno())
            if not stat.S_ISREG(file_stat.st_mode):
                raise ConfigurationError("GitHub OAuth secret must be a regular file")
            if os.name == "posix" and stat.S_IMODE(file_stat.st_mode) & 0o077:
                raise ConfigurationError("GitHub OAuth secret permissions must be owner-only")
            value = secret.read(4097)
    except OSError as exc:
        raise ConfigurationError("Unable to read GitHub OAuth secret file") from exc
    try:
        client_secret = value.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ConfigurationError("GitHub OAuth secret must be ASCII") from exc
    if not client_secret or len(value) > 4096 or any(c.isspace() for c in client_secret):
        raise ConfigurationError("GitHub OAuth secret must be nonempty and contain no whitespace")
    return LocalGitHubOAuth(
        client_id, client_secret, admin_user_id, provider or "claude", external_base_url
    )


def load_credentials(path: Path) -> tuple[ProviderCredential, ...]:
    if path.is_symlink():
        raise ConfigurationError("Credential file must not be a symbolic link")
    try:
        file_stat = path.stat()
    except OSError as exc:
        raise ConfigurationError(f"Unable to read credential file: {path}") from exc
    if not stat.S_ISREG(file_stat.st_mode):
        raise ConfigurationError("Credential path must be a regular file")
    if file_stat.st_size > _MAX_CREDENTIAL_FILE_BYTES:
        raise ConfigurationError("Credential file exceeds the 64 KiB limit")
    if os.name == "posix" and stat.S_IMODE(file_stat.st_mode) & 0o077:
        raise ConfigurationError("Credential file permissions must be restricted to its owner")

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigurationError("Credential file must contain valid UTF-8 JSON") from exc

    if not isinstance(document, dict) or document.get("version") != 1:
        raise ConfigurationError("Credential file must use schema version 1")
    entries = document.get("credentials")
    if not isinstance(entries, list) or not entries:
        raise ConfigurationError("Credential file must contain at least one credential")

    credentials: list[ProviderCredential] = []
    providers: set[str] = set()
    hashes: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ConfigurationError("Each credential must be a JSON object")
        provider = entry.get("provider")
        token_sha256 = entry.get("token_sha256")
        if not isinstance(provider, str) or not _PROVIDER_ID.fullmatch(provider):
            raise ConfigurationError(
                "Provider IDs must be 2-32 lowercase letters, digits, or hyphens"
            )
        if not isinstance(token_sha256, str) or not _SHA256.fullmatch(token_sha256):
            raise ConfigurationError(f"Credential for provider '{provider}' needs a SHA-256 digest")
        normalized_digest = token_sha256.lower()
        if provider in providers:
            raise ConfigurationError(f"Duplicate provider credential: {provider}")
        if normalized_digest in hashes:
            raise ConfigurationError("Each provider must have a distinct bearer token")
        providers.add(provider)
        hashes.add(normalized_digest)
        credentials.append(ProviderCredential(provider, normalized_digest))

    return tuple(credentials)


def _parse_hosts(value: str | None, defaults: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return defaults
    entries = tuple(part.strip().lower().rstrip(".") for part in value.split(",") if part.strip())
    if not entries:
        raise ConfigurationError("At least one allowed host is required")
    hosts: list[str] = []
    for entry in entries:
        host = entry.strip("[]")
        try:
            ipaddress.ip_address(host)
        except ValueError:
            if ":" in host or not all(_HOST_LABEL.fullmatch(label) for label in host.split(".")):
                raise ConfigurationError("Allowed hosts must be hostnames or IP literals")
        hosts.append(host)
    return tuple(dict.fromkeys(hosts))


def _parse_origins(value: str | None) -> tuple[str, ...]:
    if value is None or not value.strip():
        return ()
    origins: list[str] = []
    for item in value.split(","):
        origin = item.strip()
        parsed = urlsplit(origin)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.hostname is None
            or parsed.path
            or parsed.query
            or parsed.fragment
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ConfigurationError("Allowed origins must be exact http(s) origins without paths")
        try:
            _ = parsed.port
        except ValueError as exc:
            raise ConfigurationError("Allowed origin contains an invalid port") from exc
        origins.append(f"{parsed.scheme.lower()}://{parsed.netloc.lower()}")
    return tuple(dict.fromkeys(origins))
