from __future__ import annotations

import argparse
import hashlib
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from platformdirs import user_state_path
from pydantic import BaseModel, ConfigDict, ValidationError

_PROVIDER_ID = re.compile(r"^[a-z][a-z0-9-]{1,31}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_MAX_STATE_BYTES = 16 * 1024


class ClientStateError(RuntimeError):
    """Raised when local session state is missing, invalid, or out of scope."""


class ClientStateConflictError(ClientStateError):
    """Raised when an existing local session binding cannot be safely changed."""


class ClientSessionState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1]
    conversation_hash: str
    endpoint: str
    provider: str
    workspace_hash: str
    registration_key: str
    session_id: str | None = None


class SessionStateStore:
    def __init__(self, directory: Path | None = None) -> None:
        self._directory = directory.expanduser() if directory is not None else _default_state_directory()

    def initialize(
        self, conversation_key: str, endpoint: str, provider: str, workspace: Path
    ) -> ClientSessionState:
        scope = _build_scope(conversation_key, endpoint, provider, workspace)
        path = self._path_for(scope["conversation_hash"])
        existing = self._read_if_exists(path)
        if existing is not None:
            _ensure_scope_matches(existing, scope)
            return existing

        registration_key = hashlib.sha256(
            "\0".join(
                (
                    "polygentic-register-session-v1",
                    scope["conversation_hash"],
                    scope["endpoint"],
                    scope["provider"],
                    scope["workspace_hash"],
                )
            ).encode("utf-8")
        ).hexdigest()
        state = ClientSessionState(
            version=1,
            conversation_hash=scope["conversation_hash"],
            endpoint=scope["endpoint"],
            provider=scope["provider"],
            workspace_hash=scope["workspace_hash"],
            registration_key=registration_key,
        )
        self._write(path, state)
        return state

    def load(
        self, conversation_key: str, endpoint: str, provider: str, workspace: Path
    ) -> ClientSessionState:
        scope = _build_scope(conversation_key, endpoint, provider, workspace)
        state = self._read_if_exists(self._path_for(scope["conversation_hash"]))
        if state is None:
            raise ClientStateError("No saved session state exists for this conversation")
        _ensure_scope_matches(state, scope)
        return state

    def bind_session(
        self,
        conversation_key: str,
        endpoint: str,
        provider: str,
        workspace: Path,
        session_id: str,
    ) -> ClientSessionState:
        if not _IDENTIFIER.fullmatch(session_id):
            raise ValueError("session ID must be 1-128 ASCII letters, digits, or . _ : - characters")
        scope = _build_scope(conversation_key, endpoint, provider, workspace)
        path = self._path_for(scope["conversation_hash"])
        state = self._read_if_exists(path)
        if state is None:
            raise ClientStateError("Initialize local session state before binding a server session")
        _ensure_scope_matches(state, scope)
        if state.session_id is not None and state.session_id != session_id:
            raise ClientStateConflictError("This conversation is already bound to another session")
        if state.session_id == session_id:
            return state
        updated = state.model_copy(update={"session_id": session_id})
        self._write(path, updated)
        return updated

    def _path_for(self, conversation_hash: str) -> Path:
        return self._directory / f"{conversation_hash}.json"

    def _ensure_directory(self, *, create: bool) -> None:
        if self._directory.is_symlink():
            raise ClientStateError("Client state directory must not be a symbolic link")
        if create:
            self._directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        elif not self._directory.exists():
            return
        if self._directory.is_symlink() or not self._directory.is_dir():
            raise ClientStateError("Client state path must be a directory")
        if os.name == "posix":
            directory_stat = self._directory.stat()
            if directory_stat.st_uid != os.getuid():
                raise ClientStateError("Client state directory must be owned by the current user")
            os.chmod(self._directory, 0o700)

    def _read_if_exists(self, path: Path) -> ClientSessionState | None:
        self._ensure_directory(create=False)
        try:
            entry_stat = path.lstat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise ClientStateError("Unable to open local session state") from exc
        if not stat.S_ISREG(entry_stat.st_mode):
            raise ClientStateError("Local session state must be a regular, non-symlink file")
        try:
            fd = os.open(
                path,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
            )
        except OSError as exc:
            raise ClientStateError("Unable to open local session state") from exc
        with os.fdopen(fd, "rb") as state_file:
            file_stat = os.fstat(state_file.fileno())
            if not stat.S_ISREG(file_stat.st_mode):
                raise ClientStateError("Local session state must be a regular file")
            if os.name == "posix":
                if file_stat.st_uid != os.getuid():
                    raise ClientStateError("Local session state must be owned by the current user")
                if stat.S_IMODE(file_stat.st_mode) & 0o077:
                    raise ClientStateError("Local session state permissions must be owner-only")
            raw_state = state_file.read(_MAX_STATE_BYTES + 1)
        if len(raw_state) > _MAX_STATE_BYTES:
            raise ClientStateError("Local session state exceeds the size limit")
        try:
            return ClientSessionState.model_validate_json(raw_state)
        except ValidationError as exc:
            raise ClientStateError("Local session state has an invalid format") from exc

    def _write(self, path: Path, state: ClientSessionState) -> None:
        self._ensure_directory(create=True)
        raw_state = (state.model_dump_json(indent=2) + "\n").encode("utf-8")
        if len(raw_state) > _MAX_STATE_BYTES:
            raise ClientStateError("Local session state exceeds the size limit")
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.stem}.",
            suffix=".tmp",
            dir=self._directory,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as temporary_file:
                temporary_file.write(raw_state)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, path)
            if os.name == "posix":
                directory_fd = os.open(
                    self._directory,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                )
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            temporary_path.unlink(missing_ok=True)


def _build_scope(
    conversation_key: str, endpoint: str, provider: str, workspace: Path
) -> dict[str, str]:
    if not conversation_key or not conversation_key.strip() or len(conversation_key) > 256:
        raise ValueError("conversation key must contain 1-256 non-whitespace characters")
    if not _PROVIDER_ID.fullmatch(provider):
        raise ValueError("provider must be a 2-32 character lowercase provider ID")
    normalized_endpoint = _normalize_endpoint(endpoint)
    workspace_path = workspace.expanduser().resolve()
    if not workspace_path.is_dir():
        raise ValueError("workspace must be an existing directory")
    return {
        "conversation_hash": hashlib.sha256(conversation_key.encode("utf-8")).hexdigest(),
        "endpoint": normalized_endpoint,
        "provider": provider,
        "workspace_hash": hashlib.sha256(str(workspace_path).encode("utf-8")).hexdigest(),
    }


def _normalize_endpoint(endpoint: str) -> str:
    if endpoint != endpoint.strip():
        raise ValueError("endpoint must not contain surrounding whitespace")
    parsed = urlsplit(endpoint)
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("endpoint has an invalid port") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("endpoint must be an HTTP(S) URL without credentials, query, or fragment")
    return urlunsplit(
        (parsed.scheme.lower(), parsed.netloc.lower(), parsed.path or "/", "", "")
    )


def _ensure_scope_matches(state: ClientSessionState, scope: dict[str, str]) -> None:
    for field in ("conversation_hash", "endpoint", "provider", "workspace_hash"):
        if getattr(state, field) != scope[field]:
            raise ClientStateConflictError(f"Saved session state does not match the requested {field}")


def _default_state_directory() -> Path:
    configured_directory = os.environ.get("CONTEXT_SERVER_CLIENT_STATE_DIR")
    if configured_directory:
        return Path(configured_directory).expanduser()
    return user_state_path("Polygentic", appauthor=False) / "client-sessions"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Manage local logical-session bindings without storing bearer tokens."
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=_default_state_directory(),
        help="owner-only state directory (or CONTEXT_SERVER_CLIENT_STATE_DIR)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("init", "show", "bind"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--conversation", required=True)
        command_parser.add_argument("--endpoint", required=True)
        command_parser.add_argument("--provider", required=True)
        command_parser.add_argument("--workspace", type=Path, required=True)
        if command == "bind":
            command_parser.add_argument("--session-id", required=True)
    args = parser.parse_args(argv)
    store = SessionStateStore(args.state_dir)
    try:
        if args.command == "init":
            state = store.initialize(
                args.conversation, args.endpoint, args.provider, args.workspace
            )
        elif args.command == "show":
            state = store.load(
                args.conversation, args.endpoint, args.provider, args.workspace
            )
        else:
            state = store.bind_session(
                args.conversation,
                args.endpoint,
                args.provider,
                args.workspace,
                args.session_id,
            )
    except (ClientStateError, ValueError, OSError) as exc:
        parser.error(str(exc))
    print(state.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
