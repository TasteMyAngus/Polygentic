import os
import stat
from pathlib import Path

import pytest

from context_server.client_state import (
    ClientStateConflictError,
    ClientStateError,
    SessionStateStore,
)


def test_local_session_state_is_idempotent_and_resumes_after_reopen(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state_dir = tmp_path / "client-state"
    first_store = SessionStateStore(state_dir)

    initial = first_store.initialize(
        "chat-1", "http://127.0.0.1:8000/mcp", "codex", workspace
    )
    replay = SessionStateStore(state_dir).initialize(
        "chat-1", "http://127.0.0.1:8000/mcp", "codex", workspace
    )
    assert replay.registration_key == initial.registration_key
    assert initial.session_id is None
    other_conversation = first_store.initialize(
        "chat-2", "http://127.0.0.1:8000/mcp", "codex", workspace
    )
    assert other_conversation.registration_key != initial.registration_key

    bound = first_store.bind_session(
        "chat-1", "http://127.0.0.1:8000/mcp", "codex", workspace, "server-session-1"
    )
    restored = SessionStateStore(state_dir).load(
        "chat-1", "http://127.0.0.1:8000/mcp", "codex", workspace
    )

    assert bound.session_id == "server-session-1"
    assert restored == bound
    state_file = next(state_dir.glob("*.json"))
    stored_text = state_file.read_text(encoding="utf-8")
    assert "chat-1" not in stored_text
    assert str(workspace) not in stored_text
    assert "bearer" not in stored_text.lower()
    if os.name == "posix":
        assert stat.S_IMODE(state_dir.stat().st_mode) == 0o700
        assert stat.S_IMODE(state_file.stat().st_mode) == 0o600


def test_local_session_state_rejects_scope_changes_and_rebinding(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    other_workspace = tmp_path / "other-workspace"
    workspace.mkdir()
    other_workspace.mkdir()
    store = SessionStateStore(tmp_path / "client-state")
    store.initialize("chat-2", "http://127.0.0.1:8000/mcp", "codex", workspace)

    with pytest.raises(ClientStateConflictError, match="endpoint"):
        store.load("chat-2", "https://127.0.0.1:8000/mcp", "codex", workspace)
    with pytest.raises(ClientStateConflictError, match="provider"):
        store.load("chat-2", "http://127.0.0.1:8000/mcp", "claude", workspace)
    with pytest.raises(ClientStateConflictError, match="workspace"):
        store.load("chat-2", "http://127.0.0.1:8000/mcp", "codex", other_workspace)

    store.bind_session(
        "chat-2", "http://127.0.0.1:8000/mcp", "codex", workspace, "server-session-1"
    )
    with pytest.raises(ClientStateConflictError, match="already bound"):
        store.bind_session(
            "chat-2", "http://127.0.0.1:8000/mcp", "codex", workspace, "server-session-2"
        )


@pytest.mark.skipif(os.name != "posix", reason="POSIX file permissions are unavailable")
def test_local_session_state_rejects_broad_file_permissions(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state_dir = tmp_path / "client-state"
    store = SessionStateStore(state_dir)
    store.initialize("chat-3", "http://127.0.0.1:8000/mcp", "codex", workspace)
    state_file = next(state_dir.glob("*.json"))
    state_file.chmod(0o644)

    with pytest.raises(ClientStateError, match="owner-only"):
        store.load("chat-3", "http://127.0.0.1:8000/mcp", "codex", workspace)


@pytest.mark.skipif(os.name != "posix", reason="POSIX symbolic links are unavailable")
def test_local_session_state_rejects_symbolic_links(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state_dir = tmp_path / "client-state"
    store = SessionStateStore(state_dir)
    store.initialize("chat-4", "http://127.0.0.1:8000/mcp", "codex", workspace)
    state_file = next(state_dir.glob("*.json"))
    state_file.unlink()
    link_target = tmp_path / "link-target"
    link_target.write_text("{}", encoding="utf-8")
    state_file.symlink_to(link_target)

    with pytest.raises(ClientStateError, match="non-symlink"):
        store.load("chat-4", "http://127.0.0.1:8000/mcp", "codex", workspace)
