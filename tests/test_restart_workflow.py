from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from context_server.client_state import SessionStateStore

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SERVER_SCRIPT = Path(__file__).parent / "fixtures" / "restart_server.py"


def _unused_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _start_server(
    data_dir: Path, port: int, token: str, claude_token: str
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [
            sys.executable,
            str(_SERVER_SCRIPT),
            "--data-dir",
            str(data_dir),
            "--port",
            str(port),
            "--token-sha256",
            hashlib.sha256(token.encode()).hexdigest(),
            "--claude-token-sha256",
            hashlib.sha256(claude_token.encode()).hexdigest(),
        ],
        cwd=_REPOSITORY_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _start_configured_server(
    credentials_file: Path, data_dir: Path, model_cache_dir: Path, port: int
) -> subprocess.Popen[str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("CONTEXT_SERVER_")
    }
    environment.update(
        {
            "CONTEXT_SERVER_CREDENTIALS_FILE": str(credentials_file),
            "CONTEXT_SERVER_DATA_DIR": str(data_dir),
            "CONTEXT_SERVER_MODEL_CACHE_DIR": str(model_cache_dir),
            "CONTEXT_SERVER_PORT": str(port),
        }
    )
    return subprocess.Popen(
        [sys.executable, "-m", "context_server.server"],
        cwd=_REPOSITORY_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _write_provider_credentials(path: Path, tokens: tuple[str, ...]) -> None:
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "credentials": [
                    {
                        "provider": "codex",
                        "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
                    }
                    for token in tokens
                ],
            }
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)


def _wait_for_server(process: subprocess.Popen[str], port: int) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output, _ = process.communicate()
            raise RuntimeError(f"Test MCP server exited during startup:\n{output}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError("Test MCP server did not start within 10 seconds")


def _stop_server(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        process.terminate()
    try:
        process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()


def _structured(result: Any) -> dict[str, Any]:
    assert result.structured_content is not None
    return result.structured_content


async def _authenticated_provider(endpoint: str, token: str) -> str:
    async with Client(StreamableHttpTransport(endpoint, auth=token)) as client:
        provider = _structured(await client.call_tool("whoami", {})).get("provider")
        if not isinstance(provider, str):
            raise TypeError("MCP whoami did not return a provider")
        return provider


async def _is_unauthorized(endpoint: str, token: str) -> bool:
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "credential-lifecycle-test", "version": "1"},
        },
    }
    async with httpx.AsyncClient(timeout=5) as client:
        response = await client.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json, text/event-stream",
            },
            json=request,
        )
    return response.status_code == 401


@pytest.mark.asyncio
async def test_local_session_state_recovers_after_real_server_restart(
    tmp_path: Path,
) -> None:
    token = "codex-server-restart-test-token"
    claude_token = "claude-server-restart-test-token"
    port = _unused_port()
    endpoint = f"http://127.0.0.1:{port}/mcp"
    data_dir = tmp_path / "server-data"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state_store = SessionStateStore(tmp_path / "client-state")
    local_state = state_store.initialize("chat-a", endpoint, "codex", workspace)
    first_process = _start_server(data_dir, port, token, claude_token)
    processes = [first_process]

    try:
        _wait_for_server(first_process, port)
        async with Client(StreamableHttpTransport(endpoint, auth=token)) as client:
            first_registration = await client.call_tool(
                "register_session",
                {"label": "Restart test", "idempotency_key": local_state.registration_key},
            )
            lost_response_session_id = _structured(first_registration)["session_id"]
        _stop_server(first_process)

        assert state_store.load("chat-a", endpoint, "codex", workspace).session_id is None
        second_process = _start_server(data_dir, port, token, claude_token)
        processes.append(second_process)
        _wait_for_server(second_process, port)
        async with Client(StreamableHttpTransport(endpoint, auth=token)) as client:
            retried_registration = await client.call_tool(
                "register_session",
                {"label": "Restart test", "idempotency_key": local_state.registration_key},
            )
            session_id = _structured(retried_registration)["session_id"]
            assert session_id == lost_response_session_id
            state_store.bind_session("chat-a", endpoint, "codex", workspace, session_id)

            resumed = await client.call_tool("resume_session", {"session_id": session_id})
            assert _structured(resumed)["session_id"] == session_id
            written = await client.call_tool(
                "put_context",
                {
                    "session_id": session_id,
                    "title": "Release decision",
                    "content": "The release date is June 12.",
                    "record_type": "decision",
                    "idempotency_key": "restart-test-decision",
                },
            )
            document_id = _structured(written)["document_id"]
            checkpoint = await client.call_tool(
                "put_checkpoint",
                {
                    "session_id": session_id,
                    "task_id": "restart-test",
                    "content": "Objective: verify persistence. Next: resume after restart.",
                    "expected_head_revision": 0,
                    "idempotency_key": "restart-test-checkpoint",
                },
            )
            assert _structured(checkpoint)["head_revision"] == 1

        async with Client(StreamableHttpTransport(endpoint, auth=claude_token)) as client:
            claude_registration = await client.call_tool(
                "register_session",
                {"label": "Cross-provider reader", "idempotency_key": "restart-test-claude"},
            )
            claude_session_id = _structured(claude_registration)["session_id"]
            search = await client.call_tool(
                "search_context",
                {
                    "session_id": claude_session_id,
                    "query": "release date June",
                    "mode": "keyword",
                },
            )
            assert _structured(search)["results"][0]["document_id"] == document_id
        _stop_server(second_process)

        restored_state = state_store.load("chat-a", endpoint, "codex", workspace)
        assert restored_state.session_id == session_id
        third_process = _start_server(data_dir, port, token, claude_token)
        processes.append(third_process)
        _wait_for_server(third_process, port)
        async with Client(StreamableHttpTransport(endpoint, auth=token)) as client:
            resumed_after_restart = await client.call_tool(
                "resume_session", {"session_id": restored_state.session_id}
            )
            assert _structured(resumed_after_restart)["session_id"] == session_id
            restored_record = await client.call_tool(
                "get_context",
                {"session_id": session_id, "document_id": document_id},
            )
            assert _structured(restored_record)["content"] == "The release date is June 12."
            assembly = await client.call_tool(
                "assemble_context",
                {
                    "session_id": session_id,
                    "task_id": "restart-test",
                    "query": "release date June",
                },
            )
            restored_assembly = _structured(assembly)
            assert restored_assembly["checkpoint"]["head_revision"] == 1
            assert restored_assembly["evidence"][0]["document_id"] == document_id
    finally:
        for process in processes:
            _stop_server(process)


@pytest.mark.asyncio
async def test_credential_rotation_and_revocation_after_restart(
    tmp_path: Path,
) -> None:
    old_token = "old-credential-rotation-test-token"
    new_token = "new-credential-rotation-test-token"
    credentials_file = tmp_path / "credentials.json"
    data_dir = tmp_path / "server-data"
    model_cache_dir = tmp_path / "model-cache"
    port = _unused_port()
    endpoint = f"http://127.0.0.1:{port}/mcp"
    processes: list[subprocess.Popen[str]] = []

    try:
        _write_provider_credentials(credentials_file, (old_token,))
        first = _start_configured_server(
            credentials_file, data_dir, model_cache_dir, port
        )
        processes.append(first)
        _wait_for_server(first, port)
        assert await _authenticated_provider(endpoint, old_token) == "codex"
        assert await _is_unauthorized(endpoint, new_token)
        _stop_server(first)

        _write_provider_credentials(credentials_file, (new_token,))
        rotated = _start_configured_server(
            credentials_file, data_dir, model_cache_dir, port
        )
        processes.append(rotated)
        _wait_for_server(rotated, port)
        assert await _authenticated_provider(endpoint, new_token) == "codex"
        assert await _is_unauthorized(endpoint, old_token)
    finally:
        for process in processes:
            _stop_server(process)
