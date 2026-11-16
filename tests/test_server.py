import hashlib
from collections.abc import Sequence

import httpx
import pytest
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.exceptions import ToolError
from fastmcp.server.auth import AccessToken
from fastmcp.server.auth.providers.github import GitHubProvider
from key_value.aio.stores.memory import MemoryStore

from context_server.config import LocalGitHubOAuth, ProviderCredential, Settings
from context_server.server import create_app


class FakeEmbedder:
    async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return [tuple([1.0] + [0.0] * 383) for _ in texts]


def structured_data(result) -> dict:
    assert result.structured_content is not None
    return result.structured_content


@pytest.mark.asyncio
async def test_authenticated_streamable_http_tools() -> None:
    token = "codex-test-token"
    settings = Settings(
        credentials=(ProviderCredential("codex", hashlib.sha256(token.encode()).hexdigest()),)
    )
    app = create_app(settings)

    def httpx_client_factory(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
        follow_redirects: bool = True,
    ) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://127.0.0.1:8000",
            headers=headers,
            timeout=timeout,
            auth=auth,
            follow_redirects=follow_redirects,
        )

    transport = StreamableHttpTransport(
        "http://127.0.0.1:8000/mcp",
        auth=token,
        httpx_client_factory=httpx_client_factory,
    )

    async with app.router.lifespan_context(app), Client(transport) as client:
        identity = await client.call_tool("whoami")
        echoed = await client.call_tool("echo", {"text": "synthetic smoke test"})
        listed = await client.list_tools()

    assert identity.data == {"provider": "codex"}
    assert echoed.data == {"text": "synthetic smoke test", "length": 20}
    tools = {tool.name: tool for tool in listed}
    assert set(tools) == {
        "whoami",
        "echo",
        "register_session",
        "list_sessions",
        "resume_session",
        "put_context",
        "get_context",
        "search_context",
        "supersede_context",
        "put_checkpoint",
        "assemble_context",
    }
    read_only_tools = {
        "whoami",
        "echo",
        "list_sessions",
        "get_context",
        "search_context",
        "assemble_context",
    }
    assert all(
        tools[name].annotations is not None
        and tools[name].annotations.readOnlyHint is True
        and tools[name].annotations.destructiveHint is False
        for name in read_only_tools
    )


@pytest.mark.asyncio
async def test_mcp_memory_workflow_is_provider_bound_and_cross_provider_readable(
    tmp_path,
) -> None:
    codex_token = "codex-memory-test-token"
    claude_token = "claude-memory-test-token"
    data_dir = tmp_path / "memory"
    app_a = create_app(
        Settings(
            credentials=(
                ProviderCredential("codex", hashlib.sha256(codex_token.encode()).hexdigest()),
            ),
            data_dir=data_dir,
        ),
        embedder=FakeEmbedder(),
    )
    app_b = create_app(
        Settings(
            credentials=(
                ProviderCredential("claude", hashlib.sha256(claude_token.encode()).hexdigest()),
            ),
            data_dir=data_dir,
        ),
        embedder=FakeEmbedder(),
    )

    def transport_for(app, token: str) -> StreamableHttpTransport:
        def httpx_client_factory(
            headers: dict[str, str] | None = None,
            timeout: httpx.Timeout | None = None,
            auth: httpx.Auth | None = None,
            follow_redirects: bool = True,
        ) -> httpx.AsyncClient:
            return httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://127.0.0.1:8000",
                headers=headers,
                timeout=timeout,
                auth=auth,
                follow_redirects=follow_redirects,
            )

        return StreamableHttpTransport(
            "http://127.0.0.1:8000/mcp",
            auth=token,
            httpx_client_factory=httpx_client_factory,
        )

    async with app_a.router.lifespan_context(app_a), Client(
        transport_for(app_a, codex_token)
    ) as client_a:
        registration = await client_a.call_tool(
            "register_session",
            {"label": "Decision writer", "idempotency_key": "register-codex"},
        )
        session_a = structured_data(registration)["session_id"]
        written = await client_a.call_tool(
            "put_context",
            {
                "session_id": session_a,
                "title": "Launch decision",
                "content": "The release date is June 12.",
                "record_type": "decision",
                "idempotency_key": "decision-codex",
            },
        )
        old_document_id = structured_data(written)["document_id"]
        replacement = await client_a.call_tool(
            "put_context",
            {
                "session_id": session_a,
                "title": "Launch decision, revised",
                "content": "The release date is July 14.",
                "record_type": "decision",
                "idempotency_key": "decision-codex-revised",
            },
        )
        replacement_id = structured_data(replacement)["document_id"]
        superseded = await client_a.call_tool(
            "supersede_context",
            {
                "session_id": session_a,
                "newer_document_id": replacement_id,
                "older_document_ids": [old_document_id],
                "expected_revisions": {old_document_id: 1},
                "idempotency_key": "supersede-codex-decision",
            },
        )
        assert structured_data(superseded)["superseded_document_ids"] == [old_document_id]

        async with app_b.router.lifespan_context(app_b), Client(
            transport_for(app_b, claude_token)
        ) as client_b:
            registration_b = await client_b.call_tool(
                "register_session",
                {"label": "Decision reader", "idempotency_key": "register-claude"},
            )
            session_b = structured_data(registration_b)["session_id"]
            found = await client_b.call_tool(
                "search_context",
                {
                    "session_id": session_b,
                    "query": "release date June",
                    "mode": "keyword",
                },
            )
            search_results = structured_data(found)["results"]
            assert search_results[0]["document_id"] == replacement_id
            assert search_results[0]["author_provider"] == "codex"

            history = await client_b.call_tool(
                "search_context",
                {
                    "session_id": session_b,
                    "query": "release date June",
                    "mode": "keyword",
                    "include_superseded": True,
                },
            )
            history_results = structured_data(history)["results"]
            historical = next(
                result for result in history_results if result["document_id"] == old_document_id
            )
            assert historical["status"] == "superseded"
            assert historical["superseded_by"] == [replacement_id]

            with pytest.raises(ToolError, match="Write access"):
                await client_b.call_tool(
                    "put_context",
                    {
                        "session_id": session_b,
                        "document_id": replacement_id,
                        "expected_revision": 1,
                        "title": "Unauthorized",
                        "content": "A different provider cannot update this.",
                        "record_type": "decision",
                        "idempotency_key": "unauthorized-update",
                    },
                )


@pytest.mark.asyncio
async def test_http_boundary_rejects_missing_auth_and_bad_origin() -> None:
    token = "codex-test-token"
    app = create_app(
        Settings(
            credentials=(ProviderCredential("codex", hashlib.sha256(token.encode()).hexdigest()),),
            allowed_origins=("http://localhost:3000",),
        )
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://127.0.0.1:8000",
        ) as client,
    ):
        unauthorized = await client.get("/mcp")
        forbidden = await client.get("/mcp", headers={"Origin": "https://attacker.example"})

    assert unauthorized.status_code == 401
    assert forbidden.status_code == 403


@pytest.mark.asyncio
async def test_echo_tool_enforces_size_limit() -> None:
    token = "codex-test-token"
    app = create_app(
        Settings(
            credentials=(ProviderCredential("codex", hashlib.sha256(token.encode()).hexdigest()),)
        )
    )

    def httpx_client_factory(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
        follow_redirects: bool = True,
    ) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://127.0.0.1:8000",
            headers=headers,
            timeout=timeout,
            auth=auth,
            follow_redirects=follow_redirects,
        )

    transport = StreamableHttpTransport(
        "http://127.0.0.1:8000/mcp",
        auth=token,
        httpx_client_factory=httpx_client_factory,
    )
    async with app.router.lifespan_context(app), Client(transport) as client:
        accepted = await client.call_tool("echo", {"text": "x" * 2048})
        with pytest.raises(ToolError, match="2048-character limit"):
            await client.call_tool("echo", {"text": "x" * 2049})

    assert accepted.data == {"text": "x" * 2048, "length": 2048}


@pytest.mark.asyncio
async def test_local_oauth_and_bearer_auth_share_mcp_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def verify_github(self: GitHubProvider, token: str) -> AccessToken | None:
        if token != "oauth-test-token":
            return None
        return AccessToken(token=token, client_id="github", scopes=["user"], claims={"sub": "42"})

    monkeypatch.setattr(GitHubProvider, "verify_token", verify_github)
    bearer = "codex-test-token"
    app = create_app(
        Settings(
            credentials=(ProviderCredential("codex", hashlib.sha256(bearer.encode()).hexdigest()),),
            local_github_oauth=LocalGitHubOAuth("clientid", "not-a-real-secret", "42"),
        ),
        oauth_client_storage=MemoryStore(),
    )

    def httpx_client_factory(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
        follow_redirects: bool = True,
    ) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://localhost:8000",
            headers=headers,
            timeout=timeout,
            auth=auth,
            follow_redirects=follow_redirects,
        )

    async with app.router.lifespan_context(app):
        async with httpx_client_factory() as client:
            metadata = await client.get("/.well-known/oauth-authorization-server")
            assert metadata.status_code == 200
            assert metadata.json()["issuer"] == "http://localhost:8000/"
        for token, expected in ((bearer, "codex"), ("oauth-test-token", "claude")):
            transport = StreamableHttpTransport(
                "http://localhost:8000/mcp", auth=token, httpx_client_factory=httpx_client_factory
            )
            async with Client(transport) as client:
                identity = await client.call_tool("whoami")
                assert identity.data == {"provider": expected}


@pytest.mark.asyncio
async def test_oauth_discovery_uses_external_origin_for_hosted_claude() -> None:
    app = create_app(
        Settings(
            credentials=(ProviderCredential("codex", hashlib.sha256(b"token").hexdigest()),),
            allowed_hosts=("mcp.example.com",),
            local_github_oauth=LocalGitHubOAuth(
                "clientid",
                "not-a-real-secret",
                "42",
                external_base_url="https://mcp.example.com",
            ),
        ),
        oauth_client_storage=MemoryStore(),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="https://mcp.example.com",
        ) as client,
    ):
        unauthorized = await client.get("/mcp")
        resource = await client.get("/.well-known/oauth-protected-resource/mcp")
        authorization = await client.get("/.well-known/oauth-authorization-server")
        registration = await client.post(
            "/register",
            json={
                "redirect_uris": ["https://claude.ai/api/mcp/auth_callback"],
                "client_name": "Claude hosted test",
                "token_endpoint_auth_method": "none",
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
            },
        )

    assert unauthorized.status_code == 401
    assert unauthorized.headers["www-authenticate"] == (
        'Bearer resource_metadata="https://mcp.example.com/.well-known/'
        'oauth-protected-resource/mcp"'
    )
    assert resource.json()["resource"] == "https://mcp.example.com/mcp"
    assert resource.json()["authorization_servers"] == ["https://mcp.example.com/"]
    assert authorization.json()["issuer"] == "https://mcp.example.com/"
    assert authorization.json()["registration_endpoint"] == "https://mcp.example.com/register"
    assert authorization.json()["revocation_endpoint"] == "https://mcp.example.com/revoke"
    assert authorization.json()["code_challenge_methods_supported"] == ["S256"]
    assert registration.status_code == 201
    assert registration.json()["redirect_uris"] == ["https://claude.ai/api/mcp/auth_callback"]
