import hashlib

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
    assert {tool.name for tool in listed} == {"whoami", "echo"}
    assert all(
        tool.annotations is not None
        and tool.annotations.readOnlyHint is True
        and tool.annotations.destructiveHint is False
        for tool in listed
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
