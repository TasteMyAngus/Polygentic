import hashlib
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from fastmcp import FastMCP
from fastmcp import settings as fastmcp_settings
from fastmcp.server.auth import AccessToken
from fastmcp.server.auth.jwt_issuer import JWTIssuer
from fastmcp.server.auth.oauth_proxy.models import (
    JTIMapping,
    OAuthTransaction,
    RefreshTokenMetadata,
    UpstreamTokenSet,
)
from fastmcp.server.auth.providers.github import GitHubProvider
from key_value.aio.stores.memory import MemoryStore
from mcp.server.auth.provider import AuthorizationParams, AuthorizeError, RefreshToken
from mcp.shared.auth import OAuthClientInformationFull
from pydantic import AnyUrl

from context_server.auth import LocalGitHubOAuthProvider, ProviderTokenVerifier
from context_server.config import LocalGitHubOAuth, ProviderCredential


@pytest.mark.asyncio
async def test_verifier_binds_identity_to_matching_digest() -> None:
    token = "test-codex-token"
    verifier = ProviderTokenVerifier(
        [ProviderCredential("codex", hashlib.sha256(token.encode()).hexdigest())]
    )

    access_token = await verifier.verify_token(token)

    assert access_token is not None
    assert access_token.client_id == "codex"
    assert access_token.claims == {"provider": "codex"}
    assert await verifier.verify_token("different-token") is None


@pytest.mark.asyncio
async def test_verifier_rejects_empty_and_oversized_tokens() -> None:
    verifier = ProviderTokenVerifier(
        [ProviderCredential("codex", hashlib.sha256(b"valid-token").hexdigest())]
    )

    assert await verifier.verify_token("") is None
    assert await verifier.verify_token("x" * 4097) is None


@pytest.mark.asyncio
async def test_local_oauth_requires_admin_and_preserves_bearer_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def verify_github(self: GitHubProvider, token: str) -> AccessToken | None:
        if token == "oauth-admin":
            return AccessToken(
                token="upstream-secret",
                client_id="github",
                scopes=["user"],
                claims={"sub": "42", "email": "not-needed"},
            )
        if token == "oauth-other":
            return AccessToken(
                token=token, client_id="github", scopes=["user"], claims={"sub": "43"}
            )
        return None

    monkeypatch.setattr(GitHubProvider, "verify_token", verify_github)
    bearer = "local-bearer"
    auth = LocalGitHubOAuthProvider(
        LocalGitHubOAuth("clientid", "not-a-real-secret", "42"),
        [ProviderCredential("codex", hashlib.sha256(bearer.encode()).hexdigest())],
        8000,
        client_storage=MemoryStore(),
    )

    codex = await auth.verify_token(bearer)
    claude = await auth.verify_token("oauth-admin")
    assert codex is not None and codex.claims["provider"] == "codex"
    assert claude is not None and claude.claims["provider"] == "claude"
    assert claude.token == "oauth-admin"
    assert claude.claims == {"provider": "claude", "sub": "42"}
    assert await auth.verify_token("oauth-other") is None
    assert await auth.verify_token("invalid") is None

    copilot_auth = LocalGitHubOAuthProvider(
        LocalGitHubOAuth("clientid", "not-a-real-secret", "42", provider="copilot"),
        [],
        8000,
        client_storage=MemoryStore(),
    )
    copilot = await copilot_auth.verify_token("oauth-admin")
    assert copilot is not None and copilot.claims["provider"] == "copilot"


@pytest.mark.asyncio
async def test_local_oauth_checks_real_proxy_token_and_upstream_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth = LocalGitHubOAuthProvider(
        LocalGitHubOAuth("clientid", "not-a-real-secret", "42"),
        [],
        8000,
        client_storage=MemoryStore(),
    )
    auth.set_mcp_path("/mcp")
    await auth._upstream_token_store.put(
        key="upstream-id",
        value=UpstreamTokenSet(
            upstream_token_id="upstream-id",
            access_token="upstream-token",
            refresh_token=None,
            refresh_token_expires_at=None,
            expires_at=time.time() + 300,
            token_type="Bearer",
            scope="user",
            client_id="client",
            created_at=time.time(),
        ),
    )
    await auth._jti_mapping_store.put(
        key="test-jti",
        value=JTIMapping(jti="test-jti", upstream_token_id="upstream-id", created_at=time.time()),
    )
    token = auth.jwt_issuer.issue_access_token(
        client_id="client", scopes=["user"], jti="test-jti", expires_in=300
    )
    subject: str | None = "42"

    async def verify_upstream(token: str) -> AccessToken | None:
        assert token == "upstream-token"
        if subject is None:
            return None
        return AccessToken(token=token, client_id=subject, scopes=["user"], claims={"sub": subject})

    monkeypatch.setattr(auth._token_validator, "verify_token", verify_upstream)
    verified = await auth.verify_token(token)
    assert verified is not None
    assert verified.token == token
    assert verified.claims == {"provider": "claude", "sub": "42"}
    subject = "43"
    assert await auth.verify_token(token) is None
    subject = None
    assert await auth.verify_token(token) is None
    assert await auth.verify_token(token + "invalid") is None


@pytest.mark.asyncio
async def test_local_oauth_rejects_token_for_another_resource() -> None:
    auth = LocalGitHubOAuthProvider(
        LocalGitHubOAuth("clientid", "not-a-real-secret", "42"),
        [],
        8000,
        client_storage=MemoryStore(),
    )
    auth.set_mcp_path("/mcp")
    wrong_resource_issuer = JWTIssuer(
        issuer=auth.jwt_issuer.issuer,
        audience="https://other.example/mcp",
        signing_key=auth.jwt_issuer._signing_key,
    )
    token = wrong_resource_issuer.issue_access_token(
        client_id="client",
        scopes=["user"],
        jti="wrong-resource-jti",
        expires_in=300,
    )

    assert await auth.verify_token(token) is None


@pytest.mark.asyncio
async def test_oauth_revocation_endpoint_invalidates_access_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth = LocalGitHubOAuthProvider(
        LocalGitHubOAuth(
            "clientid",
            "not-a-real-secret",
            "42",
            external_base_url="https://mcp.example.com",
        ),
        [],
        8000,
        client_storage=MemoryStore(),
    )

    async def verify_upstream(token: str) -> AccessToken:
        assert token == "upstream-token"
        return AccessToken(token=token, client_id="42", scopes=["user"], claims={"sub": "42"})

    monkeypatch.setattr(auth._token_validator, "verify_token", verify_upstream)
    app = FastMCP(name="test", auth=auth).http_app(path="/mcp")
    upstream_token_id = "revocation-upstream-id"
    access_jti = "revocation-access-jti"
    upstream_token = UpstreamTokenSet(
        upstream_token_id=upstream_token_id,
        access_token="upstream-token",
        refresh_token=None,
        refresh_token_expires_at=None,
        expires_at=time.time() + 300,
        token_type="Bearer",
        scope="user",
        client_id="claude-client",
        created_at=time.time(),
    )
    await auth._upstream_token_store.put(key=upstream_token_id, value=upstream_token, ttl=300)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://localhost:8000",
        ) as client,
    ):
        registration = await client.post(
            "/register",
            json={
                "redirect_uris": ["https://claude.ai/api/mcp/auth_callback"],
                "client_name": "Claude test",
                "token_endpoint_auth_method": "none",
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
            },
        )
        assert registration.status_code == 201
        client_id = registration.json()["client_id"]
        second_registration = await client.post(
            "/register",
            json={
                "redirect_uris": ["https://claude.ai/api/mcp/auth_callback"],
                "client_name": "Another Claude test",
                "token_endpoint_auth_method": "none",
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
            },
        )
        assert second_registration.status_code == 201
        other_client_id = second_registration.json()["client_id"]
        access_token = auth.jwt_issuer.issue_access_token(
            client_id=client_id,
            scopes=["user"],
            jti=access_jti,
            expires_in=300,
        )
        await auth._jti_mapping_store.put(
            key=access_jti,
            value=JTIMapping(
                jti=access_jti,
                upstream_token_id=upstream_token_id,
                created_at=time.time(),
            ),
            ttl=300,
        )
        wrong_client_response = await client.post(
            "/revoke",
            data={
                "client_id": other_client_id,
                "client_secret": "",
                "token": access_token,
                "token_type_hint": "access_token",
            },
        )
        assert wrong_client_response.status_code == 200
        assert await auth.load_access_token(access_token) is not None

        response = await client.post(
            "/revoke",
            data={
                "client_id": client_id,
                "client_secret": "",
                "token": access_token,
                "token_type_hint": "access_token",
            },
        )

    assert response.status_code == 200
    assert await auth._jti_mapping_store.get(key=access_jti) is None
    assert await auth._upstream_token_store.get(key=upstream_token_id) is None
    assert await auth.load_access_token(access_token) is None


@pytest.mark.asyncio
async def test_revoking_refresh_token_invalidates_related_access_token() -> None:
    auth = LocalGitHubOAuthProvider(
        LocalGitHubOAuth("clientid", "not-a-real-secret", "42"),
        [],
        8000,
        client_storage=MemoryStore(),
    )
    auth.set_mcp_path("/mcp")
    upstream_token_id = "refresh-revocation-upstream-id"
    access_jti = "refresh-revocation-access-jti"
    refresh_jti = "refresh-revocation-refresh-jti"
    await auth._upstream_token_store.put(
        key=upstream_token_id,
        value=UpstreamTokenSet(
            upstream_token_id=upstream_token_id,
            access_token="upstream-token",
            refresh_token="upstream-refresh-token",
            refresh_token_expires_at=time.time() + 300,
            expires_at=time.time() + 300,
            token_type="Bearer",
            scope="user",
            client_id="claude-client",
            created_at=time.time(),
        ),
        ttl=300,
    )
    for jti in (access_jti, refresh_jti):
        await auth._jti_mapping_store.put(
            key=jti,
            value=JTIMapping(
                jti=jti,
                upstream_token_id=upstream_token_id,
                created_at=time.time(),
            ),
            ttl=300,
        )
    access_token = auth.jwt_issuer.issue_access_token(
        client_id="claude-client",
        scopes=["user"],
        jti=access_jti,
        expires_in=300,
    )
    refresh_token = auth.jwt_issuer.issue_refresh_token(
        client_id="claude-client",
        scopes=["user"],
        jti=refresh_jti,
        expires_in=300,
    )
    token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()
    await auth._refresh_token_store.put(
        key=token_hash,
        value=RefreshTokenMetadata(
            client_id="claude-client",
            scopes=["user"],
            expires_at=int(time.time()) + 300,
            created_at=time.time(),
        ),
        ttl=300,
    )

    await auth.revoke_token(
        RefreshToken(
            token=refresh_token,
            client_id="claude-client",
            scopes=["user"],
            expires_at=int(time.time()) + 300,
        )
    )

    assert await auth._refresh_token_store.get(key=token_hash) is None
    assert await auth._jti_mapping_store.get(key=refresh_jti) is None
    assert await auth._upstream_token_store.get(key=upstream_token_id) is None
    assert await auth.load_access_token(access_token) is None


@pytest.mark.asyncio
async def test_oauth_transaction_survives_provider_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fastmcp_settings, "home", tmp_path)
    oauth = LocalGitHubOAuth("clientid", "stable-test-client-secret", "42")
    first = LocalGitHubOAuthProvider(oauth, [], 8000)
    transaction = OAuthTransaction(
        txn_id="restart-test",
        client_id="claude-client",
        client_redirect_uri="https://claude.ai/api/mcp/auth_callback",
        client_state="client-state",
        code_challenge="challenge",
        code_challenge_method="S256",
        scopes=["user"],
        created_at=time.time(),
    )
    await first._transaction_store.put(key=transaction.txn_id, value=transaction, ttl=60)

    restarted = LocalGitHubOAuthProvider(oauth, [], 8000)
    restored = await restarted._transaction_store.get(key=transaction.txn_id)

    assert restored == transaction


@pytest.mark.asyncio
async def test_oauth_callback_rejects_browser_without_consent_cookie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange = AsyncMock(side_effect=RuntimeError("synthetic token exchange"))

    @asynccontextmanager
    async def mock_upstream() -> AsyncIterator[SimpleNamespace]:
        yield SimpleNamespace(fetch_token=exchange)

    auth = LocalGitHubOAuthProvider(
        LocalGitHubOAuth("clientid", "not-a-real-secret", "42"),
        [],
        8000,
        client_storage=MemoryStore(),
    )
    monkeypatch.setattr(auth, "_upstream_oauth_client", mock_upstream)
    app = FastMCP(name="test", auth=auth).http_app(path="/mcp")
    redirect = AnyUrl("http://127.0.0.1:32123/callback")
    client = OAuthClientInformationFull(client_id="client", redirect_uris=[redirect])
    params = AuthorizationParams(
        state="client-state",
        scopes=["user"],
        code_challenge="challenge",
        redirect_uri=redirect,
        redirect_uri_provided_explicitly=True,
    )

    async with app.router.lifespan_context(app):
        consent_url = await auth.authorize(client, params)
        transaction_id = parse_qs(urlsplit(consent_url).query)["txn_id"][0]
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://localhost:8000"
        ) as first:
            page = await first.get(consent_url)
            assert page.status_code == 200
            transaction = await auth._transaction_store.get(key=transaction_id)
            assert transaction is not None and transaction.csrf_token is not None
            approved = await first.post(
                "/consent",
                data={
                    "txn_id": transaction_id,
                    "csrf_token": transaction.csrf_token,
                    "action": "approve",
                },
            )
            assert approved.status_code == 302
            assert "__MCP_CONSENT_BINDING" in first.cookies

            async with httpx.AsyncClient(
                transport=transport, base_url="http://localhost:8000"
            ) as other:
                callback = await other.get(
                    "/auth/callback", params={"state": transaction_id, "code": "synthetic-code"}
                )
                assert callback.status_code == 403
                exchange.assert_not_awaited()

            callback = await first.get(
                "/auth/callback", params={"state": transaction_id, "code": "synthetic-code"}
            )
            assert callback.status_code == 500
            exchange.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "redirect",
    [
        "https://example.com/callback",
        "http://localhost:1234@attacker.example/callback",
        "http://localhost/callback",
    ],
)
async def test_local_oauth_rejects_nonlocal_redirects(redirect: str) -> None:
    auth = LocalGitHubOAuthProvider(
        LocalGitHubOAuth("clientid", "not-a-real-secret", "42"),
        [],
        8000,
        client_storage=MemoryStore(),
    )
    params = AuthorizationParams(
        state=None,
        scopes=None,
        code_challenge="challenge",
        redirect_uri=AnyUrl(redirect),
        redirect_uri_provided_explicitly=True,
    )
    client = OAuthClientInformationFull(client_id="client", redirect_uris=[AnyUrl(redirect)])

    with pytest.raises(AuthorizeError, match="Invalid local redirect URI"):
        await auth.authorize(client, params)


@pytest.mark.asyncio
async def test_local_oauth_allows_loopback_redirect(monkeypatch: pytest.MonkeyPatch) -> None:
    async def authorize_github(
        self: GitHubProvider, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        return "http://localhost:8000/consent"

    monkeypatch.setattr(GitHubProvider, "authorize", authorize_github)
    auth = LocalGitHubOAuthProvider(
        LocalGitHubOAuth("clientid", "not-a-real-secret", "42"),
        [],
        8000,
        client_storage=MemoryStore(),
    )
    redirect = AnyUrl("http://127.0.0.1:32123/callback")
    client = OAuthClientInformationFull(client_id="client", redirect_uris=[redirect])
    params = AuthorizationParams(
        state=None,
        scopes=None,
        code_challenge="challenge",
        redirect_uri=redirect,
        redirect_uri_provided_explicitly=True,
    )
    assert await auth.authorize(client, params) == "http://localhost:8000/consent"


@pytest.mark.asyncio
async def test_local_oauth_allows_claude_hosted_redirect_only_with_external_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def authorize_github(
        self: GitHubProvider, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        return "https://mcp.example.com/consent"

    monkeypatch.setattr(GitHubProvider, "authorize", authorize_github)
    redirect = AnyUrl("https://claude.ai/api/mcp/auth_callback")
    client = OAuthClientInformationFull(client_id="client", redirect_uris=[redirect])
    params = AuthorizationParams(
        state=None,
        scopes=["user"],
        code_challenge="challenge",
        redirect_uri=redirect,
        redirect_uri_provided_explicitly=True,
    )
    hosted = LocalGitHubOAuthProvider(
        LocalGitHubOAuth(
            "clientid", "not-a-real-secret", "42", external_base_url="https://mcp.example.com"
        ),
        [],
        8000,
        client_storage=MemoryStore(),
    )
    local_only = LocalGitHubOAuthProvider(
        LocalGitHubOAuth("clientid", "not-a-real-secret", "42"),
        [],
        8000,
        client_storage=MemoryStore(),
    )

    assert await hosted.authorize(client, params) == "https://mcp.example.com/consent"
    with pytest.raises(AuthorizeError, match="Invalid local redirect URI"):
        await local_only.authorize(client, params)
