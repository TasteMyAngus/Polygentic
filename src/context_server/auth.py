from __future__ import annotations

import hashlib
import hmac
from collections.abc import Sequence
from urllib.parse import urlsplit

from fastmcp.server.auth import AccessToken, TokenVerifier
from fastmcp.server.auth.providers.github import GitHubProvider
from joserfc.errors import JoseError
from key_value.aio.protocols import AsyncKeyValue
from mcp.server.auth.provider import (
    AccessToken as OAuthAccessToken,
)
from mcp.server.auth.provider import (
    AuthorizationParams,
    AuthorizeError,
    RefreshToken,
)
from mcp.server.auth.settings import RevocationOptions
from mcp.shared.auth import OAuthClientInformationFull

from context_server.config import LocalGitHubOAuth, ProviderCredential


class ProviderTokenVerifier(TokenVerifier):
    """Verify high-entropy bearer tokens using provider-bound digests."""

    def __init__(self, credentials: Sequence[ProviderCredential]) -> None:
        super().__init__(required_scopes=["mcp:connect"])
        self._credentials = tuple(
            (credential.provider, bytes.fromhex(credential.token_sha256))
            for credential in credentials
        )

    async def verify_token(self, token: str) -> AccessToken | None:
        if not token or len(token) > 4096:
            return None

        presented_digest = hashlib.sha256(token.encode("utf-8")).digest()
        provider: str | None = None
        for candidate_provider, expected_digest in self._credentials:
            if hmac.compare_digest(presented_digest, expected_digest):
                provider = candidate_provider

        if provider is None:
            return None

        return AccessToken(
            token=token,
            client_id=provider,
            scopes=["mcp:connect"],
            expires_at=None,
            claims={"provider": provider},
        )


class LocalGitHubOAuthProvider(GitHubProvider):
    """Local-only GitHub OAuth alongside provider bearer credentials."""

    def __init__(
        self,
        oauth: LocalGitHubOAuth,
        credentials: Sequence[ProviderCredential],
        port: int,
        client_storage: AsyncKeyValue | None = None,
    ) -> None:
        base_url = oauth.external_base_url or f"http://localhost:{port}"
        allowed_redirect_uris = ["http://localhost:*", "http://127.0.0.1:*"]
        if oauth.external_base_url is not None:
            allowed_redirect_uris.append("https://claude.ai/api/mcp/auth_callback")
        self._allow_claude_hosted_redirect = oauth.external_base_url is not None
        super().__init__(
            client_id=oauth.client_id,
            client_secret=oauth.client_secret,
            base_url=base_url,
            allowed_client_redirect_uris=allowed_redirect_uris,
            client_storage=client_storage,
        )
        self.required_scopes = []
        self.revocation_options = RevocationOptions(enabled=True)
        self._bearer_verifier = ProviderTokenVerifier(credentials)
        self._admin_user_id = oauth.admin_user_id
        self._provider = oauth.provider

    async def load_access_token(self, token: str) -> AccessToken | None:
        access_token = await super().load_access_token(token)
        if access_token is None:
            return None

        payload = self.jwt_issuer.verify_token(token)
        jti = payload.get("jti")
        client_id = payload.get("client_id")
        if not isinstance(jti, str) or not isinstance(client_id, str):
            return None

        claims = dict(access_token.claims or {})
        claims["_fastmcp_jti"] = jti
        return AccessToken(
            token=access_token.token,
            client_id=client_id,
            scopes=access_token.scopes,
            expires_at=access_token.expires_at,
            resource=access_token.resource,
            subject=access_token.subject,
            claims=claims,
        )

    async def verify_token(self, token: str) -> AccessToken | None:
        if not token or len(token) > 4096:
            return None
        bearer = await self._bearer_verifier.verify_token(token)
        if bearer is not None:
            return bearer
        authenticated = await super().verify_token(token)
        if authenticated is None or authenticated.claims.get("sub") != self._admin_user_id:
            return None
        return AccessToken(
            token=token,
            client_id=authenticated.client_id,
            scopes=authenticated.scopes,
            expires_at=authenticated.expires_at,
            claims={"provider": self._provider, "sub": self._admin_user_id},
        )

    async def revoke_token(self, token: OAuthAccessToken | RefreshToken) -> None:
        if isinstance(token, RefreshToken):
            try:
                payload = self.jwt_issuer.verify_token(token.token, expected_token_use="refresh")
            except JoseError:
                return
            jti = payload.get("jti")
        else:
            jti = (token.claims or {}).get("_fastmcp_jti")

        if not isinstance(jti, str):
            return

        mapping = await self._jti_mapping_store.get(key=jti)
        if mapping is not None:
            await self._upstream_token_store.delete(key=mapping.upstream_token_id)
            await self._jti_mapping_store.delete(key=jti)
        if isinstance(token, RefreshToken):
            await super().revoke_token(token)

    async def authorize(
        self, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        if (
            self._allow_claude_hosted_redirect
            and str(params.redirect_uri) == "https://claude.ai/api/mcp/auth_callback"
        ):
            return await super().authorize(client, params)

        redirect = urlsplit(str(params.redirect_uri))
        try:
            port = redirect.port
        except ValueError as exc:
            raise AuthorizeError("invalid_request", "Invalid local redirect URI") from exc
        if (
            redirect.scheme != "http"
            or redirect.hostname not in {"localhost", "127.0.0.1"}
            or redirect.username is not None
            or redirect.password is not None
            or port is None
            or redirect.fragment
        ):
            raise AuthorizeError("invalid_request", "Invalid local redirect URI")
        return await super().authorize(client, params)
