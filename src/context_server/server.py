from __future__ import annotations

import uvicorn
from fastmcp import FastMCP
from fastmcp.server.auth.auth import AuthProvider
from fastmcp.server.dependencies import get_access_token
from key_value.aio.protocols import AsyncKeyValue
from mcp.types import ToolAnnotations
from starlette.applications import Starlette

from context_server.auth import LocalGitHubOAuthProvider, ProviderTokenVerifier
from context_server.config import Settings
from context_server.middleware import RequestBoundaryMiddleware

MAX_ECHO_CHARACTERS = 2048


def create_app(settings: Settings, oauth_client_storage: AsyncKeyValue | None = None) -> Starlette:
    verifier: AuthProvider
    if settings.local_github_oauth is None:
        verifier = ProviderTokenVerifier(settings.credentials)
    else:
        verifier = LocalGitHubOAuthProvider(
            settings.local_github_oauth,
            settings.credentials,
            settings.port,
            client_storage=oauth_client_storage,
        )
    mcp = FastMCP(name="Context Server", auth=verifier)
    read_only = ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
    )

    @mcp.tool(annotations=read_only)
    def whoami() -> dict[str, str]:
        """Return the provider identity bound to the authenticated bearer token."""
        access_token = get_access_token()
        if access_token is None:
            raise RuntimeError("Authenticated provider identity is unavailable")
        provider = access_token.claims.get("provider")
        if not isinstance(provider, str):
            raise TypeError("Authenticated provider identity is invalid")
        return {"provider": provider}

    @mcp.tool(annotations=read_only)
    def echo(text: str) -> dict[str, str | int]:
        """Echo synthetic test text, limited to 2,048 characters."""
        if len(text) > MAX_ECHO_CHARACTERS:
            raise ValueError(f"text exceeds the {MAX_ECHO_CHARACTERS}-character limit")
        return {"text": text, "length": len(text)}

    app = mcp.http_app(path="/mcp")
    app.add_middleware(
        RequestBoundaryMiddleware,
        allowed_hosts=settings.allowed_hosts,
        allowed_origins=settings.allowed_origins,
    )
    return app


def main() -> None:
    settings = Settings.from_environment()
    app = create_app(settings)
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="info",
        access_log=settings.local_github_oauth is None,
    )


if __name__ == "__main__":
    main()
