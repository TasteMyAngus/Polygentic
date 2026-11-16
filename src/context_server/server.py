from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastmcp import FastMCP
from fastmcp.server.auth.auth import AuthProvider
from fastmcp.server.dependencies import get_access_token
from key_value.aio.protocols import AsyncKeyValue
from mcp.types import ToolAnnotations
from starlette.applications import Starlette

from context_server.auth import LocalGitHubOAuthProvider, ProviderTokenVerifier
from context_server.config import Settings
from context_server.embeddings import Embedder, FastEmbedder
from context_server.memory import (
    CheckpointWriteResult,
    ContextAssembly,
    ContextRecord,
    ContextWriteResult,
    MemoryService,
    Metadata,
    RecordType,
    SearchMode,
    SearchResponse,
    SessionResult,
    SupersessionResult,
)
from context_server.middleware import RequestBoundaryMiddleware
from context_server.storage import MemoryDatabase

MAX_ECHO_CHARACTERS = 2048


def create_app(
    settings: Settings,
    oauth_client_storage: AsyncKeyValue | None = None,
    memory_database: MemoryDatabase | None = None,
    embedder: Embedder | None = None,
) -> Starlette:
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
    database = memory_database or MemoryDatabase(settings.data_dir)
    memory = MemoryService(
        database,
        embedder or FastEmbedder(settings.model_cache_dir),
    )
    mcp = FastMCP(name="Context Server", auth=verifier)
    read_only = ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
    )
    write_operation = ToolAnnotations(
        readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False
    )

    def authenticated_provider() -> str:
        access_token = get_access_token()
        if access_token is None:
            raise RuntimeError("Authenticated provider identity is unavailable")
        provider = access_token.claims.get("provider")
        if not isinstance(provider, str):
            raise TypeError("Authenticated provider identity is invalid")
        return provider

    @mcp.tool(annotations=read_only)
    def whoami() -> dict[str, str]:
        """Return the provider identity bound to the authenticated bearer token."""
        return {"provider": authenticated_provider()}

    @mcp.tool(annotations=read_only)
    def echo(text: str) -> dict[str, str | int]:
        """Echo synthetic test text, limited to 2,048 characters."""
        if len(text) > MAX_ECHO_CHARACTERS:
            raise ValueError(f"text exceeds the {MAX_ECHO_CHARACTERS}-character limit")
        return {"text": text, "length": len(text)}

    @mcp.tool(annotations=write_operation)
    async def register_session(label: str, idempotency_key: str) -> SessionResult:
        """Create or idempotently recover a logical session and its owned spaces."""
        return await memory.register_session(
            authenticated_provider(), label, idempotency_key
        )

    @mcp.tool(annotations=read_only)
    async def list_sessions() -> list[SessionResult]:
        """List logical sessions owned by the authenticated provider."""
        return await memory.list_sessions(authenticated_provider())

    @mcp.tool(annotations=write_operation)
    async def resume_session(session_id: str) -> SessionResult:
        """Resume a logical session owned by the authenticated provider."""
        return await memory.resume_session(authenticated_provider(), session_id)

    @mcp.tool(annotations=write_operation)
    async def put_context(
        session_id: str,
        title: str,
        content: str,
        record_type: RecordType,
        idempotency_key: str,
        space_id: str | None = None,
        document_id: str | None = None,
        expected_revision: int | None = None,
        source_ref: str | None = None,
        metadata: Metadata | None = None,
    ) -> ContextWriteResult:
        """Create or revision-update a typed text record in an owned space."""
        return await memory.put_context(
            provider=authenticated_provider(),
            session_id=session_id,
            title=title,
            content=content,
            record_type=record_type,
            idempotency_key=idempotency_key,
            space_id=space_id,
            document_id=document_id,
            expected_revision=expected_revision,
            source_ref=source_ref,
            metadata=metadata,
        )

    @mcp.tool(annotations=read_only)
    async def get_context(
        session_id: str,
        document_id: str,
        offset: int = 0,
        max_chars: int = 16_000,
        include_superseded: bool = False,
    ) -> ContextRecord:
        """Read a record page, optionally including superseded history."""
        return await memory.get_context(
            authenticated_provider(),
            session_id,
            document_id,
            offset,
            max_chars,
            include_superseded,
        )

    @mcp.tool(annotations=read_only)
    async def search_context(
        session_id: str,
        query: str,
        mode: SearchMode = "hybrid",
        top_k: int = 10,
        space_ids: list[str] | None = None,
        record_types: list[RecordType] | None = None,
        include_superseded: bool = False,
    ) -> SearchResponse:
        """Search authorized spaces; superseded history is opt-in."""
        return await memory.search_context(
            provider=authenticated_provider(),
            session_id=session_id,
            query=query,
            mode=mode,
            top_k=top_k,
            space_ids=space_ids,
            record_types=record_types,
            include_superseded=include_superseded,
        )

    @mcp.tool(annotations=write_operation)
    async def supersede_context(
        session_id: str,
        newer_document_id: str,
        older_document_ids: list[str],
        expected_revisions: dict[str, int],
        idempotency_key: str,
    ) -> SupersessionResult:
        """Explicitly supersede same-space records with expected-revision checks."""
        return await memory.supersede_context(
            provider=authenticated_provider(),
            session_id=session_id,
            newer_document_id=newer_document_id,
            older_document_ids=older_document_ids,
            expected_revisions=expected_revisions,
            idempotency_key=idempotency_key,
        )

    @mcp.tool(annotations=write_operation)
    async def put_checkpoint(
        session_id: str,
        task_id: str,
        content: str,
        expected_head_revision: int,
        idempotency_key: str,
    ) -> CheckpointWriteResult:
        """Write a typed task checkpoint and atomically advance its revisioned head."""
        return await memory.put_checkpoint(
            provider=authenticated_provider(),
            session_id=session_id,
            task_id=task_id,
            content=content,
            expected_head_revision=expected_head_revision,
            idempotency_key=idempotency_key,
        )

    @mcp.tool(annotations=read_only)
    async def assemble_context(
        session_id: str,
        task_id: str,
        query: str,
        max_chars: int = 6000,
    ) -> ContextAssembly:
        """Return the exact task checkpoint plus bounded supporting evidence."""
        return await memory.assemble_context(
            provider=authenticated_provider(),
            session_id=session_id,
            task_id=task_id,
            query=query,
            max_chars=max_chars,
        )

    app = mcp.http_app(path="/mcp")
    app.add_middleware(
        RequestBoundaryMiddleware,
        allowed_hosts=settings.allowed_hosts,
        allowed_origins=settings.allowed_origins,
    )
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def memory_lifespan(application: Starlette) -> AsyncIterator[None]:
        try:
            async with original_lifespan(application):
                yield
        finally:
            database.close()

    app.router.lifespan_context = memory_lifespan
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
