from __future__ import annotations

import asyncio
import hashlib
import json
import re
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel

from context_server.config import EMBEDDING_MODEL
from context_server.embeddings import Embedder
from context_server.storage import (
    CheckpointSnapshot,
    ChunkInput,
    MemoryDatabase,
    RecordSnapshot,
    SearchCandidate,
    SessionInfo,
    SpaceInfo,
    SpaceInvitationInfo,
    SupersessionReceipt,
    WriteReceipt,
)

type RecordType = Literal[
    "note",
    "decision",
    "constraint",
    "finding",
    "unresolved",
    "lesson",
    "checkpoint",
    "handoff",
]
type SearchMode = Literal["hybrid", "keyword", "semantic"]
type SharedSpaceKind = Literal["project", "shared"]
type MetadataValue = str | int | float | bool | None
type Metadata = dict[str, MetadataValue]

MAX_RECORD_BYTES = 256 * 1024
MAX_METADATA_BYTES = 16 * 1024
MAX_TITLE_CHARACTERS = 256
MAX_SOURCE_REFERENCE_CHARACTERS = 2048
MAX_SESSION_LABEL_CHARACTERS = 128
MAX_TASK_ID_CHARACTERS = 256
MAX_QUERY_CHARACTERS = 4096
MAX_TOP_K = 50
DEFAULT_TOP_K = 10
MAX_SNIPPET_CHARACTERS = 1000
MAX_COMBINED_SNIPPET_CHARACTERS = 12_000
DEFAULT_ASSEMBLY_CHARACTERS = 6000
MAX_ASSEMBLY_CHARACTERS = 12_000
MAX_CONTEXT_PAGE_CHARACTERS = 32_000
CHUNK_CHARACTERS = 1200
CHUNK_OVERLAP_CHARACTERS = 120
MAX_CHUNKS_PER_RECORD = 256
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_QUERY_TERM = re.compile(r"[^\W_]+(?:[-_][^\W_]+)*", re.UNICODE)


class SessionResult(BaseModel):
    session_id: str
    provider: str
    label: str
    provider_space_id: str
    session_space_id: str
    created_at: str
    last_seen_at: str


class ContextWriteResult(BaseModel):
    document_id: str
    revision: int
    space_id: str


class ContextRecord(BaseModel):
    document_id: str
    space_id: str
    revision: int
    title: str
    content: str
    record_type: RecordType
    metadata: Metadata
    source_ref: str | None
    author_provider: str
    author_session_id: str
    created_at: str
    task_id: str | None
    status: Literal["active", "superseded"]
    supersedes: list[str]
    superseded_by: list[str]
    offset: int
    next_offset: int | None
    truncated: bool


class SearchHit(BaseModel):
    document_id: str
    space_id: str
    revision: int
    title: str
    record_type: RecordType
    snippet: str
    char_start: int
    char_end: int
    source_ref: str | None
    author_provider: str
    author_session_id: str
    created_at: str
    score: float
    status: Literal["active", "superseded"]
    supersedes: list[str]
    superseded_by: list[str]


class SearchResponse(BaseModel):
    query: str
    mode: SearchMode
    results: list[SearchHit]
    truncated: bool
    snippet_characters: int


class CheckpointWriteResult(BaseModel):
    document_id: str
    document_revision: int
    head_revision: int


class SupersessionResult(BaseModel):
    newer_document_id: str
    superseded_document_ids: list[str]


class RegisteredSessionResult(BaseModel):
    session_id: str
    provider: str


class SpaceMemberResult(BaseModel):
    session_id: str
    provider: str
    role: Literal["owner", "member"]


class SpaceResult(BaseModel):
    space_id: str
    kind: SharedSpaceKind
    name: str
    project_space_id: str | None
    owner_session_id: str
    owner_provider: str
    archived_at: str | None
    members: list[SpaceMemberResult]


class SpaceInvitationResult(BaseModel):
    invitation_id: str
    space_id: str
    space_name: str
    owner_session_id: str
    inviter_session_id: str
    invited_session_id: str
    status: Literal["pending", "accepted", "revoked", "left"]
    created_at: str


class AssembledCheckpoint(BaseModel):
    document_id: str
    document_revision: int
    head_revision: int
    title: str
    content: str
    truncated: bool


class ContextAssembly(BaseModel):
    session_id: str
    task_id: str
    checkpoint: AssembledCheckpoint | None
    missing_checkpoint: bool
    evidence: list[SearchHit]
    truncated: bool
    total_characters: int


class MemoryService:
    def __init__(
        self,
        database: MemoryDatabase,
        embedder: Embedder,
        embedding_model: str = EMBEDDING_MODEL,
    ) -> None:
        self._database = database
        self._embedder = embedder
        self._embedding_model = embedding_model

    async def register_session(
        self, provider: str, label: str, idempotency_key: str
    ) -> SessionResult:
        _validate_identifier(idempotency_key, "idempotency key")
        _validate_bounded_text(label, MAX_SESSION_LABEL_CHARACTERS, "session label")
        session = await asyncio.to_thread(
            self._database.register_session, provider, label, idempotency_key
        )
        return _session_result(session)

    async def list_sessions(self, provider: str) -> list[SessionResult]:
        sessions = await asyncio.to_thread(self._database.list_sessions, provider)
        return [_session_result(session) for session in sessions]

    async def resume_session(self, provider: str, session_id: str) -> SessionResult:
        _validate_identifier(session_id, "session ID")
        session = await asyncio.to_thread(
            self._database.resume_session, provider, session_id
        )
        return _session_result(session)

    async def list_registered_sessions(self) -> list[RegisteredSessionResult]:
        sessions = await asyncio.to_thread(self._database.list_registered_sessions)
        return [
            RegisteredSessionResult(session_id=session.session_id, provider=session.provider)
            for session in sessions
        ]

    async def create_space(
        self,
        provider: str,
        session_id: str,
        kind: SharedSpaceKind,
        name: str,
        idempotency_key: str,
        project_space_id: str | None = None,
    ) -> SpaceResult:
        _validate_identifier(session_id, "session ID")
        _validate_identifier(idempotency_key, "idempotency key")
        if kind not in {"project", "shared"}:
            raise ValueError("kind must be project or shared")
        _validate_bounded_text(name, MAX_TITLE_CHARACTERS, "space name")
        if project_space_id is not None:
            _validate_identifier(project_space_id, "project space ID")
        payload_hash = _payload_hash(
            {
                "operation": "create_space",
                "session_id": session_id,
                "kind": kind,
                "name": name,
                "project_space_id": project_space_id,
            }
        )
        space = await asyncio.to_thread(
            self._database.create_space,
            provider,
            session_id,
            kind,
            name,
            idempotency_key,
            payload_hash,
            project_space_id,
        )
        return _space_result(space)

    async def list_spaces(self, provider: str, session_id: str) -> list[SpaceResult]:
        _validate_identifier(session_id, "session ID")
        spaces = await asyncio.to_thread(self._database.list_spaces, provider, session_id)
        return [_space_result(space) for space in spaces]

    async def get_space(
        self, provider: str, session_id: str, space_id: str
    ) -> SpaceResult:
        _validate_identifier(session_id, "session ID")
        _validate_identifier(space_id, "space ID")
        space = await asyncio.to_thread(
            self._database.get_space, provider, session_id, space_id
        )
        return _space_result(space)

    async def invite_session(
        self, provider: str, session_id: str, space_id: str, target_session_id: str
    ) -> SpaceInvitationResult:
        _validate_identifier(session_id, "session ID")
        _validate_identifier(space_id, "space ID")
        _validate_identifier(target_session_id, "target session ID")
        invitation = await asyncio.to_thread(
            self._database.invite_session,
            provider,
            session_id,
            space_id,
            target_session_id,
        )
        return _invitation_result(invitation)

    async def list_invitations(
        self, provider: str, session_id: str
    ) -> list[SpaceInvitationResult]:
        _validate_identifier(session_id, "session ID")
        invitations = await asyncio.to_thread(
            self._database.list_invitations, provider, session_id
        )
        return [_invitation_result(invitation) for invitation in invitations]

    async def accept_invitation(
        self, provider: str, session_id: str, invitation_id: str
    ) -> SpaceResult:
        _validate_identifier(session_id, "session ID")
        _validate_identifier(invitation_id, "invitation ID")
        space = await asyncio.to_thread(
            self._database.accept_invitation, provider, session_id, invitation_id
        )
        return _space_result(space)

    async def revoke_membership(
        self, provider: str, session_id: str, space_id: str, member_session_id: str
    ) -> None:
        _validate_identifier(session_id, "session ID")
        _validate_identifier(space_id, "space ID")
        _validate_identifier(member_session_id, "member session ID")
        await asyncio.to_thread(
            self._database.revoke_membership,
            provider,
            session_id,
            space_id,
            member_session_id,
        )

    async def leave_space(self, provider: str, session_id: str, space_id: str) -> None:
        _validate_identifier(session_id, "session ID")
        _validate_identifier(space_id, "space ID")
        await asyncio.to_thread(self._database.leave_space, provider, session_id, space_id)

    async def archive_space(
        self, provider: str, session_id: str, space_id: str
    ) -> str:
        _validate_identifier(session_id, "session ID")
        _validate_identifier(space_id, "space ID")
        return await asyncio.to_thread(
            self._database.archive_space, provider, session_id, space_id
        )

    async def put_context(
        self,
        *,
        provider: str,
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
        _validate_identifier(session_id, "session ID")
        _validate_identifier(idempotency_key, "idempotency key")
        if document_id is not None:
            _validate_identifier(document_id, "document ID")
        if space_id is not None:
            _validate_identifier(space_id, "space ID")
        _validate_bounded_text(title, MAX_TITLE_CHARACTERS, "title")
        _validate_content(content)
        _validate_record_type(record_type)
        if record_type == "checkpoint":
            raise ValueError("Use put_checkpoint for revisioned task checkpoints")
        _validate_expected_revision(expected_revision)
        if source_ref is not None:
            _validate_bounded_text(
                source_ref, MAX_SOURCE_REFERENCE_CHARACTERS, "source reference"
            )
        metadata_json = _encode_metadata(metadata or {})
        payload_hash = _payload_hash(
            {
                "operation": "put_context",
                "session_id": session_id,
                "space_id": space_id,
                "document_id": document_id,
                "expected_revision": expected_revision,
                "title": title,
                "content": content,
                "record_type": record_type,
                "source_ref": source_ref,
                "metadata_json": metadata_json,
            }
        )
        resolved_space_id, replay = await asyncio.to_thread(
            self._database.prepare_put,
            provider,
            session_id,
            space_id,
            document_id,
            expected_revision,
            idempotency_key,
            payload_hash,
        )
        if replay is not None:
            return _write_result(replay, space_id)
        if resolved_space_id is None:
            raise RuntimeError("Memory write preflight returned no space")
        chunks = _chunk_content(content)
        vectors = await self._embedder.embed([chunk.text for chunk in chunks])
        actual_document_id = document_id or str(uuid4())
        receipt = await asyncio.to_thread(
            self._database.put_context,
            provider=provider,
            session_id=session_id,
            space_id=resolved_space_id,
            document_id=actual_document_id,
            is_new=document_id is None,
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
            payload_hash=payload_hash,
            title=title,
            content=content,
            record_type=record_type,
            metadata_json=metadata_json,
            source_ref=source_ref,
            chunks=chunks,
            vectors=vectors,
            embedding_model=self._embedding_model,
        )
        return _write_result(receipt, resolved_space_id)

    async def get_context(
        self,
        provider: str,
        session_id: str,
        document_id: str,
        offset: int,
        max_chars: int,
        include_superseded: bool = False,
    ) -> ContextRecord:
        _validate_identifier(session_id, "session ID")
        _validate_identifier(document_id, "document ID")
        if offset < 0:
            raise ValueError("offset must be nonnegative")
        if not 1 <= max_chars <= MAX_CONTEXT_PAGE_CHARACTERS:
            raise ValueError(f"max_chars must be between 1 and {MAX_CONTEXT_PAGE_CHARACTERS}")
        record = await asyncio.to_thread(
            self._database.get_record,
            provider,
            session_id,
            document_id,
            include_superseded,
        )
        return _context_record(record, offset, max_chars)

    async def search_context(
        self,
        *,
        provider: str,
        session_id: str,
        query: str,
        mode: SearchMode = "hybrid",
        top_k: int = DEFAULT_TOP_K,
        space_ids: list[str] | None = None,
        record_types: list[RecordType] | None = None,
        include_superseded: bool = False,
    ) -> SearchResponse:
        _validate_identifier(session_id, "session ID")
        if len(query) > MAX_QUERY_CHARACTERS:
            raise ValueError(f"query exceeds the {MAX_QUERY_CHARACTERS}-character limit")
        if not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")
        if mode not in {"hybrid", "keyword", "semantic"}:
            raise ValueError("mode must be hybrid, keyword, or semantic")
        if space_ids is not None:
            if len(space_ids) > MAX_TOP_K:
                raise ValueError(f"space_ids cannot contain more than {MAX_TOP_K} entries")
            for space_id in space_ids:
                _validate_identifier(space_id, "space ID")
        if record_types is not None:
            if len(record_types) > 8:
                raise ValueError("record_types cannot contain more than 8 entries")
            for record_type in record_types:
                _validate_record_type(record_type)
        await asyncio.to_thread(self._database.session_space, provider, session_id)
        normalized_query = query.strip()
        if not normalized_query:
            return SearchResponse(
                query=query,
                mode=mode,
                results=[],
                truncated=False,
                snippet_characters=0,
            )
        match_query = _fts_match_query(normalized_query)
        candidate_limit = min(MAX_TOP_K * 4, max(20, top_k * 3))
        keyword_candidates: list[SearchCandidate] = []
        semantic_candidates: list[SearchCandidate] = []
        if mode in {"hybrid", "keyword"} and match_query:
            keyword_candidates = await asyncio.to_thread(
                self._database.search_keyword,
                provider,
                session_id,
                match_query,
                candidate_limit,
                space_ids,
                record_types,
                include_superseded,
            )
        if mode in {"hybrid", "semantic"}:
            query_vector = (await self._embedder.embed([normalized_query]))[0]
            semantic_candidates = await asyncio.to_thread(
                self._database.search_semantic,
                provider,
                session_id,
                query_vector,
                candidate_limit,
                space_ids,
                record_types,
                include_superseded,
            )
        ranked = _reciprocal_rank_fusion(keyword_candidates, semantic_candidates, top_k)
        hits, truncated, snippet_characters = _make_search_hits(ranked, top_k)
        return SearchResponse(
            query=normalized_query,
            mode=mode,
            results=hits,
            truncated=truncated,
            snippet_characters=snippet_characters,
        )

    async def put_checkpoint(
        self,
        *,
        provider: str,
        session_id: str,
        task_id: str,
        content: str,
        expected_head_revision: int,
        idempotency_key: str,
    ) -> CheckpointWriteResult:
        _validate_identifier(session_id, "session ID")
        _validate_bounded_text(task_id, MAX_TASK_ID_CHARACTERS, "task ID")
        _validate_identifier(idempotency_key, "idempotency key")
        _validate_content(content)
        if expected_head_revision < 0:
            raise ValueError("expected_head_revision must be nonnegative")
        metadata_json = _encode_metadata({"task_id": task_id})
        payload_hash = _payload_hash(
            {
                "operation": "put_checkpoint",
                "session_id": session_id,
                "task_id": task_id,
                "content": content,
                "expected_head_revision": expected_head_revision,
            }
        )
        _space_id, replay = await asyncio.to_thread(
            self._database.prepare_checkpoint,
            provider,
            session_id,
            task_id,
            expected_head_revision,
            idempotency_key,
            payload_hash,
        )
        if replay is not None:
            return _checkpoint_result(replay)
        chunks = _chunk_content(content)
        vectors = await self._embedder.embed([chunk.text for chunk in chunks])
        receipt = await asyncio.to_thread(
            self._database.put_checkpoint,
            provider=provider,
            session_id=session_id,
            task_id=task_id,
            expected_head_revision=expected_head_revision,
            document_id=str(uuid4()),
            idempotency_key=idempotency_key,
            payload_hash=payload_hash,
            title=f"Checkpoint: {task_id}"[:MAX_TITLE_CHARACTERS],
            content=content,
            metadata_json=metadata_json,
            chunks=chunks,
            vectors=vectors,
            embedding_model=self._embedding_model,
        )
        return _checkpoint_result(receipt)

    async def supersede_context(
        self,
        *,
        provider: str,
        session_id: str,
        newer_document_id: str,
        older_document_ids: list[str],
        expected_revisions: dict[str, int],
        idempotency_key: str,
    ) -> SupersessionResult:
        _validate_identifier(session_id, "session ID")
        _validate_identifier(newer_document_id, "replacement document ID")
        _validate_identifier(idempotency_key, "idempotency key")
        if not older_document_ids or len(older_document_ids) > MAX_TOP_K:
            raise ValueError(f"older_document_ids must contain 1-{MAX_TOP_K} documents")
        if len(set(older_document_ids)) != len(older_document_ids):
            raise ValueError("older_document_ids must be unique")
        for document_id in older_document_ids:
            _validate_identifier(document_id, "superseded document ID")
        if set(expected_revisions) != set(older_document_ids):
            raise ValueError("Provide one expected revision for each superseded document")
        for revision in expected_revisions.values():
            _validate_expected_revision(revision)
        ordered_ids = sorted(older_document_ids)
        payload_hash = _payload_hash(
            {
                "operation": "supersede_context",
                "newer_document_id": newer_document_id,
                "older_document_ids": ordered_ids,
                "expected_revisions": {
                    document_id: expected_revisions[document_id] for document_id in ordered_ids
                },
            }
        )
        receipt: SupersessionReceipt = await asyncio.to_thread(
            self._database.supersede_context,
            provider=provider,
            session_id=session_id,
            newer_document_id=newer_document_id,
            older_document_ids=ordered_ids,
            expected_revisions=expected_revisions,
            idempotency_key=idempotency_key,
            payload_hash=payload_hash,
        )
        return SupersessionResult(
            newer_document_id=receipt.newer_document_id,
            superseded_document_ids=list(receipt.superseded_document_ids),
        )

    async def assemble_context(
        self,
        *,
        provider: str,
        session_id: str,
        task_id: str,
        query: str,
        max_chars: int = DEFAULT_ASSEMBLY_CHARACTERS,
    ) -> ContextAssembly:
        _validate_identifier(session_id, "session ID")
        _validate_bounded_text(task_id, MAX_TASK_ID_CHARACTERS, "task ID")
        if len(query) > MAX_QUERY_CHARACTERS:
            raise ValueError(f"query exceeds the {MAX_QUERY_CHARACTERS}-character limit")
        if not 1 <= max_chars <= MAX_ASSEMBLY_CHARACTERS:
            raise ValueError(f"max_chars must be between 1 and {MAX_ASSEMBLY_CHARACTERS}")
        checkpoint = await asyncio.to_thread(
            self._database.get_checkpoint, provider, session_id, task_id
        )
        evidence_response = await self.search_context(
            provider=provider,
            session_id=session_id,
            query=query,
            mode="hybrid",
            top_k=10,
            record_types=["decision", "constraint", "finding", "unresolved", "lesson"],
        )
        return _assemble_context(session_id, task_id, checkpoint, evidence_response, max_chars)


def _session_result(session: SessionInfo) -> SessionResult:
    return SessionResult(
        session_id=session.session_id,
        provider=session.provider,
        label=session.label,
        provider_space_id=session.provider_space_id,
        session_space_id=session.session_space_id,
        created_at=session.created_at,
        last_seen_at=session.last_seen_at,
    )


def _space_result(space: SpaceInfo) -> SpaceResult:
    return SpaceResult(
        space_id=space.space_id,
        kind=space.kind,  # type: ignore[arg-type]
        name=space.name,
        project_space_id=space.project_space_id,
        owner_session_id=space.owner_session_id,
        owner_provider=space.owner_provider,
        archived_at=space.archived_at,
        members=[
            SpaceMemberResult(
                session_id=member.session_id,
                provider=member.provider,
                role=member.role,  # type: ignore[arg-type]
            )
            for member in space.members
        ],
    )


def _invitation_result(invitation: SpaceInvitationInfo) -> SpaceInvitationResult:
    return SpaceInvitationResult(
        invitation_id=invitation.invitation_id,
        space_id=invitation.space_id,
        space_name=invitation.space_name,
        owner_session_id=invitation.owner_session_id,
        inviter_session_id=invitation.inviter_session_id,
        invited_session_id=invitation.invited_session_id,
        status=invitation.status,  # type: ignore[arg-type]
        created_at=invitation.created_at,
    )


def _write_result(receipt: WriteReceipt, space_id: str | None) -> ContextWriteResult:
    return ContextWriteResult(
        document_id=receipt.document_id,
        revision=receipt.revision,
        space_id=receipt.space_id or space_id or "",
    )


def _checkpoint_result(receipt: WriteReceipt) -> CheckpointWriteResult:
    if receipt.head_revision is None:
        raise RuntimeError("Checkpoint write did not return a head revision")
    return CheckpointWriteResult(
        document_id=receipt.document_id,
        document_revision=receipt.revision,
        head_revision=receipt.head_revision,
    )


def _validate_identifier(value: str, label: str) -> None:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} must be 1-128 ASCII letters, digits, or . _ : - characters")


def _validate_bounded_text(value: str, limit: int, label: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{label} must not be empty")
    if len(value) > limit:
        raise ValueError(f"{label} exceeds the {limit}-character limit")


def _validate_content(content: str) -> None:
    if not content or not content.strip():
        raise ValueError("content must not be empty")
    if "\x00" in content:
        raise ValueError("content must not contain NUL characters")
    if len(content.encode("utf-8")) > MAX_RECORD_BYTES:
        raise ValueError(f"content exceeds the {MAX_RECORD_BYTES}-byte limit")


def _validate_expected_revision(expected_revision: int | None) -> None:
    if expected_revision is not None and expected_revision < 1:
        raise ValueError("expected_revision must be positive")


def _validate_record_type(record_type: str) -> None:
    if record_type not in {
        "note",
        "decision",
        "constraint",
        "finding",
        "unresolved",
        "lesson",
        "checkpoint",
        "handoff",
    }:
        raise ValueError("record_type is not supported")


def _encode_metadata(metadata: Metadata) -> str:
    try:
        encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must contain only finite JSON scalar values") from exc
    if len(encoded.encode("utf-8")) > MAX_METADATA_BYTES:
        raise ValueError(f"metadata exceeds the {MAX_METADATA_BYTES}-byte limit")
    return encoded


def _payload_hash(payload: dict[str, object]) -> str:
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _chunk_content(content: str) -> list[ChunkInput]:
    chunks: list[ChunkInput] = []
    start = 0
    while start < len(content):
        end = min(start + CHUNK_CHARACTERS, len(content))
        if end < len(content):
            boundary = content.rfind(" ", start + (CHUNK_CHARACTERS * 2 // 3), end)
            if boundary > start:
                end = boundary
        if end <= start:
            end = min(start + CHUNK_CHARACTERS, len(content))
        chunk_text = content[start:end]
        chunks.append(
            ChunkInput(
                text=chunk_text,
                char_start=start,
                char_end=end,
                content_hash=hashlib.sha256(chunk_text.encode("utf-8")).hexdigest(),
            )
        )
        if end == len(content):
            break
        start = max(start + 1, end - CHUNK_OVERLAP_CHARACTERS)
        while start < end and content[start].isspace():
            start += 1
        if len(chunks) > MAX_CHUNKS_PER_RECORD:
            raise ValueError(f"content creates more than {MAX_CHUNKS_PER_RECORD} chunks")
    return chunks


def _fts_match_query(query: str) -> str:
    terms = list(dict.fromkeys(_QUERY_TERM.findall(query)))
    return " OR ".join(f'"{term}"' for term in terms)


def _reciprocal_rank_fusion(
    keyword: list[SearchCandidate],
    semantic: list[SearchCandidate],
    top_k: int,
) -> list[tuple[SearchCandidate, float]]:
    scores: dict[str, float] = {}
    candidates: dict[str, SearchCandidate] = {}
    for ranking in (keyword, semantic):
        for rank, candidate in enumerate(ranking, start=1):
            candidates.setdefault(candidate.document_id, candidate)
            scores[candidate.document_id] = scores.get(candidate.document_id, 0.0) + (
                1.0 / (60 + rank)
            )
    ranked = sorted(scores, key=lambda document_id: (-scores[document_id], document_id))
    return [(candidates[document_id], scores[document_id]) for document_id in ranked[:top_k]]


def _make_search_hits(
    candidates: list[tuple[SearchCandidate, float]], top_k: int
) -> tuple[list[SearchHit], bool, int]:
    hits: list[SearchHit] = []
    total_characters = 0
    truncated = False
    for candidate, score in candidates[:top_k]:
        remaining = MAX_COMBINED_SNIPPET_CHARACTERS - total_characters
        if remaining <= 0:
            truncated = True
            break
        snippet = candidate.chunk_text[: min(MAX_SNIPPET_CHARACTERS, remaining)]
        if len(snippet) < len(candidate.chunk_text):
            truncated = True
        hits.append(
            SearchHit(
                document_id=candidate.document_id,
                space_id=candidate.space_id,
                revision=candidate.revision,
                title=candidate.title,
                record_type=candidate.record_type,  # type: ignore[arg-type]
                snippet=snippet,
                char_start=candidate.char_start,
                char_end=candidate.char_start + len(snippet),
                source_ref=candidate.source_ref,
                author_provider=candidate.author_provider,
                author_session_id=candidate.author_session_id,
                created_at=candidate.created_at,
                score=score,
                status=candidate.status,  # type: ignore[arg-type]
                supersedes=list(candidate.supersedes),
                superseded_by=list(candidate.superseded_by),
            )
        )
        total_characters += len(snippet)
    return hits, truncated, total_characters


def _context_record(
    record: RecordSnapshot, offset: int, max_chars: int
) -> ContextRecord:
    if offset > len(record.content):
        raise ValueError("offset exceeds the context record length")
    end = min(offset + max_chars, len(record.content))
    return ContextRecord(
        document_id=record.document_id,
        space_id=record.space_id,
        revision=record.revision,
        title=record.title,
        content=record.content[offset:end],
        record_type=record.record_type,  # type: ignore[arg-type]
        metadata=json.loads(record.metadata_json),
        source_ref=record.source_ref,
        author_provider=record.author_provider,
        author_session_id=record.author_session_id,
        created_at=record.created_at,
        task_id=record.task_id,
        status=record.status,  # type: ignore[arg-type]
        supersedes=list(record.supersedes),
        superseded_by=list(record.superseded_by),
        offset=offset,
        next_offset=end if end < len(record.content) else None,
        truncated=end < len(record.content),
    )


def _assemble_context(
    session_id: str,
    task_id: str,
    checkpoint: CheckpointSnapshot | None,
    evidence_response: SearchResponse,
    max_chars: int,
) -> ContextAssembly:
    remaining = max_chars
    checkpoint_result: AssembledCheckpoint | None = None
    evidence: list[SearchHit] = []
    truncated = evidence_response.truncated
    total_characters = 0
    if checkpoint is not None:
        content = checkpoint.record.content[:remaining]
        checkpoint_truncated = len(content) < len(checkpoint.record.content)
        checkpoint_result = AssembledCheckpoint(
            document_id=checkpoint.record.document_id,
            document_revision=checkpoint.record.revision,
            head_revision=checkpoint.head_revision,
            title=checkpoint.record.title,
            content=content,
            truncated=checkpoint_truncated,
        )
        remaining -= len(content)
        total_characters += len(content)
        truncated = truncated or checkpoint_truncated

    for hit in evidence_response.results:
        if remaining <= 0:
            truncated = True
            break
        snippet = hit.snippet[:remaining]
        evidence.append(hit.model_copy(update={"snippet": snippet}))
        remaining -= len(snippet)
        total_characters += len(snippet)
        if len(snippet) < len(hit.snippet):
            truncated = True
            break

    return ContextAssembly(
        session_id=session_id,
        task_id=task_id,
        checkpoint=checkpoint_result,
        missing_checkpoint=checkpoint is None,
        evidence=evidence,
        truncated=truncated,
        total_characters=total_characters,
    )
