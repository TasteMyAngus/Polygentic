from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path

import pytest

from context_server.memory import MemoryService
from context_server.storage import (
    MemoryAccessError,
    MemoryConflictError,
    MemoryDatabase,
    MemoryNotFoundError,
)


class FakeEmbedder:
    async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text: str) -> tuple[float, ...]:
        values = [0.0] * 384
        terms = set(re.findall(r"[a-z0-9]+", text.lower()))
        if terms.intersection({"launch", "window", "release", "schedule", "deadline"}):
            values[0] = 1.0
        for term in terms:
            index = int.from_bytes(hashlib.sha256(term.encode()).digest()[:4], "big") % 383 + 1
            values[index] += 1.0
        if not any(values):
            values[1] = 1.0
        return tuple(values)


def _service(data_dir: Path | None = None) -> tuple[MemoryDatabase, MemoryService]:
    database = MemoryDatabase(data_dir)
    return database, MemoryService(database, FakeEmbedder())


@pytest.mark.asyncio
async def test_session_registration_is_idempotent_and_provider_bound() -> None:
    database, memory = _service()
    try:
        first = await memory.register_session("codex", "Planning", "register-1")
        replay = await memory.register_session("codex", "Planning", "register-1")
        assert replay.session_id == first.session_id
        assert replay.session_space_id == first.session_space_id
        assert len(await memory.list_sessions("codex")) == 1

        with pytest.raises(MemoryConflictError, match="idempotency key"):
            await memory.register_session("codex", "Different label", "register-1")
        with pytest.raises(MemoryNotFoundError, match="authenticated provider"):
            await memory.resume_session("claude", first.session_id)
    finally:
        database.close()


@pytest.mark.asyncio
async def test_keyword_search_validates_session_without_searchable_terms() -> None:
    database, memory = _service()
    try:
        with pytest.raises(MemoryNotFoundError, match="authenticated provider"):
            await memory.search_context(
                provider="codex",
                session_id="unknown-session",
                query="✨",
                mode="keyword",
            )
    finally:
        database.close()


@pytest.mark.asyncio
async def test_cross_provider_search_is_readable_but_writes_remain_owned() -> None:
    database, memory = _service()
    try:
        codex = await memory.register_session("codex", "Writer", "register-codex")
        record = await memory.put_context(
            provider="codex",
            session_id=codex.session_id,
            title="Launch decision",
            content="The launch date moved to June after the readiness review.",
            record_type="decision",
            idempotency_key="write-1",
            metadata={"approved": True},
            source_ref="release-review",
        )
        replay = await memory.put_context(
            provider="codex",
            session_id=codex.session_id,
            title="Launch decision",
            content="The launch date moved to June after the readiness review.",
            record_type="decision",
            idempotency_key="write-1",
            metadata={"approved": True},
            source_ref="release-review",
        )
        assert replay.document_id == record.document_id
        assert replay.revision == 1
        assert replay.space_id == codex.session_space_id

        claude = await memory.register_session("claude", "Reader", "register-claude")
        results = await memory.search_context(
            provider="claude",
            session_id=claude.session_id,
            query="launch date June",
            mode="keyword",
        )
        assert results.results[0].document_id == record.document_id
        assert results.results[0].author_provider == "codex"

        with pytest.raises(MemoryAccessError, match="Write access"):
            await memory.put_context(
                provider="claude",
                session_id=claude.session_id,
                title="Unauthorized replacement",
                content="This must not replace the record.",
                record_type="decision",
                idempotency_key="write-unauthorized",
                document_id=record.document_id,
                expected_revision=1,
            )
        with pytest.raises(MemoryConflictError, match="idempotency key"):
            await memory.put_context(
                provider="codex",
                session_id=codex.session_id,
                title="Changed payload",
                content="Different content with the same key.",
                record_type="decision",
                idempotency_key="write-1",
            )
    finally:
        database.close()


@pytest.mark.asyncio
async def test_semantic_search_revision_conflicts_and_record_pagination() -> None:
    database, memory = _service()
    try:
        session = await memory.register_session("codex", "Research", "register-2")
        long_text = "The launch window was moved to June. " + ("supporting detail " * 1800)
        record = await memory.put_context(
            provider="codex",
            session_id=session.session_id,
            title="Schedule update",
            content=long_text,
            record_type="finding",
            idempotency_key="write-long",
        )
        semantic = await memory.search_context(
            provider="codex",
            session_id=session.session_id,
            query="release schedule",
            mode="semantic",
        )
        assert semantic.results[0].document_id == record.document_id

        first_page = await memory.get_context(
            "codex", session.session_id, record.document_id, 0, 1000
        )
        assert first_page.content == long_text[:1000]
        assert first_page.next_offset == 1000
        second_page = await memory.get_context(
            "codex", session.session_id, record.document_id, 1000, 1000
        )
        assert second_page.content == long_text[1000:2000]

        updated = await memory.put_context(
            provider="codex",
            session_id=session.session_id,
            title="Schedule update",
            content="The release schedule is now confirmed for July.",
            record_type="finding",
            idempotency_key="update-1",
            document_id=record.document_id,
            expected_revision=1,
        )
        assert updated.revision == 2
        with pytest.raises(MemoryConflictError, match="stale"):
            await memory.put_context(
                provider="codex",
                session_id=session.session_id,
                title="Stale write",
                content="This stale update must fail.",
                record_type="finding",
                idempotency_key="update-stale",
                document_id=record.document_id,
                expected_revision=1,
            )
    finally:
        database.close()


@pytest.mark.asyncio
async def test_checkpoint_head_is_atomic_idempotent_and_exactly_assembled() -> None:
    database, memory = _service()
    try:
        session = await memory.register_session("claude", "Implementation", "register-3")
        await memory.put_context(
            provider="claude",
            session_id=session.session_id,
            title="Decision",
            content="The launch date is June 12.",
            record_type="decision",
            idempotency_key="decision-1",
        )
        first = await memory.put_checkpoint(
            provider="claude",
            session_id=session.session_id,
            task_id="memory-slice",
            content="Objective: implement memory. Next: verify retrieval tests.",
            expected_head_revision=0,
            idempotency_key="checkpoint-1",
        )
        replay = await memory.put_checkpoint(
            provider="claude",
            session_id=session.session_id,
            task_id="memory-slice",
            content="Objective: implement memory. Next: verify retrieval tests.",
            expected_head_revision=0,
            idempotency_key="checkpoint-1",
        )
        assert replay == first
        assert first.head_revision == 1

        updated = await memory.put_checkpoint(
            provider="claude",
            session_id=session.session_id,
            task_id="memory-slice",
            content="Objective: implement memory. Done: persistence. Next: verify retrieval.",
            expected_head_revision=1,
            idempotency_key="checkpoint-2",
        )
        assert updated.head_revision == 2
        with pytest.raises(MemoryConflictError, match="stale"):
            await memory.put_checkpoint(
                provider="claude",
                session_id=session.session_id,
                task_id="memory-slice",
                content="Stale checkpoint.",
                expected_head_revision=1,
                idempotency_key="checkpoint-stale",
            )

        assembled = await memory.assemble_context(
            provider="claude",
            session_id=session.session_id,
            task_id="memory-slice",
            query="launch date",
            max_chars=500,
        )
        assert assembled.checkpoint is not None
        assert assembled.checkpoint.head_revision == 2
        assert "Done: persistence" in assembled.checkpoint.content
        assert assembled.evidence[0].document_id
        assert assembled.total_characters <= 500
    finally:
        database.close()


@pytest.mark.asyncio
async def test_persistent_store_reopens_with_sessions_and_vectors(tmp_path: Path) -> None:
    data_dir = tmp_path / "memory"
    first_database, first_memory = _service(data_dir)
    session = await first_memory.register_session("codex", "Persistent", "register-4")
    record = await first_memory.put_context(
        provider="codex",
        session_id=session.session_id,
        title="Persistent note",
        content="This note remains after the database connection is reopened.",
        record_type="note",
        idempotency_key="persistent-write",
    )
    first_database.close()

    second_database, second_memory = _service(data_dir)
    try:
        resumed = await second_memory.resume_session("codex", session.session_id)
        restored = await second_memory.get_context(
            "codex", resumed.session_id, record.document_id, 0, 1000
        )
        assert restored.content == "This note remains after the database connection is reopened."
        assert restored.revision == 1
    finally:
        second_database.close()
