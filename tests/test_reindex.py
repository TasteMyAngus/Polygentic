from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from context_server.config import ProviderCredential, Settings
from context_server.embeddings import EMBEDDING_DIMENSIONS
from context_server.memory import MemoryService
from context_server.reindex import reindex_database
from context_server.server import create_app
from context_server.storage import MemoryConflictError, MemoryDatabase, MemoryStorageError


class AxisEmbedder:
    def __init__(self, alpha_axis: int) -> None:
        self._alpha_axis = alpha_axis

    async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        vectors = []
        for text in texts:
            axis = self._alpha_axis if "alpha" in text.lower() else 1 - self._alpha_axis
            vector = [0.0] * EMBEDDING_DIMENSIONS
            vector[axis] = 1.0
            vectors.append(tuple(vector))
        return vectors


class InvalidEmbedder:
    async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return [(1.0,) for _ in texts]


async def _create_indexed_database(data_dir: Path) -> tuple[MemoryDatabase, str, str]:
    database = MemoryDatabase(data_dir)
    memory = MemoryService(database, AxisEmbedder(alpha_axis=0), "old-model")
    session = await memory.register_session("codex", "Reindex test", "reindex-registration")
    alpha = await memory.put_context(
        provider="codex",
        session_id=session.session_id,
        title="Alpha decision",
        content="The alpha decision is to keep the blue deployment path.",
        record_type="decision",
        idempotency_key="write-alpha",
    )
    await memory.put_context(
        provider="codex",
        session_id=session.session_id,
        title="Beta decision",
        content="The beta decision is to keep the green deployment path.",
        record_type="decision",
        idempotency_key="write-beta",
    )
    return database, session.session_id, alpha.document_id


@pytest.mark.asyncio
async def test_reindex_builds_a_validated_copy_and_preserves_original(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "source"
    database, session_id, alpha_document_id = await _create_indexed_database(data_dir)
    original_memory = MemoryService(database, AxisEmbedder(alpha_axis=0), "old-model")
    original_search = await original_memory.search_context(
        provider="codex", session_id=session_id, query="alpha", mode="semantic", top_k=2
    )
    assert original_search.results[0].document_id == alpha_document_id
    database.close()

    reindexed_path = tmp_path / "reindexed.sqlite3"
    assert (
        await reindex_database(data_dir, reindexed_path, "new-model", AxisEmbedder(alpha_axis=1))
        == reindexed_path
    )
    assert reindexed_path.stat().st_mode & 0o777 == 0o600

    original = MemoryDatabase(data_dir)
    try:
        original.validate_embedding_model("old-model")
    finally:
        original.close()

    restored_dir = tmp_path / "restored"
    MemoryDatabase.restore_from_backup(reindexed_path, restored_dir)
    restored = MemoryDatabase(restored_dir)
    try:
        restored.validate_embedding_model("new-model")
        with pytest.raises(MemoryStorageError, match="not 'old-model'"):
            restored.validate_embedding_model("old-model")
        memory = MemoryService(restored, AxisEmbedder(alpha_axis=1), "new-model")
        results = await memory.search_context(
            provider="codex", session_id=session_id, query="alpha", mode="semantic", top_k=2
        )
        assert results.results[0].document_id == alpha_document_id
    finally:
        restored.close()


@pytest.mark.asyncio
async def test_reindex_failure_leaves_source_unchanged_and_produces_no_output(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "source"
    database, _, _ = await _create_indexed_database(data_dir)
    database.close()
    reindexed_path = tmp_path / "invalid-reindex.sqlite3"

    with pytest.raises(MemoryStorageError, match="incompatible embedding"):
        await reindex_database(data_dir, reindexed_path, "invalid-model", InvalidEmbedder())

    assert not reindexed_path.exists()
    unchanged = MemoryDatabase(data_dir)
    try:
        unchanged.validate_embedding_model("old-model")
    finally:
        unchanged.close()


@pytest.mark.asyncio
async def test_reindex_refuses_an_open_server_database(tmp_path: Path) -> None:
    data_dir = tmp_path / "source"
    database, _, _ = await _create_indexed_database(data_dir)
    reindexed_path = tmp_path / "should-not-exist.sqlite3"
    try:
        with pytest.raises(MemoryConflictError, match="stop all server processes"):
            await reindex_database(
                data_dir, reindexed_path, "new-model", AxisEmbedder(alpha_axis=1)
            )
        assert not reindexed_path.exists()
    finally:
        database.close()


@pytest.mark.asyncio
async def test_server_rejects_a_model_that_does_not_match_stored_vectors(
    tmp_path: Path,
) -> None:
    database, _, _ = await _create_indexed_database(tmp_path / "memory")
    database.close()

    settings = Settings(
        credentials=(ProviderCredential("codex", "0" * 64),),
        data_dir=tmp_path / "memory",
        embedding_model="new-model",
    )
    with pytest.raises(MemoryStorageError, match="context-reindex"):
        create_app(settings)
