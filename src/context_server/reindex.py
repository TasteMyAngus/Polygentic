from __future__ import annotations

import argparse
import asyncio
import os
import tempfile
from pathlib import Path

from platformdirs import user_cache_path, user_data_path

from context_server.config import EMBEDDING_MODEL
from context_server.embeddings import (
    MAX_EMBEDDING_BATCH_SIZE,
    Embedder,
    EmbeddingError,
    FastEmbedder,
)
from context_server.storage import MemoryConflictError, MemoryDatabase, MemoryStorageError


async def reindex_database(
    data_dir: Path,
    destination: Path,
    model_name: str,
    embedder: Embedder,
    *,
    overwrite: bool = False,
) -> Path:
    if not model_name.strip() or len(model_name) > 256:
        raise MemoryStorageError("Embedding model name must be 1-256 characters")
    data_dir = data_dir.expanduser()
    database_path = data_dir / "context.sqlite3"
    if not database_path.is_file():
        raise MemoryStorageError(f"No context database found in {data_dir}")
    output_path = destination.expanduser().absolute()
    if output_path.resolve() == database_path.resolve():
        raise MemoryStorageError("Reindex output must differ from the live database")

    source = MemoryDatabase(data_dir)
    try:
        with tempfile.TemporaryDirectory(prefix=".context-reindex-", dir=data_dir) as temporary_dir:
            work_dir = Path(temporary_dir)
            snapshot_path = work_dir / "source.sqlite3"
            source.backup_to(snapshot_path)
            source.close()
            MemoryDatabase.require_stopped(data_dir)

            staging_dir = work_dir / "staging"
            MemoryDatabase.restore_from_backup(snapshot_path, staging_dir)
            staged = MemoryDatabase(staging_dir)
            try:
                chunks = staged.chunks_for_reindex()
                rebuilt: list[tuple[int, tuple[float, ...]]] = []
                for start in range(0, len(chunks), MAX_EMBEDDING_BATCH_SIZE):
                    batch = chunks[start : start + MAX_EMBEDDING_BATCH_SIZE]
                    vectors = await embedder.embed([text for _, text in batch])
                    if len(vectors) != len(batch):
                        raise EmbeddingError(
                            "Embedding model returned an unexpected result count during reindex"
                        )
                    rebuilt.extend(
                        (chunk_id, vector)
                        for (chunk_id, _), vector in zip(batch, vectors, strict=True)
                    )
                staged.replace_chunk_embeddings(rebuilt, model_name)
                staged.validate_embedding_model(model_name)
                return staged.backup_to(output_path, overwrite=overwrite)
            finally:
                staged.close()
    finally:
        source.close()


def reindex_main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a validated, owner-only database backup with regenerated embeddings."
    )
    parser.add_argument("destination", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--model-cache-dir", type=Path)
    parser.add_argument("--model-name")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.data_dir is not None:
        data_dir = args.data_dir.expanduser()
    elif os.environ.get("CONTEXT_SERVER_DATA_DIR"):
        data_dir = Path(os.environ["CONTEXT_SERVER_DATA_DIR"]).expanduser()
    else:
        data_dir = user_data_path("Polygentic", appauthor=False)
    if args.model_cache_dir is not None:
        model_cache_dir = args.model_cache_dir.expanduser()
    elif os.environ.get("CONTEXT_SERVER_MODEL_CACHE_DIR"):
        model_cache_dir = Path(os.environ["CONTEXT_SERVER_MODEL_CACHE_DIR"]).expanduser()
    else:
        model_cache_dir = user_cache_path("Polygentic", appauthor=False) / "models"
    model_name = (
        args.model_name
        if args.model_name is not None
        else os.environ.get("CONTEXT_SERVER_EMBEDDING_MODEL", EMBEDDING_MODEL)
    )
    try:
        output_path = asyncio.run(
            reindex_database(
                data_dir,
                args.destination,
                model_name,
                FastEmbedder(model_cache_dir, model_name),
                overwrite=args.overwrite,
            )
        )
    except (EmbeddingError, MemoryConflictError, MemoryStorageError, OSError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(output_path)
