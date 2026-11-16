from __future__ import annotations

import asyncio
import math
import os
from collections.abc import Sequence
from pathlib import Path
from threading import Lock
from typing import Protocol

from fastembed import TextEmbedding

from context_server.config import EMBEDDING_MODEL

EMBEDDING_DIMENSIONS = 384
MAX_EMBEDDING_BATCH_SIZE = 256


class EmbeddingError(RuntimeError):
    """Raised when local embedding inference fails or returns invalid data."""


class Embedder(Protocol):
    async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]: ...


class FastEmbedder:
    def __init__(self, cache_dir: Path | None = None, model_name: str = EMBEDDING_MODEL) -> None:
        self._cache_dir = cache_dir
        self._model_name = model_name
        self._model: TextEmbedding | None = None
        self._lock = Lock()

    async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        if not texts:
            return []
        if len(texts) > MAX_EMBEDDING_BATCH_SIZE:
            raise EmbeddingError(
                f"Embedding batch exceeds the {MAX_EMBEDDING_BATCH_SIZE}-item limit"
            )
        return await asyncio.to_thread(self._embed_sync, tuple(texts))

    def _embed_sync(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        with self._lock:
            model = self._get_model()
            vectors = [
                tuple(float(value) for value in vector)
                for vector in model.embed(list(texts), batch_size=32)
            ]

        if len(vectors) != len(texts):
            raise EmbeddingError("Embedding model returned an unexpected result count")
        for vector in vectors:
            if len(vector) != EMBEDDING_DIMENSIONS or not all(math.isfinite(v) for v in vector):
                raise EmbeddingError("Embedding model returned an invalid vector")
        return vectors

    def _get_model(self) -> TextEmbedding:
        if self._model is not None:
            return self._model
        if self._cache_dir is not None:
            if self._cache_dir.is_symlink():
                raise EmbeddingError("Embedding cache directory must not be a symbolic link")
            self._cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            if os.name == "posix":
                os.chmod(self._cache_dir, 0o700)
        self._model = TextEmbedding(
            model_name=self._model_name,
            cache_dir=str(self._cache_dir) if self._cache_dir is not None else None,
            threads=2,
        )
        return self._model
