from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import cast
from uuid import uuid4

import sqlite_vec  # type: ignore[import-untyped]

from context_server.embeddings import EMBEDDING_DIMENSIONS

_SCHEMA_VERSION = 2
_MAX_TOP_K = 50


class MemoryStorageError(RuntimeError):
    """Raised when the persistent memory store cannot safely serve a request."""


class MemoryConflictError(RuntimeError):
    """Raised when an idempotency key or expected revision conflicts."""


class MemoryNotFoundError(RuntimeError):
    """Raised when a session or context record is not available to the caller."""


class MemoryAccessError(RuntimeError):
    """Raised when a caller attempts to write outside its owned spaces."""


@dataclass(frozen=True)
class SessionInfo:
    session_id: str
    provider: str
    label: str
    provider_space_id: str
    session_space_id: str
    created_at: str
    last_seen_at: str


@dataclass(frozen=True)
class WriteReceipt:
    document_id: str
    revision: int
    head_revision: int | None = None
    space_id: str | None = None


@dataclass(frozen=True)
class SupersessionReceipt:
    newer_document_id: str
    superseded_document_ids: tuple[str, ...]


@dataclass(frozen=True)
class ChunkInput:
    text: str
    char_start: int
    char_end: int
    content_hash: str


@dataclass(frozen=True)
class RecordSnapshot:
    document_id: str
    space_id: str
    revision: int
    title: str
    content: str
    record_type: str
    metadata_json: str
    source_ref: str | None
    author_provider: str
    author_session_id: str
    created_at: str
    task_id: str | None
    status: str
    supersedes: tuple[str, ...] = ()
    superseded_by: tuple[str, ...] = ()


@dataclass(frozen=True)
class CheckpointSnapshot:
    head_revision: int
    record: RecordSnapshot


@dataclass(frozen=True)
class SearchCandidate:
    document_id: str
    space_id: str
    revision: int
    title: str
    record_type: str
    metadata_json: str
    source_ref: str | None
    author_provider: str
    author_session_id: str
    created_at: str
    chunk_text: str
    char_start: int
    char_end: int
    metric: float
    status: str
    supersedes: tuple[str, ...]
    superseded_by: tuple[str, ...]


class MemoryDatabase:
    def __init__(self, data_dir: Path | None = None) -> None:
        self._lock = RLock()
        self._closed = False
        self._database_path: Path | None = None
        if data_dir is None:
            database = ":memory:"
        else:
            self._prepare_data_dir(data_dir)
            self._database_path = data_dir / "context.sqlite3"
            if self._database_path.is_symlink():
                raise MemoryStorageError("Memory database must not be a symbolic link")
            database = str(self._database_path)

        try:
            self._connection = sqlite3.connect(
                database,
                timeout=10,
                check_same_thread=False,
                isolation_level=None,
            )
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA busy_timeout = 10000")
            self._connection.enable_load_extension(True)
            sqlite_vec.load(self._connection)
            self._connection.enable_load_extension(False)
            if self._database_path is not None:
                journal_mode = self._connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
                if journal_mode.lower() != "wal":
                    raise MemoryStorageError("SQLite could not enable WAL journaling")
                self._connection.execute("PRAGMA synchronous = FULL")
            self._initialize_schema()
            if self._database_path is not None and os.name == "posix":
                os.chmod(self._database_path, 0o600)
        except Exception:
            if hasattr(self, "_connection"):
                self._connection.close()
            raise

    @staticmethod
    def _prepare_data_dir(data_dir: Path) -> None:
        if data_dir.is_symlink():
            raise MemoryStorageError("Memory data directory must not be a symbolic link")
        data_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not data_dir.is_dir():
            raise MemoryStorageError("Memory data path must be a directory")
        if os.name == "posix":
            directory_stat = data_dir.stat()
            if directory_stat.st_uid != os.getuid():
                raise MemoryStorageError("Memory data directory must be owned by the service user")
            if stat.S_IMODE(directory_stat.st_mode) & 0o077:
                raise MemoryStorageError("Memory data directory permissions must be owner-only")

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._ensure_open()
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
            except Exception:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    def _ensure_open(self) -> None:
        if self._closed:
            raise MemoryStorageError("Memory database is closed")

    def _initialize_schema(self) -> None:
        current_version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
        if current_version > _SCHEMA_VERSION:
            raise MemoryStorageError(
                f"Memory database schema {current_version} is newer than supported "
                f"schema {_SCHEMA_VERSION}"
            )
        if current_version == _SCHEMA_VERSION:
            return
        if current_version == 1:
            self._migrate_v1_to_v2()
            return
        self._connection.executescript(
            """
            BEGIN IMMEDIATE;
            CREATE TABLE spaces (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL CHECK (kind IN ('provider', 'session')),
                provider TEXT NOT NULL,
                owner_session_id TEXT,
                created_at TEXT NOT NULL,
                CHECK (
                    (kind = 'provider' AND owner_session_id IS NULL) OR
                    (kind = 'session' AND owner_session_id IS NOT NULL)
                )
            );
            CREATE UNIQUE INDEX provider_space_owner
                ON spaces(provider) WHERE kind = 'provider';
            CREATE UNIQUE INDEX session_space_owner
                ON spaces(owner_session_id) WHERE kind = 'session';

            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                label TEXT NOT NULL,
                provider_space_id TEXT NOT NULL REFERENCES spaces(id),
                session_space_id TEXT NOT NULL REFERENCES spaces(id),
                registration_key TEXT NOT NULL,
                registration_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                UNIQUE(provider, registration_key)
            );
            CREATE INDEX sessions_by_provider
                ON sessions(provider, last_seen_at DESC);

            CREATE TABLE documents (
                id TEXT PRIMARY KEY,
                space_id TEXT NOT NULL REFERENCES spaces(id),
                current_revision INTEGER NOT NULL CHECK (current_revision > 0),
                status TEXT NOT NULL CHECK (status IN ('active', 'superseded', 'deleted')),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE document_revisions (
                document_id TEXT NOT NULL REFERENCES documents(id),
                revision INTEGER NOT NULL CHECK (revision > 0),
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                record_type TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                source_ref TEXT,
                author_provider TEXT NOT NULL,
                author_session_id TEXT NOT NULL REFERENCES sessions(id),
                task_id TEXT,
                created_at TEXT NOT NULL,
                PRIMARY KEY(document_id, revision)
            );
            CREATE INDEX document_revisions_by_author
                ON document_revisions(author_provider, author_session_id, created_at DESC);
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                revision INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
                char_start INTEGER NOT NULL CHECK (char_start >= 0),
                char_end INTEGER NOT NULL CHECK (char_end >= char_start),
                chunk_text TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                model_name TEXT NOT NULL,
                UNIQUE(document_id, revision, chunk_index),
                FOREIGN KEY(document_id, revision)
                    REFERENCES document_revisions(document_id, revision)
            );
            CREATE VIRTUAL TABLE chunk_fts USING fts5(title, chunk_text, tokenize = 'unicode61');
            CREATE VIRTUAL TABLE chunk_vectors
                USING vec0(embedding float[384]);

            CREATE TABLE write_requests (
                provider TEXT NOT NULL,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                idempotency_key TEXT NOT NULL,
                operation TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                document_id TEXT NOT NULL REFERENCES documents(id),
                space_id TEXT NOT NULL REFERENCES spaces(id),
                document_revision INTEGER NOT NULL,
                head_revision INTEGER,
                created_at TEXT NOT NULL,
                PRIMARY KEY(provider, session_id, idempotency_key)
            );
            CREATE TABLE checkpoint_heads (
                session_id TEXT NOT NULL REFERENCES sessions(id),
                task_id TEXT NOT NULL,
                head_revision INTEGER NOT NULL CHECK (head_revision > 0),
                document_id TEXT NOT NULL,
                document_revision INTEGER NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(session_id, task_id),
                FOREIGN KEY(document_id, document_revision)
                    REFERENCES document_revisions(document_id, revision)
            );
            PRAGMA user_version = 1;
            COMMIT;
            """
        )
        self._migrate_v1_to_v2()

    def _migrate_v1_to_v2(self) -> None:
        self._connection.executescript(
            """
            BEGIN IMMEDIATE;
            CREATE TABLE IF NOT EXISTS supersession_edges (
                newer_document_id TEXT NOT NULL REFERENCES documents(id),
                older_document_id TEXT NOT NULL REFERENCES documents(id),
                created_by_provider TEXT NOT NULL,
                created_by_session_id TEXT NOT NULL REFERENCES sessions(id),
                created_at TEXT NOT NULL,
                PRIMARY KEY(newer_document_id, older_document_id),
                CHECK (newer_document_id <> older_document_id)
            );
            CREATE INDEX IF NOT EXISTS supersession_edges_by_older
                ON supersession_edges(older_document_id, newer_document_id);

            CREATE TABLE IF NOT EXISTS document_status_events (
                id INTEGER PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES documents(id),
                content_revision INTEGER NOT NULL,
                prior_status TEXT NOT NULL CHECK (prior_status IN ('active', 'superseded')),
                new_status TEXT NOT NULL CHECK (new_status IN ('superseded', 'deleted')),
                actor_provider TEXT NOT NULL,
                actor_session_id TEXT NOT NULL REFERENCES sessions(id),
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS document_status_events_by_document
                ON document_status_events(document_id, id);

            CREATE TABLE IF NOT EXISTS supersession_requests (
                provider TEXT NOT NULL,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                idempotency_key TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(provider, session_id, idempotency_key)
            );
            PRAGMA user_version = 2;
            COMMIT;
            """
        )

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat(timespec="microseconds")

    @staticmethod
    def _session_info(row: sqlite3.Row) -> SessionInfo:
        return SessionInfo(
            session_id=row["id"],
            provider=row["provider"],
            label=row["label"],
            provider_space_id=row["provider_space_id"],
            session_space_id=row["session_space_id"],
            created_at=row["created_at"],
            last_seen_at=row["last_seen_at"],
        )

    def _get_session(
        self, connection: sqlite3.Connection, provider: str, session_id: str
    ) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM sessions WHERE id = ? AND provider = ?",
            (session_id, provider),
        ).fetchone()
        if row is None:
            raise MemoryNotFoundError("Session not found for authenticated provider")
        return cast(sqlite3.Row, row)

    def register_session(self, provider: str, label: str, idempotency_key: str) -> SessionInfo:
        registration_hash = hashlib.sha256(
            json.dumps({"label": label}, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM sessions WHERE provider = ? AND registration_key = ?",
                (provider, idempotency_key),
            ).fetchone()
            if existing is not None:
                if existing["registration_hash"] != registration_hash:
                    raise MemoryConflictError("Registration idempotency key was reused")
                return self._session_info(existing)

            provider_space = connection.execute(
                "SELECT id FROM spaces WHERE provider = ? AND kind = 'provider'",
                (provider,),
            ).fetchone()
            provider_space_id = (
                str(provider_space["id"]) if provider_space is not None else str(uuid4())
            )
            session_id = str(uuid4())
            session_space_id = str(uuid4())
            now = self._now()
            if provider_space is None:
                connection.execute(
                    "INSERT INTO spaces(id, kind, provider, created_at) VALUES (?, 'provider', ?, ?)",
                    (provider_space_id, provider, now),
                )
            connection.execute(
                "INSERT INTO spaces(id, kind, provider, owner_session_id, created_at) "
                "VALUES (?, 'session', ?, ?, ?)",
                (session_space_id, provider, session_id, now),
            )
            connection.execute(
                """
                INSERT INTO sessions(
                    id, provider, label, provider_space_id, session_space_id,
                    registration_key, registration_hash, created_at, last_seen_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    provider,
                    label,
                    provider_space_id,
                    session_space_id,
                    idempotency_key,
                    registration_hash,
                    now,
                    now,
                ),
            )
            return SessionInfo(
                session_id=session_id,
                provider=provider,
                label=label,
                provider_space_id=provider_space_id,
                session_space_id=session_space_id,
                created_at=now,
                last_seen_at=now,
            )

    def list_sessions(self, provider: str) -> list[SessionInfo]:
        with self._lock:
            self._ensure_open()
            rows = self._connection.execute(
                "SELECT * FROM sessions WHERE provider = ? ORDER BY last_seen_at DESC, id",
                (provider,),
            ).fetchall()
            return [self._session_info(row) for row in rows]

    def resume_session(self, provider: str, session_id: str) -> SessionInfo:
        with self._transaction() as connection:
            row = self._get_session(connection, provider, session_id)
            now = self._now()
            connection.execute(
                "UPDATE sessions SET last_seen_at = ? WHERE id = ?",
                (now, session_id),
            )
            updated = dict(row)
            updated["last_seen_at"] = now
            return SessionInfo(
                session_id=str(updated["id"]),
                provider=str(updated["provider"]),
                label=str(updated["label"]),
                provider_space_id=str(updated["provider_space_id"]),
                session_space_id=str(updated["session_space_id"]),
                created_at=str(updated["created_at"]),
                last_seen_at=now,
            )

    def session_space(self, provider: str, session_id: str) -> str:
        with self._lock:
            self._ensure_open()
            row = self._get_session(self._connection, provider, session_id)
            return str(row["session_space_id"])

    def _write_space(
        self,
        connection: sqlite3.Connection,
        provider: str,
        session: sqlite3.Row,
        requested_space_id: str | None,
    ) -> str:
        space_id = requested_space_id or str(session["session_space_id"])
        row = connection.execute(
            "SELECT * FROM spaces WHERE id = ?",
            (space_id,),
        ).fetchone()
        if row is None or row["provider"] != provider:
            raise MemoryAccessError("Write access to the requested space is not allowed")
        if row["kind"] == "session" and row["owner_session_id"] != session["id"]:
            raise MemoryAccessError("Write access to the requested space is not allowed")
        return space_id

    def _idempotent_receipt(
        self,
        connection: sqlite3.Connection,
        provider: str,
        session_id: str,
        idempotency_key: str,
        operation: str,
        payload_hash: str,
    ) -> WriteReceipt | None:
        row = connection.execute(
            """
            SELECT operation, payload_hash, document_id, document_revision, head_revision, space_id
            FROM write_requests
            WHERE provider = ? AND session_id = ? AND idempotency_key = ?
            """,
            (provider, session_id, idempotency_key),
        ).fetchone()
        supersession_request = connection.execute(
            """
            SELECT 1 FROM supersession_requests
            WHERE provider = ? AND session_id = ? AND idempotency_key = ?
            """,
            (provider, session_id, idempotency_key),
        ).fetchone()
        if supersession_request is not None:
            raise MemoryConflictError("Write idempotency key was reused for another operation")
        if row is None:
            return None
        if row["operation"] != operation or row["payload_hash"] != payload_hash:
            raise MemoryConflictError("Write idempotency key was reused with a different request")
        return WriteReceipt(
            document_id=row["document_id"],
            revision=row["document_revision"],
            head_revision=row["head_revision"],
            space_id=row["space_id"],
        )

    def prepare_put(
        self,
        provider: str,
        session_id: str,
        requested_space_id: str | None,
        document_id: str | None,
        expected_revision: int | None,
        idempotency_key: str,
        payload_hash: str,
    ) -> tuple[str | None, WriteReceipt | None]:
        with self._lock:
            self._ensure_open()
            session = self._get_session(self._connection, provider, session_id)
            receipt = self._idempotent_receipt(
                self._connection,
                provider,
                session_id,
                idempotency_key,
                "put_context",
                payload_hash,
            )
            if receipt is not None:
                return None, receipt

            if document_id is None:
                if expected_revision is not None:
                    raise MemoryConflictError("A new record cannot specify an expected revision")
                return self._write_space(
                    self._connection, provider, session, requested_space_id
                ), None

            document = self._connection.execute(
                "SELECT space_id, current_revision, status FROM documents WHERE id = ?",
                (document_id,),
            ).fetchone()
            if document is None or document["status"] != "active":
                raise MemoryNotFoundError("Context record not found")
            resolved_space = self._write_space(
                self._connection, provider, session, str(document["space_id"])
            )
            if requested_space_id is not None and requested_space_id != resolved_space:
                raise MemoryConflictError("A context record cannot move between spaces")
            if expected_revision is None or expected_revision != document["current_revision"]:
                raise MemoryConflictError("Context record revision is stale")
            return resolved_space, None

    def put_context(
        self,
        *,
        provider: str,
        session_id: str,
        space_id: str,
        document_id: str,
        is_new: bool,
        expected_revision: int | None,
        idempotency_key: str,
        payload_hash: str,
        title: str,
        content: str,
        record_type: str,
        metadata_json: str,
        source_ref: str | None,
        chunks: Sequence[ChunkInput],
        vectors: Sequence[Sequence[float]],
    ) -> WriteReceipt:
        with self._transaction() as connection:
            session = self._get_session(connection, provider, session_id)
            receipt = self._idempotent_receipt(
                connection, provider, session_id, idempotency_key, "put_context", payload_hash
            )
            if receipt is not None:
                return receipt
            self._write_space(connection, provider, session, space_id)
            if is_new:
                if expected_revision is not None:
                    raise MemoryConflictError("A new record cannot specify an expected revision")
                revision = 1
                now = self._now()
                connection.execute(
                    "INSERT INTO documents(id, space_id, current_revision, status, created_at, "
                    "updated_at) VALUES (?, ?, 1, 'active', ?, ?)",
                    (document_id, space_id, now, now),
                )
            else:
                existing = connection.execute(
                    "SELECT space_id, current_revision, status FROM documents WHERE id = ?",
                    (document_id,),
                ).fetchone()
                if existing is None or existing["status"] != "active":
                    raise MemoryNotFoundError("Context record not found")
                if existing["space_id"] != space_id:
                    raise MemoryConflictError("A context record cannot move between spaces")
                if (
                    expected_revision is None
                    or existing["current_revision"] != expected_revision
                ):
                    raise MemoryConflictError("Context record revision is stale")
                revision = int(existing["current_revision"]) + 1
                now = self._now()
                connection.execute(
                    "UPDATE documents SET current_revision = ?, updated_at = ? WHERE id = ?",
                    (revision, now, document_id),
                )
            self._insert_revision(
                connection,
                document_id=document_id,
                revision=revision,
                title=title,
                content=content,
                record_type=record_type,
                metadata_json=metadata_json,
                source_ref=source_ref,
                provider=provider,
                session_id=session_id,
                task_id=None,
                chunks=chunks,
                vectors=vectors,
                created_at=now,
            )
            connection.execute(
                """
                INSERT INTO write_requests(
                    provider, session_id, idempotency_key, operation, payload_hash,
                    document_id, space_id, document_revision, created_at
                ) VALUES (?, ?, ?, 'put_context', ?, ?, ?, ?, ?)
                """,
                (
                    provider,
                    session_id,
                    idempotency_key,
                    payload_hash,
                    document_id,
                    space_id,
                    revision,
                    now,
                ),
            )
            return WriteReceipt(document_id, revision, space_id=space_id)

    def prepare_checkpoint(
        self,
        provider: str,
        session_id: str,
        task_id: str,
        expected_head_revision: int,
        idempotency_key: str,
        payload_hash: str,
    ) -> tuple[str | None, WriteReceipt | None]:
        with self._lock:
            self._ensure_open()
            session = self._get_session(self._connection, provider, session_id)
            receipt = self._idempotent_receipt(
                self._connection,
                provider,
                session_id,
                idempotency_key,
                "put_checkpoint",
                payload_hash,
            )
            if receipt is not None:
                return None, receipt
            row = self._connection.execute(
                "SELECT head_revision FROM checkpoint_heads WHERE session_id = ? AND task_id = ?",
                (session_id, task_id),
            ).fetchone()
            actual_revision = int(row["head_revision"]) if row is not None else 0
            if actual_revision != expected_head_revision:
                raise MemoryConflictError("Checkpoint head revision is stale")
            return str(session["session_space_id"]), None

    def put_checkpoint(
        self,
        *,
        provider: str,
        session_id: str,
        task_id: str,
        expected_head_revision: int,
        document_id: str,
        idempotency_key: str,
        payload_hash: str,
        title: str,
        content: str,
        metadata_json: str,
        chunks: Sequence[ChunkInput],
        vectors: Sequence[Sequence[float]],
    ) -> WriteReceipt:
        with self._transaction() as connection:
            session = self._get_session(connection, provider, session_id)
            receipt = self._idempotent_receipt(
                connection, provider, session_id, idempotency_key, "put_checkpoint", payload_hash
            )
            if receipt is not None:
                return receipt
            current = connection.execute(
                "SELECT head_revision FROM checkpoint_heads WHERE session_id = ? AND task_id = ?",
                (session_id, task_id),
            ).fetchone()
            actual_revision = int(current["head_revision"]) if current is not None else 0
            if actual_revision != expected_head_revision:
                raise MemoryConflictError("Checkpoint head revision is stale")

            now = self._now()
            connection.execute(
                "INSERT INTO documents(id, space_id, current_revision, status, created_at, "
                "updated_at) VALUES (?, ?, 1, 'active', ?, ?)",
                (document_id, session["session_space_id"], now, now),
            )
            self._insert_revision(
                connection,
                document_id=document_id,
                revision=1,
                title=title,
                content=content,
                record_type="checkpoint",
                metadata_json=metadata_json,
                source_ref=None,
                provider=provider,
                session_id=session_id,
                task_id=task_id,
                chunks=chunks,
                vectors=vectors,
                created_at=now,
            )
            head_revision = actual_revision + 1
            connection.execute(
                """
                INSERT INTO checkpoint_heads(
                    session_id, task_id, head_revision, document_id, document_revision, updated_at
                ) VALUES (?, ?, ?, ?, 1, ?)
                ON CONFLICT(session_id, task_id) DO UPDATE SET
                    head_revision = excluded.head_revision,
                    document_id = excluded.document_id,
                    document_revision = excluded.document_revision,
                    updated_at = excluded.updated_at
                """,
                (session_id, task_id, head_revision, document_id, now),
            )
            connection.execute(
                """
                INSERT INTO write_requests(
                    provider, session_id, idempotency_key, operation, payload_hash,
                    document_id, space_id, document_revision, head_revision, created_at
                ) VALUES (?, ?, ?, 'put_checkpoint', ?, ?, ?, 1, ?, ?)
                """,
                (
                    provider,
                    session_id,
                    idempotency_key,
                    payload_hash,
                    document_id,
                    session["session_space_id"],
                    head_revision,
                    now,
                ),
            )
            return WriteReceipt(
                document_id,
                1,
                head_revision,
                str(session["session_space_id"]),
            )

    def supersede_context(
        self,
        *,
        provider: str,
        session_id: str,
        newer_document_id: str,
        older_document_ids: Sequence[str],
        expected_revisions: dict[str, int],
        idempotency_key: str,
        payload_hash: str,
    ) -> SupersessionReceipt:
        if not older_document_ids or len(older_document_ids) > _MAX_TOP_K:
            raise ValueError(f"older_document_ids must contain 1-{_MAX_TOP_K} documents")
        if len(set(older_document_ids)) != len(older_document_ids):
            raise ValueError("older_document_ids must be unique")
        if set(expected_revisions) != set(older_document_ids):
            raise ValueError("An expected revision is required for every superseded document")
        if newer_document_id in older_document_ids:
            raise ValueError("A document cannot supersede itself")

        with self._transaction() as connection:
            session = self._get_session(connection, provider, session_id)
            existing = connection.execute(
                """
                SELECT payload_hash, result_json FROM supersession_requests
                WHERE provider = ? AND session_id = ? AND idempotency_key = ?
                """,
                (provider, session_id, idempotency_key),
            ).fetchone()
            write_request = connection.execute(
                """
                SELECT 1 FROM write_requests
                WHERE provider = ? AND session_id = ? AND idempotency_key = ?
                """,
                (provider, session_id, idempotency_key),
            ).fetchone()
            if write_request is not None:
                raise MemoryConflictError("Write idempotency key was reused for another operation")
            if existing is not None:
                if existing["payload_hash"] != payload_hash:
                    raise MemoryConflictError(
                        "Supersession idempotency key was reused with a different request"
                    )
                try:
                    result = json.loads(existing["result_json"])
                    return SupersessionReceipt(
                        newer_document_id=result["newer_document_id"],
                        superseded_document_ids=tuple(result["superseded_document_ids"]),
                    )
                except (KeyError, TypeError, json.JSONDecodeError) as exc:
                    raise MemoryStorageError("Stored supersession result is invalid") from exc

            newer = connection.execute(
                """
                SELECT d.space_id, d.status, r.record_type
                FROM documents AS d
                JOIN document_revisions AS r
                  ON r.document_id = d.id AND r.revision = d.current_revision
                WHERE d.id = ?
                """,
                (newer_document_id,),
            ).fetchone()
            if newer is None or newer["status"] == "deleted":
                raise MemoryNotFoundError("Replacement context record not found")
            if newer["record_type"] == "checkpoint":
                raise MemoryConflictError("Task checkpoints cannot participate in supersession")
            space_id = self._write_space(
                connection, provider, session, str(newer["space_id"])
            )

            older_rows: list[sqlite3.Row] = []
            for older_document_id in older_document_ids:
                older = connection.execute(
                    """
                    SELECT d.space_id, d.current_revision, d.status, r.record_type
                    FROM documents AS d
                    JOIN document_revisions AS r
                      ON r.document_id = d.id AND r.revision = d.current_revision
                    WHERE d.id = ?
                    """,
                    (older_document_id,),
                ).fetchone()
                if older is None or older["status"] == "deleted":
                    raise MemoryNotFoundError("Superseded context record not found")
                if older["record_type"] == "checkpoint":
                    raise MemoryConflictError("Task checkpoints cannot participate in supersession")
                older_space = self._write_space(
                    connection, provider, session, str(older["space_id"])
                )
                if older_space != space_id:
                    raise MemoryConflictError("Supersession records must be in the same space")
                if expected_revisions[older_document_id] != older["current_revision"]:
                    raise MemoryConflictError("Superseded record revision is stale")
                edge_exists = connection.execute(
                    """
                    SELECT 1 FROM supersession_edges
                    WHERE newer_document_id = ? AND older_document_id = ?
                    """,
                    (newer_document_id, older_document_id),
                ).fetchone()
                if edge_exists is not None:
                    raise MemoryConflictError("Supersession edge already exists")
                cycle = connection.execute(
                    """
                    WITH RECURSIVE reachable(document_id) AS (
                        SELECT ?
                        UNION
                        SELECT edge.older_document_id
                        FROM supersession_edges AS edge
                        JOIN reachable ON edge.newer_document_id = reachable.document_id
                    )
                    SELECT 1 FROM reachable WHERE document_id = ? LIMIT 1
                    """,
                    (older_document_id, newer_document_id),
                ).fetchone()
                if cycle is not None:
                    raise MemoryConflictError("Supersession would create a cycle")
                older_rows.append(older)

            now = self._now()
            for older_document_id, older in zip(older_document_ids, older_rows, strict=True):
                connection.execute(
                    """
                    INSERT INTO supersession_edges(
                        newer_document_id, older_document_id, created_by_provider,
                        created_by_session_id, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (newer_document_id, older_document_id, provider, session_id, now),
                )
                if older["status"] == "active":
                    connection.execute(
                        "UPDATE documents SET status = 'superseded', updated_at = ? WHERE id = ?",
                        (now, older_document_id),
                    )
                    connection.execute(
                        """
                        INSERT INTO document_status_events(
                            document_id, content_revision, prior_status, new_status,
                            actor_provider, actor_session_id, created_at
                        ) VALUES (?, ?, 'active', 'superseded', ?, ?, ?)
                        """,
                        (
                            older_document_id,
                            older["current_revision"],
                            provider,
                            session_id,
                            now,
                        ),
                    )

            receipt = SupersessionReceipt(
                newer_document_id=newer_document_id,
                superseded_document_ids=tuple(older_document_ids),
            )
            result_json = json.dumps(
                {
                    "newer_document_id": receipt.newer_document_id,
                    "superseded_document_ids": receipt.superseded_document_ids,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            connection.execute(
                """
                INSERT INTO supersession_requests(
                    provider, session_id, idempotency_key, payload_hash, result_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (provider, session_id, idempotency_key, payload_hash, result_json, now),
            )
            return receipt

    def _insert_revision(
        self,
        connection: sqlite3.Connection,
        *,
        document_id: str,
        revision: int,
        title: str,
        content: str,
        record_type: str,
        metadata_json: str,
        source_ref: str | None,
        provider: str,
        session_id: str,
        task_id: str | None,
        chunks: Sequence[ChunkInput],
        vectors: Sequence[Sequence[float]],
        created_at: str,
    ) -> None:
        if len(chunks) != len(vectors) or not chunks:
            raise MemoryStorageError("Every stored chunk must have exactly one embedding")
        connection.execute(
            """
            INSERT INTO document_revisions(
                document_id, revision, title, content, record_type, metadata_json,
                source_ref, author_provider, author_session_id, task_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                revision,
                title,
                content,
                record_type,
                metadata_json,
                source_ref,
                provider,
                session_id,
                task_id,
                created_at,
            ),
        )
        for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
            cursor = connection.execute(
                """
                INSERT INTO chunks(
                    document_id, revision, chunk_index, char_start, char_end,
                    chunk_text, content_hash, model_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    revision,
                    index,
                    chunk.char_start,
                    chunk.char_end,
                    chunk.text,
                    chunk.content_hash,
                    "BAAI/bge-small-en-v1.5",
                ),
            )
            if cursor.lastrowid is None:
                raise MemoryStorageError("SQLite did not assign a chunk identifier")
            chunk_id = cursor.lastrowid
            connection.execute(
                "INSERT INTO chunk_fts(rowid, title, chunk_text) VALUES (?, ?, ?)",
                (chunk_id, title, chunk.text),
            )
            connection.execute(
                "INSERT INTO chunk_vectors(rowid, embedding) VALUES (?, ?)",
                (chunk_id, sqlite_vec.serialize_float32(vector)),
            )

    def get_record(
        self,
        provider: str,
        session_id: str,
        document_id: str,
        include_superseded: bool = False,
    ) -> RecordSnapshot:
        with self._lock:
            self._ensure_open()
            self._get_session(self._connection, provider, session_id)
            status_filter = (
                "d.status IN ('active', 'superseded')"
                if include_superseded
                else "d.status = 'active'"
            )
            row = self._connection.execute(
                f"""
                SELECT d.id, d.space_id, r.revision, r.title, r.content, r.record_type,
                       r.metadata_json, r.source_ref, r.author_provider, r.author_session_id,
                       r.created_at, r.task_id, d.status
                FROM documents AS d
                JOIN document_revisions AS r
                  ON r.document_id = d.id AND r.revision = d.current_revision
                JOIN spaces AS s ON s.id = d.space_id
                WHERE d.id = ? AND {status_filter}
                  AND s.kind IN ('provider', 'session')
                """,
                (document_id,),
            ).fetchone()
            if row is None:
                raise MemoryNotFoundError("Context record not found")
            snapshot = self._record_snapshot(row)
            supersedes, superseded_by = self._supersession_references(
                self._connection, document_id
            )
            return replace(
                snapshot,
                supersedes=supersedes,
                superseded_by=superseded_by,
            )

    def get_checkpoint(
        self, provider: str, session_id: str, task_id: str
    ) -> CheckpointSnapshot | None:
        with self._lock:
            self._ensure_open()
            self._get_session(self._connection, provider, session_id)
            row = self._connection.execute(
                """
                SELECT h.head_revision, d.status, d.space_id, d.id, r.revision, r.title, r.content,
                       r.record_type, r.metadata_json, r.source_ref, r.author_provider,
                       r.author_session_id, r.created_at, r.task_id
                FROM checkpoint_heads AS h
                JOIN documents AS d ON d.id = h.document_id AND d.status = 'active'
                JOIN document_revisions AS r
                  ON r.document_id = h.document_id AND r.revision = h.document_revision
                JOIN spaces AS s ON s.id = d.space_id
                WHERE h.session_id = ? AND h.task_id = ? AND s.kind = 'session'
                """,
                (session_id, task_id),
            ).fetchone()
            if row is None:
                return None
            return CheckpointSnapshot(
                head_revision=int(row["head_revision"]),
                record=self._record_snapshot(row),
            )

    @staticmethod
    def _record_snapshot(row: sqlite3.Row) -> RecordSnapshot:
        return RecordSnapshot(
            document_id=row["id"],
            space_id=row["space_id"],
            revision=row["revision"],
            title=row["title"],
            content=row["content"],
            record_type=row["record_type"],
            metadata_json=row["metadata_json"],
            source_ref=row["source_ref"],
            author_provider=row["author_provider"],
            author_session_id=row["author_session_id"],
            created_at=row["created_at"],
            task_id=row["task_id"],
            status=row["status"],
        )

    @staticmethod
    def _supersession_references(
        connection: sqlite3.Connection, document_id: str
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        supersedes = connection.execute(
            "SELECT older_document_id FROM supersession_edges "
            "WHERE newer_document_id = ? ORDER BY older_document_id",
            (document_id,),
        ).fetchall()
        superseded_by = connection.execute(
            "SELECT newer_document_id FROM supersession_edges "
            "WHERE older_document_id = ? ORDER BY newer_document_id",
            (document_id,),
        ).fetchall()
        return (
            tuple(str(row["older_document_id"]) for row in supersedes),
            tuple(str(row["newer_document_id"]) for row in superseded_by),
        )

    def search_keyword(
        self,
        provider: str,
        session_id: str,
        match_query: str,
        limit: int,
        space_ids: Sequence[str] | None,
        record_types: Sequence[str] | None,
        include_superseded: bool = False,
    ) -> list[SearchCandidate]:
        if not match_query:
            return []
        with self._lock:
            self._ensure_open()
            self._get_session(self._connection, provider, session_id)
            where, filters = self._search_filters(space_ids, record_types)
            status_filter = (
                "d.status IN ('active', 'superseded')"
                if include_superseded
                else "d.status = 'active'"
            )
            sql = f"""
                SELECT c.id AS chunk_id, d.id AS document_id, d.space_id, r.revision,
                       r.title, r.record_type, r.metadata_json, r.source_ref,
                       r.author_provider, r.author_session_id, r.created_at,
                       c.chunk_text, c.char_start, c.char_end,
                       d.status,
                       (SELECT group_concat(edge.older_document_id, ',')
                        FROM supersession_edges AS edge
                        WHERE edge.newer_document_id = d.id) AS supersedes,
                       (SELECT group_concat(edge.newer_document_id, ',')
                        FROM supersession_edges AS edge
                        WHERE edge.older_document_id = d.id) AS superseded_by,
                       bm25(chunk_fts) AS metric
                FROM chunk_fts
                JOIN chunks AS c ON c.id = chunk_fts.rowid
                JOIN documents AS d ON d.id = c.document_id
                JOIN document_revisions AS r
                  ON r.document_id = c.document_id AND r.revision = c.revision
                JOIN spaces AS s ON s.id = d.space_id
                WHERE chunk_fts MATCH ? AND d.current_revision = c.revision
                  AND {status_filter} AND s.kind IN ('provider', 'session')
                  AND {where}
                ORDER BY metric ASC, c.id ASC
                LIMIT ?
            """
            rows = self._connection.execute(
                sql, (match_query, *filters, limit * 20)
            ).fetchall()
            return self._deduplicate_candidates(rows, limit)

    def search_semantic(
        self,
        provider: str,
        session_id: str,
        query_vector: Sequence[float],
        limit: int,
        space_ids: Sequence[str] | None,
        record_types: Sequence[str] | None,
        include_superseded: bool = False,
    ) -> list[SearchCandidate]:
        if len(query_vector) != EMBEDDING_DIMENSIONS:
            raise MemoryStorageError("Query embedding has an incompatible dimension")
        with self._lock:
            self._ensure_open()
            self._get_session(self._connection, provider, session_id)
            where, filters = self._search_filters(space_ids, record_types)
            vector = sqlite_vec.serialize_float32(query_vector)
            status_filter = (
                "d.status IN ('active', 'superseded')"
                if include_superseded
                else "d.status = 'active'"
            )
            sql = f"""
                WITH distances AS (
                    SELECT c.id AS chunk_id, d.id AS document_id, d.space_id, r.revision,
                           r.title, r.record_type, r.metadata_json, r.source_ref,
                           r.author_provider, r.author_session_id, r.created_at,
                           c.chunk_text, c.char_start, c.char_end,
                           d.status,
                           (SELECT group_concat(edge.older_document_id, ',')
                            FROM supersession_edges AS edge
                            WHERE edge.newer_document_id = d.id) AS supersedes,
                           (SELECT group_concat(edge.newer_document_id, ',')
                            FROM supersession_edges AS edge
                            WHERE edge.older_document_id = d.id) AS superseded_by,
                           vec_distance_cosine(v.embedding, ?) AS metric
                    FROM chunk_vectors AS v
                    JOIN chunks AS c ON c.id = v.rowid
                    JOIN documents AS d ON d.id = c.document_id
                    JOIN document_revisions AS r
                      ON r.document_id = c.document_id AND r.revision = c.revision
                    JOIN spaces AS s ON s.id = d.space_id
                    WHERE d.current_revision = c.revision AND {status_filter}
                      AND s.kind IN ('provider', 'session') AND {where}
                ),
                ranked AS (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY document_id ORDER BY metric ASC, chunk_id ASC
                    ) AS document_rank
                    FROM distances
                )
                SELECT * FROM ranked
                WHERE document_rank = 1
                ORDER BY metric ASC, document_id ASC
                LIMIT ?
            """
            rows = self._connection.execute(sql, (vector, *filters, limit)).fetchall()
            return [self._search_candidate(row) for row in rows]

    @staticmethod
    def _search_filters(
        space_ids: Sequence[str] | None,
        record_types: Sequence[str] | None,
    ) -> tuple[str, tuple[object, ...]]:
        clauses: list[str] = []
        values: list[object] = []
        if space_ids is not None:
            if not space_ids:
                return "0", ()
            placeholders = ",".join("?" for _ in space_ids)
            clauses.append(f"d.space_id IN ({placeholders})")
            values.extend(space_ids)
        if record_types is None:
            clauses.append("r.record_type NOT IN ('checkpoint', 'handoff')")
        else:
            if not record_types:
                return "0", ()
            placeholders = ",".join("?" for _ in record_types)
            clauses.append(f"r.record_type IN ({placeholders})")
            values.extend(record_types)
        return " AND ".join(clauses) if clauses else "1", tuple(values)

    @classmethod
    def _deduplicate_candidates(
        cls, rows: Sequence[sqlite3.Row], limit: int
    ) -> list[SearchCandidate]:
        candidates: dict[str, SearchCandidate] = {}
        for row in rows:
            document_id = str(row["document_id"])
            if document_id not in candidates:
                candidates[document_id] = cls._search_candidate(row)
                if len(candidates) >= limit:
                    break
        return list(candidates.values())

    @staticmethod
    def _search_candidate(row: sqlite3.Row) -> SearchCandidate:
        return SearchCandidate(
            document_id=row["document_id"],
            space_id=row["space_id"],
            revision=row["revision"],
            title=row["title"],
            record_type=row["record_type"],
            metadata_json=row["metadata_json"],
            source_ref=row["source_ref"],
            author_provider=row["author_provider"],
            author_session_id=row["author_session_id"],
            created_at=row["created_at"],
            chunk_text=row["chunk_text"],
            char_start=row["char_start"],
            char_end=row["char_end"],
            metric=float(row["metric"]),
            status=row["status"],
            supersedes=tuple(sorted((row["supersedes"] or "").split(",")))
            if row["supersedes"]
            else (),
            superseded_by=tuple(sorted((row["superseded_by"] or "").split(",")))
            if row["superseded_by"]
            else (),
        )
