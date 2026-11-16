from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import stat
import tempfile
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

_SCHEMA_VERSION = 3
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
class SpaceMemberInfo:
    session_id: str
    provider: str
    role: str


@dataclass(frozen=True)
class SpaceInfo:
    space_id: str
    kind: str
    name: str
    project_space_id: str | None
    owner_session_id: str
    owner_provider: str
    archived_at: str | None
    members: tuple[SpaceMemberInfo, ...]


@dataclass(frozen=True)
class SpaceInvitationInfo:
    invitation_id: str
    space_id: str
    space_name: str
    owner_session_id: str
    inviter_session_id: str
    invited_session_id: str
    status: str
    created_at: str


@dataclass(frozen=True)
class RegisteredSessionInfo:
    session_id: str
    provider: str


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

    def validate_embedding_model(self, model_name: str) -> None:
        if not model_name.strip():
            raise MemoryStorageError("Embedding model name cannot be empty")
        with self._lock:
            self._ensure_open()
            stored_models = tuple(
                row[0]
                for row in self._connection.execute(
                    "SELECT DISTINCT model_name FROM chunks ORDER BY model_name"
                )
            )
        if len(stored_models) > 1:
            raise MemoryStorageError(
                "Memory database contains mixed embedding models; run context-reindex"
            )
        if stored_models and stored_models[0] != model_name:
            raise MemoryStorageError(
                f"Memory database uses embedding model '{stored_models[0]}', not "
                f"'{model_name}'; run context-reindex before starting the server"
            )

    def chunks_for_reindex(self) -> list[tuple[int, str]]:
        with self._lock:
            self._ensure_open()
            rows = self._connection.execute(
                "SELECT id, chunk_text FROM chunks ORDER BY id"
            ).fetchall()
            return [(int(row["id"]), str(row["chunk_text"])) for row in rows]

    def replace_chunk_embeddings(
        self,
        embeddings: Sequence[tuple[int, Sequence[float]]],
        model_name: str,
    ) -> None:
        if not model_name.strip():
            raise MemoryStorageError("Embedding model name cannot be empty")
        chunk_ids = tuple(chunk_id for chunk_id, _ in embeddings)
        if len(set(chunk_ids)) != len(chunk_ids):
            raise MemoryStorageError("Reindex result contains duplicate chunk identifiers")
        for _, vector in embeddings:
            if len(vector) != EMBEDDING_DIMENSIONS or not all(
                math.isfinite(value) for value in vector
            ):
                raise MemoryStorageError("Reindex produced an incompatible embedding vector")

        with self._transaction() as connection:
            stored_ids = tuple(
                int(row["id"]) for row in connection.execute("SELECT id FROM chunks ORDER BY id")
            )
            if stored_ids != tuple(sorted(chunk_ids)):
                raise MemoryConflictError(
                    "Chunk set changed during reindex; stop the server and retry"
                )
            connection.execute("DELETE FROM chunk_vectors")
            for chunk_id, vector in embeddings:
                connection.execute(
                    "INSERT INTO chunk_vectors(rowid, embedding) VALUES (?, ?)",
                    (chunk_id, sqlite_vec.serialize_float32(vector)),
                )
            connection.execute("UPDATE chunks SET model_name = ?", (model_name,))
            vector_count = int(
                connection.execute("SELECT count(*) FROM chunk_vectors").fetchone()[0]
            )
            if vector_count != len(stored_ids):
                raise MemoryStorageError("Reindex did not rebuild every chunk embedding")

    @classmethod
    def require_stopped(cls, data_dir: Path) -> None:
        database_path = data_dir / "context.sqlite3"
        if not database_path.exists():
            return
        try:
            current = sqlite3.connect(database_path)
            try:
                checkpoint = current.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                if checkpoint is None or checkpoint[0] != 0:
                    raise MemoryConflictError(
                        "Unable to checkpoint database; stop all server processes"
                    )
            finally:
                current.close()
        except sqlite3.Error as exc:
            raise MemoryStorageError(
                "Unable to prepare database; stop all server processes"
            ) from exc
        for suffix in ("-wal", "-shm"):
            sidecar = Path(f"{database_path}{suffix}")
            if sidecar.is_symlink():
                raise MemoryStorageError("Database sidecars must not be symbolic links")
            if sidecar.exists():
                raise MemoryConflictError("Database sidecars remain; stop all server processes")

    def backup_to(self, destination: Path, *, overwrite: bool = False) -> Path:
        if self._database_path is None:
            raise MemoryStorageError("An in-memory database cannot be backed up to a file")
        backup_path = destination.expanduser().absolute()
        if backup_path.resolve() == self._database_path.resolve():
            raise MemoryStorageError("Backup destination must differ from the live database")
        if backup_path.is_symlink():
            raise MemoryStorageError("Backup destination must not be a symbolic link")
        if backup_path.exists():
            backup_stat = backup_path.stat()
            if not stat.S_ISREG(backup_stat.st_mode):
                raise MemoryStorageError("Backup destination must be a regular file")
            if os.name == "posix" and backup_stat.st_uid != os.getuid():
                raise MemoryStorageError("Backup destination must be owned by the service user")
            if not overwrite:
                raise MemoryConflictError(
                    "Backup destination already exists; pass overwrite=True to replace it"
                )

        parent = backup_path.parent
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if parent.is_symlink() or not parent.is_dir():
            raise MemoryStorageError("Backup destination parent must be a directory")
        if os.name == "posix" and parent.stat().st_uid != os.getuid():
            raise MemoryStorageError("Backup destination parent must be owned by the service user")

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{backup_path.name}.",
                suffix=".tmp",
                dir=parent,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                os.chmod(temporary_path, 0o600)
            with self._lock:
                self._ensure_open()
                target = sqlite3.connect(temporary_path)
                try:
                    self._connection.backup(target)
                    target.enable_load_extension(True)
                    sqlite_vec.load(target)
                    target.enable_load_extension(False)
                    check = target.execute("PRAGMA quick_check").fetchone()
                    if check is None or check[0] != "ok":
                        raise MemoryStorageError("Database backup failed its integrity check")
                    if target.execute("PRAGMA foreign_key_check").fetchone() is not None:
                        raise MemoryStorageError("Database backup contains foreign-key violations")
                finally:
                    target.close()
            with temporary_path.open("rb") as backup_file:
                os.fsync(backup_file.fileno())
            os.replace(temporary_path, backup_path)
            temporary_path = None
            os.chmod(backup_path, 0o600)
            if os.name == "posix":
                parent_fd = os.open(parent, os.O_RDONLY)
                try:
                    os.fsync(parent_fd)
                finally:
                    os.close(parent_fd)
            return backup_path
        except sqlite3.Error as exc:
            raise MemoryStorageError("Unable to create a consistent SQLite backup") from exc
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @classmethod
    def restore_from_backup(
        cls,
        source: Path,
        data_dir: Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        backup_path = source.expanduser()
        if backup_path.is_symlink():
            raise MemoryStorageError("Backup source must not be a symbolic link")
        try:
            source_stat = backup_path.stat()
        except OSError as exc:
            raise MemoryStorageError("Unable to read database backup") from exc
        if not stat.S_ISREG(source_stat.st_mode):
            raise MemoryStorageError("Backup source must be a regular file")
        if os.name == "posix":
            if source_stat.st_uid != os.getuid():
                raise MemoryStorageError("Backup source must be owned by the service user")
            if stat.S_IMODE(source_stat.st_mode) & 0o077:
                raise MemoryStorageError("Backup source permissions must be owner-only")

        resolved_source = backup_path.resolve()
        cls._prepare_data_dir(data_dir)
        database_path = data_dir / "context.sqlite3"
        if resolved_source == database_path.resolve():
            raise MemoryStorageError("Restore source must differ from the destination database")
        if database_path.is_symlink():
            raise MemoryStorageError("Destination database must not be a symbolic link")
        if database_path.exists():
            database_stat = database_path.stat()
            if not stat.S_ISREG(database_stat.st_mode):
                raise MemoryStorageError("Destination database must be a regular file")
            if os.name == "posix" and database_stat.st_uid != os.getuid():
                raise MemoryStorageError("Destination database must be owned by the service user")
            if not overwrite:
                raise MemoryConflictError(
                    "Destination database already exists; pass overwrite=True to replace it"
                )

        try:
            with tempfile.TemporaryDirectory(
                prefix=".context-restore-", dir=data_dir
            ) as temporary_dir:
                stage_dir = Path(temporary_dir)
                cls._prepare_data_dir(stage_dir)
                staged_path = stage_dir / "context.sqlite3"
                ready_path = stage_dir / "restored.sqlite3"
                source_connection = sqlite3.connect(
                    f"{resolved_source.as_uri()}?mode=ro", uri=True
                )
                try:
                    schema_version = int(
                        source_connection.execute("PRAGMA user_version").fetchone()[0]
                    )
                    if not 1 <= schema_version <= _SCHEMA_VERSION:
                        raise MemoryStorageError("Backup has an unsupported database schema")
                    source_connection.enable_load_extension(True)
                    sqlite_vec.load(source_connection)
                    source_connection.enable_load_extension(False)
                    if source_connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                        raise MemoryStorageError("Backup failed its integrity check")
                    if (
                        source_connection.execute("PRAGMA foreign_key_check").fetchone()
                        is not None
                    ):
                        raise MemoryStorageError("Backup contains foreign-key violations")
                    staged_connection = sqlite3.connect(staged_path)
                    try:
                        source_connection.backup(staged_connection)
                    finally:
                        staged_connection.close()
                finally:
                    source_connection.close()

                staged = cls(stage_dir)
                try:
                    check = staged._connection.execute("PRAGMA quick_check").fetchone()
                    if check is None or check[0] != "ok":
                        raise MemoryStorageError("Restored database failed its integrity check")
                    if (
                        staged._connection.execute("PRAGMA foreign_key_check").fetchone()
                        is not None
                    ):
                        raise MemoryStorageError(
                            "Restored database contains foreign-key violations"
                        )
                    ready = sqlite3.connect(ready_path)
                    try:
                        staged._connection.backup(ready)
                        ready.enable_load_extension(True)
                        sqlite_vec.load(ready)
                        ready.enable_load_extension(False)
                        if ready.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                            raise MemoryStorageError(
                                "Final restored database failed its integrity check"
                            )
                    finally:
                        ready.close()
                finally:
                    staged.close()

                cls.require_stopped(data_dir)
                os.replace(ready_path, database_path)
                os.chmod(database_path, 0o600)
                if os.name == "posix":
                    directory_fd = os.open(data_dir, os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
        except sqlite3.DatabaseError as exc:
            raise MemoryStorageError("Backup is not a valid SQLite database") from exc
        except sqlite3.Error as exc:
            raise MemoryStorageError("Unable to validate or restore the database backup") from exc
        return database_path

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
            self._migrate_v2_to_v3()
            return
        if current_version == 2:
            self._migrate_v2_to_v3()
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
        self._migrate_v2_to_v3()

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

    def _migrate_v2_to_v3(self) -> None:
        self._connection.executescript(
            """
            BEGIN IMMEDIATE;
            ALTER TABLE spaces ADD COLUMN space_type TEXT NOT NULL DEFAULT 'session'
                CHECK (space_type IN ('provider', 'session', 'project', 'shared'));
            ALTER TABLE spaces ADD COLUMN name TEXT NOT NULL DEFAULT '';
            ALTER TABLE spaces ADD COLUMN archived_at TEXT;
            ALTER TABLE spaces ADD COLUMN project_space_id TEXT REFERENCES spaces(id)
                CHECK (project_space_id IS NULL OR space_type = 'shared');
            UPDATE spaces SET space_type = 'provider' WHERE kind = 'provider';
            DROP INDEX session_space_owner;
            CREATE UNIQUE INDEX session_space_owner
                ON spaces(owner_session_id) WHERE space_type = 'session';

            CREATE TABLE space_memberships (
                space_id TEXT NOT NULL REFERENCES spaces(id),
                session_id TEXT NOT NULL REFERENCES sessions(id),
                role TEXT NOT NULL CHECK (role IN ('owner', 'member')),
                added_by_session_id TEXT NOT NULL REFERENCES sessions(id),
                created_at TEXT NOT NULL,
                PRIMARY KEY(space_id, session_id)
            );
            CREATE INDEX space_memberships_by_session
                ON space_memberships(session_id, space_id);

            CREATE TABLE space_invitations (
                id TEXT PRIMARY KEY,
                space_id TEXT NOT NULL REFERENCES spaces(id),
                invited_session_id TEXT NOT NULL REFERENCES sessions(id),
                invited_by_session_id TEXT NOT NULL REFERENCES sessions(id),
                status TEXT NOT NULL CHECK (status IN ('pending', 'accepted', 'revoked', 'left')),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE UNIQUE INDEX one_pending_space_invitation
                ON space_invitations(space_id, invited_session_id)
                WHERE status = 'pending';
            CREATE INDEX space_invitations_by_target
                ON space_invitations(invited_session_id, status, created_at DESC);
            CREATE INDEX space_invitations_by_inviter
                ON space_invitations(invited_by_session_id, status, created_at DESC);

            CREATE TABLE space_membership_events (
                id INTEGER PRIMARY KEY,
                space_id TEXT NOT NULL REFERENCES spaces(id),
                subject_session_id TEXT NOT NULL REFERENCES sessions(id),
                actor_session_id TEXT NOT NULL REFERENCES sessions(id),
                event TEXT NOT NULL CHECK (
                    event IN ('created', 'invited', 'accepted', 'revoked', 'left', 'archived')
                ),
                created_at TEXT NOT NULL
            );
            CREATE INDEX space_membership_events_by_space
                ON space_membership_events(space_id, id);

            CREATE TABLE space_creation_requests (
                provider TEXT NOT NULL,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                idempotency_key TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                space_id TEXT NOT NULL REFERENCES spaces(id),
                created_at TEXT NOT NULL,
                PRIMARY KEY(provider, session_id, idempotency_key)
            );
            PRAGMA user_version = 3;
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
                    "INSERT INTO spaces(id, kind, provider, space_type, created_at) "
                    "VALUES (?, 'provider', ?, 'provider', ?)",
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

    @staticmethod
    def _space_info(
        connection: sqlite3.Connection, row: sqlite3.Row
    ) -> SpaceInfo:
        members = connection.execute(
            """
            SELECT m.session_id, s.provider, m.role
            FROM space_memberships AS m
            JOIN sessions AS s ON s.id = m.session_id
            WHERE m.space_id = ?
            ORDER BY CASE m.role WHEN 'owner' THEN 0 ELSE 1 END, m.session_id
            """,
            (row["id"],),
        ).fetchall()
        return SpaceInfo(
            space_id=str(row["id"]),
            kind=str(row["space_type"]),
            name=str(row["name"]),
            project_space_id=row["project_space_id"],
            owner_session_id=str(row["owner_session_id"]),
            owner_provider=str(row["provider"]),
            archived_at=row["archived_at"],
            members=tuple(
                SpaceMemberInfo(
                    session_id=str(member["session_id"]),
                    provider=str(member["provider"]),
                    role=str(member["role"]),
                )
                for member in members
            ),
        )

    @staticmethod
    def _invitation_info(row: sqlite3.Row) -> SpaceInvitationInfo:
        return SpaceInvitationInfo(
            invitation_id=str(row["id"]),
            space_id=str(row["space_id"]),
            space_name=str(row["name"]),
            owner_session_id=str(row["owner_session_id"]),
            inviter_session_id=str(row["invited_by_session_id"]),
            invited_session_id=str(row["invited_session_id"]),
            status=str(row["status"]),
            created_at=str(row["created_at"]),
        )

    def list_registered_sessions(self) -> list[RegisteredSessionInfo]:
        with self._lock:
            self._ensure_open()
            rows = self._connection.execute(
                "SELECT id, provider FROM sessions ORDER BY provider, id"
            ).fetchall()
            return [
                RegisteredSessionInfo(
                    session_id=str(row["id"]), provider=str(row["provider"])
                )
                for row in rows
            ]

    def create_space(
        self,
        provider: str,
        session_id: str,
        kind: str,
        name: str,
        idempotency_key: str,
        payload_hash: str,
        project_space_id: str | None = None,
    ) -> SpaceInfo:
        if kind not in {"project", "shared"}:
            raise ValueError("Shared space kind must be project or shared")
        if project_space_id is not None and kind != "shared":
            raise ValueError("Only ad-hoc shared spaces can reference a project space")
        with self._transaction() as connection:
            self._get_session(connection, provider, session_id)
            request = connection.execute(
                """
                SELECT payload_hash, space_id FROM space_creation_requests
                WHERE provider = ? AND session_id = ? AND idempotency_key = ?
                """,
                (provider, session_id, idempotency_key),
            ).fetchone()
            if request is not None:
                if request["payload_hash"] != payload_hash:
                    raise MemoryConflictError(
                        "Space creation idempotency key was reused with a different request"
                    )
                existing_space = connection.execute(
                    "SELECT * FROM spaces WHERE id = ?", (request["space_id"],)
                ).fetchone()
                if existing_space is None:
                    raise MemoryStorageError("Created shared space could not be read")
                return self._space_info(connection, existing_space)
            if project_space_id is not None:
                project = connection.execute(
                    """
                    SELECT 1 FROM spaces AS s
                    WHERE s.id = ? AND s.space_type = 'project' AND s.archived_at IS NULL
                      AND (
                        s.owner_session_id = ?
                        OR EXISTS (
                            SELECT 1 FROM space_memberships AS m
                            WHERE m.space_id = s.id AND m.session_id = ?
                        )
                      )
                    """,
                    (project_space_id, session_id, session_id),
                ).fetchone()
                if project is None:
                    raise MemoryNotFoundError("Project space not found")
            space_id = str(uuid4())
            now = self._now()
            connection.execute(
                """
                INSERT INTO spaces(
                    id, kind, provider, owner_session_id, space_type, name,
                    project_space_id, created_at
                ) VALUES (?, 'session', ?, ?, ?, ?, ?, ?)
                """,
                (
                    space_id,
                    provider,
                    session_id,
                    kind,
                    name,
                    project_space_id,
                    now,
                ),
            )
            connection.execute(
                """
                INSERT INTO space_memberships(
                    space_id, session_id, role, added_by_session_id, created_at
                ) VALUES (?, ?, 'owner', ?, ?)
                """,
                (space_id, session_id, session_id, now),
            )
            connection.execute(
                """
                INSERT INTO space_membership_events(
                    space_id, subject_session_id, actor_session_id, event, created_at
                ) VALUES (?, ?, ?, 'created', ?)
                """,
                (space_id, session_id, session_id, now),
            )
            connection.execute(
                """
                INSERT INTO space_creation_requests(
                    provider, session_id, idempotency_key, payload_hash, space_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (provider, session_id, idempotency_key, payload_hash, space_id, now),
            )
            row = connection.execute(
                "SELECT * FROM spaces WHERE id = ?", (space_id,)
            ).fetchone()
            if row is None:
                raise MemoryStorageError("Created shared space could not be read")
            return self._space_info(connection, row)

    def list_spaces(self, provider: str, session_id: str) -> list[SpaceInfo]:
        with self._lock:
            self._ensure_open()
            self._get_session(self._connection, provider, session_id)
            rows = self._connection.execute(
                """
                SELECT s.*
                FROM spaces AS s
                WHERE s.space_type IN ('project', 'shared')
                  AND (
                    (s.archived_at IS NULL AND EXISTS (
                        SELECT 1 FROM space_memberships AS m
                        WHERE m.space_id = s.id AND m.session_id = ?
                    ))
                    OR s.owner_session_id = ?
                  )
                ORDER BY s.name COLLATE NOCASE, s.id
                """,
                (session_id, session_id),
            ).fetchall()
            return [self._space_info(self._connection, row) for row in rows]

    def get_space(
        self, provider: str, session_id: str, space_id: str
    ) -> SpaceInfo:
        with self._lock:
            self._ensure_open()
            self._get_session(self._connection, provider, session_id)
            row = self._connection.execute(
                """
                SELECT s.*
                FROM spaces AS s
                WHERE s.id = ? AND s.space_type IN ('project', 'shared')
                  AND (
                    s.owner_session_id = ?
                    OR (s.archived_at IS NULL AND EXISTS (
                        SELECT 1 FROM space_memberships AS m
                        WHERE m.space_id = s.id AND m.session_id = ?
                    ))
                  )
                """,
                (space_id, session_id, session_id),
            ).fetchone()
            if row is None:
                raise MemoryNotFoundError("Shared space not found")
            return self._space_info(self._connection, row)

    def invite_session(
        self, provider: str, session_id: str, space_id: str, target_session_id: str
    ) -> SpaceInvitationInfo:
        with self._transaction() as connection:
            self._get_session(connection, provider, session_id)
            space = connection.execute(
                """
                SELECT * FROM spaces
                WHERE id = ? AND space_type IN ('project', 'shared')
                  AND owner_session_id = ? AND archived_at IS NULL
                """,
                (space_id, session_id),
            ).fetchone()
            if space is None:
                raise MemoryNotFoundError("Shared space not found")
            target = connection.execute(
                "SELECT id FROM sessions WHERE id = ?", (target_session_id,)
            ).fetchone()
            if target is None:
                raise MemoryNotFoundError("Invitation target session not found")
            member = connection.execute(
                "SELECT 1 FROM space_memberships WHERE space_id = ? AND session_id = ?",
                (space_id, target_session_id),
            ).fetchone()
            if member is not None:
                raise MemoryConflictError("Session is already a space member")
            existing = connection.execute(
                """
                SELECT * FROM space_invitations
                WHERE space_id = ? AND invited_session_id = ? AND status = 'pending'
                """,
                (space_id, target_session_id),
            ).fetchone()
            if existing is not None:
                return self._invitation_info(
                    self._join_invitation_space(connection, existing)
                )

            invitation_id = str(uuid4())
            now = self._now()
            connection.execute(
                """
                INSERT INTO space_invitations(
                    id, space_id, invited_session_id, invited_by_session_id,
                    status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'pending', ?, ?)
                """,
                (invitation_id, space_id, target_session_id, session_id, now, now),
            )
            connection.execute(
                """
                INSERT INTO space_membership_events(
                    space_id, subject_session_id, actor_session_id, event, created_at
                ) VALUES (?, ?, ?, 'invited', ?)
                """,
                (space_id, target_session_id, session_id, now),
            )
            row = connection.execute(
                "SELECT * FROM space_invitations WHERE id = ?", (invitation_id,)
            ).fetchone()
            if row is None:
                raise MemoryStorageError("Created invitation could not be read")
            return self._invitation_info(self._join_invitation_space(connection, row))

    @staticmethod
    def _join_invitation_space(
        connection: sqlite3.Connection, invitation: sqlite3.Row
    ) -> sqlite3.Row:
        row = connection.execute(
            """
            SELECT i.*, s.name, s.owner_session_id
            FROM space_invitations AS i
            JOIN spaces AS s ON s.id = i.space_id
            WHERE i.id = ?
            """,
            (invitation["id"],),
        ).fetchone()
        if row is None:
            raise MemoryStorageError("Invitation space could not be read")
        return cast(sqlite3.Row, row)

    def list_invitations(
        self, provider: str, session_id: str
    ) -> list[SpaceInvitationInfo]:
        with self._lock:
            self._ensure_open()
            self._get_session(self._connection, provider, session_id)
            rows = self._connection.execute(
                """
                SELECT i.*, s.name, s.owner_session_id
                FROM space_invitations AS i
                JOIN spaces AS s ON s.id = i.space_id
                WHERE i.invited_session_id = ? OR i.invited_by_session_id = ?
                ORDER BY i.created_at DESC, i.id
                """,
                (session_id, session_id),
            ).fetchall()
            return [self._invitation_info(row) for row in rows]

    def accept_invitation(
        self, provider: str, session_id: str, invitation_id: str
    ) -> SpaceInfo:
        with self._transaction() as connection:
            self._get_session(connection, provider, session_id)
            invitation = connection.execute(
                """
                SELECT i.space_id, i.status, s.archived_at
                FROM space_invitations AS i
                JOIN spaces AS s ON s.id = i.space_id
                WHERE i.id = ? AND i.invited_session_id = ?
                """,
                (invitation_id, session_id),
            ).fetchone()
            if invitation is None or invitation["archived_at"] is not None:
                raise MemoryNotFoundError("Pending invitation not found")
            space_id = str(invitation["space_id"])
            if invitation["status"] == "accepted":
                member = connection.execute(
                    """
                    SELECT 1 FROM space_memberships
                    WHERE space_id = ? AND session_id = ?
                    """,
                    (space_id, session_id),
                ).fetchone()
                if member is None:
                    raise MemoryNotFoundError("Pending invitation not found")
                row = connection.execute(
                    "SELECT * FROM spaces WHERE id = ?", (space_id,)
                ).fetchone()
                if row is None:
                    raise MemoryStorageError("Accepted invitation space could not be read")
                return self._space_info(connection, row)
            if invitation["status"] != "pending":
                raise MemoryNotFoundError("Pending invitation not found")
            now = self._now()
            connection.execute(
                """
                INSERT INTO space_memberships(
                    space_id, session_id, role, added_by_session_id, created_at
                ) VALUES (?, ?, 'member', ?, ?)
                """,
                (space_id, session_id, session_id, now),
            )
            connection.execute(
                """
                UPDATE space_invitations
                SET status = 'accepted', updated_at = ?
                WHERE id = ? AND status = 'pending'
                """,
                (now, invitation_id),
            )
            connection.execute(
                """
                INSERT INTO space_membership_events(
                    space_id, subject_session_id, actor_session_id, event, created_at
                ) VALUES (?, ?, ?, 'accepted', ?)
                """,
                (space_id, session_id, session_id, now),
            )
            row = connection.execute(
                "SELECT * FROM spaces WHERE id = ?", (space_id,)
            ).fetchone()
            if row is None:
                raise MemoryStorageError("Accepted invitation space could not be read")
            return self._space_info(connection, row)

    def revoke_membership(
        self,
        provider: str,
        session_id: str,
        space_id: str,
        member_session_id: str,
    ) -> None:
        with self._transaction() as connection:
            self._get_session(connection, provider, session_id)
            space = connection.execute(
                """
                SELECT owner_session_id FROM spaces
                WHERE id = ? AND space_type IN ('project', 'shared')
                  AND owner_session_id = ? AND archived_at IS NULL
                """,
                (space_id, session_id),
            ).fetchone()
            if space is None:
                raise MemoryNotFoundError("Shared space not found")
            if member_session_id == session_id:
                raise MemoryConflictError("The owner cannot revoke their own membership")
            deleted = connection.execute(
                "DELETE FROM space_memberships WHERE space_id = ? AND session_id = ?",
                (space_id, member_session_id),
            )
            if deleted.rowcount != 1:
                raise MemoryNotFoundError("Space membership not found")
            now = self._now()
            connection.execute(
                """
                UPDATE space_invitations SET status = 'revoked', updated_at = ?
                WHERE space_id = ? AND invited_session_id = ? AND status = 'accepted'
                """,
                (now, space_id, member_session_id),
            )
            connection.execute(
                """
                INSERT INTO space_membership_events(
                    space_id, subject_session_id, actor_session_id, event, created_at
                ) VALUES (?, ?, ?, 'revoked', ?)
                """,
                (space_id, member_session_id, session_id, now),
            )

    def leave_space(self, provider: str, session_id: str, space_id: str) -> None:
        with self._transaction() as connection:
            self._get_session(connection, provider, session_id)
            space = connection.execute(
                """
                SELECT owner_session_id FROM spaces
                WHERE id = ? AND space_type IN ('project', 'shared')
                  AND archived_at IS NULL
                """,
                (space_id,),
            ).fetchone()
            if space is None:
                raise MemoryNotFoundError("Shared space not found")
            if space["owner_session_id"] == session_id:
                raise MemoryConflictError("Space owner cannot leave their space")
            deleted = connection.execute(
                "DELETE FROM space_memberships WHERE space_id = ? AND session_id = ?",
                (space_id, session_id),
            )
            if deleted.rowcount != 1:
                raise MemoryNotFoundError("Space membership not found")
            now = self._now()
            connection.execute(
                """
                UPDATE space_invitations SET status = 'left', updated_at = ?
                WHERE space_id = ? AND invited_session_id = ? AND status = 'accepted'
                """,
                (now, space_id, session_id),
            )
            connection.execute(
                """
                INSERT INTO space_membership_events(
                    space_id, subject_session_id, actor_session_id, event, created_at
                ) VALUES (?, ?, ?, 'left', ?)
                """,
                (space_id, session_id, session_id, now),
            )

    def archive_space(self, provider: str, session_id: str, space_id: str) -> str:
        with self._transaction() as connection:
            self._get_session(connection, provider, session_id)
            space = connection.execute(
                """
                SELECT * FROM spaces
                WHERE id = ? AND space_type IN ('project', 'shared')
                  AND owner_session_id = ?
                """,
                (space_id, session_id),
            ).fetchone()
            if space is None:
                raise MemoryNotFoundError("Shared space not found")
            if space["archived_at"] is not None:
                return str(space["archived_at"])
            now = self._now()
            pending_invites = connection.execute(
                """
                SELECT invited_session_id FROM space_invitations
                WHERE space_id = ? AND status = 'pending'
                """,
                (space_id,),
            ).fetchall()
            connection.execute(
                """
                UPDATE space_invitations SET status = 'revoked', updated_at = ?
                WHERE space_id = ? AND status = 'pending'
                """,
                (now, space_id),
            )
            for invitation in pending_invites:
                connection.execute(
                    """
                    INSERT INTO space_membership_events(
                        space_id, subject_session_id, actor_session_id, event, created_at
                    ) VALUES (?, ?, ?, 'revoked', ?)
                    """,
                    (space_id, invitation["invited_session_id"], session_id, now),
                )
            connection.execute(
                "UPDATE spaces SET archived_at = ? WHERE id = ?", (now, space_id)
            )
            connection.execute(
                """
                INSERT INTO space_membership_events(
                    space_id, subject_session_id, actor_session_id, event, created_at
                ) VALUES (?, ?, ?, 'archived', ?)
                """,
                (space_id, session_id, session_id, now),
            )
            return now

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
        if row is None or row["archived_at"] is not None:
            raise MemoryAccessError("Write access to the requested space is not allowed")
        if row["space_type"] == "provider":
            if row["provider"] != provider:
                raise MemoryAccessError("Write access to the requested space is not allowed")
            return space_id
        if row["space_type"] == "session":
            if row["owner_session_id"] != session["id"]:
                raise MemoryAccessError("Write access to the requested space is not allowed")
            return space_id
        member = connection.execute(
            "SELECT 1 FROM space_memberships WHERE space_id = ? AND session_id = ?",
            (space_id, session["id"]),
        ).fetchone()
        if member is None:
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
        embedding_model: str,
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
                embedding_model=embedding_model,
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
        embedding_model: str,
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
                embedding_model=embedding_model,
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
        embedding_model: str,
        created_at: str,
    ) -> None:
        if len(chunks) != len(vectors) or not chunks:
            raise MemoryStorageError("Every stored chunk must have exactly one embedding")
        if not embedding_model.strip():
            raise MemoryStorageError("Embedding model name cannot be empty")
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
                    embedding_model,
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
                  AND s.archived_at IS NULL
                  AND (
                    s.space_type IN ('provider', 'session')
                    OR EXISTS (
                        SELECT 1 FROM space_memberships AS m
                        WHERE m.space_id = s.id AND m.session_id = ?
                    )
                  )
                """,
                (document_id, session_id),
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
                WHERE h.session_id = ? AND h.task_id = ? AND s.space_type = 'session'
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
                  AND {status_filter} AND s.archived_at IS NULL
                  AND (
                    s.space_type IN ('provider', 'session')
                    OR EXISTS (
                        SELECT 1 FROM space_memberships AS m
                        WHERE m.space_id = s.id AND m.session_id = ?
                    )
                  )
                  AND {where}
                ORDER BY metric ASC, c.id ASC
                LIMIT ?
            """
            rows = self._connection.execute(
                sql, (match_query, session_id, *filters, limit * 20)
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
                      AND s.archived_at IS NULL
                      AND (
                        s.space_type IN ('provider', 'session')
                        OR EXISTS (
                            SELECT 1 FROM space_memberships AS m
                            WHERE m.space_id = s.id AND m.session_id = ?
                        )
                      )
                      AND {where}
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
            rows = self._connection.execute(
                sql, (vector, session_id, *filters, limit)
            ).fetchall()
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
