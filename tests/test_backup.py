from __future__ import annotations

import os
import stat
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from context_server.backup import backup_main, restore_main
from context_server.memory import MemoryService
from context_server.storage import (
    MemoryConflictError,
    MemoryDatabase,
    MemoryNotFoundError,
    MemoryStorageError,
)


class BackupEmbedder:
    async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return [tuple([1.0] + [0.0] * 383) for _ in texts]


async def _create_record(
    data_dir: Path, label: str, registration_key: str, content: str
) -> tuple[MemoryDatabase, str, str]:
    database = MemoryDatabase(data_dir)
    memory = MemoryService(database, BackupEmbedder())
    session = await memory.register_session("codex", label, registration_key)
    record = await memory.put_context(
        provider="codex",
        session_id=session.session_id,
        title="Backup verification",
        content=content,
        record_type="decision",
        idempotency_key=f"write-{registration_key}",
    )
    return database, session.session_id, record.document_id


@pytest.mark.asyncio
async def test_consistent_backup_restores_records_and_search_indexes(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    database, session_id, document_id = await _create_record(
        source_dir,
        "Original",
        "original-registration",
        "The glacier rollout is approved after the readiness review.",
    )
    backup_path = tmp_path / "backups" / "context.sqlite3"
    try:
        assert database.backup_to(backup_path) == backup_path
    finally:
        database.close()

    assert stat.S_IMODE(backup_path.stat().st_mode) == 0o600
    restored_dir = tmp_path / "restored"
    assert MemoryDatabase.restore_from_backup(backup_path, restored_dir) == (
        restored_dir / "context.sqlite3"
    )
    assert stat.S_IMODE((restored_dir / "context.sqlite3").stat().st_mode) == 0o600

    restored = MemoryDatabase(restored_dir)
    memory = MemoryService(restored, BackupEmbedder())
    try:
        record = await memory.get_context("codex", session_id, document_id, 0, 1000)
        assert record.content == "The glacier rollout is approved after the readiness review."
        search = await memory.search_context(
            provider="codex",
            session_id=session_id,
            query="glacier rollout readiness",
            mode="hybrid",
        )
        assert search.results[0].document_id == document_id
    finally:
        restored.close()


@pytest.mark.asyncio
async def test_restore_requires_explicit_overwrite_and_replaces_atomically(
    tmp_path: Path,
) -> None:
    source, source_session, _source_record = await _create_record(
        tmp_path / "source",
        "Source",
        "source-registration",
        "The source database should replace the existing destination.",
    )
    backup_path = tmp_path / "backup.sqlite3"
    try:
        source.backup_to(backup_path)
    finally:
        source.close()

    destination, old_session, _old_record = await _create_record(
        tmp_path / "destination",
        "Old destination",
        "old-registration",
        "This existing destination must remain until replacement is requested.",
    )
    with pytest.raises(MemoryConflictError, match="sidecars remain"):
        MemoryDatabase.restore_from_backup(
            backup_path, tmp_path / "destination", overwrite=True
        )
    destination.close()

    with pytest.raises(MemoryConflictError, match="overwrite=True"):
        MemoryDatabase.restore_from_backup(backup_path, tmp_path / "destination")

    unchanged = MemoryDatabase(tmp_path / "destination")
    try:
        assert unchanged.resume_session("codex", old_session).session_id == old_session
    finally:
        unchanged.close()

    MemoryDatabase.restore_from_backup(
        backup_path, tmp_path / "destination", overwrite=True
    )
    restored = MemoryDatabase(tmp_path / "destination")
    try:
        assert restored.resume_session("codex", source_session).session_id == source_session
        with pytest.raises(MemoryNotFoundError, match="Session not found"):
            restored.resume_session("codex", old_session)
    finally:
        restored.close()


def test_backup_and_restore_reject_unsafe_paths_and_invalid_files(tmp_path: Path) -> None:
    memory_only = MemoryDatabase()
    try:
        with pytest.raises(MemoryStorageError, match="in-memory"):
            memory_only.backup_to(tmp_path / "memory.sqlite3")
    finally:
        memory_only.close()

    data_dir = tmp_path / "source"
    persistent = MemoryDatabase(data_dir)
    live_path = data_dir / "context.sqlite3"
    try:
        with pytest.raises(MemoryStorageError, match="differ"):
            persistent.backup_to(live_path)

        existing_backup = tmp_path / "existing-backup.sqlite3"
        existing_backup.write_bytes(b"keep this backup")
        os.chmod(existing_backup, 0o600)
        with pytest.raises(MemoryConflictError, match="overwrite=True"):
            persistent.backup_to(existing_backup)
        assert existing_backup.read_bytes() == b"keep this backup"
        persistent.backup_to(existing_backup, overwrite=True)
        assert existing_backup.read_bytes().startswith(b"SQLite format 3")

        symlink_target = tmp_path / "symlink-target"
        symlink_target.write_bytes(b"target")
        symlink_target.chmod(0o600)
        symlink = tmp_path / "backup-symlink.sqlite3"
        symlink.symlink_to(symlink_target)
        with pytest.raises(MemoryStorageError, match="symbolic link"):
            persistent.backup_to(symlink)
    finally:
        persistent.close()
    existing_database_bytes = live_path.read_bytes()

    invalid_backup = tmp_path / "invalid.sqlite3"
    invalid_backup.write_bytes(b"not a SQLite database")
    os.chmod(invalid_backup, 0o600)
    invalid_symlink = tmp_path / "invalid-symlink.sqlite3"
    invalid_symlink.symlink_to(invalid_backup)
    with pytest.raises(MemoryStorageError, match="symbolic link"):
        MemoryDatabase.restore_from_backup(invalid_symlink, data_dir, overwrite=True)
    with pytest.raises(MemoryStorageError, match="valid SQLite database"):
        MemoryDatabase.restore_from_backup(
            invalid_backup, data_dir, overwrite=True
        )
    assert live_path.read_bytes() == existing_database_bytes


@pytest.mark.asyncio
async def test_backup_and_restore_cli_use_explicit_data_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source_dir = tmp_path / "source"
    source, session_id, document_id = await _create_record(
        source_dir,
        "CLI source",
        "cli-registration",
        "The command line backup must preserve searchable context.",
    )
    source.close()

    backup_path = tmp_path / "cli-backup.sqlite3"
    monkeypatch.setattr(
        sys,
        "argv",
        ["context-backup", str(backup_path), "--data-dir", str(source_dir)],
    )
    backup_main()
    assert capsys.readouterr().out.strip() == str(backup_path)

    restored_dir = tmp_path / "cli-restored"
    monkeypatch.setattr(
        sys,
        "argv",
        ["context-restore", str(backup_path), "--data-dir", str(restored_dir)],
    )
    restore_main()
    assert capsys.readouterr().out.strip() == str(restored_dir / "context.sqlite3")

    restored = MemoryDatabase(restored_dir)
    memory = MemoryService(restored, BackupEmbedder())
    try:
        record = await memory.get_context("codex", session_id, document_id, 0, 1000)
        assert record.content == "The command line backup must preserve searchable context."
    finally:
        restored.close()
