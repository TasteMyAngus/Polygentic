from __future__ import annotations

import argparse
import os
from pathlib import Path

from platformdirs import user_data_path

from context_server.storage import MemoryConflictError, MemoryDatabase, MemoryStorageError


def _data_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser()
    configured = os.environ.get("CONTEXT_SERVER_DATA_DIR")
    if configured:
        return Path(configured).expanduser()
    return user_data_path("Polygentic", appauthor=False)


def backup_main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an owner-only, consistent backup of the context database."
    )
    parser.add_argument("destination", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    data_dir = _data_dir(args.data_dir)
    if not (data_dir / "context.sqlite3").is_file():
        parser.exit(1, f"error: no context database found in {data_dir}\n")
    try:
        database = MemoryDatabase(data_dir)
    except (MemoryStorageError, OSError) as exc:
        parser.exit(1, f"error: {exc}\n")
    try:
        backup_path = database.backup_to(args.destination, overwrite=args.overwrite)
    except (MemoryConflictError, MemoryStorageError, OSError) as exc:
        parser.exit(1, f"error: {exc}\n")
    finally:
        database.close()
    print(backup_path)


def restore_main() -> None:
    parser = argparse.ArgumentParser(
        description="Restore a validated context database backup."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the existing database after validation; stop all server processes first.",
    )
    args = parser.parse_args()
    try:
        restored_path = MemoryDatabase.restore_from_backup(
            args.source, _data_dir(args.data_dir), overwrite=args.overwrite
        )
    except (MemoryConflictError, MemoryStorageError, OSError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(restored_path)
