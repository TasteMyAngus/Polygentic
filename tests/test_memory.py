from __future__ import annotations

import hashlib
import re
import sqlite3
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
async def test_shared_space_invitation_and_exact_session_acl() -> None:
    database, memory = _service()
    try:
        owner = await memory.register_session("codex", "Project owner", "register-owner")
        sibling = await memory.register_session("codex", "Owner sibling", "register-sibling")
        guest = await memory.register_session("claude", "Project guest", "register-guest")
        project = await memory.create_space(
            "codex", owner.session_id, "project", "Release project", "create-release-project"
        )
        assert project.kind == "project"
        assert [(member.session_id, member.role) for member in project.members] == [
            (owner.session_id, "owner")
        ]
        assert {
            (session.session_id, session.provider)
            for session in await memory.list_registered_sessions()
        } == {
            (owner.session_id, "codex"),
            (sibling.session_id, "codex"),
            (guest.session_id, "claude"),
        }

        assert await memory.list_spaces("codex", sibling.session_id) == []
        with pytest.raises(MemoryNotFoundError, match="Project space not found"):
            await memory.create_space(
                "codex",
                sibling.session_id,
                "shared",
                "Unlinked room",
                "create-unlinked-room",
                project_space_id=project.space_id,
            )

        project_invitation = await memory.invite_session(
            "codex", owner.session_id, project.space_id, guest.session_id
        )
        await memory.accept_invitation(
            "claude", guest.session_id, project_invitation.invitation_id
        )
        space = await memory.create_space(
            "codex",
            owner.session_id,
            "shared",
            "Release incident room",
            "create-release-incident-room",
            project_space_id=project.space_id,
        )
        replayed_space = await memory.create_space(
            "codex",
            owner.session_id,
            "shared",
            "Release incident room",
            "create-release-incident-room",
            project_space_id=project.space_id,
        )
        assert replayed_space.space_id == space.space_id
        with pytest.raises(MemoryConflictError, match="Space creation idempotency key"):
            await memory.create_space(
                "codex",
                owner.session_id,
                "shared",
                "Different room",
                "create-release-incident-room",
                project_space_id=project.space_id,
            )
        assert space.project_space_id == project.space_id
        assert {
            available.space_id
            for available in await memory.list_spaces("claude", guest.session_id)
        } == {project.space_id}
        with pytest.raises(MemoryNotFoundError, match="Shared space not found"):
            await memory.get_space("claude", guest.session_id, space.space_id)

        invitation = await memory.invite_session(
            "codex", owner.session_id, space.space_id, guest.session_id
        )
        replay = await memory.invite_session(
            "codex", owner.session_id, space.space_id, guest.session_id
        )
        assert replay.invitation_id == invitation.invitation_id
        assert replay.status == "pending"
        assert (await memory.list_invitations("claude", guest.session_id))[0] == invitation
        assert {
            available.space_id
            for available in await memory.list_spaces("claude", guest.session_id)
        } == {project.space_id}

        with pytest.raises(MemoryNotFoundError, match="Pending invitation"):
            await memory.accept_invitation("codex", owner.session_id, invitation.invitation_id)

        accepted = await memory.accept_invitation(
            "claude", guest.session_id, invitation.invitation_id
        )
        accepted_replay = await memory.accept_invitation(
            "claude", guest.session_id, invitation.invitation_id
        )
        assert {member.session_id for member in accepted.members} == {
            owner.session_id,
            guest.session_id,
        }
        assert accepted_replay == accepted
        record = await memory.put_context(
            provider="claude",
            session_id=guest.session_id,
            space_id=space.space_id,
            title="Project milestone",
            content="The orbital release milestone is scheduled for Friday.",
            record_type="decision",
            idempotency_key="shared-write",
        )
        owner_search = await memory.search_context(
            provider="codex",
            session_id=owner.session_id,
            query="orbital release milestone",
            mode="keyword",
        )
        assert owner_search.results[0].document_id == record.document_id

        for mode in ("keyword", "semantic"):
            hidden = await memory.search_context(
                provider="codex",
                session_id=sibling.session_id,
                query="orbital release milestone",
                mode=mode,
            )
            assert hidden.results == []
            hidden_by_explicit_space = await memory.search_context(
                provider="codex",
                session_id=sibling.session_id,
                query="orbital release milestone",
                mode=mode,
                space_ids=[space.space_id],
            )
            assert hidden_by_explicit_space.results == []
        with pytest.raises(MemoryNotFoundError, match="Context record not found"):
            await memory.get_context(
                "codex", sibling.session_id, record.document_id, 0, 1000
            )
        with pytest.raises(MemoryAccessError, match="Write access"):
            await memory.put_context(
                provider="codex",
                session_id=sibling.session_id,
                space_id=space.space_id,
                title="Sibling write",
                content="Provider identity does not grant shared-space membership.",
                record_type="note",
                idempotency_key="sibling-shared-write",
            )
        with pytest.raises(MemoryNotFoundError, match="Shared space not found"):
            await memory.invite_session(
                "claude", guest.session_id, space.space_id, sibling.session_id
            )

        await memory.revoke_membership(
            "codex", owner.session_id, space.space_id, guest.session_id
        )
        assert (await memory.list_invitations("claude", guest.session_id))[0].status == "revoked"
        assert [member.session_id for member in (await memory.get_space(
            "codex", owner.session_id, space.space_id
        )).members] == [owner.session_id]
        with pytest.raises(MemoryNotFoundError, match="Context record not found"):
            await memory.get_context(
                "claude", guest.session_id, record.document_id, 0, 1000
            )
    finally:
        database.close()


@pytest.mark.asyncio
async def test_shared_space_leave_owner_rules_and_archive() -> None:
    database, memory = _service()
    try:
        owner = await memory.register_session("codex", "Space owner", "register-space-owner")
        member = await memory.register_session("claude", "Space member", "register-space-member")
        pending = await memory.register_session(
            "copilot", "Pending invitee", "register-pending-invitee"
        )
        space = await memory.create_space(
            "codex",
            owner.session_id,
            "shared",
            "Incident response",
            "create-incident-response",
        )
        invitation = await memory.invite_session(
            "codex", owner.session_id, space.space_id, member.session_id
        )
        await memory.accept_invitation("claude", member.session_id, invitation.invitation_id)

        with pytest.raises(MemoryConflictError, match="owner cannot leave"):
            await memory.leave_space("codex", owner.session_id, space.space_id)
        await memory.leave_space("claude", member.session_id, space.space_id)
        assert (await memory.list_invitations("claude", member.session_id))[0].status == "left"
        with pytest.raises(MemoryNotFoundError, match="Shared space not found"):
            await memory.get_space("claude", member.session_id, space.space_id)

        archived_record = await memory.put_context(
            provider="codex",
            session_id=owner.session_id,
            space_id=space.space_id,
            title="Incident runbook",
            content="The glacier incident response starts with the control-plane checklist.",
            record_type="constraint",
            idempotency_key="archive-record",
        )
        pending_invitation = await memory.invite_session(
            "codex", owner.session_id, space.space_id, pending.session_id
        )
        archived_at = await memory.archive_space("codex", owner.session_id, space.space_id)
        assert archived_at
        assert (await memory.list_invitations("copilot", pending.session_id))[0].status == "revoked"
        with pytest.raises(MemoryNotFoundError, match="Pending invitation"):
            await memory.accept_invitation(
                "copilot", pending.session_id, pending_invitation.invitation_id
            )
        archived = await memory.get_space("codex", owner.session_id, space.space_id)
        assert archived.archived_at == archived_at
        assert await memory.archive_space("codex", owner.session_id, space.space_id) == archived_at
        archived_search = await memory.search_context(
            provider="codex",
            session_id=owner.session_id,
            query="glacier incident control-plane",
            mode="keyword",
        )
        assert all(hit.document_id != archived_record.document_id for hit in archived_search.results)
        with pytest.raises(MemoryNotFoundError, match="Context record not found"):
            await memory.get_context(
                "codex", owner.session_id, archived_record.document_id, 0, 1000
            )
        with pytest.raises(MemoryNotFoundError, match="Shared space not found"):
            await memory.invite_session(
                "codex", owner.session_id, space.space_id, member.session_id
            )
        with pytest.raises(MemoryAccessError, match="Write access"):
            await memory.put_context(
                provider="codex",
                session_id=owner.session_id,
                space_id=space.space_id,
                title="Archived write",
                content="Archived spaces reject new writes.",
                record_type="note",
                idempotency_key="archived-write",
            )
    finally:
        database.close()


@pytest.mark.asyncio
async def test_supersession_is_idempotent_and_history_is_opt_in() -> None:
    database, memory = _service()
    try:
        session = await memory.register_session("codex", "Supersession", "register-supersession")
        older = await memory.put_context(
            provider="codex",
            session_id=session.session_id,
            title="Legacy migration decision",
            content="The migration plan used maplewood as its storage engine.",
            record_type="decision",
            idempotency_key="supersession-old",
        )
        newer = await memory.put_context(
            provider="codex",
            session_id=session.session_id,
            title="Current migration decision",
            content="The migration plan uses bluebird as its storage engine.",
            record_type="decision",
            idempotency_key="supersession-new",
        )
        request = {
            "provider": "codex",
            "session_id": session.session_id,
            "newer_document_id": newer.document_id,
            "older_document_ids": [older.document_id],
            "expected_revisions": {older.document_id: 1},
            "idempotency_key": "supersession-edge-1",
        }

        result = await memory.supersede_context(**request)
        replay = await memory.supersede_context(**request)
        assert replay == result
        assert result.superseded_document_ids == [older.document_id]

        default_search = await memory.search_context(
            provider="codex",
            session_id=session.session_id,
            query="maplewood",
            mode="keyword",
        )
        assert all(hit.document_id != older.document_id for hit in default_search.results)
        history_search = await memory.search_context(
            provider="codex",
            session_id=session.session_id,
            query="maplewood",
            mode="keyword",
            include_superseded=True,
        )
        historical_hit = next(hit for hit in history_search.results if hit.document_id == older.document_id)
        assert historical_hit.status == "superseded"
        assert historical_hit.superseded_by == [newer.document_id]

        with pytest.raises(MemoryNotFoundError, match="Context record not found"):
            await memory.get_context("codex", session.session_id, older.document_id, 0, 1000)
        historical_record = await memory.get_context(
            "codex",
            session.session_id,
            older.document_id,
            0,
            1000,
            include_superseded=True,
        )
        assert historical_record.status == "superseded"
        assert historical_record.superseded_by == [newer.document_id]
        current_record = await memory.get_context(
            "codex", session.session_id, newer.document_id, 0, 1000
        )
        assert current_record.supersedes == [older.document_id]

        event_count = database._connection.execute(
            "SELECT count(*) FROM document_status_events WHERE document_id = ?",
            (older.document_id,),
        ).fetchone()[0]
        assert event_count == 1
        with pytest.raises(MemoryConflictError, match="idempotency key"):
            await memory.supersede_context(
                **{**request, "expected_revisions": {older.document_id: 2}}
            )
        with pytest.raises(MemoryConflictError, match="idempotency key"):
            await memory.put_context(
                provider="codex",
                session_id=session.session_id,
                title="Wrong operation",
                content="This key was already used for supersession.",
                record_type="note",
                idempotency_key="supersession-edge-1",
            )
    finally:
        database.close()


@pytest.mark.asyncio
async def test_supersession_rejects_cycles_stale_revisions_and_cross_space_edges() -> None:
    database, memory = _service()
    try:
        session = await memory.register_session("codex", "Supersession graph", "register-graph")
        first = await memory.put_context(
            provider="codex",
            session_id=session.session_id,
            title="First",
            content="First historical decision.",
            record_type="decision",
            idempotency_key="graph-first",
        )
        second = await memory.put_context(
            provider="codex",
            session_id=session.session_id,
            title="Second",
            content="Second replacement decision.",
            record_type="decision",
            idempotency_key="graph-second",
        )
        third = await memory.put_context(
            provider="codex",
            session_id=session.session_id,
            title="Third",
            content="Third replacement decision.",
            record_type="decision",
            idempotency_key="graph-third",
        )
        await memory.supersede_context(
            provider="codex",
            session_id=session.session_id,
            newer_document_id=second.document_id,
            older_document_ids=[first.document_id],
            expected_revisions={first.document_id: 1},
            idempotency_key="graph-edge-1",
        )
        await memory.supersede_context(
            provider="codex",
            session_id=session.session_id,
            newer_document_id=third.document_id,
            older_document_ids=[second.document_id],
            expected_revisions={second.document_id: 1},
            idempotency_key="graph-edge-2",
        )
        with pytest.raises(MemoryConflictError, match="cycle"):
            await memory.supersede_context(
                provider="codex",
                session_id=session.session_id,
                newer_document_id=first.document_id,
                older_document_ids=[third.document_id],
                expected_revisions={third.document_id: 1},
                idempotency_key="graph-cycle",
            )

        stale = await memory.put_context(
            provider="codex",
            session_id=session.session_id,
            title="Stale candidate",
            content="A second old decision.",
            record_type="decision",
            idempotency_key="graph-stale",
        )
        with pytest.raises(MemoryConflictError, match="revision is stale"):
            await memory.supersede_context(
                provider="codex",
                session_id=session.session_id,
                newer_document_id=third.document_id,
                older_document_ids=[first.document_id, stale.document_id],
                expected_revisions={first.document_id: 1, stale.document_id: 2},
                idempotency_key="graph-stale-revisions",
            )
        first_after_failure = await memory.get_context(
            "codex",
            session.session_id,
            first.document_id,
            0,
            1000,
            include_superseded=True,
        )
        assert first_after_failure.superseded_by == [second.document_id]
        assert stale.document_id not in first_after_failure.superseded_by

        provider_record = await memory.put_context(
            provider="codex",
            session_id=session.session_id,
            space_id=session.provider_space_id,
            title="Provider-space decision",
            content="Provider-wide legacy decision.",
            record_type="decision",
            idempotency_key="graph-provider",
        )
        with pytest.raises(MemoryConflictError, match="same space"):
            await memory.supersede_context(
                provider="codex",
                session_id=session.session_id,
                newer_document_id=third.document_id,
                older_document_ids=[provider_record.document_id],
                expected_revisions={provider_record.document_id: 1},
                idempotency_key="graph-cross-space",
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


def test_schema_v1_migrates_to_shared_space_schema_v3(tmp_path: Path) -> None:
    data_dir = tmp_path / "legacy-memory"
    initial = MemoryDatabase(data_dir)
    initial.register_session("codex", "Legacy session", "legacy-registration")
    initial.close()

    connection = sqlite3.connect(data_dir / "context.sqlite3")
    connection.execute("DROP TABLE space_membership_events")
    connection.execute("DROP TABLE space_invitations")
    connection.execute("DROP TABLE space_memberships")
    connection.execute("DROP TABLE space_creation_requests")
    connection.execute("DROP INDEX session_space_owner")
    connection.execute("ALTER TABLE spaces DROP COLUMN archived_at")
    connection.execute("ALTER TABLE spaces DROP COLUMN name")
    connection.execute("ALTER TABLE spaces DROP COLUMN project_space_id")
    connection.execute("ALTER TABLE spaces DROP COLUMN space_type")
    connection.execute(
        "CREATE UNIQUE INDEX session_space_owner "
        "ON spaces(owner_session_id) WHERE kind = 'session'"
    )
    connection.execute("DROP TABLE supersession_requests")
    connection.execute("DROP TABLE document_status_events")
    connection.execute("DROP TABLE supersession_edges")
    connection.execute("PRAGMA user_version = 1")
    connection.commit()
    connection.close()

    migrated = MemoryDatabase(data_dir)
    try:
        version = migrated._connection.execute("PRAGMA user_version").fetchone()[0]
        tables = {
            row["name"]
            for row in migrated._connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert version == 3
        assert {
            "supersession_edges",
            "document_status_events",
            "supersession_requests",
            "space_memberships",
            "space_invitations",
            "space_membership_events",
            "space_creation_requests",
        } <= tables
        space_types = {
            row["space_type"]
            for row in migrated._connection.execute("SELECT space_type FROM spaces")
        }
        assert space_types == {"provider", "session"}
    finally:
        migrated.close()
