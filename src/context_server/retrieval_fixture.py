from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from context_server.memory import RecordType

CorpusScope = Literal["provider", "session"]
ExpectedStatus = Literal["active", "superseded"]

CORPUS_VERSION = 1
MINIMUM_EVIDENCE_HIT_AT_5 = 0.90


@dataclass(frozen=True)
class FixtureRecord:
    key: str
    provider: str
    scope: CorpusScope
    title: str
    content: str
    record_type: RecordType


@dataclass(frozen=True)
class FixtureEvidence:
    record_key: str
    span: str
    status: ExpectedStatus = "active"


@dataclass(frozen=True)
class FixtureQuestion:
    key: str
    category: str
    query: str
    required: tuple[FixtureEvidence, ...]
    include_superseded: bool = False


@dataclass(frozen=True)
class FixtureSupersession:
    newer_key: str
    older_keys: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalFixture:
    version: int
    records: tuple[FixtureRecord, ...]
    questions: tuple[FixtureQuestion, ...]
    supersessions: tuple[FixtureSupersession, ...]


type ExactRecordSpec = tuple[str, str, CorpusScope, str, str, str, str]
type ParaphraseRecordSpec = tuple[str, str, CorpusScope, str, str, str, str]
type SupersessionSpec = tuple[
    str, str, CorpusScope, str, str, str, str, str, str, str
]
type DistractorSpec = tuple[str, str, CorpusScope, str, str]


def build_retrieval_fixture() -> RetrievalFixture:
    records: list[FixtureRecord] = []
    questions: list[FixtureQuestion] = []
    supersessions: list[FixtureSupersession] = []

    exact_specs: tuple[ExactRecordSpec, ...] = (
        (
            "exact-01",
            "codex",
            "provider",
            "Telemetry ticket ownership",
            "Ticket QZ-1041 is assigned to the telemetry team.",
            "Who owns ticket QZ-1041?",
            "Ticket QZ-1041 is assigned to the telemetry team.",
        ),
        (
            "exact-02",
            "claude",
            "session",
            "Runbook link",
            "Runbook RB-2208 is the approved recovery guide for the billing queue.",
            "Which recovery guide is approved for queue billing, RB-2208?",
            "Runbook RB-2208 is the approved recovery guide for the billing queue.",
        ),
        (
            "exact-03",
            "codex",
            "session",
            "Cache namespace",
            "The cache namespace for product stream LYNX-73 is `lynx:stable`.",
            "What cache namespace belongs to product stream LYNX-73?",
            "The cache namespace for product stream LYNX-73 is `lynx:stable`.",
        ),
        (
            "exact-04",
            "claude",
            "provider",
            "Secret rotation schedule",
            "Secret rotation ticket SEC-5812 schedules the signing key for November 28.",
            "When is the signing key rotation in SEC-5812?",
            "Secret rotation ticket SEC-5812 schedules the signing key for November 28.",
        ),
        (
            "exact-05",
            "codex",
            "provider",
            "Database failover region",
            "Failover group PG-441 uses region eu-west-2 as its recovery target.",
            "Which region is the recovery target for PG-441?",
            "Failover group PG-441 uses region eu-west-2 as its recovery target.",
        ),
        (
            "exact-06",
            "claude",
            "session",
            "Long-record evidence",
            "Background: " + ("routine archive detail " * 78)
            + "The evidence for case CHUNK-903 is stored in the signed ledger.",
            "Where is the evidence for case CHUNK-903 stored?",
            "The evidence for case CHUNK-903 is stored in the signed ledger.",
        ),
    )
    for key, provider, scope, title, content, query, span in exact_specs:
        records.append(FixtureRecord(key, provider, scope, title, content, "finding"))
        questions.append(
            FixtureQuestion(
                key=f"question-{key}",
                category="exact_identifier",
                query=query,
                required=(FixtureEvidence(key, span),),
            )
        )

    paraphrase_specs: tuple[ParaphraseRecordSpec, ...] = (
        (
            "paraphrase-01",
            "codex",
            "session",
            "Retry policy",
            "The team selected exponential backoff capped at 90 seconds for transient HTTP failures.",
            "How should temporary network errors be retried?",
            "exponential backoff capped at 90 seconds",
        ),
        (
            "paraphrase-02",
            "claude",
            "provider",
            "Service discovery design",
            "For service discovery, the chosen interface is stateless REST with no per-client server session.",
            "What architecture avoids retaining a caller's server-side conversation state?",
            "stateless REST with no per-client server session",
        ),
        (
            "paraphrase-03",
            "codex",
            "provider",
            "Administrator audit retention",
            "Administrator audit events are retained for seven years before archival.",
            "How long are records of administrative actions kept?",
            "retained for seven years",
        ),
        (
            "paraphrase-04",
            "claude",
            "session",
            "Production rollout",
            "Production traffic will move through three canary stages before the full release.",
            "Which gradual launch strategy was selected?",
            "three canary stages",
        ),
        (
            "paraphrase-05",
            "codex",
            "session",
            "Workload credentials",
            "Workload credentials rotate every 90 days, with a short overlap during replacement.",
            "How frequently should machine credentials be renewed?",
            "rotate every 90 days",
        ),
        (
            "paraphrase-06",
            "claude",
            "provider",
            "Embedding location",
            "The service computes embeddings locally and caches vectors on the device; it uses no remote generation model.",
            "Where does semantic indexing run?",
            "computes embeddings locally",
        ),
    )
    for key, provider, scope, title, content, query, span in paraphrase_specs:
        records.append(FixtureRecord(key, provider, scope, title, content, "decision"))
        questions.append(
            FixtureQuestion(
                key=f"question-{key}",
                category="paraphrased_decision",
                query=query,
                required=(FixtureEvidence(key, span),),
            )
        )

    conflict_specs = (
        ("ORION", "retention period", "30 days", "90 days"),
        ("GATEWAY-ETA", "request timeout", "2 seconds", "5 seconds"),
        ("CLUSTER-SABLE", "backup frequency", "daily", "weekly"),
        ("FRAUD-MODEL", "review threshold", "0.70", "0.85"),
        ("MIRROR-PIPE", "maximum payload", "10 megabytes", "20 megabytes"),
        ("NORTHWIND", "primary region", "eu-north-1", "eu-west-1"),
    )
    for index, (subject, decision, left_value, right_value) in enumerate(conflict_specs, start=1):
        left_key = f"conflict-{index:02d}-a"
        right_key = f"conflict-{index:02d}-b"
        subject_name = subject.replace("-", " ").title()
        left_span = f"The {decision} for {subject_name} is {left_value}."
        right_span = f"The {decision} for {subject_name} is {right_value}."
        records.extend(
            (
                FixtureRecord(
                    left_key,
                    "codex",
                    "provider",
                    f"{subject_name} {decision}",
                    left_span,
                    "decision",
                ),
                FixtureRecord(
                    right_key,
                    "claude",
                    "session",
                    f"{subject_name} {decision}",
                    right_span,
                    "decision",
                ),
            )
        )
        questions.append(
            FixtureQuestion(
                key=f"question-conflict-{index:02d}",
                category="active_conflict",
                query=f"What active values are recorded for the {decision} of {subject_name}?",
                required=(
                    FixtureEvidence(left_key, left_span),
                    FixtureEvidence(right_key, right_span),
                ),
            )
        )

    supersession_specs: tuple[SupersessionSpec, ...] = (
        (
            "supersession-01",
            "codex",
            "session",
            "ROTATION-ALPHA",
            "The prior owner of ROTATION-ALPHA was team Cedar.",
            "The current owner of ROTATION-ALPHA is team Birch.",
            "Who currently owns ROTATION-ALPHA?",
            "team Birch",
            "Who previously owned ROTATION-ALPHA?",
            "team Cedar",
        ),
        (
            "supersession-02",
            "claude",
            "provider",
            "ARCHIVE-BETA",
            "The previous archive target for ARCHIVE-BETA was Glacier.",
            "The active archive target for ARCHIVE-BETA is Coldline.",
            "What is the current archive target for ARCHIVE-BETA?",
            "Coldline",
            "What was the historical archive target for ARCHIVE-BETA?",
            "Glacier",
        ),
        (
            "supersession-03",
            "codex",
            "provider",
            "LIMIT-GAMMA",
            "The earlier concurrency limit for LIMIT-GAMMA was 12 workers.",
            "The current concurrency limit for LIMIT-GAMMA is 24 workers.",
            "What concurrency limit is now in force for LIMIT-GAMMA?",
            "24 workers",
            "What was the superseded concurrency limit for LIMIT-GAMMA?",
            "12 workers",
        ),
        (
            "supersession-04",
            "claude",
            "session",
            "FORMAT-DELTA",
            "The former export format for FORMAT-DELTA was CSV.",
            "The current export format for FORMAT-DELTA is Parquet.",
            "Which export format replaced CSV for FORMAT-DELTA?",
            "Parquet",
            "Which export format was used before the change to FORMAT-DELTA?",
            "CSV",
        ),
        (
            "supersession-05",
            "codex",
            "session",
            "REGION-EPSILON",
            "The previous destination for REGION-EPSILON was us-east-1.",
            "The current destination for REGION-EPSILON is us-west-2.",
            "Where is REGION-EPSILON currently hosted?",
            "us-west-2",
            "Where was REGION-EPSILON hosted before the move?",
            "us-east-1",
        ),
        (
            "supersession-06",
            "claude",
            "provider",
            "TIMEOUT-ZETA",
            "The old request timeout for TIMEOUT-ZETA was 3 seconds.",
            "The active request timeout for TIMEOUT-ZETA is 8 seconds.",
            "What request timeout is active for TIMEOUT-ZETA?",
            "8 seconds",
            "What was the historical timeout for TIMEOUT-ZETA?",
            "3 seconds",
        ),
    )
    for index, (
        prefix,
        provider,
        scope,
        subject,
        older_content,
        newer_content,
        current_query,
        current_span,
        history_query,
        history_span,
    ) in enumerate(supersession_specs, start=1):
        older_key = f"{prefix}-old"
        newer_key = f"{prefix}-new"
        records.extend(
            (
                FixtureRecord(
                    older_key, provider, scope, f"{subject} previous", older_content, "decision"
                ),
                FixtureRecord(
                    newer_key, provider, scope, f"{subject} current", newer_content, "decision"
                ),
            )
        )
        supersessions.append(FixtureSupersession(newer_key, (older_key,)))
        if index <= 3:
            questions.append(
                FixtureQuestion(
                    key=f"question-{prefix}-current",
                    category="superseded_decision",
                    query=current_query,
                    required=(FixtureEvidence(newer_key, current_span),),
                )
            )
        else:
            questions.append(
                FixtureQuestion(
                    key=f"question-{prefix}-history",
                    category="superseded_decision",
                    query=history_query,
                    required=(FixtureEvidence(older_key, history_span, "superseded"),),
                    include_superseded=True,
                )
            )

    distractors: tuple[DistractorSpec, ...] = (
        (
            "distractor-01",
            "codex",
            "session",
            "Orion data steward",
            "Dataset Orion is stewarded by the analytics group, independent of any retention rule.",
        ),
        (
            "distractor-02",
            "claude",
            "provider",
            "Gateway dashboard",
            "Gateway ETA charts display percentile latency but do not define a request timeout.",
        ),
        (
            "distractor-03",
            "codex",
            "provider",
            "Sable region map",
            "Cluster Sable is located in eu-west-1; the map says nothing about backup cadence.",
        ),
        (
            "distractor-04",
            "claude",
            "session",
            "Northwind deploy checklist",
            "Northwind deployment requires an owner review before any region change is applied.",
        ),
    )
    records.extend(
        FixtureRecord(key, provider, scope, title, content, "note")
        for key, provider, scope, title, content in distractors
    )

    return RetrievalFixture(
        version=CORPUS_VERSION,
        records=tuple(records),
        questions=tuple(questions),
        supersessions=tuple(supersessions),
    )
