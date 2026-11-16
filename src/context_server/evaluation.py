from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

from platformdirs import user_cache_path

from context_server.embeddings import FastEmbedder
from context_server.memory import MemoryService, SearchHit
from context_server.retrieval_fixture import (
    CORPUS_VERSION,
    MINIMUM_EVIDENCE_HIT_AT_5,
    FixtureEvidence,
    RetrievalFixture,
    build_retrieval_fixture,
)
from context_server.storage import MemoryDatabase

_MINIMUM_EXACT_IDENTIFIER_HIT_AT_5 = 1.0
_MINIMUM_ACTIVE_CONFLICT_COMPLETION_AT_10 = 1.0
_MINIMUM_CURRENT_DECISION_HIT = 1.0
_MINIMUM_HISTORICAL_EVIDENCE_HIT = 1.0


def _validate_fixture(fixture: RetrievalFixture) -> None:
    if fixture.version != CORPUS_VERSION:
        raise ValueError(f"Unsupported retrieval fixture version: {fixture.version}")
    if len(fixture.records) != 40 or len(fixture.questions) != 24:
        raise ValueError("Retrieval fixture must contain exactly 40 records and 24 questions")
    record_keys = [record.key for record in fixture.records]
    question_keys = [question.key for question in fixture.questions]
    if len(set(record_keys)) != len(record_keys) or len(set(question_keys)) != len(question_keys):
        raise ValueError("Retrieval fixture keys must be unique")
    categories = Counter(question.category for question in fixture.questions)
    if categories != {
        "exact_identifier": 6,
        "paraphrased_decision": 6,
        "active_conflict": 6,
        "superseded_decision": 6,
    }:
        raise ValueError("Retrieval fixture must contain six questions in each required category")


def _evidence_rank(
    evidence: FixtureEvidence,
    hits: list[SearchHit],
    document_ids: dict[str, str],
    record_providers: dict[str, str],
) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if (
            hit.document_id == document_ids[evidence.record_key]
            and evidence.span in hit.snippet
            and hit.status == evidence.status
            and hit.author_provider == record_providers[evidence.record_key]
        ):
            return rank
    return None


async def evaluate_retrieval(
    *, model_cache_dir: Path | None = None
) -> dict[str, Any]:
    fixture = build_retrieval_fixture()
    _validate_fixture(fixture)
    configured_cache = os.environ.get("CONTEXT_SERVER_MODEL_CACHE_DIR")
    cache_dir = (
        model_cache_dir
        if model_cache_dir is not None
        else Path(configured_cache).expanduser()
        if configured_cache
        else user_cache_path("Polygentic", appauthor=False) / "models"
    )
    with tempfile.TemporaryDirectory(prefix="polygentic-retrieval-eval-") as temporary_dir:
        database = MemoryDatabase(Path(temporary_dir) / "memory")
        try:
            memory = MemoryService(database, FastEmbedder(cache_dir))
            sessions = {}
            for provider in ("codex", "claude"):
                sessions[provider] = await memory.register_session(
                    provider, f"Retrieval evaluation {provider}", f"eval-register-{provider}"
                )

            document_ids: dict[str, str] = {}
            for record in fixture.records:
                session = sessions[record.provider]
                space_id = (
                    session.provider_space_id
                    if record.scope == "provider"
                    else session.session_space_id
                )
                stored = await memory.put_context(
                    provider=record.provider,
                    session_id=session.session_id,
                    space_id=space_id,
                    title=record.title,
                    content=record.content,
                    record_type=record.record_type,
                    idempotency_key=f"eval-write-{record.key}",
                )
                document_ids[record.key] = stored.document_id

            records_by_key = {record.key: record for record in fixture.records}
            for supersession in fixture.supersessions:
                newer = records_by_key[supersession.newer_key]
                session = sessions[newer.provider]
                older_ids = [document_ids[key] for key in supersession.older_keys]
                await memory.supersede_context(
                    provider=newer.provider,
                    session_id=session.session_id,
                    newer_document_id=document_ids[supersession.newer_key],
                    older_document_ids=older_ids,
                    expected_revisions={document_id: 1 for document_id in older_ids},
                    idempotency_key=f"eval-supersede-{supersession.newer_key}",
                )

            default_superseded_hits = 0
            unrecognized_hits = 0
            budget_violations = 0
            category_counts: Counter[str] = Counter()
            category_hits_at_5: Counter[str] = Counter()
            category_complete_at_10: Counter[str] = Counter()
            category_required_evidence: Counter[str] = Counter()
            category_visible_evidence_at_5: Counter[str] = Counter()
            question_results: list[dict[str, Any]] = []
            latencies_ms: list[float] = []
            failures: list[dict[str, Any]] = []
            required_evidence_count = 0
            visible_evidence_at_5_count = 0
            visible_evidence_at_10_count = 0
            reciprocal_rank_sum = 0.0
            valid_document_ids = set(document_ids.values())
            record_providers = {record.key: record.provider for record in fixture.records}
            record_keys_by_id = {
                document_id: key for key, document_id in document_ids.items()
            }

            for question in fixture.questions:
                started = time.perf_counter()
                response = await memory.search_context(
                    provider="codex",
                    session_id=sessions["codex"].session_id,
                    query=question.query,
                    mode="hybrid",
                    top_k=10,
                    include_superseded=question.include_superseded,
                )
                elapsed_ms = (time.perf_counter() - started) * 1000
                latencies_ms.append(elapsed_ms)
                category_counts[question.category] += 1
                evidence_ranks = [
                    _evidence_rank(
                        evidence, response.results, document_ids, record_providers
                    )
                    for evidence in question.required
                ]
                evidence_diagnostics = [
                    {
                        "record_key": evidence.record_key,
                        "span": evidence.span,
                        "expected_status": evidence.status,
                        "expected_provider": record_providers[evidence.record_key],
                        "rank": rank,
                        "observed_provider": (
                            response.results[rank - 1].author_provider
                            if rank is not None
                            else None
                        ),
                        "provenance_matches": rank is not None,
                    }
                    for evidence, rank in zip(
                        question.required, evidence_ranks, strict=True
                    )
                ]
                visible_at_5 = sum(
                    rank is not None and rank <= 5 for rank in evidence_ranks
                )
                visible_at_10 = sum(
                    rank is not None and rank <= 10 for rank in evidence_ranks
                )
                required_evidence_count += len(evidence_ranks)
                visible_evidence_at_5_count += visible_at_5
                visible_evidence_at_10_count += visible_at_10
                category_required_evidence[question.category] += len(evidence_ranks)
                category_visible_evidence_at_5[question.category] += visible_at_5
                reciprocal_rank_sum += sum(
                    1 / rank for rank in evidence_ranks if rank is not None
                )
                complete_at_5 = visible_at_5 == len(question.required)
                complete_at_10 = visible_at_10 == len(question.required)
                if complete_at_5:
                    category_hits_at_5[question.category] += 1
                if complete_at_10:
                    category_complete_at_10[question.category] += 1

                superseded_in_default = (
                    0
                    if question.include_superseded
                    else sum(hit.status == "superseded" for hit in response.results)
                )
                default_superseded_hits += superseded_in_default
                unrecognized = sum(hit.document_id not in valid_document_ids for hit in response.results)
                unrecognized_hits += unrecognized
                if response.snippet_characters > 12_000 or any(
                    len(hit.snippet) > 1000 for hit in response.results
                ):
                    budget_violations += 1

                missing = [
                    item
                    for item, rank in zip(
                        evidence_diagnostics, evidence_ranks, strict=True
                    )
                    if rank is None
                ]
                max_snippet_characters = max(
                    (len(hit.snippet) for hit in response.results), default=0
                )
                question_result = {
                    "question": question.key,
                    "category": question.category,
                    "include_superseded": question.include_superseded,
                    "required_evidence": len(question.required),
                    "visible_at_5": visible_at_5,
                    "visible_at_10": visible_at_10,
                    "complete_at_5": complete_at_5,
                    "complete_at_10": complete_at_10,
                    "evidence": evidence_diagnostics,
                    "ranked_results": [
                        {
                            "rank": rank,
                            "record_key": record_keys_by_id.get(
                                hit.document_id, "unrecognized"
                            ),
                            "title": hit.title,
                            "score": hit.score,
                            "status": hit.status,
                            "author_provider": hit.author_provider,
                        }
                        for rank, hit in enumerate(response.results, start=1)
                    ],
                    "latency_ms": round(elapsed_ms, 2),
                    "response_budget": {
                        "snippet_characters": response.snippet_characters,
                        "max_snippet_characters": max_snippet_characters,
                        "truncated": response.truncated,
                    },
                    "missing_evidence": missing,
                }
                question_results.append(question_result)
                if not complete_at_5:
                    failures.append(
                        {
                            "question": question.key,
                            "failure": "required evidence was not fully visible in the top five",
                            "missing": [
                                evidence.record_key
                                for evidence, rank in zip(
                                    question.required, evidence_ranks, strict=True
                                )
                                if rank is None or rank > 5
                            ],
                        }
                    )
                if (
                    question.category == "superseded_decision"
                    and not complete_at_10
                ):
                    failures.append(
                        {
                            "question": question.key,
                            "failure": "current or historical supersession evidence was not retrieved",
                            "missing": [
                                evidence.record_key
                                for evidence, rank in zip(
                                    question.required, evidence_ranks, strict=True
                                )
                                if rank is None or rank > 10
                            ],
                        }
                    )
                if (
                    question.category == "active_conflict"
                    and not complete_at_10
                ):
                    failures.append(
                        {
                            "question": question.key,
                            "failure": "both active conflicting records were not visible in the top ten",
                            "missing": [
                                evidence.record_key
                                for evidence, rank in zip(
                                    question.required, evidence_ranks, strict=True
                                )
                                if rank is None or rank > 10
                            ],
                        }
                    )
                if superseded_in_default:
                    failures.append(
                        {
                            "question": question.key,
                            "failure": "default retrieval returned superseded evidence",
                            "count": superseded_in_default,
                        }
                    )

            category_rate_at_5 = {
                category: category_hits_at_5[category] / count
                for category, count in category_counts.items()
            }
            category_rate_at_10 = {
                category: category_complete_at_10[category] / count
                for category, count in category_counts.items()
            }
            overall_hit_at_5 = (
                visible_evidence_at_5_count / required_evidence_count
            )
            category_evidence_hit_at_5 = {
                category: category_visible_evidence_at_5[category] / count
                for category, count in category_required_evidence.items()
            }
            ordered_latencies_ms = sorted(latencies_ms)
            p95_latency_index = max(0, round(0.95 * len(ordered_latencies_ms)) - 1)
            exact_hit_at_5 = category_rate_at_5["exact_identifier"]
            conflict_complete_at_10 = category_rate_at_10["active_conflict"]
            history_questions = [
                question
                for question in fixture.questions
                if question.category == "superseded_decision" and question.include_superseded
            ]
            history_hit_count = sum(
                1
                for result in question_results
                if result["category"] == "superseded_decision"
                and result["include_superseded"]
                and result["complete_at_10"]
            )
            history_rate = history_hit_count / len(history_questions)
            current_question_count = category_counts["superseded_decision"] - len(history_questions)
            current_hit_count = sum(
                1
                for result in question_results
                if result["category"] == "superseded_decision"
                and not result["include_superseded"]
                and result["complete_at_10"]
            )
            current_decision_hit = current_hit_count / current_question_count
            thresholds = {
                "overall_evidence_hit_at_5": MINIMUM_EVIDENCE_HIT_AT_5,
                "exact_identifier_hit_at_5": _MINIMUM_EXACT_IDENTIFIER_HIT_AT_5,
                "active_conflict_complete_at_10": _MINIMUM_ACTIVE_CONFLICT_COMPLETION_AT_10,
                "current_decision_hit": _MINIMUM_CURRENT_DECISION_HIT,
                "historical_evidence_hit": _MINIMUM_HISTORICAL_EVIDENCE_HIT,
            }
            metrics = {
                "overall_evidence_hit_at_5": overall_hit_at_5,
                "category_evidence_hit_at_5": category_evidence_hit_at_5,
                "question_complete_at_5": sum(category_hits_at_5.values())
                / len(fixture.questions),
                "labeled_evidence_recall_at_10": (
                    visible_evidence_at_10_count / required_evidence_count
                ),
                "mean_reciprocal_rank": reciprocal_rank_sum / required_evidence_count,
                "exact_identifier_hit_at_5": exact_hit_at_5,
                "active_conflict_complete_at_10": conflict_complete_at_10,
                "current_decision_hit": current_decision_hit,
                "historical_evidence_hit": history_rate,
                "default_superseded_hits": default_superseded_hits,
                "unrecognized_hits": unrecognized_hits,
                "response_budget_violations": budget_violations,
                "latency_ms": {
                    "mean": round(sum(latencies_ms) / len(latencies_ms), 2),
                    "p95": round(ordered_latencies_ms[p95_latency_index], 2),
                },
            }
            checks = {
                "overall_evidence_hit_at_5": overall_hit_at_5 >= MINIMUM_EVIDENCE_HIT_AT_5,
                "exact_identifier_hit_at_5": exact_hit_at_5
                >= _MINIMUM_EXACT_IDENTIFIER_HIT_AT_5,
                "active_conflict_complete_at_10": conflict_complete_at_10
                >= _MINIMUM_ACTIVE_CONFLICT_COMPLETION_AT_10,
                "current_decision_hit": current_decision_hit >= _MINIMUM_CURRENT_DECISION_HIT,
                "historical_evidence_hit": history_rate >= _MINIMUM_HISTORICAL_EVIDENCE_HIT,
                "zero_default_superseded_hits": default_superseded_hits == 0,
                "zero_unrecognized_hits": unrecognized_hits == 0,
                "response_budgets_respected": budget_violations == 0,
            }
            return {
                "corpus_version": fixture.version,
                "record_count": len(fixture.records),
                "question_count": len(fixture.questions),
                "thresholds": thresholds,
                "metrics": metrics,
                "checks": checks,
                "passed": all(checks.values()),
                "threshold_failures": [
                    name for name, passed in checks.items() if not passed
                ],
                "question_results": question_results,
                "question_failures": failures,
            }
        finally:
            database.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate hybrid retrieval against the versioned local synthetic corpus."
    )
    parser.add_argument("--model-cache-dir", type=Path)
    args = parser.parse_args()
    report = asyncio.run(evaluate_retrieval(model_cache_dir=args.model_cache_dir))
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
