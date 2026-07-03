#!/usr/bin/env python3
"""Offline health probe for corpus-augmented prompt injection.

This probes a local corpus index directly, without any live server calls or
inference. It runs a small set of representative coding queries against the
v3 sharded corpus by default and reports latency and retrieval quality signals
that can be used to decide whether the corpus is healthy enough for online
prompt injection.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.services.corpus_retrieval import (  # noqa: E402
    CorpusConfig,
    CorpusRetriever,
    extract_code_query,
)

DEFAULT_INDEX_PATH = Path("/mnt/raid0/llm/cache/corpus/v3_sharded")
DEFAULT_MAX_RESULTS = 3
DEFAULT_P95_THRESHOLD_MS = 5000.0
DEFAULT_MIN_SNIPPETS_PER_QUERY = 1.0


@dataclass(frozen=True)
class ProbeQuery:
    """A representative code-oriented retrieval query."""

    id: str
    query: str


@dataclass
class ProbeRecord:
    """Per-query probe result."""

    id: str
    query: str
    normalized_query: str
    elapsed_ms: float
    snippets_returned: int
    candidate_count: int | None
    failure_reason: str
    failure_detail: str
    loaded: bool
    format: str
    shards_queried: int | None = None
    shards_failed: int | None = None
    shards_unavailable: int | None = None


@dataclass
class ProbeSummary:
    """Aggregate health probe summary."""

    index_path: str
    dry_run: bool
    query_count: int
    success_count: int
    failure_count: int
    total_snippets_returned: int
    avg_snippets_returned: float
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    candidate_count_total: int | None
    candidate_count_sampled_queries: int
    failure_reasons: dict[str, int]
    usable_for_online_prompt_injection: bool
    p95_threshold_ms: float
    min_snippets_per_query: float
    records: list[ProbeRecord] = field(default_factory=list)

    def to_dict(self, include_records: bool = False) -> dict[str, Any]:
        data = asdict(self)
        if not include_records:
            data.pop("records", None)
        return data


DEFAULT_QUERIES: tuple[ProbeQuery, ...] = (
    ProbeQuery(
        id="async_retry",
        query=(
            "Write a Python async HTTP client with retry logic, exponential backoff, "
            "and circuit breaker pattern. Include type hints and a usage example."
        ),
    ),
    ProbeQuery(
        id="bst_iterator",
        query=(
            "Implement a binary search tree in Python with an in-order iterator "
            "that uses O(h) memory. Include insert, search, delete, and the iterator protocol."
        ),
    ),
    ProbeQuery(
        id="lru_cache",
        query=(
            "Write a thread-safe LRU cache in Python using a doubly-linked list "
            "and a dictionary. Support get, put, resize, and a decorator version."
        ),
    ),
    ProbeQuery(
        id="json_parser",
        query=(
            "Write a recursive descent JSON parser in Python from scratch. "
            "Handle strings, numbers, booleans, null, arrays, and objects."
        ),
    ),
    ProbeQuery(
        id="rate_limiter",
        query=(
            "Implement a token bucket rate limiter in Python with per-key limits, "
            "burst capacity, and automatic refill."
        ),
    ),
    ProbeQuery(
        id="graph_shortest",
        query=(
            "Write Dijkstra's algorithm and A* search in Python for weighted directed graphs. "
            "Include a priority queue implementation and path reconstruction."
        ),
    ),
)


def _load_queries(path: Path) -> list[ProbeQuery]:
    """Load queries from JSON, JSONL, or a plain text file."""
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    def _coerce(item: Any, idx: int) -> ProbeQuery | None:
        if isinstance(item, str):
            query = item.strip()
            return ProbeQuery(id=f"query_{idx:03d}", query=query) if query else None
        if isinstance(item, dict):
            query = str(item.get("query") or item.get("prompt") or "").strip()
            if not query:
                return None
            return ProbeQuery(
                id=str(item.get("id") or f"query_{idx:03d}"),
                query=query,
            )
        return None

    if path.suffix.lower() in {".json", ".jsonc"}:
        raw = json.loads(text)
        if not isinstance(raw, list):
            raise ValueError(f"expected a JSON array in {path}")
        queries: list[ProbeQuery] = []
        for idx, entry in enumerate(raw, start=1):
            query = _coerce(entry, idx)
            if query is not None:
                queries.append(query)
        return queries

    if path.suffix.lower() == ".jsonl":
        queries: list[ProbeQuery] = []
        for idx, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            queries.append(_coerce(json.loads(line), idx))
        return [query for query in queries if query is not None]

    return [
        ProbeQuery(id=f"query_{idx:03d}", query=line.strip())
        for idx, line in enumerate(text.splitlines(), start=1)
        if line.strip()
    ]


def _select_queries(
    queries: list[ProbeQuery],
    *,
    limit: int | None,
    query_prefix: str | None,
) -> list[ProbeQuery]:
    selected = queries
    if query_prefix:
        needle = query_prefix.lower()
        selected = [query for query in selected if needle in query.id.lower() or needle in query.query.lower()]
    if limit is not None and limit >= 0:
        selected = selected[:limit]
    return selected


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _query_retriever(
    retriever: CorpusRetriever,
    probe_query: ProbeQuery,
) -> ProbeRecord:
    normalized_query = extract_code_query(probe_query.query)
    start = time.perf_counter()
    try:
        snippets = retriever.retrieve(normalized_query)
    except Exception as exc:  # pragma: no cover - defensive, not expected in fixtures
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return ProbeRecord(
            id=probe_query.id,
            query=probe_query.query,
            normalized_query=normalized_query,
            elapsed_ms=elapsed_ms,
            snippets_returned=0,
            candidate_count=None,
            failure_reason="exception",
            failure_detail=f"{type(exc).__name__}: {exc}",
            loaded=False,
            format="",
        )

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    diag = retriever.last_diagnostics
    candidate_count = None
    shards_queried = None
    shards_failed = None
    shards_unavailable = None
    failure_reason = ""
    failure_detail = ""
    loaded = bool(diag.loaded) if diag is not None else False
    format_name = diag.format if diag is not None else ""

    if diag is not None:
        if diag.format == "sharded_sqlite":
            candidate_count = diag.candidates_found
        shards_queried = diag.shards_queried
        shards_failed = diag.shards_failed
        shards_unavailable = diag.shards_unavailable
        failure_reason = diag.failure_reason
        failure_detail = diag.failure_detail

    return ProbeRecord(
        id=probe_query.id,
        query=probe_query.query,
        normalized_query=normalized_query,
        elapsed_ms=elapsed_ms,
        snippets_returned=len(snippets),
        candidate_count=candidate_count,
        failure_reason=failure_reason,
        failure_detail=failure_detail,
        loaded=loaded,
        format=format_name,
        shards_queried=shards_queried,
        shards_failed=shards_failed,
        shards_unavailable=shards_unavailable,
    )


def run_probe(
    *,
    index_path: Path = DEFAULT_INDEX_PATH,
    queries: list[ProbeQuery] | None = None,
    limit: int | None = None,
    query_prefix: str | None = None,
    dry_run: bool = False,
    enabled: bool = True,
    max_results: int = DEFAULT_MAX_RESULTS,
    min_score: float = 0.0,
    p95_threshold_ms: float = DEFAULT_P95_THRESHOLD_MS,
    min_snippets_per_query: float = DEFAULT_MIN_SNIPPETS_PER_QUERY,
) -> ProbeSummary:
    selected_queries = _select_queries(list(queries or DEFAULT_QUERIES), limit=limit, query_prefix=query_prefix)
    if dry_run:
        return ProbeSummary(
            index_path=str(index_path),
            dry_run=True,
            query_count=len(selected_queries),
            success_count=0,
            failure_count=0,
            total_snippets_returned=0,
            avg_snippets_returned=0.0,
            p50_latency_ms=None,
            p95_latency_ms=None,
            candidate_count_total=None,
            candidate_count_sampled_queries=0,
            failure_reasons={},
            usable_for_online_prompt_injection=False,
            p95_threshold_ms=p95_threshold_ms,
            min_snippets_per_query=min_snippets_per_query,
            records=[],
        )

    retriever = CorpusRetriever(
        CorpusConfig(
            enabled=enabled,
            index_path=str(index_path),
            max_snippets=max_results,
            min_score=min_score,
        )
    )

    records = [_query_retriever(retriever, query) for query in selected_queries]
    latencies_ms = [record.elapsed_ms for record in records]
    snippet_counts = [record.snippets_returned for record in records]
    candidate_counts = [record.candidate_count for record in records if record.candidate_count is not None]
    failure_reasons = Counter(
        record.failure_reason for record in records if record.failure_reason
    )

    query_count = len(records)
    total_snippets_returned = sum(snippet_counts)
    avg_snippets_returned = (total_snippets_returned / query_count) if query_count else 0.0
    p50_latency_ms = _percentile(latencies_ms, 50.0)
    p95_latency_ms = _percentile(latencies_ms, 95.0)
    success_count = sum(1 for record in records if record.snippets_returned > 0 and not record.failure_reason)
    failure_count = sum(1 for record in records if record.failure_reason)
    usable = (
        query_count > 0
        and p95_latency_ms is not None
        and p95_latency_ms <= p95_threshold_ms
        and avg_snippets_returned >= min_snippets_per_query
        and failure_count == 0
    )

    return ProbeSummary(
        index_path=str(index_path),
        dry_run=False,
        query_count=query_count,
        success_count=success_count,
        failure_count=failure_count,
        total_snippets_returned=total_snippets_returned,
        avg_snippets_returned=avg_snippets_returned,
        p50_latency_ms=p50_latency_ms,
        p95_latency_ms=p95_latency_ms,
        candidate_count_total=sum(candidate_counts) if candidate_counts else None,
        candidate_count_sampled_queries=len(candidate_counts),
        failure_reasons=dict(sorted(failure_reasons.items())),
        usable_for_online_prompt_injection=usable,
        p95_threshold_ms=p95_threshold_ms,
        min_snippets_per_query=min_snippets_per_query,
        records=records,
    )


def render_markdown(summary: ProbeSummary) -> str:
    """Render a compact markdown-ish summary."""
    lines = [
        "# Corpus Health Probe",
        f"- index_path: `{summary.index_path}`",
        f"- dry_run: `{summary.dry_run}`",
        f"- query_count: `{summary.query_count}`",
        f"- success_count: `{summary.success_count}`",
        f"- failure_count: `{summary.failure_count}`",
        f"- snippets_returned_total: `{summary.total_snippets_returned}`",
        f"- snippets_returned_avg: `{summary.avg_snippets_returned:.2f}`",
        f"- p50_latency_ms: `{summary.p50_latency_ms:.3f}`" if summary.p50_latency_ms is not None else "- p50_latency_ms: `n/a`",
        f"- p95_latency_ms: `{summary.p95_latency_ms:.3f}`" if summary.p95_latency_ms is not None else "- p95_latency_ms: `n/a`",
        f"- candidate_count_total: `{summary.candidate_count_total}`" if summary.candidate_count_total is not None else "- candidate_count_total: `n/a`",
        f"- candidate_count_sampled_queries: `{summary.candidate_count_sampled_queries}`",
        f"- failure_reasons: `{json.dumps(summary.failure_reasons, sort_keys=True)}`",
        f"- p95_threshold_ms: `{summary.p95_threshold_ms}`",
        f"- min_snippets_per_query: `{summary.min_snippets_per_query}`",
        f"- usable_for_online_prompt_injection: `{summary.usable_for_online_prompt_injection}`",
    ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--queries-file", type=Path)
    parser.add_argument("--query-prefix", help="Filter default queries by substring before running.")
    parser.add_argument("--limit", type=int, help="Limit the number of queries to run.")
    parser.add_argument("--dry-run", action="store_true", help="Select queries but do not touch the index.")
    parser.add_argument("--json", action="store_true", help="Emit JSON summary only.")
    parser.add_argument("--max-results", type=int, default=DEFAULT_MAX_RESULTS)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--p95-threshold-ms", type=float, default=DEFAULT_P95_THRESHOLD_MS)
    parser.add_argument("--min-snippets-per-query", type=float, default=DEFAULT_MIN_SNIPPETS_PER_QUERY)
    parser.add_argument("--disabled", action="store_true", help="Force corpus retrieval disabled for probing.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    queries = _load_queries(args.queries_file) if args.queries_file else None
    summary = run_probe(
        index_path=args.index_path,
        queries=queries,
        limit=args.limit,
        query_prefix=args.query_prefix,
        dry_run=args.dry_run,
        enabled=not args.disabled,
        max_results=args.max_results,
        min_score=args.min_score,
        p95_threshold_ms=args.p95_threshold_ms,
        min_snippets_per_query=args.min_snippets_per_query,
    )

    if args.json:
        print(json.dumps(summary.to_dict(include_records=True), indent=2, sort_keys=True))
    else:
        print(render_markdown(summary))
        print()
        print(json.dumps(summary.to_dict(include_records=False), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
