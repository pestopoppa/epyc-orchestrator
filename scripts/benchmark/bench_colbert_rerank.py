#!/usr/bin/env python3
"""Benchmark ColBERT reranker CPU latency.

The benchmark exercises ``src.tools.web.colbert_reranker.rerank_snippets`` with
deterministic synthetic search snippets. It refuses to run when the selected
ONNX model is unavailable, unless ``--skip-if-unavailable`` is set, because the
reranker gracefully falls back to original ordering and that would under-report
real latency.

Examples:
    python scripts/benchmark/bench_colbert_rerank.py \
        --model-path /mnt/raid0/llm/models/lateon-onnx-int8 \
        --model-slot lateon --snippets 20 --iterations 50 --json

    python scripts/benchmark/bench_colbert_rerank.py \
        --model-path /mnt/raid0/llm/models/reason-mxbai-colbert-v0-32m-onnx-int8 \
        --model-slot reason_mxbai --skip-if-unavailable
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

LATEON_ENV = "LATEON_MODEL_PATH"
REASON_MXBAI_ENV = "REASON_MXBAI_MODEL_PATH"


class ModelUnavailableError(RuntimeError):
    """Raised when the selected reranker model files are not present."""


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for ColBERT rerank latency measurement."""

    model_path: Path | None = None
    model_slot: str = "lateon"
    iterations: int = 30
    warmup: int = 3
    snippets: int = 20
    queries: int = 4
    top_k: int = 5
    skip_if_unavailable: bool = False


@dataclass(frozen=True)
class BenchmarkReport:
    """Serializable benchmark output."""

    status: str
    model_slot: str
    model_dir: str
    snippets_per_call: int
    top_k: int
    warmup_calls: int
    measured_calls: int
    total_snippets: int
    first_call_ms: float
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    snippets_per_second: float


QUERIES = [
    "How does ColBERT late interaction improve web search reranking?",
    "What causes transformer inference latency to increase with context length?",
    "How should an orchestration router detect stale benchmark evidence?",
    "Which signals distinguish web search backend noise from infrastructure failure?",
    "What makes a CPU-only reranker practical for a local coding agent?",
    "How can negative transfer corrupt prompt mutation selection?",
    "What evidence is needed before replacing a retrieval model?",
    "How does MaxSim scoring compare query tokens against document tokens?",
]

SNIPPET_TOPICS = [
    (
        "Late interaction retrieval",
        "ColBERT keeps token-level document embeddings and applies MaxSim at query time.",
    ),
    (
        "CPU inference latency",
        "Longer context windows increase attention and cache pressure during decoding.",
    ),
    (
        "Routing evidence hygiene",
        "A router should quarantine contaminated trials before updating policy memory.",
    ),
    (
        "Search backend health",
        "Repeated HTTP failures across engines indicate infrastructure work, not query quality.",
    ),
    (
        "Local web research",
        "A compact ONNX reranker can improve snippet ordering without calling an LLM.",
    ),
    (
        "Prompt mutation safety",
        "Benchmark-specific rules should not be generalized without cited trial support.",
    ),
    (
        "Retrieval model migration",
        "Replacement candidates need paired latency and relevance evidence before promotion.",
    ),
    (
        "MaxSim scoring",
        "Each query token contributes its strongest similarity against document tokens.",
    ),
]


def build_snippets(count: int) -> list[dict[str, str]]:
    """Create deterministic search-like snippets for latency measurement."""
    snippets: list[dict[str, str]] = []
    for idx in range(count):
        title, body = SNIPPET_TOPICS[idx % len(SNIPPET_TOPICS)]
        snippets.append(
            {
                "title": f"{title} #{idx + 1}",
                "snippet": (
                    f"{body} Measurement item {idx + 1} includes enough text "
                    "to exercise tokenization and document encoding paths."
                ),
                "url": f"https://example.invalid/colbert-bench/{idx + 1}",
            }
        )
    return snippets


def build_queries(count: int) -> list[str]:
    """Return a deterministic query subset, repeating if count exceeds the seed set."""
    if count <= 0:
        raise ValueError("queries must be positive")
    return [QUERIES[idx % len(QUERIES)] for idx in range(count)]


def _percentile(values: list[float], percentile: float) -> float:
    """Nearest-rank percentile for small benchmark samples."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, round((percentile / 100.0) * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def configure_model_env(config: BenchmarkConfig) -> None:
    """Apply an explicit model-path slot override before importing reranker."""
    if config.model_path is None:
        return

    os.environ.pop(LATEON_ENV, None)
    os.environ.pop(REASON_MXBAI_ENV, None)

    if config.model_slot == "lateon":
        os.environ[LATEON_ENV] = str(config.model_path)
    elif config.model_slot == "reason_mxbai":
        os.environ[REASON_MXBAI_ENV] = str(config.model_path)
    else:
        raise ValueError(f"unsupported model_slot: {config.model_slot}")


def load_reranker(config: BenchmarkConfig) -> ModuleType:
    """Import the reranker after any env override is in place."""
    configure_model_env(config)
    module = importlib.import_module("src.tools.web.colbert_reranker")
    return importlib.reload(module)


def run_benchmark(config: BenchmarkConfig, reranker: Any) -> BenchmarkReport:
    """Run warmup + measured rerank calls and return latency metrics."""
    if config.iterations <= 0:
        raise ValueError("iterations must be positive")
    if config.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if config.snippets <= 0:
        raise ValueError("snippets must be positive")
    if config.queries <= 0:
        raise ValueError("queries must be positive")
    if config.top_k <= 0:
        raise ValueError("top_k must be positive")

    model_slot = str(getattr(reranker, "_MODEL_SLOT", "unknown"))
    model_dir = str(getattr(reranker, "_MODEL_DIR", "unknown"))

    if not reranker.is_available():
        raise ModelUnavailableError(f"ColBERT model unavailable: slot={model_slot} dir={model_dir}")

    snippets = build_snippets(config.snippets)
    queries = build_queries(config.queries)
    measured_latencies_ms: list[float] = []
    first_call_ms = 0.0

    total_calls = config.warmup + config.iterations
    for call_idx in range(total_calls):
        query = queries[call_idx % len(queries)]
        start = time.perf_counter()
        reranker.rerank_snippets(query, snippets, top_k=config.top_k)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if call_idx == 0:
            first_call_ms = elapsed_ms
        if call_idx >= config.warmup:
            measured_latencies_ms.append(elapsed_ms)

    total_measured_ms = sum(measured_latencies_ms)
    total_snippets = config.iterations * config.snippets
    snippets_per_second = (
        (total_snippets / total_measured_ms) * 1000.0
        if total_measured_ms > 0
        else 0.0
    )

    return BenchmarkReport(
        status="ok",
        model_slot=model_slot,
        model_dir=model_dir,
        snippets_per_call=config.snippets,
        top_k=config.top_k,
        warmup_calls=config.warmup,
        measured_calls=config.iterations,
        total_snippets=total_snippets,
        first_call_ms=round(first_call_ms, 3),
        mean_ms=round(statistics.fmean(measured_latencies_ms), 3),
        median_ms=round(statistics.median(measured_latencies_ms), 3),
        p95_ms=round(_percentile(measured_latencies_ms, 95), 3),
        min_ms=round(min(measured_latencies_ms), 3),
        max_ms=round(max(measured_latencies_ms), 3),
        snippets_per_second=round(snippets_per_second, 3),
    )


def make_skipped_report(config: BenchmarkConfig, reranker: Any) -> BenchmarkReport:
    """Return a zero-valued report for automation that intentionally skips."""
    return BenchmarkReport(
        status="skipped_unavailable",
        model_slot=str(getattr(reranker, "_MODEL_SLOT", "unknown")),
        model_dir=str(getattr(reranker, "_MODEL_DIR", "unknown")),
        snippets_per_call=config.snippets,
        top_k=config.top_k,
        warmup_calls=config.warmup,
        measured_calls=0,
        total_snippets=0,
        first_call_ms=0.0,
        mean_ms=0.0,
        median_ms=0.0,
        p95_ms=0.0,
        min_ms=0.0,
        max_ms=0.0,
        snippets_per_second=0.0,
    )


def format_text(report: BenchmarkReport) -> str:
    """Format a compact human-readable report."""
    return "\n".join(
        [
            f"status: {report.status}",
            f"model: {report.model_slot} ({report.model_dir})",
            f"workload: {report.measured_calls} calls x {report.snippets_per_call} snippets",
            f"first_call_ms: {report.first_call_ms:.3f}",
            f"mean_ms: {report.mean_ms:.3f}",
            f"median_ms: {report.median_ms:.3f}",
            f"p95_ms: {report.p95_ms:.3f}",
            f"min_ms: {report.min_ms:.3f}",
            f"max_ms: {report.max_ms:.3f}",
            f"snippets_per_second: {report.snippets_per_second:.3f}",
        ]
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, help="Model directory to benchmark")
    parser.add_argument(
        "--model-slot",
        choices=["lateon", "reason_mxbai"],
        default="lateon",
        help="Env slot to use when --model-path is provided",
    )
    parser.add_argument("--iterations", type=int, default=30, help="Measured rerank calls")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup calls excluded from metrics")
    parser.add_argument("--snippets", type=int, default=20, help="Snippets per rerank call")
    parser.add_argument("--queries", type=int, default=4, help="Distinct synthetic queries to cycle")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k snippets returned by reranker")
    parser.add_argument(
        "--skip-if-unavailable",
        action="store_true",
        help="Return success with skipped_unavailable report when model files are absent",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = BenchmarkConfig(
        model_path=args.model_path,
        model_slot=args.model_slot,
        iterations=args.iterations,
        warmup=args.warmup,
        snippets=args.snippets,
        queries=args.queries,
        top_k=args.top_k,
        skip_if_unavailable=args.skip_if_unavailable,
    )
    reranker = load_reranker(config)

    try:
        report = run_benchmark(config, reranker)
    except ModelUnavailableError as exc:
        if not config.skip_if_unavailable:
            print(str(exc), file=sys.stderr)
            return 2
        report = make_skipped_report(config, reranker)

    if args.json:
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
    else:
        print(format_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
