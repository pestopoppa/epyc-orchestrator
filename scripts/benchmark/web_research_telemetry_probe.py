#!/usr/bin/env python3
"""Run direct web_research telemetry probes without exercising role routing.

This is the tool-only half of the web_research gate: search, fetch, synthesis,
and relevance counters. It intentionally bypasses the orchestrator /chat route
so request-path and routing behavior can be validated separately.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ORCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ORCH_ROOT))

DEFAULT_SENTINEL = ORCH_ROOT / "orchestration" / "deep_research_sentinel.yaml"
DEFAULT_OUTPUT_ROOT = ORCH_ROOT / "benchmarks" / "results" / "eval"


def _utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_deep_research_sentinel(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text()) or []
    prompts: list[dict[str, Any]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        prompt = str(item.get("prompt") or "").strip()
        if not prompt:
            continue
        prompts.append(
            {
                "id": str(item.get("id") or f"prompt_{idx:03d}"),
                "suite": str(item.get("suite") or "deep_research"),
                "style": str(item.get("style") or ""),
                "query": prompt,
            }
        )
    return prompts


def _load_jsonl_queries(path: Path) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    with path.open() as fh:
        for idx, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if isinstance(raw, str):
                query = raw
                item_id = f"query_{idx:03d}"
            elif isinstance(raw, dict):
                query = str(raw.get("query") or raw.get("prompt") or "").strip()
                item_id = str(raw.get("id") or f"query_{idx:03d}")
            else:
                continue
            if query:
                prompts.append(
                    {
                        "id": item_id,
                        "suite": str(raw.get("suite") or "custom") if isinstance(raw, dict) else "custom",
                        "style": str(raw.get("style") or "") if isinstance(raw, dict) else "",
                        "query": query,
                    }
                )
    return prompts


def _bounded_prompts(prompts: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None or limit <= 0:
        return prompts
    return prompts[:limit]


def _shard_prompts(
    prompts: list[dict[str, Any]],
    *,
    shard_count: int,
    shard_index: int,
) -> list[dict[str, Any]]:
    if shard_count <= 1:
        return prompts
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("--shard-index must be in [0, --shard-count)")
    return [prompt for idx, prompt in enumerate(prompts) if idx % shard_count == shard_index]


def _socket_reachable(url: str, timeout_s: float) -> bool:
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def _source_metadata(result: dict[str, Any]) -> list[dict[str, Any]]:
    sources = []
    for source in result.get("sources") or []:
        if not isinstance(source, dict):
            continue
        sources.append(
            {
                "title": source.get("title"),
                "url": source.get("url"),
                "relevant": source.get("relevant"),
                "has_synthesis": "synthesis" in source,
            }
        )
    return sources


def _record_for_prompt(prompt: dict[str, Any], result: dict[str, Any], elapsed_ms: float) -> dict[str, Any]:
    return {
        "id": prompt["id"],
        "suite": prompt.get("suite"),
        "style": prompt.get("style"),
        "query": prompt["query"],
        "success": bool(result.get("success")),
        "error": result.get("error"),
        "search_backend": result.get("search_backend"),
        "search_result_count": int(result.get("search_result_count") or 0),
        "pages_attempted": int(result.get("pages_attempted") or 0),
        "pages_fetched": int(result.get("pages_fetched") or 0),
        "pages_fetched_successful": int(result.get("pages_fetched_successful") or 0),
        "pages_synthesized": int(result.get("pages_synthesized") or 0),
        "pages_irrelevant": int(result.get("pages_irrelevant") or 0),
        "irrelevant_rate": float(result.get("irrelevant_rate") or 0.0),
        "fetch_failures": int(result.get("fetch_failures") or 0),
        "synthesis_failures": int(result.get("synthesis_failures") or 0),
        "dedup_paragraphs_removed": int(result.get("dedup_paragraphs_removed") or 0),
        "dedup_chars_saved": int(result.get("dedup_chars_saved") or 0),
        "total_elapsed_ms": float(result.get("total_elapsed_ms") or elapsed_ms),
        "probe_elapsed_ms": elapsed_ms,
        "reranked": bool(result.get("reranked")),
        "no_results_reason": result.get("no_results_reason"),
        "sources": _source_metadata(result),
    }


def _sum(records: list[dict[str, Any]], key: str) -> int:
    return sum(int(record.get(key) or 0) for record in records)


def _summary(
    *,
    args: argparse.Namespace,
    run_id: str,
    out_dir: Path,
    records: list[dict[str, Any]],
    started_at: str,
) -> dict[str, Any]:
    synthesized = _sum(records, "pages_synthesized")
    irrelevant = _sum(records, "pages_irrelevant")
    irrelevant_rate = (irrelevant / synthesized) if synthesized else 0.0
    return {
        "run_id": run_id,
        "created_at": started_at,
        "finished_at": _utc_iso(),
        "mode": "tool_only",
        "source": args.source,
        "source_path": str(args.query_file or args.deep_research_sentinel),
        "records_jsonl": str(out_dir / "records.jsonl"),
        "query_count": len(records),
        "success_count": sum(1 for record in records if record.get("success")),
        "search_backends": sorted({record.get("search_backend") for record in records if record.get("search_backend")}),
        "max_results": args.max_results,
        "max_pages": args.max_pages,
        "synthesis_workers": args.synthesis_workers,
        "worker_timeout_s": args.worker_timeout,
        "worker_url": args.worker_url,
        "search_results_total": _sum(records, "search_result_count"),
        "pages_attempted_total": _sum(records, "pages_attempted"),
        "pages_fetched_total": _sum(records, "pages_fetched"),
        "pages_fetched_successful_total": _sum(records, "pages_fetched_successful"),
        "pages_synthesized_total": synthesized,
        "pages_irrelevant_total": irrelevant,
        "irrelevant_rate": round(irrelevant_rate, 4),
        "fetch_failures_total": _sum(records, "fetch_failures"),
        "synthesis_failures_total": _sum(records, "synthesis_failures"),
        "dedup_paragraphs_removed_total": _sum(records, "dedup_paragraphs_removed"),
        "dedup_chars_saved_total": _sum(records, "dedup_chars_saved"),
        "avg_elapsed_ms": (
            sum(float(record.get("probe_elapsed_ms") or 0.0) for record in records) / len(records)
            if records
            else 0.0
        ),
        "s5_irrelevance_threshold": args.irrelevant_threshold,
        "s5_gate_signal": irrelevant_rate >= args.irrelevant_threshold if synthesized else False,
        "command": [Path(sys.argv[0]).name, *sys.argv[1:]],
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=f"web-research-direct-{_utc_compact()}")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--source",
        choices=("deep-research-sentinel", "jsonl"),
        default="deep-research-sentinel",
    )
    parser.add_argument("--deep-research-sentinel", type=Path, default=DEFAULT_SENTINEL)
    parser.add_argument("--query-file", type=Path)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means all loaded prompts")
    parser.add_argument("--max-results", type=int, default=5)
    parser.add_argument("--max-pages", type=int, default=3)
    parser.add_argument("--synthesis-workers", type=int, default=1)
    parser.add_argument("--worker-timeout", type=int, default=120)
    parser.add_argument("--worker-url", default="http://localhost:8082/completion")
    parser.add_argument("--require-worker", action="store_true")
    parser.add_argument("--irrelevant-threshold", type=float, default=0.20)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.source == "jsonl" and args.query_file is None:
        raise SystemExit("--query-file is required with --source jsonl")

    if args.require_worker and not _socket_reachable(args.worker_url, timeout_s=3.0):
        raise SystemExit(f"worker endpoint is not reachable: {args.worker_url}")

    from src.tools.web import research

    research._MAX_SYNTH_WORKERS = max(1, int(args.synthesis_workers))
    research._WORKER_TIMEOUT = max(1, int(args.worker_timeout))
    research._WORKER_URL = args.worker_url

    prompts = (
        _load_jsonl_queries(args.query_file)
        if args.source == "jsonl"
        else _load_deep_research_sentinel(args.deep_research_sentinel)
    )
    prompts = _shard_prompts(
        prompts,
        shard_count=max(1, int(args.shard_count)),
        shard_index=int(args.shard_index),
    )
    prompts = _bounded_prompts(prompts, args.limit)
    if not prompts:
        raise SystemExit("no prompts loaded")

    out_dir = args.output_root / args.run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    started_at = _utc_iso()
    records: list[dict[str, Any]] = []

    records_path = out_dir / "records.jsonl"
    with records_path.open("w") as fh:
        for idx, prompt in enumerate(prompts, start=1):
            print(f"[{idx}/{len(prompts)}] {prompt['id']}: {prompt['query'][:90]}", flush=True)
            start = time.perf_counter()
            result = research._web_research_impl(
                prompt["query"],
                max_results=args.max_results,
                max_pages=args.max_pages,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            record = _record_for_prompt(prompt, result, elapsed_ms)
            records.append(record)
            fh.write(json.dumps(record, sort_keys=True) + os.linesep)
            fh.flush()

    summary = _summary(
        args=args,
        run_id=args.run_id,
        out_dir=out_dir,
        records=records,
        started_at=started_at,
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + os.linesep)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
