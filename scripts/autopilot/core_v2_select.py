#!/usr/bin/env python3
"""Select a candidate core_v2 fixed eval core from calibration outcomes.

This is a zero-inference utility. It expects prior calibration/eval journal rows
with compact ``eval_details.question_results`` vectors, estimates per-item
correctness, selects medium-difficulty items, and writes a designed-core JSONL
that EvalTower can load with ``AUTOPILOT_T1_CORE_ID``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ORCH_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JOURNAL = ORCH_ROOT / "orchestration" / "autopilot_journal.jsonl"
DEFAULT_CORE_OUT = ORCH_ROOT / "benchmarks" / "prompts" / "core_v2.jsonl"
DEFAULT_REPORT_OUT = ORCH_ROOT / "orchestration" / "reports" / "core_v2_selection.json"
DEFAULT_POOL_CANDIDATES = (
    ORCH_ROOT / "benchmarks" / "prompts" / "question_pool.jsonl",
    Path("/mnt/raid0/llm/epyc-inference-research/benchmarks/prompts/question_pool.jsonl"),
)
POOL_METADATA_KEY = "__pool_metadata__"


@dataclass
class ItemStats:
    qid: str
    suite: str
    attempts: int = 0
    correct: int = 0
    partitions: set[str] = field(default_factory=set)

    @property
    def p_correct(self) -> float:
        return self.correct / self.attempts if self.attempts else 0.0

    @property
    def difficulty_distance(self) -> float:
        return abs(self.p_correct - 0.5)

    def as_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "suite": self.suite,
            "attempts": self.attempts,
            "correct": self.correct,
            "p_correct": round(self.p_correct, 6),
            "partitions": sorted(self.partitions),
        }


def _stable_question_qid(suite: str, prompt_text: str) -> str:
    payload = f"{suite}\x00{prompt_text}".encode("utf-8", errors="replace")
    return hashlib.sha1(payload).hexdigest()[:16]


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "correct"}
    return False


def iter_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
    return rows


def _eval_details(row: dict[str, Any]) -> dict[str, Any]:
    details = row.get("eval_details") or {}
    return details if isinstance(details, dict) else {}


def _nested_details(row: dict[str, Any]) -> dict[str, Any]:
    details = _eval_details(row).get("details") or {}
    return details if isinstance(details, dict) else {}


def question_results(row: dict[str, Any]) -> list[dict[str, Any]]:
    details = _eval_details(row)
    nested = _nested_details(row)
    for key in ("question_results", "per_question_results", "per_question"):
        raw = details.get(key)
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict)]
        raw = nested.get(key)
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict)]
    return []


def collect_item_stats(
    rows: list[dict[str, Any]],
    *,
    include_partitions: set[str],
) -> dict[tuple[str, str], ItemStats]:
    stats: dict[tuple[str, str], ItemStats] = {}
    for row in rows:
        if row.get("trial_id") is None:
            continue
        for result in question_results(row):
            suite = str(result.get("suite") or "unknown")
            qid = str(
                result.get("qid")
                or result.get("question_id")
                or result.get("id")
                or ""
            ).strip()
            if not qid:
                continue
            partition = str(result.get("partition") or "core")
            if partition not in include_partitions:
                continue
            item = stats.setdefault((suite, qid), ItemStats(qid=qid, suite=suite))
            item.attempts += 1
            item.correct += int(_safe_bool(result.get("correct")))
            item.partitions.add(partition)
    return stats


def _candidate_sort_key(item: ItemStats) -> tuple[float, int, str, str]:
    return (item.difficulty_distance, -item.attempts, item.suite, item.qid)


def select_core_items(
    stats: dict[tuple[str, str], ItemStats],
    *,
    target_size: int,
    min_attempts: int,
    p_min: float,
    p_max: float,
    max_per_suite: int,
) -> list[ItemStats]:
    by_suite: dict[str, list[ItemStats]] = {}
    for item in stats.values():
        if item.attempts < min_attempts:
            continue
        if not (p_min <= item.p_correct <= p_max):
            continue
        by_suite.setdefault(item.suite, []).append(item)

    for suite in by_suite:
        by_suite[suite].sort(key=_candidate_sort_key)

    selected: list[ItemStats] = []
    per_suite_counts = {suite: 0 for suite in by_suite}
    suites = sorted(by_suite)
    while len(selected) < target_size and suites:
        progressed = False
        for suite in list(suites):
            if len(selected) >= target_size:
                break
            if max_per_suite > 0 and per_suite_counts[suite] >= max_per_suite:
                suites.remove(suite)
                continue
            if not by_suite[suite]:
                suites.remove(suite)
                continue
            selected.append(by_suite[suite].pop(0))
            per_suite_counts[suite] += 1
            progressed = True
        if not progressed:
            break
    return selected


def _default_pool_path() -> Path | None:
    for path in DEFAULT_POOL_CANDIDATES:
        if path.exists():
            return path
    return None


def load_pool_lookup(pool_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    with pool_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict) or row.get(POOL_METADATA_KEY):
                continue
            suite = str(row.get("suite") or "unknown")
            qid = str(row.get("id") or row.get("question_id") or "").strip()
            if qid:
                lookup.setdefault((suite, qid), row)
                lookup.setdefault((suite, f"{suite}/{qid}"), row)
            prompt = str(row.get("prompt") or "")
            if prompt:
                lookup.setdefault((suite, _stable_question_qid(suite, prompt)), row)
    return lookup


def build_report(
    *,
    rows: list[dict[str, Any]],
    selected: list[ItemStats],
    stats: dict[tuple[str, str], ItemStats],
    unresolved: list[ItemStats],
    args: argparse.Namespace,
) -> dict[str, Any]:
    eligible = [
        item
        for item in stats.values()
        if item.attempts >= args.min_attempts
        and args.p_min <= item.p_correct <= args.p_max
    ]
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "core_id": args.core_id,
        "parameters": {
            "target_size": args.target_size,
            "min_attempts": args.min_attempts,
            "p_min": args.p_min,
            "p_max": args.p_max,
            "max_per_suite": args.max_per_suite,
            "include_partitions": sorted(set(args.include_partition)),
        },
        "source_rows": len(rows),
        "observed_items": len(stats),
        "eligible_items": len(eligible),
        "selected_count": len(selected),
        "unresolved_selected_count": len(unresolved),
        "shortfall": max(0, args.target_size - len(selected)),
        "selected": [item.as_dict() for item in selected],
        "unresolved_selected": [item.as_dict() for item in unresolved],
    }


def write_core_jsonl(
    *,
    path: Path,
    core_id: str,
    selected: list[ItemStats],
    pool_lookup: dict[tuple[str, str], dict[str, Any]],
    report: dict[str, Any],
) -> list[ItemStats]:
    path.parent.mkdir(parents=True, exist_ok=True)
    unresolved: list[ItemStats] = []
    with path.open("w", encoding="utf-8") as handle:
        metadata = {
            "__core_metadata__": True,
            "core_id": core_id,
            "generated_at": report["generated_at"],
            "generator": "scripts/autopilot/core_v2_select.py",
            "target_size": report["parameters"]["target_size"],
            "selected_count": report["selected_count"],
            "selection_report": report,
        }
        handle.write(json.dumps(metadata, sort_keys=True) + "\n")
        for item in selected:
            question = pool_lookup.get((item.suite, item.qid))
            if question is None:
                unresolved.append(item)
                continue
            row = dict(question)
            row.setdefault("suite", item.suite)
            row["core_selection"] = {
                "qid": item.qid,
                "attempts": item.attempts,
                "correct": item.correct,
                "p_correct": round(item.p_correct, 6),
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return unresolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--journal",
        action="append",
        type=Path,
        default=[],
        help="Journal/calibration JSONL file. May be passed more than once.",
    )
    parser.add_argument("--pool", type=Path, default=None, help="Question-pool JSONL.")
    parser.add_argument("--out-core", type=Path, default=DEFAULT_CORE_OUT)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--core-id", default="core_v2")
    parser.add_argument("--target-size", type=int, default=40)
    parser.add_argument("--min-attempts", type=int, default=2)
    parser.add_argument("--p-min", type=float, default=0.2)
    parser.add_argument("--p-max", type=float, default=0.8)
    parser.add_argument("--max-per-suite", type=int, default=0)
    parser.add_argument(
        "--include-partition",
        action="append",
        default=["core"],
        help="Question-results partition to include; defaults to core.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    journals = args.journal or [DEFAULT_JOURNAL]
    rows = iter_jsonl(journals)
    stats = collect_item_stats(rows, include_partitions=set(args.include_partition))
    selected = select_core_items(
        stats,
        target_size=max(0, args.target_size),
        min_attempts=max(1, args.min_attempts),
        p_min=args.p_min,
        p_max=args.p_max,
        max_per_suite=max(0, args.max_per_suite),
    )
    pool_path = args.pool or _default_pool_path()
    pool_lookup = load_pool_lookup(pool_path) if pool_path else {}
    unresolved = [item for item in selected if (item.suite, item.qid) not in pool_lookup]
    report = build_report(
        rows=rows,
        selected=selected,
        stats=stats,
        unresolved=unresolved,
        args=args,
    )
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.out_core:
        if not pool_lookup:
            raise SystemExit("question pool unavailable; pass --pool or omit --out-core")
        write_core_jsonl(
            path=args.out_core,
            core_id=args.core_id,
            selected=selected,
            pool_lookup=pool_lookup,
            report=report,
        )
    print(
        f"selected={report['selected_count']} eligible={report['eligible_items']} "
        f"observed={report['observed_items']} unresolved={report['unresolved_selected_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
