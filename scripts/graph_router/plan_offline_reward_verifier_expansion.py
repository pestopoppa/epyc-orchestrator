#!/usr/bin/env python3
"""Plan prompt-free verifier-data expansion for sparse routing actions.

The A9 verifier data currently has sparse escalation-role coverage. This tool
scans existing benchmark result files for source records that can be converted
into offline reward-oracle rows, maps historical role labels to current
canonical actions, excludes rows already present in a feature manifest, and
emits a prompt-free candidate manifest plus a source-file recommendation list.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.graph_router.action_space import (
    load_live_canonical_actions,
    normalize_action,
)
from scripts.graph_router.build_offline_reward_oracle_rows import (
    _binary_reward,
    _load_records,
    _role_results,
)

SCHEMA_VERSION = "offline_reward_verifier_expansion_plan.v1"
CANDIDATE_SCHEMA_VERSION = "offline_reward_verifier_expansion_candidate.v1"
PRIVATE_FIELDS = {"answer", "expected", "prompt", "reference", "response"}
DEFAULT_RESULTS_ROOT = Path("/mnt/raid0/llm/epyc-inference-research/benchmarks/results")
DEFAULT_EXISTING_MANIFEST = (
    PROJECT_ROOT
    / "orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621"
    / "offline_reward_feature_manifest.jsonl"
)
RAW_ROLE_ALIASES = {
    "coder_primary": "coder_escalation",
}
ROLE_SUFFIXES = (":delegated", ":direct", ":repl", ":react")
ACTION_SPACE_LOGGER = logging.getLogger("scripts.graph_router.action_space")


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def _existing_keys(path: Path | None) -> set[tuple[str, int, str]]:
    if path is None or not path.exists():
        return set()
    keys: set[tuple[str, int, str]] = set()
    for row in _load_jsonl(path):
        source_path = str(row.get("source_path") or "")
        offset = row.get("source_record_offset")
        role_key = str(row.get("role_key") or "")
        if isinstance(offset, int) and source_path and role_key:
            keys.add((source_path, offset, role_key))
    return keys


def _iter_input_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(
                sorted(
                    child
                    for child in path.rglob("*")
                    if child.suffix in {".json", ".jsonl"} and child.is_file()
                )
            )
        elif path.suffix in {".json", ".jsonl"} and path.is_file():
            files.append(path)
    return files


def _canonical_action(raw_role: str, canonical_actions: set[str]) -> str | None:
    alias = RAW_ROLE_ALIASES.get(raw_role, raw_role)
    candidates: list[str] = []
    for suffix in ROLE_SUFFIXES:
        if alias.endswith(suffix):
            candidates.append(alias[: -len(suffix)])
            break
    candidates.append(alias)
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        previous_disabled = ACTION_SPACE_LOGGER.disabled
        ACTION_SPACE_LOGGER.disabled = True
        try:
            canonical = normalize_action(candidate, include_seeded_frontdoor=True)
        finally:
            ACTION_SPACE_LOGGER.disabled = previous_disabled
        if canonical in canonical_actions:
            return canonical
    return None


def _candidate_id(path: Path, record_index: int, role_key: str) -> str:
    safe_role = role_key.replace(":", "_").replace("/", "_")
    return f"{path.stem}:{record_index}:{safe_role}"


def _scan_file(
    path: Path,
    *,
    canonical_actions: set[str],
    target_actions: set[str],
    existing: set[tuple[str, int, str]],
) -> tuple[list[dict[str, Any]], Counter]:
    candidates: list[dict[str, Any]] = []
    stats = Counter()
    try:
        records = list(_load_records(path))
    except Exception:
        stats["skipped_unreadable_file"] += 1
        return candidates, stats

    for offset, record in enumerate(records):
        if not isinstance(record, dict):
            stats["skipped_non_object_record"] += 1
            continue
        reference = str(record.get("expected") or record.get("reference") or "")
        prompt = str(record.get("prompt") or "")
        if not reference:
            stats["skipped_missing_reference"] += 1
            continue
        if not prompt:
            stats["skipped_missing_prompt"] += 1
            continue
        role_results = _role_results(record)
        if not role_results:
            stats["skipped_missing_role_results"] += 1
            continue
        rewards = record.get("rewards") if isinstance(record.get("rewards"), dict) else {}
        record_index = offset + 1
        for role_key, role_result in role_results.items():
            raw_role = str(role_key)
            if (str(path), offset, raw_role) in existing:
                stats["skipped_existing_row"] += 1
                continue
            if not isinstance(role_result, dict):
                stats["skipped_bad_role_result"] += 1
                continue
            response = str(role_result.get("answer") or role_result.get("response") or "")
            if not response:
                stats["skipped_missing_response"] += 1
                continue
            canonical = _canonical_action(raw_role, canonical_actions)
            if canonical is None:
                stats["skipped_unmapped_role"] += 1
                continue
            if target_actions and canonical not in target_actions:
                stats["skipped_non_target_action"] += 1
                continue
            reward_value = rewards.get(raw_role)
            binary_reward = _binary_reward(role_result, reward_value)
            candidate = {
                "schema_version": CANDIDATE_SCHEMA_VERSION,
                "candidate_id": _candidate_id(path, record_index, raw_role),
                "source_path": str(path),
                "source_record_index": record_index,
                "source_record_offset": offset,
                "source_record_index_base": "one_based",
                "question_id": str(record.get("question_id") or record.get("qid") or ""),
                "suite": str(record.get("suite") or "unknown"),
                "role_key": raw_role,
                "canonical_action": canonical,
                "source_passed": role_result.get("passed"),
                "source_error_present": bool(role_result.get("error")),
                "source_elapsed_seconds": role_result.get("elapsed_seconds"),
                "binary_reward": binary_reward,
                "prompt_sha256": _hash_text(prompt),
                "reference_sha256": _hash_text(reference),
                "response_sha256": _hash_text(response),
                "prompt_chars": len(prompt),
                "reference_chars": len(reference),
                "response_chars": len(response),
                "scoring_next_step": "build_offline_reward_oracle_rows -> score_offline_reward_oracle_token_coverage",
            }
            if reward_value is not None:
                try:
                    candidate["q_reward"] = float(reward_value)
                except (TypeError, ValueError):
                    stats["skipped_non_numeric_q_reward"] += 1
            candidates.append(candidate)
            stats["candidate_rows"] += 1
    return candidates, stats


def _assert_prompt_free(rows: Iterable[dict[str, Any]]) -> None:
    for index, row in enumerate(rows, start=1):
        present = sorted(PRIVATE_FIELDS & set(row))
        if present:
            raise ValueError(f"candidate row {index}: private fields present: {present}")


def _cap_candidates(
    candidates: list[dict[str, Any]],
    *,
    max_per_action: int,
) -> list[dict[str, Any]]:
    counts = Counter()
    capped: list[dict[str, Any]] = []
    for row in sorted(
        candidates,
        key=lambda r: (
            str(r["canonical_action"]),
            str(r["source_path"]),
            int(r["source_record_offset"]),
            str(r["role_key"]),
        ),
    ):
        action = str(row["canonical_action"])
        if counts[action] >= max_per_action:
            continue
        capped.append(row)
        counts[action] += 1
    return capped


def _recommend_sources(
    candidates: list[dict[str, Any]],
    *,
    existing_action_counts: dict[str, int],
    target_actions: list[str],
    min_action_rows: int,
) -> list[dict[str, Any]]:
    by_source: dict[str, Counter] = defaultdict(Counter)
    for row in candidates:
        by_source[str(row["source_path"])][str(row["canonical_action"])] += 1

    deficits = {
        action: max(0, min_action_rows - int(existing_action_counts.get(action, 0)))
        for action in target_actions
    }
    selected: list[dict[str, Any]] = []
    remaining_sources = set(by_source)
    while remaining_sources and any(value > 0 for value in deficits.values()):
        best_source: str | None = None
        best_score = 0
        for source in sorted(remaining_sources):
            score = sum(
                min(by_source[source].get(action, 0), deficits[action])
                for action in target_actions
            )
            if score > best_score:
                best_score = score
                best_source = source
        if best_source is None or best_score <= 0:
            break
        counts = by_source[best_source]
        selected.append(
            {
                "source_path": best_source,
                "target_action_counts": {
                    action: int(counts.get(action, 0))
                    for action in target_actions
                    if counts.get(action, 0)
                },
                "deficit_reduction": int(best_score),
            }
        )
        for action in target_actions:
            deficits[action] = max(0, deficits[action] - int(counts.get(action, 0)))
        remaining_sources.remove(best_source)
    return selected


def build_plan(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    canonical_actions = set(load_live_canonical_actions())
    target_actions = [part.strip() for part in args.target_actions.split(",") if part.strip()]
    unknown_targets = sorted(set(target_actions) - canonical_actions)
    if unknown_targets:
        raise ValueError(f"unknown target action(s): {', '.join(unknown_targets)}")

    existing = _existing_keys(args.existing_manifest)
    files = _iter_input_files(args.input)
    all_candidates: list[dict[str, Any]] = []
    stats = Counter()
    for path in files:
        rows, file_stats = _scan_file(
            path,
            canonical_actions=canonical_actions,
            target_actions=set(target_actions),
            existing=existing,
        )
        all_candidates.extend(rows)
        stats.update(file_stats)
        stats["files_scanned"] += 1

    all_candidates = _cap_candidates(all_candidates, max_per_action=args.max_candidates_per_action)
    _assert_prompt_free(all_candidates)

    action_counts = Counter(str(row["canonical_action"]) for row in all_candidates)
    source_counts: dict[str, Counter] = defaultdict(Counter)
    for row in all_candidates:
        source_counts[str(row["source_path"])][str(row["canonical_action"])] += 1

    existing_action_counts = {}
    if args.existing_summary and args.existing_summary.exists():
        summary = json.loads(args.existing_summary.read_text(encoding="utf-8"))
        existing_action_counts = {
            str(k): int(v)
            for k, v in summary.get("canonical_action_counts", {}).items()
        }

    summary = {
        "schema_version": SCHEMA_VERSION,
        "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "inputs": [str(path) for path in args.input],
        "files_scanned": int(stats["files_scanned"]),
        "existing_manifest": str(args.existing_manifest) if args.existing_manifest else None,
        "existing_rows_excluded": int(stats["skipped_existing_row"]),
        "target_actions": target_actions,
        "min_action_rows": args.min_action_rows,
        "max_candidates_per_action": args.max_candidates_per_action,
        "candidate_rows": len(all_candidates),
        "candidate_action_counts": dict(sorted(action_counts.items())),
        "existing_action_counts": dict(sorted(existing_action_counts.items())),
        "recommended_sources": _recommend_sources(
            all_candidates,
            existing_action_counts=existing_action_counts,
            target_actions=target_actions,
            min_action_rows=args.min_action_rows,
        ),
        "stats": {key: int(value) for key, value in sorted(stats.items())},
        "privacy": {
            "private_fields_excluded": sorted(PRIVATE_FIELDS),
            "text_represented_by_sha256_and_lengths": True,
            "commits_prompt_text": False,
        },
        "next_step": (
            "Run build_offline_reward_oracle_rows on recommended source_path values, "
            "score with reference_token_coverage, export labels, rebuild feature manifest/NPZ, "
            "then rerun verifier robustness."
        ),
    }
    return all_candidates, summary


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Offline Verifier Expansion Plan",
        "",
        f"- Candidate rows: `{summary['candidate_rows']}`",
        f"- Target actions: `{summary['target_actions']}`",
        f"- Candidate action counts: `{summary['candidate_action_counts']}`",
        f"- Existing action counts: `{summary['existing_action_counts']}`",
        f"- Recommended source count: `{len(summary['recommended_sources'])}`",
        "",
        "## Recommended Sources",
        "",
    ]
    for source in summary["recommended_sources"]:
        lines.append(
            f"- `{source['source_path']}` -> `{source['target_action_counts']}`"
        )
    lines.extend(
        [
            "",
            "This artifact is prompt-free. It identifies candidate source rows for",
            "offline scoring and does not commit prompt/reference/response text.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan prompt-free verifier-data expansion for sparse actions",
    )
    parser.add_argument("--input", action="append", type=Path, default=[DEFAULT_RESULTS_ROOT])
    parser.add_argument("--existing-manifest", type=Path, default=DEFAULT_EXISTING_MANIFEST)
    parser.add_argument(
        "--existing-summary",
        type=Path,
        default=DEFAULT_EXISTING_MANIFEST.with_name("offline_reward_verifier_data_summary.json"),
    )
    parser.add_argument("--candidates-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path)
    parser.add_argument("--target-actions", default="architect_general,coder_escalation")
    parser.add_argument("--min-action-rows", type=int, default=30)
    parser.add_argument("--max-candidates-per-action", type=int, default=200)
    args = parser.parse_args(argv)

    try:
        candidates, summary = build_plan(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not candidates:
        print("error: no expansion candidates found", file=sys.stderr)
        return 2
    write_jsonl(args.candidates_jsonl, candidates)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.summary_md:
        write_markdown(args.summary_md, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
