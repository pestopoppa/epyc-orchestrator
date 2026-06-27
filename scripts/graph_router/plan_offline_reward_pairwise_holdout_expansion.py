#!/usr/bin/env python3
"""Plan A9 pairwise expansion for failed independent holdout strata.

The pairwise ranker has signal on random group-disjoint splits but mixed
independent holdout evidence. This planner selects prompt-free source rows that
can add non-overlapping cross-action pairwise evidence for the weak strata
without training a model or changing runtime routing.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.graph_router.action_space import load_live_canonical_actions  # noqa: E402
from scripts.graph_router.build_offline_reward_feature_manifest import (  # noqa: E402
    PRIVATE_FIELDS,
    _source_family_onehot,
)
from scripts.graph_router.build_offline_reward_oracle_rows import (  # noqa: E402
    _binary_reward,
    _load_records,
    _role_results,
)
from scripts.graph_router.build_offline_reward_pairwise_contract import (  # noqa: E402
    _group_key,
)
from scripts.graph_router.plan_offline_reward_verifier_expansion import (  # noqa: E402
    CANDIDATE_SCHEMA_VERSION,
    _canonical_action,
    _candidate_id,
    _existing_keys,
    _iter_input_files,
    _parse_csv,
)


SUMMARY_SCHEMA_VERSION = "offline_reward_pairwise_holdout_expansion_plan.v1"
DEFAULT_RESULTS_ROOT = Path("/mnt/raid0/llm/epyc-inference-research/benchmarks/results")
DEFAULT_REPORT_DIR = (
    PROJECT_ROOT
    / "orchestration/reports/offline_reward_oracle_token_coverage_final_labels_20260621"
)
DEFAULT_EXISTING_MANIFEST = (
    DEFAULT_REPORT_DIR / "offline_reward_feature_manifest_with_seeding_eval_expansion.jsonl"
)
DEFAULT_EXISTING_PAIRWISE = (
    DEFAULT_REPORT_DIR / "offline_reward_pairwise_preference_contract_score_ordered.jsonl"
)
DEFAULT_TARGET_SOURCE_FAMILIES = "seeding_eval"
DEFAULT_TARGET_SUITES = "thinking"


class PairwiseHoldoutExpansionError(ValueError):
    """Raised when a pairwise holdout expansion plan cannot be built."""


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise PairwiseHoldoutExpansionError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _record_group_key(
    *,
    source_path: Path,
    offset: int,
    question_id: str,
    prompt_sha256: str,
    expected_sha256: str,
) -> str:
    return "|".join(
        [
            str(source_path),
            str(offset),
            question_id,
            prompt_sha256,
            expected_sha256,
        ]
    )


def _existing_manifest_groups(
    path: Path | None,
    *,
    canonical_actions: set[str],
) -> tuple[dict[str, set[str]], Counter, Counter]:
    if path is None or not path.exists():
        return {}, Counter(), Counter()
    groups: dict[str, set[str]] = defaultdict(set)
    source_family_counts: Counter = Counter()
    suite_counts: Counter = Counter()
    for row in _load_jsonl(path):
        group_key = _group_key(row)
        role_key = str(row.get("role_key") or "")
        canonical = _canonical_action(role_key, canonical_actions) if role_key else None
        if canonical:
            groups[group_key].add(canonical)
        context = row.get("feature_context")
        source_family = None
        if isinstance(context, dict):
            source_family = context.get("source_family")
        if isinstance(source_family, str) and source_family:
            source_family_counts[source_family] += 1
        suite = str(row.get("suite") or "")
        if suite:
            suite_counts[suite] += 1
    return groups, source_family_counts, suite_counts


def _existing_pairwise_groups(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    groups: set[str] = set()
    for row in _load_jsonl(path):
        group_key = str(row.get("group_key") or "")
        if group_key:
            groups.add(group_key)
    return groups


def _canonical_action_pair(actions: Iterable[str]) -> str:
    values = sorted(str(action) for action in actions)
    if len(values) != 2:
        raise PairwiseHoldoutExpansionError("action pair must contain exactly two actions")
    return f"{values[0]}>{values[1]}"


def _load_collection_targets(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    targets = payload.get("collection_targets")
    if not isinstance(targets, list):
        raise PairwiseHoldoutExpansionError(f"{path}: missing collection_targets list")
    out: list[dict[str, Any]] = []
    for index, target in enumerate(targets):
        if not isinstance(target, dict):
            raise PairwiseHoldoutExpansionError(f"{path}: collection_targets[{index}] must be object")
        field = str(target.get("stratum_field") or "")
        value = str(target.get("stratum_value") or "")
        action_pair = str(target.get("action_pair") or "")
        if field not in {"source_family", "suite"}:
            raise PairwiseHoldoutExpansionError(
                f"{path}: collection_targets[{index}] has unsupported stratum_field {field!r}"
            )
        if not value:
            raise PairwiseHoldoutExpansionError(
                f"{path}: collection_targets[{index}] missing stratum_value"
            )
        parts = [part for part in action_pair.split(">") if part]
        if len(parts) != 2:
            raise PairwiseHoldoutExpansionError(
                f"{path}: collection_targets[{index}] invalid action_pair {action_pair!r}"
            )
        out.append(
            {
                "stratum_field": field,
                "stratum_value": value,
                "action_pair": _canonical_action_pair(parts),
                "needs_direction": target.get("needs_direction") or [],
                "suggested_min_rows": target.get("suggested_min_rows"),
            }
        )
    return out


def _pair_keys(actions: set[str]) -> set[str]:
    return {_canonical_action_pair(pair) for pair in combinations(sorted(actions), 2)}


def _candidate_from_role(
    *,
    path: Path,
    offset: int,
    record: dict[str, Any],
    role_key: str,
    role_result: dict[str, Any],
    canonical_action: str,
    source_family: str,
    prompt: str,
    reference: str,
) -> dict[str, Any]:
    rewards = record.get("rewards") if isinstance(record.get("rewards"), dict) else {}
    reward_value = rewards.get(role_key)
    record_index = offset + 1
    response = str(role_result.get("answer") or role_result.get("response") or "")
    candidate = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "candidate_id": _candidate_id(path, record_index, role_key),
        "source_path": str(path),
        "source_record_index": record_index,
        "source_record_offset": offset,
        "source_record_index_base": "one_based",
        "question_id": str(record.get("question_id") or record.get("qid") or ""),
        "suite": str(record.get("suite") or "unknown"),
        "role_key": role_key,
        "canonical_action": canonical_action,
        "source_family": source_family,
        "source_passed": role_result.get("passed"),
        "source_error_present": bool(role_result.get("error")),
        "source_elapsed_seconds": role_result.get("elapsed_seconds"),
        "binary_reward": _binary_reward(role_result, reward_value),
        "prompt_sha256": _hash_text(prompt),
        "reference_sha256": _hash_text(reference),
        "expected_sha256": _hash_text(reference),
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
            pass
    return candidate


def _scan_candidate_groups(
    paths: Iterable[Path],
    *,
    canonical_actions: set[str],
    target_actions: set[str],
    target_source_families: set[str],
    target_suites: set[str],
    target_match_mode: str,
    existing_keys: set[tuple[str, int, str]],
) -> tuple[dict[str, list[dict[str, Any]]], Counter]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stats: Counter = Counter()
    for path in _iter_input_files(paths):
        stats["files_scanned"] += 1
        source_family, _ = _source_family_onehot(path)
        if (
            target_match_mode == "all"
            and target_source_families
            and source_family not in target_source_families
        ):
            stats["skipped_non_target_source_family_file"] += 1
            continue
        try:
            records = list(_load_records(path))
        except Exception:
            stats["skipped_unreadable_file"] += 1
            continue
        for offset, record in enumerate(records):
            if not isinstance(record, dict):
                stats["skipped_non_object_record"] += 1
                continue
            suite = str(record.get("suite") or "unknown")
            if target_match_mode == "all":
                source_family_match = (
                    not target_source_families or source_family in target_source_families
                )
                suite_match = not target_suites or suite in target_suites
                target_match = source_family_match and suite_match
            else:
                source_family_match = (
                    bool(target_source_families) and source_family in target_source_families
                )
                suite_match = bool(target_suites) and suite in target_suites
                target_match = source_family_match or suite_match
            if not target_match:
                stats["skipped_non_target_suite_record"] += 1
                continue
            prompt = str(record.get("prompt") or "")
            reference = str(record.get("expected") or record.get("reference") or "")
            if not prompt:
                stats["skipped_missing_prompt"] += 1
                continue
            if not reference:
                stats["skipped_missing_reference"] += 1
                continue
            role_results = _role_results(record)
            if not role_results:
                stats["skipped_missing_role_results"] += 1
                continue
            group_key = _record_group_key(
                source_path=path,
                offset=offset,
                question_id=str(record.get("question_id") or record.get("qid") or ""),
                prompt_sha256=_hash_text(prompt),
                expected_sha256=_hash_text(reference),
            )
            for role_key, role_result in role_results.items():
                raw_role = str(role_key)
                if (str(path), offset, raw_role) in existing_keys:
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
                groups[group_key].append(
                    _candidate_from_role(
                        path=path,
                        offset=offset,
                        record=record,
                        role_key=raw_role,
                        role_result=role_result,
                        canonical_action=canonical,
                        source_family=source_family,
                        prompt=prompt,
                        reference=reference,
                    )
                )
                stats["candidate_rows"] += 1
    return groups, stats


def _assert_prompt_free(rows: Iterable[dict[str, Any]]) -> None:
    for row_number, row in enumerate(rows, start=1):
        present = sorted(PRIVATE_FIELDS & set(row))
        if present:
            raise PairwiseHoldoutExpansionError(
                f"candidate row {row_number}: private fields present: {', '.join(present)}"
            )


def build_plan(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    canonical_actions = set(load_live_canonical_actions())
    target_actions = set(_parse_csv(args.target_actions))
    unknown_actions = sorted(target_actions - canonical_actions)
    if unknown_actions:
        raise PairwiseHoldoutExpansionError(
            f"unknown target action(s): {', '.join(unknown_actions)}"
        )
    target_source_families = set(_parse_csv(args.target_source_families))
    target_suites = set(_parse_csv(args.target_suites))
    collection_targets = _load_collection_targets(args.collection_targets_json)
    collection_target_pairs: dict[tuple[str, str], set[str]] = defaultdict(set)
    for target in collection_targets:
        key = (str(target["stratum_field"]), str(target["stratum_value"]))
        collection_target_pairs[key].add(str(target["action_pair"]))
        if target["stratum_field"] == "source_family":
            target_source_families.add(str(target["stratum_value"]))
        elif target["stratum_field"] == "suite":
            target_suites.add(str(target["stratum_value"]))
    if args.target_match_mode not in {"any", "all"}:
        raise PairwiseHoldoutExpansionError("target_match_mode must be 'any' or 'all'")
    input_paths = args.input or [DEFAULT_RESULTS_ROOT]
    existing_keys = _existing_keys(args.existing_manifest)
    manifest_groups, manifest_source_family_counts, manifest_suite_counts = (
        _existing_manifest_groups(args.existing_manifest, canonical_actions=canonical_actions)
    )
    pairwise_groups = _existing_pairwise_groups(args.existing_pairwise_jsonl)
    candidate_groups, stats = _scan_candidate_groups(
        input_paths,
        canonical_actions=canonical_actions,
        target_actions=target_actions,
        target_source_families=target_source_families,
        target_suites=target_suites,
        target_match_mode=args.target_match_mode,
        existing_keys=existing_keys,
    )

    selected_groups: list[dict[str, Any]] = []
    selected_candidates: list[dict[str, Any]] = []
    skipped_pairwise_overlap = 0
    skipped_no_cross_action = 0
    skipped_no_collection_target_pair = 0
    for group_key in sorted(candidate_groups):
        candidates = sorted(
            candidate_groups[group_key],
            key=lambda row: (str(row["canonical_action"]), str(row["role_key"])),
        )
        if args.require_non_overlapping_pairwise_groups and group_key in pairwise_groups:
            skipped_pairwise_overlap += 1
            continue
        candidate_actions = {str(row["canonical_action"]) for row in candidates}
        existing_actions = manifest_groups.get(group_key, set())
        potential_actions = set(existing_actions) | candidate_actions
        if len(potential_actions) < 2:
            skipped_no_cross_action += 1
            continue
        group_pair_keys = _pair_keys(potential_actions)
        source_family = str(candidates[0]["source_family"])
        suite = str(candidates[0]["suite"])
        matched_collection_targets: list[dict[str, Any]] = []
        if collection_targets:
            for field, value in (("source_family", source_family), ("suite", suite)):
                matched_pairs = sorted(group_pair_keys & collection_target_pairs.get((field, value), set()))
                for pair in matched_pairs:
                    matched_collection_targets.append(
                        {
                            "stratum_field": field,
                            "stratum_value": value,
                            "action_pair": pair,
                        }
                    )
            if not matched_collection_targets:
                skipped_no_collection_target_pair += 1
                continue
        selected_groups.append(
            {
                "group_key": group_key,
                "candidate_rows": len(candidates),
                "candidate_actions": sorted(candidate_actions),
                "existing_manifest_actions": sorted(existing_actions),
                "potential_actions": sorted(potential_actions),
                "potential_action_pairs": sorted(group_pair_keys),
                "matched_collection_targets": matched_collection_targets,
                "source_family": source_family,
                "suite": suite,
                "source_path": str(candidates[0]["source_path"]),
                "source_record_offset": int(candidates[0]["source_record_offset"]),
            }
        )
        selected_candidates.extend(candidates)
        if args.max_groups is not None and len(selected_groups) >= args.max_groups:
            break
        if args.max_candidate_rows is not None and len(selected_candidates) >= args.max_candidate_rows:
            selected_candidates = selected_candidates[: args.max_candidate_rows]
            break

    _assert_prompt_free(selected_candidates)
    source_family_counts = Counter(str(row["source_family"]) for row in selected_candidates)
    suite_counts = Counter(str(row["suite"]) for row in selected_candidates)
    action_counts = Counter(str(row["canonical_action"]) for row in selected_candidates)
    group_source_family_counts = Counter(str(row["source_family"]) for row in selected_groups)
    group_suite_counts = Counter(str(row["suite"]) for row in selected_groups)
    matched_collection_target_counts = Counter(
        f"{target['stratum_field']}:{target['stratum_value']}:{target['action_pair']}"
        for group in selected_groups
        for target in group["matched_collection_targets"]
    )
    expected_collection_target_keys = sorted(
        f"{target['stratum_field']}:{target['stratum_value']}:{target['action_pair']}"
        for target in collection_targets
    )
    unmatched_collection_targets = [
        key for key in expected_collection_target_keys if key not in matched_collection_target_counts
    ]

    status = (
        "expansion_plan_ready"
        if selected_candidates and len(selected_groups) >= args.min_cross_action_candidate_groups
        else "insufficient_non_overlapping_cross_action_candidates"
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "inputs": [str(path) for path in input_paths],
        "existing_manifest": str(args.existing_manifest) if args.existing_manifest else None,
        "existing_pairwise_jsonl": (
            str(args.existing_pairwise_jsonl) if args.existing_pairwise_jsonl else None
        ),
        "target_source_families": sorted(target_source_families),
        "target_suites": sorted(target_suites),
        "target_match_mode": args.target_match_mode,
        "target_actions": sorted(target_actions),
        "collection_targets_json": (
            str(args.collection_targets_json) if args.collection_targets_json else None
        ),
        "collection_target_count": len(collection_targets),
        "matched_collection_target_counts": dict(sorted(matched_collection_target_counts.items())),
        "unmatched_collection_targets": unmatched_collection_targets,
        "candidate_rows": len(selected_candidates),
        "candidate_groups": len(selected_groups),
        "candidate_action_counts": dict(sorted(action_counts.items())),
        "candidate_source_family_counts": dict(sorted(source_family_counts.items())),
        "candidate_suite_counts": dict(sorted(suite_counts.items())),
        "candidate_group_source_family_counts": dict(sorted(group_source_family_counts.items())),
        "candidate_group_suite_counts": dict(sorted(group_suite_counts.items())),
        "existing_manifest_source_family_counts": dict(sorted(manifest_source_family_counts.items())),
        "existing_manifest_suite_counts": dict(sorted(manifest_suite_counts.items())),
        "existing_pairwise_group_count": len(pairwise_groups),
        "skipped_pairwise_overlap_groups": skipped_pairwise_overlap,
        "skipped_no_cross_action_groups": skipped_no_cross_action,
        "skipped_no_collection_target_pair_groups": skipped_no_collection_target_pair,
        "min_cross_action_candidate_groups": args.min_cross_action_candidate_groups,
        "decision": {
            "status": status,
            "runtime_gate_change_allowed": False,
            "recommended_next": (
                "score_selected_candidates_and_rebuild_pairwise_contract"
                if status == "expansion_plan_ready"
                else "add_more_source_records_for_failed_pairwise_holdout_strata"
            ),
        },
        "privacy": {
            "private_fields_excluded": sorted(PRIVATE_FIELDS),
            "text_represented_by_sha256_and_lengths": True,
            "commits_prompt_reference_response_text": False,
        },
        "stats": {key: int(value) for key, value in sorted(stats.items())},
        "selected_groups": selected_groups[:100],
    }
    return selected_candidates, summary


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Offline Reward Pairwise Holdout Expansion Plan",
        "",
        f"- Decision: `{summary['decision']['status']}`",
        f"- Candidate rows: `{summary['candidate_rows']}`",
        f"- Candidate groups: `{summary['candidate_groups']}`",
        f"- Target source families: `{summary['target_source_families']}`",
        f"- Target suites: `{summary['target_suites']}`",
        f"- Target match mode: `{summary['target_match_mode']}`",
        f"- Target actions: `{summary['target_actions']}`",
        f"- Collection targets: `{summary['collection_target_count']}`",
        f"- Matched collection targets: `{summary['matched_collection_target_counts']}`",
        f"- Unmatched collection targets: `{summary['unmatched_collection_targets']}`",
        f"- Candidate action counts: `{summary['candidate_action_counts']}`",
        f"- Candidate source-family counts: `{summary['candidate_source_family_counts']}`",
        f"- Candidate suite counts: `{summary['candidate_suite_counts']}`",
        f"- Existing pairwise groups: `{summary['existing_pairwise_group_count']}`",
        f"- Skipped pairwise-overlap groups: `{summary['skipped_pairwise_overlap_groups']}`",
        f"- Skipped no-cross-action groups: `{summary['skipped_no_cross_action_groups']}`",
        f"- Skipped no-collection-target-pair groups: `{summary['skipped_no_collection_target_pair_groups']}`",
        f"- Runtime gate change allowed: `{summary['decision']['runtime_gate_change_allowed']}`",
        f"- Recommended next: `{summary['decision']['recommended_next']}`",
        "",
        "## Selected Groups",
        "",
    ]
    for group in summary["selected_groups"][:20]:
        lines.append(
            f"- `{group['source_family']}/{group['suite']}` "
            f"`{group['source_path']}#{group['source_record_offset']}` "
            f"candidates `{group['candidate_actions']}` existing "
            f"`{group['existing_manifest_actions']}` targets "
            f"`{group['matched_collection_targets']}`"
        )
    if not summary["selected_groups"]:
        lines.append("- none")
    lines.extend(
        [
            "",
            "This artifact is prompt-free. It selects source/role keys for the",
            "existing offline scoring path and does not authorize a runtime gate.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan pairwise expansion candidates for failed A9 holdout strata.",
    )
    parser.add_argument("--input", action="append", type=Path, default=None)
    parser.add_argument("--existing-manifest", type=Path, default=DEFAULT_EXISTING_MANIFEST)
    parser.add_argument("--existing-pairwise-jsonl", type=Path, default=DEFAULT_EXISTING_PAIRWISE)
    parser.add_argument("--candidates-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path)
    parser.add_argument("--target-source-families", default=DEFAULT_TARGET_SOURCE_FAMILIES)
    parser.add_argument("--target-suites", default=DEFAULT_TARGET_SUITES)
    parser.add_argument(
        "--collection-targets-json",
        type=Path,
        help="Optional preference-direction audit JSON; restrict selected groups to its collection_targets.",
    )
    parser.add_argument(
        "--target-match-mode",
        choices=("any", "all"),
        default="any",
        help="Use 'any' for the union of weak strata, or 'all' for their intersection.",
    )
    parser.add_argument(
        "--target-actions",
        default="architect_general,coder_escalation,frontdoor",
    )
    parser.add_argument("--min-cross-action-candidate-groups", type=int, default=20)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--max-candidate-rows", type=int)
    parser.add_argument(
        "--allow-existing-pairwise-groups",
        action="store_false",
        dest="require_non_overlapping_pairwise_groups",
        help="Allow candidate groups that already appear in the existing pairwise contract.",
    )
    parser.set_defaults(require_non_overlapping_pairwise_groups=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        candidates, summary = build_plan(args)
    except (PairwiseHoldoutExpansionError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not candidates:
        print("error: no pairwise holdout expansion candidates found", file=sys.stderr)
        return 2
    write_jsonl(args.candidates_jsonl, candidates)
    write_json(args.summary_json, summary)
    if args.summary_md:
        write_markdown(args.summary_md, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
