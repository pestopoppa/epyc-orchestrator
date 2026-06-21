#!/usr/bin/env python3
"""Build a prompt-free pairwise A9 reward-oracle contract.

The absolute prompt/action verifier family has reached a stop condition. This
tool creates the next offline contract without training a model or changing any
runtime gate: within each source task, pair oracle-positive rows against
oracle-negative rows and emit preference examples. That changes the learning
target from absolute binary success to within-task action preference while
keeping prompt/answer/expected text out of committed artifacts.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from math import log1p
from pathlib import Path
import sys
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.graph_router.action_space import normalize_action
from scripts.graph_router.build_offline_reward_feature_manifest import (  # noqa: E402
    FEATURE_ROW_SCHEMA_VERSION,
    PRIVATE_FIELDS,
)
from scripts.graph_router.build_offline_reward_verifier_npz import (  # noqa: E402
    ROLE_ALIASES,
    ROLE_SUFFIXES,
)

PAIRWISE_ROW_SCHEMA_VERSION = "offline_reward_pairwise_preference.v1"
SUMMARY_SCHEMA_VERSION = "offline_reward_pairwise_contract_summary.v1"
CONTRACT_NAME = "within_task_pairwise_preference_v1"


class PairwiseContractError(ValueError):
    """Raised when manifest rows cannot produce a pairwise contract."""


@dataclass(frozen=True)
class PairwiseRow:
    schema_version: str
    contract_name: str
    pair_id: str
    group_key: str
    question_id: str
    suite: str
    source_path: str
    source_record_offset: int
    source_family: str
    prompt_sha256: str
    expected_sha256: str
    preferred_item_id: str
    rejected_item_id: str
    preferred_role_key: str
    rejected_role_key: str
    preferred_canonical_action: str
    rejected_canonical_action: str
    preferred_oracle_score: float
    rejected_oracle_score: float
    oracle_score_delta: float
    label_source: str
    target_source: str
    preferred_answer_chars: int
    rejected_answer_chars: int
    answer_chars_log_delta: float
    elapsed_log_delta: float
    preferred_error_present: bool
    rejected_error_present: bool


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise PairwiseContractError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def _assert_prompt_free(row: dict[str, Any], *, row_number: int) -> None:
    present = sorted(PRIVATE_FIELDS & set(row))
    if present:
        raise PairwiseContractError(
            f"manifest row {row_number}: private fields present: {', '.join(present)}"
        )


def _int_value(value: Any, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_value(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _canonical_action(role_key: str) -> str | None:
    base = role_key
    for suffix in ROLE_SUFFIXES:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    base = ROLE_ALIASES.get(base, base)
    if base == "architect_coding":  # stack-change-guard: allow retired-role remap fixture
        base = "architect_general"
    return normalize_action(base, include_seeded_frontdoor=True)


def _source_family(row: dict[str, Any]) -> str:
    context = row.get("feature_context")
    if isinstance(context, dict) and isinstance(context.get("source_family"), str):
        return str(context["source_family"])
    return "unknown"


def _group_key(row: dict[str, Any]) -> str:
    offset = _int_value(row.get("source_record_offset"), default=-1)
    return "|".join(
        [
            str(row.get("source_path") or ""),
            str(offset),
            str(row.get("question_id") or ""),
            str(row.get("prompt_sha256") or ""),
            str(row.get("expected_sha256") or ""),
        ]
    )


def _eligible_row(row: dict[str, Any], *, row_number: int) -> dict[str, Any] | None:
    _assert_prompt_free(row, row_number=row_number)
    if row.get("schema_version") != FEATURE_ROW_SCHEMA_VERSION:
        raise PairwiseContractError(
            f"manifest row {row_number}: expected schema_version={FEATURE_ROW_SCHEMA_VERSION!r}"
        )
    label = _int_value(row.get("oracle_binary_label"), default=-1)
    if label not in {0, 1}:
        return None
    role_key = str(row.get("role_key") or "")
    canonical = _canonical_action(role_key)
    if canonical is None:
        return None
    return {
        **row,
        "_oracle_binary_label": label,
        "_canonical_action": canonical,
        "_group_key": _group_key(row),
        "_source_family": _source_family(row),
    }


def _pair_id(group_key: str, preferred_item_id: str, rejected_item_id: str) -> str:
    return f"{group_key}|prefer={preferred_item_id}|reject={rejected_item_id}"


def _elapsed_log(row: dict[str, Any]) -> float:
    return log1p(max(_float_value(row.get("source_elapsed_seconds")), 0.0))


def _build_pair(group_key: str, positive: dict[str, Any], negative: dict[str, Any]) -> PairwiseRow:
    preferred_score = _float_value(positive.get("oracle_score"))
    rejected_score = _float_value(negative.get("oracle_score"))
    preferred_item_id = str(positive.get("item_id") or "")
    rejected_item_id = str(negative.get("item_id") or "")
    preferred_answer_chars = _int_value(positive.get("answer_chars"))
    rejected_answer_chars = _int_value(negative.get("answer_chars"))
    preferred_role_key = str(positive.get("role_key") or "")
    rejected_role_key = str(negative.get("role_key") or "")
    return PairwiseRow(
        schema_version=PAIRWISE_ROW_SCHEMA_VERSION,
        contract_name=CONTRACT_NAME,
        pair_id=_pair_id(group_key, preferred_item_id, rejected_item_id),
        group_key=group_key,
        question_id=str(positive.get("question_id") or negative.get("question_id") or ""),
        suite=str(positive.get("suite") or negative.get("suite") or ""),
        source_path=str(positive.get("source_path") or ""),
        source_record_offset=_int_value(positive.get("source_record_offset"), default=-1),
        source_family=str(positive.get("_source_family") or "unknown"),
        prompt_sha256=str(positive.get("prompt_sha256") or ""),
        expected_sha256=str(positive.get("expected_sha256") or ""),
        preferred_item_id=preferred_item_id,
        rejected_item_id=rejected_item_id,
        preferred_role_key=preferred_role_key,
        rejected_role_key=rejected_role_key,
        preferred_canonical_action=str(positive["_canonical_action"]),
        rejected_canonical_action=str(negative["_canonical_action"]),
        preferred_oracle_score=preferred_score,
        rejected_oracle_score=rejected_score,
        oracle_score_delta=preferred_score - rejected_score,
        label_source=str(positive.get("label_source") or negative.get("label_source") or ""),
        target_source=str(positive.get("target_source") or negative.get("target_source") or ""),
        preferred_answer_chars=preferred_answer_chars,
        rejected_answer_chars=rejected_answer_chars,
        answer_chars_log_delta=log1p(max(preferred_answer_chars, 0)) - log1p(max(rejected_answer_chars, 0)),
        elapsed_log_delta=_elapsed_log(positive) - _elapsed_log(negative),
        preferred_error_present=bool(positive.get("source_error_present")),
        rejected_error_present=bool(negative.get("source_error_present")),
    )


def build_pairwise_contract(
    manifest_rows: list[dict[str, Any]],
    *,
    max_pairs_per_group: int | None = None,
    min_pairs: int = 100,
    min_cross_action_pairs: int = 50,
    generated_at: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    generated = generated_at or datetime.now(UTC).isoformat()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped_unmapped = 0
    for row_number, row in enumerate(manifest_rows, start=1):
        eligible = _eligible_row(row, row_number=row_number)
        if eligible is None:
            skipped_unmapped += 1
            continue
        grouped[str(eligible["_group_key"])].append(eligible)

    pair_rows: list[dict[str, Any]] = []
    group_diagnostics: list[dict[str, Any]] = []
    action_pairs: Counter[str] = Counter()
    source_families: Counter[str] = Counter()
    suites: Counter[str] = Counter()
    skipped_no_contrast = 0
    for group_key, rows in sorted(grouped.items()):
        positives = [row for row in rows if row["_oracle_binary_label"] == 1]
        negatives = [row for row in rows if row["_oracle_binary_label"] == 0]
        if not positives or not negatives:
            skipped_no_contrast += 1
            continue
        emitted = 0
        for positive in sorted(positives, key=lambda row: str(row.get("item_id") or "")):
            for negative in sorted(negatives, key=lambda row: str(row.get("item_id") or "")):
                pair = _build_pair(group_key, positive, negative)
                payload = asdict(pair)
                pair_rows.append(payload)
                action_pairs[f"{pair.preferred_canonical_action}>{pair.rejected_canonical_action}"] += 1
                source_families[pair.source_family] += 1
                suites[pair.suite] += 1
                emitted += 1
                if max_pairs_per_group is not None and emitted >= max_pairs_per_group:
                    break
            if max_pairs_per_group is not None and emitted >= max_pairs_per_group:
                break
        group_diagnostics.append(
            {
                "group_key": group_key,
                "rows": len(rows),
                "positive_rows": len(positives),
                "negative_rows": len(negatives),
                "pairs": emitted,
            }
        )

    _assert_output_prompt_free(pair_rows)
    unique_action_pairs = len(action_pairs)
    cross_action_pair_rows = sum(
        count
        for action_pair, count in action_pairs.items()
        if action_pair.split(">", 1)[0] != action_pair.split(">", 1)[1]
    )
    same_action_pair_rows = len(pair_rows) - cross_action_pair_rows
    status = (
        "contract_ready"
        if (
            len(pair_rows) >= min_pairs
            and cross_action_pair_rows >= min_cross_action_pairs
            and unique_action_pairs > 0
        )
        else "insufficient_contrast"
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated,
        "contract": {
            "name": CONTRACT_NAME,
            "row_schema_version": PAIRWISE_ROW_SCHEMA_VERSION,
            "learning_target": "within_source_record_positive_over_negative_preference",
            "material_difference_from_stopped_family": [
                "uses pairwise preference labels instead of absolute binary labels",
                "controls prompt and expected answer by pairing only within the same source task",
                "keeps conflicting absolute model-input groups as contrastive evidence instead of dropping all conflicting rows",
                "does not train a classifier or authorize a runtime gate",
            ],
        },
        "inputs": {
            "manifest_rows": len(manifest_rows),
            "eligible_rows": sum(len(rows) for rows in grouped.values()),
            "skipped_unmapped_or_unlabeled_rows": skipped_unmapped,
        },
        "coverage": {
            "source_record_groups": len(grouped),
            "contrastive_groups": len(group_diagnostics),
            "skipped_no_contrast_groups": skipped_no_contrast,
            "pair_rows": len(pair_rows),
            "cross_action_pair_rows": cross_action_pair_rows,
            "same_action_pair_rows": same_action_pair_rows,
            "unique_action_pairs": unique_action_pairs,
            "action_pair_counts": dict(sorted(action_pairs.items())),
            "source_family_pair_counts": dict(sorted(source_families.items())),
            "suite_pair_counts": dict(sorted(suites.items())),
        },
        "decision": {
            "status": status,
            "min_pairs": min_pairs,
            "min_cross_action_pairs": min_cross_action_pairs,
            "runtime_gate_change_allowed": False,
            "recommended_next": (
                "train_pairwise_reward_ranker_offline"
                if status == "contract_ready"
                else "collect_more_within_task_positive_negative_contrasts"
            ),
        },
        "privacy": {
            "private_fields_excluded": sorted(PRIVATE_FIELDS),
            "text_represented_by_sha256_lengths_and_deltas": True,
            "commits_prompt_answer_expected_text": False,
        },
        "group_diagnostics": group_diagnostics[:50],
    }
    return pair_rows, summary


def _assert_output_prompt_free(rows: Iterable[dict[str, Any]]) -> None:
    for row_number, row in enumerate(rows, start=1):
        present = sorted(PRIVATE_FIELDS & set(row))
        if present:
            raise PairwiseContractError(
                f"pair row {row_number}: private fields present: {', '.join(present)}"
            )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_markdown(summary: dict[str, Any]) -> str:
    coverage = summary["coverage"]
    decision = summary["decision"]
    lines = [
        "# Offline Reward Pairwise Contract",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Contract: `{summary['contract']['name']}`",
        f"- Decision: `{decision['status']}`",
        f"- Runtime gate change allowed: `{decision['runtime_gate_change_allowed']}`",
        f"- Pair rows: `{coverage['pair_rows']}`",
        f"- Cross-action pair rows: `{coverage['cross_action_pair_rows']}`",
        f"- Same-action pair rows: `{coverage['same_action_pair_rows']}`",
        f"- Contrastive source-record groups: `{coverage['contrastive_groups']}`",
        f"- Unique action pairs: `{coverage['unique_action_pairs']}`",
        f"- Recommended next: `{decision['recommended_next']}`",
        "",
        "## Material Difference",
        "",
    ]
    lines.extend(f"- {item}" for item in summary["contract"]["material_difference_from_stopped_family"])
    lines.extend(["", "## Top Action Pairs", ""])
    for key, count in sorted(
        coverage["action_pair_counts"].items(),
        key=lambda item: (-int(item[1]), str(item[0])),
    )[:12]:
        lines.append(f"- `{key}`: `{count}`")
    if not coverage["action_pair_counts"]:
        lines.append("- none")
    lines.extend(["", "## Privacy", ""])
    for key, value in summary["privacy"].items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(summary), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build prompt-free within-task pairwise preference rows for A9."
    )
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path)
    parser.add_argument("--max-pairs-per-group", type=int)
    parser.add_argument("--min-pairs", type=int, default=100)
    parser.add_argument("--min-cross-action-pairs", type=int, default=50)
    parser.add_argument("--generated-at")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_jsonl(args.manifest_jsonl)
    pair_rows, summary = build_pairwise_contract(
        rows,
        max_pairs_per_group=args.max_pairs_per_group,
        min_pairs=max(0, int(args.min_pairs)),
        min_cross_action_pairs=max(0, int(args.min_cross_action_pairs)),
        generated_at=args.generated_at,
    )
    summary["inputs"]["manifest_jsonl"] = str(args.manifest_jsonl)
    summary["outputs"] = {
        "pairwise_jsonl": str(args.output_jsonl),
        "summary_json": str(args.summary_json),
        "summary_md": str(args.summary_md) if args.summary_md else None,
    }
    write_jsonl(args.output_jsonl, pair_rows)
    write_json(args.summary_json, summary)
    if args.summary_md:
        write_markdown(args.summary_md, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except PairwiseContractError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
