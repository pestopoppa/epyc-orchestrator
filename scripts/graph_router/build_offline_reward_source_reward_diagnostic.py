#!/usr/bin/env python3
"""Build a prompt-free A9 pairwise diagnostic from source reward values.

This is deliberately not an independent reward oracle. It answers a narrower
question: do existing candidate rows contain enough within-task reward contrast
if we order candidates by the source benchmark `q_reward` already present in the
collection artifact?
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import json
from math import isfinite
from pathlib import Path
import sys
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.graph_router.build_offline_reward_feature_manifest import (  # noqa: E402
    FEATURE_ROW_SCHEMA_VERSION,
    PRIVATE_FIELDS,
)
from scripts.graph_router.build_offline_reward_pairwise_contract import (  # noqa: E402
    PairwiseContractError,
    build_pairwise_contract,
    write_json,
    write_jsonl,
)


SUMMARY_SCHEMA_VERSION = "offline_reward_source_reward_diagnostic_summary.v1"
ORACLE_SCORE_SOURCE = "source_q_reward_passthrough"
LABEL_STATUS = "diagnostic_source_reward_passthrough"
DEFAULT_BINARY_THRESHOLD = 0.5
DEFAULT_MIN_PAIRS = 100
DEFAULT_MIN_CROSS_ACTION_PAIRS = 50


class SourceRewardDiagnosticError(ValueError):
    """Raised when source-reward diagnostic input is malformed."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise SourceRewardDiagnosticError(
                    f"{path}:{line_number}: expected object"
                )
            rows.append(value)
    if not rows:
        raise SourceRewardDiagnosticError(f"{path}: no rows")
    return rows


def _assert_prompt_free_candidate(row: dict[str, Any], *, row_number: int) -> None:
    present = sorted(PRIVATE_FIELDS & set(row))
    if present:
        raise SourceRewardDiagnosticError(
            f"candidate row {row_number}: private fields present: {', '.join(present)}"
        )


def _required_str(
    row: dict[str, Any],
    key: str,
    *,
    row_number: int,
    fallback_key: str | None = None,
) -> str:
    value = row.get(key)
    if value in (None, "") and fallback_key is not None:
        value = row.get(fallback_key)
    if value in (None, ""):
        raise SourceRewardDiagnosticError(f"candidate row {row_number}: missing {key}")
    return str(value)


def _required_int(
    row: dict[str, Any],
    key: str,
    *,
    row_number: int,
    fallback_key: str | None = None,
) -> int:
    value = row.get(key)
    if value is None and fallback_key is not None:
        value = row.get(fallback_key)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SourceRewardDiagnosticError(
            f"candidate row {row_number}: {key} must be int"
        ) from exc


def _required_float(row: dict[str, Any], key: str, *, row_number: int) -> float:
    try:
        value = float(row.get(key))
    except (TypeError, ValueError) as exc:
        raise SourceRewardDiagnosticError(
            f"candidate row {row_number}: {key} must be numeric"
        ) from exc
    if not isfinite(value):
        raise SourceRewardDiagnosticError(
            f"candidate row {row_number}: {key} must be finite"
        )
    return value


def _source_binary_label(row: dict[str, Any], score: float, threshold: float) -> int:
    source_passed = row.get("source_passed")
    if isinstance(source_passed, bool):
        return int(source_passed)
    return int(score >= threshold)


def _candidate_to_manifest_row(
    row: dict[str, Any],
    *,
    row_number: int,
    binary_threshold: float,
) -> dict[str, Any]:
    _assert_prompt_free_candidate(row, row_number=row_number)
    score = _required_float(row, "q_reward", row_number=row_number)
    item_id = _required_str(row, "candidate_id", row_number=row_number)
    role_key = _required_str(row, "role_key", row_number=row_number)
    source_path = _required_str(row, "source_path", row_number=row_number)
    offset = _required_int(row, "source_record_offset", row_number=row_number)
    source_record_index = _required_int(
        row,
        "source_record_index",
        row_number=row_number,
        fallback_key="source_record_offset",
    )
    source_family = str(row.get("source_family") or "unknown")
    binary_label = int(score >= binary_threshold)
    target_label = _source_binary_label(row, score, binary_threshold)
    return {
        "schema_version": FEATURE_ROW_SCHEMA_VERSION,
        "item_id": item_id,
        "join_key": f"{source_path}:{offset}:{role_key}:{item_id}",
        "question_id": _required_str(row, "question_id", row_number=row_number),
        "suite": str(row.get("suite") or "unknown"),
        "role_key": role_key,
        "source_path": source_path,
        "source_record_index": source_record_index,
        "source_record_offset": offset,
        "source_record_index_base": str(
            row.get("source_record_index_base") or "unknown"
        ),
        "source_record_count": _required_int(
            row,
            "source_record_count",
            row_number=row_number,
        )
        if row.get("source_record_count") is not None
        else 0,
        "source_role": role_key.split(":", 1)[0],
        "source_passed": row.get("source_passed"),
        "source_elapsed_seconds": row.get("source_elapsed_seconds"),
        "source_error_present": bool(row.get("source_error_present")),
        "prompt_sha256": _required_str(row, "prompt_sha256", row_number=row_number),
        "expected_sha256": _required_str(
            row,
            "expected_sha256",
            fallback_key="reference_sha256",
            row_number=row_number,
        ),
        "answer_sha256": _required_str(
            row,
            "answer_sha256",
            fallback_key="response_sha256",
            row_number=row_number,
        ),
        "prompt_chars": _required_int(row, "prompt_chars", row_number=row_number),
        "expected_chars": _required_int(
            row,
            "expected_chars",
            fallback_key="reference_chars",
            row_number=row_number,
        ),
        "answer_chars": _required_int(
            row,
            "answer_chars",
            fallback_key="response_chars",
            row_number=row_number,
        ),
        "feature_context": {
            "source_family": source_family,
            "context_length_chars": row.get("prompt_chars"),
        },
        "oracle_binary_label": binary_label,
        "oracle_score": score,
        "oracle_threshold": binary_threshold,
        "oracle_score_source": ORACLE_SCORE_SOURCE,
        "target_binary_label": target_label,
        "target_source": "verifier_sparse_action_expansion",
        "label_source": f"{ORACLE_SCORE_SOURCE}@{binary_threshold:g}",
        "label_status": LABEL_STATUS,
    }


def build_source_reward_diagnostic(
    candidates: Iterable[dict[str, Any]],
    *,
    binary_threshold: float = DEFAULT_BINARY_THRESHOLD,
    min_score_delta: float = 0.0,
    min_pairs: int = DEFAULT_MIN_PAIRS,
    min_cross_action_pairs: int = DEFAULT_MIN_CROSS_ACTION_PAIRS,
    max_pairs_per_group: int | None = None,
    generated_at: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not isfinite(binary_threshold):
        raise SourceRewardDiagnosticError("binary_threshold must be finite")
    if min_score_delta < 0.0:
        raise SourceRewardDiagnosticError("min_score_delta must be non-negative")

    manifest_rows = [
        _candidate_to_manifest_row(
            row,
            row_number=row_number,
            binary_threshold=binary_threshold,
        )
        for row_number, row in enumerate(candidates, start=1)
    ]
    if not manifest_rows:
        raise SourceRewardDiagnosticError("no candidate rows")

    pair_rows, summary = build_pairwise_contract(
        manifest_rows,
        max_pairs_per_group=max_pairs_per_group,
        min_pairs=max(0, int(min_pairs)),
        min_cross_action_pairs=max(0, int(min_cross_action_pairs)),
        pairing_mode="score_ordered",
        min_score_delta=max(0.0, float(min_score_delta)),
        generated_at=generated_at or datetime.now(UTC).isoformat(),
    )
    score_counts = Counter(row["oracle_score"] for row in manifest_rows)
    summary["schema_version"] = SUMMARY_SCHEMA_VERSION
    summary["diagnostic"] = {
        "score_source": ORACLE_SCORE_SOURCE,
        "label_status": LABEL_STATUS,
        "binary_threshold": binary_threshold,
        "independent_oracle": False,
        "source_reward_passthrough": True,
        "diagnostic_only": True,
        "runtime_gate_change_allowed": False,
        "interpretation": (
            "Use this to test whether the candidate set contains enough "
            "within-task source-reward contrast. It is not an adopted "
            "independent reward oracle."
        ),
        "recommended_next": (
            "decide whether A9 should train on source-q-reward pairwise labels "
            "or build a new independent scorer/source contract before ranker use"
            if summary["decision"]["status"] == "contract_ready"
            else "collect or construct rows with more within-task source-reward contrast"
        ),
        "score_value_counts": {
            str(key): int(value) for key, value in sorted(score_counts.items())
        },
    }
    summary["decision"]["runtime_gate_change_allowed"] = False
    summary["decision"]["recommended_next"] = summary["diagnostic"][
        "recommended_next"
    ]
    summary["inputs"]["candidate_rows"] = len(manifest_rows)
    summary["inputs"]["source_score_field"] = "q_reward"
    return pair_rows, summary


def render_markdown(summary: dict[str, Any]) -> str:
    coverage = summary["coverage"]
    decision = summary["decision"]
    diagnostic = summary["diagnostic"]
    lines = [
        "# Offline Reward Source-Reward Pairwise Diagnostic",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Score source: `{diagnostic['score_source']}`",
        f"- Independent oracle: `{diagnostic['independent_oracle']}`",
        f"- Diagnostic only: `{diagnostic['diagnostic_only']}`",
        f"- Decision: `{decision['status']}`",
        f"- Runtime gate change allowed: `{decision['runtime_gate_change_allowed']}`",
        f"- Pair rows: `{coverage['pair_rows']}`",
        f"- Cross-action pair rows: `{coverage['cross_action_pair_rows']}`",
        f"- Contrastive source-record groups: `{coverage['contrastive_groups']}`",
        f"- Recommended next: `{decision['recommended_next']}`",
        "",
        diagnostic["interpretation"],
        "",
        "## Top Action Pairs",
        "",
    ]
    for key, count in sorted(
        coverage["action_pair_counts"].items(),
        key=lambda item: (-int(item[1]), str(item[0])),
    )[:12]:
        lines.append(f"- `{key}`: `{count}`")
    if not coverage["action_pair_counts"]:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(summary) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a prompt-free A9 pairwise diagnostic from source q_reward."
    )
    parser.add_argument("--candidates-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path)
    parser.add_argument("--binary-threshold", type=float, default=DEFAULT_BINARY_THRESHOLD)
    parser.add_argument("--min-score-delta", type=float, default=0.0)
    parser.add_argument("--min-pairs", type=int, default=DEFAULT_MIN_PAIRS)
    parser.add_argument(
        "--min-cross-action-pairs",
        type=int,
        default=DEFAULT_MIN_CROSS_ACTION_PAIRS,
    )
    parser.add_argument("--max-pairs-per-group", type=int)
    parser.add_argument("--generated-at")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    candidates = load_jsonl(args.candidates_jsonl)
    pair_rows, summary = build_source_reward_diagnostic(
        candidates,
        binary_threshold=float(args.binary_threshold),
        min_score_delta=float(args.min_score_delta),
        min_pairs=int(args.min_pairs),
        min_cross_action_pairs=int(args.min_cross_action_pairs),
        max_pairs_per_group=args.max_pairs_per_group,
        generated_at=args.generated_at,
    )
    summary["inputs"]["candidates_jsonl"] = str(args.candidates_jsonl)
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
    except (
        SourceRewardDiagnosticError,
        PairwiseContractError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
