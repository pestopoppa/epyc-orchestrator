#!/usr/bin/env python3
"""Read-only BSV-2 paired behavior-signature report.

This CLI does not run inference and does not gate AutoPilot acceptance by itself.
It converts already-journaled same-question eval vectors into baseline/candidate
behavior signatures, compares them with `diff_signatures`, and emits the exact
evidence a future accept gate needs: shared-qid coverage, scalar paired delta,
signature severity, and blockers.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ORCH_ROOT))

try:
    from scripts.autopilot.paired_stats import (  # type: ignore[import-not-found]
        DEFAULT_JOURNAL_DIR,
        McNemarResult,
        QuestionOutcome,
        group_rows_by_fingerprint,
        iter_journal_rows,
        majority_vector,
        mcnemar_from_vectors,
        trial_vectors,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from paired_stats import (  # type: ignore[no-redef]
        DEFAULT_JOURNAL_DIR,
        McNemarResult,
        QuestionOutcome,
        group_rows_by_fingerprint,
        iter_journal_rows,
        majority_vector,
        mcnemar_from_vectors,
        trial_vectors,
    )

from src.behavior_signature import compute_behavior_signature, diff_signatures

DEFAULT_MIN_SHARED_QIDS = 35


def _route_path_from_row(row: dict[str, Any]) -> list[str] | None:
    routing = row.get("routing_distribution") or (row.get("eval_details") or {}).get(
        "routing_distribution"
    )
    if not isinstance(routing, dict):
        return None
    items = sorted(routing.items())
    return [f"{role}:{_weight_bucket(weight)}" for role, weight in items] or None


def _weight_bucket(weight: Any) -> str:
    try:
        w = float(weight)
    except (TypeError, ValueError):
        return "q?"
    return "q1" if w < 0.25 else "q2" if w < 0.5 else "q3" if w < 0.75 else "q4"


def _avg_prompt_tokens_from_rows(rows: list[dict[str, Any]]) -> float | None:
    values: list[float] = []
    for row in rows:
        for key in ("avg_prompt_tokens", "instruction_tokens"):
            value = row.get(key)
            if value is None:
                value = (row.get("eval_details") or {}).get(key)
            try:
                values.append(float(value))
                break
            except (TypeError, ValueError):
                continue
    return statistics.median(values) if values else None


def _sentinel_outcomes(vector: dict[str, QuestionOutcome]) -> dict[str, str]:
    return {
        qid: "pass" if outcome.correct else "fail"
        for qid, outcome in sorted(vector.items())
    }


def _signature(
    *,
    archive_member_id: str,
    trial_id: int | None,
    vector: dict[str, QuestionOutcome],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    route_path = _route_path_from_row(rows[-1]) if rows else None
    sig = compute_behavior_signature(
        archive_member_id=archive_member_id,
        trial_id=trial_id,
        sentinel_outcomes=_sentinel_outcomes(vector),
        route_path=route_path,
        total_tokens=_avg_prompt_tokens_from_rows(rows),
        signature_confidence="partial",
    )
    return {
        "archive_member_id": sig.archive_member_id,
        "trial_id": sig.trial_id,
        "sentinel_outcomes": sig.sentinel_outcomes or {},
        "route_path_hash": sig.route_path_hash,
        "tool_sequence_hash": sig.tool_sequence_hash,
        "escalation_path_hash": sig.escalation_path_hash,
        "latency_bucket": sig.latency_bucket,
        "token_bucket": sig.token_bucket,
        "signature_hash": sig.signature_hash,
        "signature_confidence": sig.signature_confidence,
    }


def _decision(
    *,
    severity: str,
    stats: McNemarResult,
    min_shared_qids: int,
    max_accuracy_regression: float,
) -> tuple[str, list[str]]:
    blockers: list[str] = []
    if stats.shared_qids < min_shared_qids:
        blockers.append(f"shared_qids {stats.shared_qids} < {min_shared_qids}")
    if stats.delta_b_minus_a < -abs(max_accuracy_regression):
        blockers.append(
            "candidate accuracy regression "
            f"{stats.delta_b_minus_a:.6f} < -{abs(max_accuracy_regression):.6f}"
        )
    if severity == "blocking":
        blockers.append("behavior signature severity is blocking")
    return ("pass" if not blockers else "block", blockers)


def build_trial_pair_report(
    rows: list[dict[str, Any]],
    *,
    baseline_trial: int,
    candidate_trial: int,
    min_shared_qids: int = DEFAULT_MIN_SHARED_QIDS,
    max_accuracy_regression: float = 0.0,
) -> dict[str, Any]:
    vectors = trial_vectors(rows)
    by_trial = {int(row["trial_id"]): row for row in rows if row.get("trial_id") is not None}
    try:
        baseline_vector = vectors[baseline_trial]
        candidate_vector = vectors[candidate_trial]
        baseline_row = by_trial[baseline_trial]
        candidate_row = by_trial[candidate_trial]
    except KeyError as exc:
        raise ValueError(f"trial has no question_results vector: {exc}") from exc

    stats = mcnemar_from_vectors(
        baseline_vector,
        candidate_vector,
        str(baseline_trial),
        str(candidate_trial),
    )
    baseline_sig = _signature(
        archive_member_id=f"trial:{baseline_trial}",
        trial_id=baseline_trial,
        vector=baseline_vector,
        rows=[baseline_row],
    )
    candidate_sig = _signature(
        archive_member_id=f"trial:{candidate_trial}",
        trial_id=candidate_trial,
        vector=candidate_vector,
        rows=[candidate_row],
    )
    return _paired_report(
        comparison_type="trial_pair",
        baseline_label=str(baseline_trial),
        candidate_label=str(candidate_trial),
        stats=stats,
        baseline_signature=baseline_sig,
        candidate_signature=candidate_sig,
        min_shared_qids=min_shared_qids,
        max_accuracy_regression=max_accuracy_regression,
    )


def build_fingerprint_pair_report(
    rows: list[dict[str, Any]],
    *,
    baseline_fingerprint: str,
    candidate_fingerprint: str,
    min_shared_qids: int = DEFAULT_MIN_SHARED_QIDS,
    max_accuracy_regression: float = 0.0,
) -> dict[str, Any]:
    grouped = group_rows_by_fingerprint(rows)
    baseline_rows = grouped.get(baseline_fingerprint, [])
    candidate_rows = grouped.get(candidate_fingerprint, [])
    if not baseline_rows:
        raise ValueError(f"baseline fingerprint has no question_results vector: {baseline_fingerprint}")
    if not candidate_rows:
        raise ValueError(f"candidate fingerprint has no question_results vector: {candidate_fingerprint}")

    baseline_vector = majority_vector(baseline_rows)
    candidate_vector = majority_vector(candidate_rows)
    stats = mcnemar_from_vectors(
        baseline_vector,
        candidate_vector,
        f"baseline:{baseline_fingerprint}",
        f"candidate:{candidate_fingerprint}",
    )
    baseline_sig = _signature(
        archive_member_id=f"fingerprint:{baseline_fingerprint}",
        trial_id=None,
        vector=baseline_vector,
        rows=baseline_rows,
    )
    candidate_sig = _signature(
        archive_member_id=f"fingerprint:{candidate_fingerprint}",
        trial_id=None,
        vector=candidate_vector,
        rows=candidate_rows,
    )
    report = _paired_report(
        comparison_type="fingerprint_pair",
        baseline_label=baseline_fingerprint,
        candidate_label=candidate_fingerprint,
        stats=stats,
        baseline_signature=baseline_sig,
        candidate_signature=candidate_sig,
        min_shared_qids=min_shared_qids,
        max_accuracy_regression=max_accuracy_regression,
    )
    report["baseline_trials"] = [row["trial_id"] for row in baseline_rows]
    report["candidate_trials"] = [row["trial_id"] for row in candidate_rows]
    return report


def _paired_report(
    *,
    comparison_type: str,
    baseline_label: str,
    candidate_label: str,
    stats: McNemarResult,
    baseline_signature: dict[str, Any],
    candidate_signature: dict[str, Any],
    min_shared_qids: int,
    max_accuracy_regression: float,
) -> dict[str, Any]:
    severity, reasons = diff_signatures(baseline_signature, candidate_signature)
    decision, blockers = _decision(
        severity=severity,
        stats=stats,
        min_shared_qids=min_shared_qids,
        max_accuracy_regression=max_accuracy_regression,
    )
    return {
        "bsv_paired_report_version": "bsv-2-paired-report-v1",
        "comparison_type": comparison_type,
        "baseline": baseline_label,
        "candidate": candidate_label,
        "gate_decision": decision,
        "blockers": blockers,
        "thresholds": {
            "min_shared_qids": min_shared_qids,
            "max_accuracy_regression": abs(max_accuracy_regression),
        },
        "paired_stats": asdict(stats),
        "signature_diff": {
            "severity": severity,
            "reasons": reasons,
        },
        "baseline_signature": baseline_signature,
        "candidate_signature": candidate_signature,
    }


def render_markdown(report: dict[str, Any]) -> str:
    stats = report["paired_stats"]
    diff = report["signature_diff"]
    lines = [
        "# BSV-2 Paired Behavior Report",
        "",
        f"- Comparison: {report['comparison_type']}",
        f"- Baseline: {report['baseline']}",
        f"- Candidate: {report['candidate']}",
        f"- Gate decision: {report['gate_decision']}",
        f"- Shared qids: {stats['shared_qids']}",
        f"- Candidate accuracy delta: {stats['delta_b_minus_a']:+.6f}",
        f"- Signature severity: {diff['severity']}",
        "",
        "## Blockers",
        "",
    ]
    blockers = list(report.get("blockers") or [])
    lines.extend(f"- {item}" for item in blockers) if blockers else lines.append("- none")
    lines.extend(["", "## Signature Reasons", ""])
    lines.extend(f"- {item}" for item in diff.get("reasons") or [])
    return "\n".join(lines)


def _load_rows(path: Path | str) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_journal_rows(path)]


def _cmd_trial_pair(args: argparse.Namespace) -> int:
    report = build_trial_pair_report(
        _load_rows(args.journal),
        baseline_trial=args.baseline_trial,
        candidate_trial=args.candidate_trial,
        min_shared_qids=args.min_shared_qids,
        max_accuracy_regression=args.max_accuracy_regression,
    )
    _print_report(report, markdown=args.markdown)
    return 0 if args.no_fail or report["gate_decision"] == "pass" else 1


def _cmd_fingerprint_pair(args: argparse.Namespace) -> int:
    report = build_fingerprint_pair_report(
        _load_rows(args.journal),
        baseline_fingerprint=args.baseline_fingerprint,
        candidate_fingerprint=args.candidate_fingerprint,
        min_shared_qids=args.min_shared_qids,
        max_accuracy_regression=args.max_accuracy_regression,
    )
    _print_report(report, markdown=args.markdown)
    return 0 if args.no_fail or report["gate_decision"] == "pass" else 1


def _print_report(report: dict[str, Any], *, markdown: bool) -> None:
    if markdown:
        print(render_markdown(report))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only BSV-2 paired behavior report")
    parser.add_argument("--journal", default=str(DEFAULT_JOURNAL_DIR), help="journal dir or JSONL file")
    parser.add_argument("--min-shared-qids", type=int, default=DEFAULT_MIN_SHARED_QIDS)
    parser.add_argument("--max-accuracy-regression", type=float, default=0.0)
    parser.add_argument("--markdown", action="store_true", help="render a markdown report")
    parser.add_argument("--no-fail", action="store_true", help="always exit 0 after printing")
    sub = parser.add_subparsers(dest="cmd", required=True)

    trial = sub.add_parser("trial-pair", help="compare two vector-bearing trial ids")
    trial.add_argument("baseline_trial", type=int)
    trial.add_argument("candidate_trial", type=int)
    trial.set_defaults(func=_cmd_trial_pair)

    fp = sub.add_parser("fingerprint-pair", help="compare majority vectors by config fingerprint")
    fp.add_argument("baseline_fingerprint")
    fp.add_argument("candidate_fingerprint")
    fp.set_defaults(func=_cmd_fingerprint_pair)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
