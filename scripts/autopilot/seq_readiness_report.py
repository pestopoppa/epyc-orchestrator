#!/usr/bin/env python3
"""Read-only W4/W6 sequential-verdict readiness report.

The report deliberately does not infer a production verdict. It checks whether
the journal has enough trusted vector and shadow sequential evidence to justify
cutting over from the legacy MAD improvement branch.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from statistics import median
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ORCH_ROOT))

try:
    from scripts.autopilot.paired_stats import (
        DEFAULT_JOURNAL_DIR,
        compare_fingerprints,
        extract_question_outcomes,
        group_rows_by_fingerprint,
        iter_journal_rows,
        row_fingerprint,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from paired_stats import (  # type: ignore[no-redef]
        DEFAULT_JOURNAL_DIR,
        compare_fingerprints,
        extract_question_outcomes,
        group_rows_by_fingerprint,
        iter_journal_rows,
        row_fingerprint,
    )

from src.autopilot_core.sequential_verdict import (
    DEFAULT_POLICY,
    STATE_ACCUMULATING,
    STATE_CONFIRMED,
    STATE_REFUTED,
)


DEFAULT_MIN_TRUSTED_VECTOR_TRIALS = 120
DEFAULT_MIN_SEQ_SHADOW_ROWS = 30
DEFAULT_MIN_FLIP_RATE = 0.30
DEFAULT_HOLD_FLIP_RATE = 0.10
DEFAULT_MIN_SHARED_QIDS = 35
DEFAULT_STATE_PATH = ORCH_ROOT / "orchestration" / "autopilot_state.json"

LEGACY_MAD_EXCLUSIONS = frozenset({"mad_noise", "reproduction_confirmed"})
UNTRUSTED_OUTCOME_STATUSES = frozenset({"invalid", "skipped"})


@dataclass(frozen=True)
class CandidateCluster:
    fingerprint: str
    trusted_vector_trials: list[int]
    vector_trial_count: int
    median_quality: float
    latest_quality: float
    median_questions: int
    learning_exclusions: dict[str, int]
    seq_shadow_rows: int
    latest_seq_state: str | None


@dataclass(frozen=True)
class PairwiseReplay:
    fingerprint_a: str
    fingerprint_b: str
    shared_qids: int
    discordant_pairs: int
    flip_rate: float
    delta_b_minus_a: float
    p_value_two_sided: float


def build_seq_readiness_report(
    rows: Iterable[Mapping[str, Any]],
    *,
    state: Mapping[str, Any] | None = None,
    min_trusted_vector_trials: int = DEFAULT_MIN_TRUSTED_VECTOR_TRIALS,
    min_seq_shadow_rows: int = DEFAULT_MIN_SEQ_SHADOW_ROWS,
    min_flip_rate: float = DEFAULT_MIN_FLIP_RATE,
    hold_flip_rate: float = DEFAULT_HOLD_FLIP_RATE,
    min_shared_qids: int = DEFAULT_MIN_SHARED_QIDS,
) -> dict[str, Any]:
    """Build a no-write readiness report from folded journal rows."""
    normalized = [dict(row) for row in rows if isinstance(row, Mapping)]
    vector_rows = [row for row in normalized if extract_question_outcomes(dict(row))]
    trusted_rows = [row for row in vector_rows if _is_trusted_vector_row(row)]
    untrusted_rows = [row for row in vector_rows if row not in trusted_rows]
    clusters = _candidate_clusters(trusted_rows)
    pairwise = _pairwise_replays(trusted_rows, min_shared_qids=min_shared_qids)
    shadow = _seq_shadow_disagreement(trusted_rows)
    blockers = _cutover_blockers(
        trusted_vector_trials=len(trusted_rows),
        seq_shadow_rows=shadow["seq_shadow_rows"],
        flip_rate=shadow["flip_rate"],
        pairwise_count=len(pairwise),
        min_trusted_vector_trials=min_trusted_vector_trials,
        min_seq_shadow_rows=min_seq_shadow_rows,
        min_flip_rate=min_flip_rate,
    )
    cutover_ready = not blockers

    return {
        "ok": cutover_ready,
        "cutover_ready": cutover_ready,
        "policy_version": DEFAULT_POLICY.version,
        "thresholds": {
            "min_trusted_vector_trials": min_trusted_vector_trials,
            "min_seq_shadow_rows": min_seq_shadow_rows,
            "min_flip_rate": min_flip_rate,
            "hold_flip_rate": hold_flip_rate,
            "min_shared_qids": min_shared_qids,
        },
        "raw_vector_trials": len(vector_rows),
        "trusted_vector_trials": len(trusted_rows),
        "untrusted_vector_trials": len(untrusted_rows),
        "untrusted_vector_trial_ids": _trial_ids(untrusted_rows),
        "candidate_count": len(clusters),
        "candidate_clusters": [asdict(cluster) for cluster in clusters],
        "pairwise_replays": [asdict(replay) for replay in pairwise],
        "seq_shadow": shadow,
        "w8_promotion_evidence": _w8_promotion_evidence(state, normalized),
        "cutover_blockers": blockers,
        "recommendation": _recommendation(
            cutover_ready=cutover_ready,
            trusted_vector_trials=len(trusted_rows),
            seq_shadow_rows=shadow["seq_shadow_rows"],
            flip_rate=shadow["flip_rate"],
            hold_flip_rate=hold_flip_rate,
        ),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render the readiness report for operator checkpoints."""
    thresholds = report["thresholds"]
    shadow = report["seq_shadow"]
    lines = [
        "# AutoPilot Sequential Verdict Readiness Report",
        "",
        f"- Status: {'ready' if report['cutover_ready'] else 'blocked'}",
        f"- Recommendation: {report['recommendation']}",
        (
            "- Vector trials: "
            f"trusted={report['trusted_vector_trials']}, "
            f"raw={report['raw_vector_trials']}, "
            f"untrusted={report['untrusted_vector_trials']}"
        ),
        (
            "- Sequential shadow rows: "
            f"{shadow['seq_shadow_rows']} "
            f"(disagreements={shadow['disagreements']}, "
            f"flip_rate={_fmt_rate(shadow['flip_rate'])})"
        ),
        (
            "- Thresholds: "
            f"trusted_vectors>={thresholds['min_trusted_vector_trials']}, "
            f"seq_shadow>={thresholds['min_seq_shadow_rows']}, "
            f"flip_rate>={thresholds['min_flip_rate']:.0%}, "
            f"shared_qids>={thresholds['min_shared_qids']}"
        ),
        (
            "- W8 promotion evidence: "
            f"status={report.get('w8_promotion_evidence', {}).get('status', 'unknown')}, "
            f"pending_candidate="
            f"{report.get('w8_promotion_evidence', {}).get('pending_candidate')}, "
            f"last_finalized_trial="
            f"{report.get('w8_promotion_evidence', {}).get('last_finalized_trial_id')}, "
            f"last_blocked_reason="
            f"{report.get('w8_promotion_evidence', {}).get('last_blocked_reason')}, "
            f"latest_seq_trial="
            f"{report.get('w8_promotion_evidence', {}).get('latest_seq_trial_id')}, "
            f"latest_combined_E="
            f"{report.get('w8_promotion_evidence', {}).get('latest_combined_E')}, "
            f"required_E="
            f"{report.get('w8_promotion_evidence', {}).get('latest_required_E')}, "
            f"open_requirements="
            f"{report.get('w8_promotion_evidence', {}).get('open_requirements', [])}"
        ),
        "",
        "## Candidate Clusters",
        "",
    ]
    clusters = list(report.get("candidate_clusters") or [])
    if not clusters:
        lines.append("- no trusted vector-bearing candidates")
    else:
        for cluster in clusters:
            lines.append(
                "- fp={fp} trials={trials} n={n} median_q={median_q:.3f} "
                "latest_q={latest_q:.3f} median_questions={median_questions} "
                "learning_exclusions={learning_exclusions} seq_rows={seq_rows} "
                "latest_seq={latest_seq}".format(
                    fp=cluster["fingerprint"],
                    trials=_compact_trials(cluster["trusted_vector_trials"]),
                    n=cluster["vector_trial_count"],
                    median_q=float(cluster["median_quality"]),
                    latest_q=float(cluster["latest_quality"]),
                    median_questions=cluster["median_questions"],
                    learning_exclusions=cluster["learning_exclusions"],
                    seq_rows=cluster["seq_shadow_rows"],
                    latest_seq=cluster["latest_seq_state"] or "none",
                )
            )

    blockers = list(report.get("cutover_blockers") or [])
    if blockers:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {blocker}" for blocker in blockers)

    pairwise = list(report.get("pairwise_replays") or [])
    if pairwise:
        lines.extend(["", "## Pairwise Replay", ""])
        for item in pairwise:
            lines.append(
                "- {a} -> {b}: shared={shared} discordant={discordant} "
                "flip_rate={flip_rate:.1%} delta_b_minus_a={delta:+.3f} p={p:.4g}".format(
                    a=item["fingerprint_a"],
                    b=item["fingerprint_b"],
                    shared=item["shared_qids"],
                    discordant=item["discordant_pairs"],
                    flip_rate=item["flip_rate"],
                    delta=item["delta_b_minus_a"],
                    p=item["p_value_two_sided"],
                )
            )
    return "\n".join(lines)


def _is_trusted_vector_row(row: Mapping[str, Any]) -> bool:
    if row.get("bug_corrupted_by"):
        return False
    try:
        if int(row.get("tier", 1)) < 1:
            return False
    except (TypeError, ValueError):
        return False
    status = str(row.get("outcome_status") or "ok")
    if status in UNTRUSTED_OUTCOME_STATUSES:
        return False
    return bool(extract_question_outcomes(dict(row)))


def _candidate_clusters(rows: list[Mapping[str, Any]]) -> list[CandidateCluster]:
    grouped = group_rows_by_fingerprint([dict(row) for row in rows])
    clusters: list[CandidateCluster] = []
    for fingerprint, group in sorted(
        grouped.items(),
        key=lambda item: (_latest_trial_id(item[1]), item[0]),
        reverse=True,
    ):
        qualities = [_float(row.get("quality")) for row in group]
        q_counts = [len(extract_question_outcomes(row)) for row in group]
        latest = max(group, key=_latest_trial_id)
        seq_rows = [row for row in group if _seq_state(row) is not None]
        clusters.append(
            CandidateCluster(
                fingerprint=fingerprint,
                trusted_vector_trials=_trial_ids(group),
                vector_trial_count=len(group),
                median_quality=round(float(median(qualities)), 6) if qualities else 0.0,
                latest_quality=round(_float(latest.get("quality")), 6),
                median_questions=int(median(q_counts)) if q_counts else 0,
                learning_exclusions=dict(
                    sorted(Counter(_learning_exclusion_by(row) or "none" for row in group).items())
                ),
                seq_shadow_rows=len(seq_rows),
                latest_seq_state=_seq_state(max(seq_rows, key=_latest_trial_id))
                if seq_rows
                else None,
            )
        )
    return clusters


def _pairwise_replays(
    rows: list[Mapping[str, Any]],
    *,
    min_shared_qids: int,
) -> list[PairwiseReplay]:
    grouped = group_rows_by_fingerprint([dict(row) for row in rows])
    fingerprints = sorted(
        grouped,
        key=lambda fp: (len(grouped[fp]), _latest_trial_id(grouped[fp]), fp),
        reverse=True,
    )
    replays: list[PairwiseReplay] = []
    for idx, fp_a in enumerate(fingerprints):
        for fp_b in fingerprints[idx + 1 :]:
            result = compare_fingerprints([dict(row) for row in rows], fp_b, fp_a)
            shared_qids = int(result["shared_qids"])
            if shared_qids < min_shared_qids:
                continue
            discordant = int(result["a_correct_b_wrong"]) + int(result["a_wrong_b_correct"])
            replays.append(
                PairwiseReplay(
                    fingerprint_a=fp_a,
                    fingerprint_b=fp_b,
                    shared_qids=shared_qids,
                    discordant_pairs=discordant,
                    flip_rate=round(discordant / shared_qids if shared_qids else 0.0, 6),
                    delta_b_minus_a=float(result["delta_b_minus_a"]),
                    p_value_two_sided=float(result["p_value_two_sided"]),
                )
            )
    return replays


def _seq_shadow_disagreement(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    seq_rows = [row for row in rows if _seq_state(row) is not None]
    disagreements = 0
    examples: list[dict[str, Any]] = []
    for row in seq_rows:
        legacy = _legacy_effective_verdict(row)
        seq = _seq_effective_verdict(row)
        if legacy == seq:
            continue
        disagreements += 1
        if len(examples) < 10:
            examples.append(
                {
                    "trial_id": _trial_id(row),
                    "fingerprint": row_fingerprint(dict(row)),
                    "legacy": legacy,
                    "seq": seq,
                    "seq_state": _seq_state(row),
                    "learning_exclusion": _learning_exclusion_by(row),
                }
            )
    denominator = len(seq_rows)
    return {
        "seq_shadow_rows": denominator,
        "disagreements": disagreements,
        "flip_rate": round(disagreements / denominator, 6) if denominator else None,
        "examples": examples,
        "policy_versions": sorted(
            {
                str((row.get("seq") or {}).get("policy_version") or DEFAULT_POLICY.version)
                for row in seq_rows
                if isinstance(row.get("seq"), Mapping)
            }
        ),
    }


def _safe_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _w8_promotion_evidence(
    state: Mapping[str, Any] | None,
    rows: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Project W8 promotion-eval state into read-only reports."""
    state_map = _safe_mapping(state)
    pending = _safe_mapping(state_map.get("seq_pending_promotion_fresh_eval"))
    finalized = _safe_mapping(state_map.get("seq_last_promotion_finalized"))
    blocked = _safe_mapping(state_map.get("seq_last_promotion_blocked"))
    baseline_forced = _safe_mapping(state_map.get("seq_baseline_draw_forced"))
    baseline_blocked = _safe_mapping(state_map.get("seq_baseline_draw_blocked"))
    latest_seq = _latest_w8_seq_snapshot(rows)
    finalized_delta_ci = _safe_mapping(finalized.get("delta_ci"))

    if pending:
        status = "pending_fresh_eval"
    elif finalized:
        status = "finalized"
    elif blocked:
        status = "blocked"
    else:
        status = "none"

    open_requirements = _w8_open_requirements(
        status=status,
        latest_seq=latest_seq,
        pending=pending,
        blocked=blocked,
    )

    return {
        "status": status,
        "open_requirements": open_requirements,
        "pending": pending,
        "last_finalized": finalized,
        "last_blocked": blocked,
        "pending_candidate": pending.get("candidate"),
        "pending_source_trial_id": pending.get("source_trial_id"),
        "pending_attempts": pending.get("attempts"),
        "pending_combined_E": pending.get("combined_E"),
        "last_finalized_trial_id": finalized.get("trial_id"),
        "last_finalized_candidate": finalized.get("candidate"),
        "last_finalized_combined_E": finalized.get("combined_E"),
        "last_finalized_delta_ci": finalized_delta_ci,
        "last_finalized_delta_excludes_regression": finalized_delta_ci.get(
            "excludes_regression"
        ),
        "last_blocked_trial_id": blocked.get("trial_id"),
        "last_blocked_candidate": blocked.get("candidate"),
        "last_blocked_reason": blocked.get("reason"),
        "last_blocked_combined_E": blocked.get("combined_E"),
        "latest_seq_trial_id": latest_seq.get("trial_id"),
        "latest_candidate": latest_seq.get("candidate"),
        "latest_combined_E": latest_seq.get("combined_E"),
        "latest_required_E": latest_seq.get("required_E"),
        "latest_confirmed": latest_seq.get("confirmed"),
        "latest_seq_state": latest_seq.get("state"),
        "latest_baseline_reference_state": latest_seq.get("baseline_reference_state"),
        "latest_fresh_eval": latest_seq.get("fresh_eval"),
        "baseline_reference_last_forced_trial_id": baseline_forced.get("trial_id"),
        "baseline_reference_last_forced_reason": _safe_mapping(
            baseline_forced.get("reference")
        ).get("reason"),
        "baseline_reference_last_forced_stale": _safe_mapping(
            baseline_forced.get("reference")
        ).get("stale_reference"),
        "baseline_reference_blocked_trial_id": baseline_blocked.get("trial_id"),
        "baseline_reference_blocked_reason": baseline_blocked.get("reason"),
    }


def _w8_open_requirements(
    *,
    status: str,
    latest_seq: Mapping[str, Any],
    pending: Mapping[str, Any],
    blocked: Mapping[str, Any],
) -> list[str]:
    """Return unmet W8 promotion-eval requirements as stable report keys."""
    if status == "finalized":
        return []

    requirements: list[str] = []
    if pending:
        requirements.append("pending_fresh_eval_queued")
    if blocked.get("reason"):
        requirements.append(f"last_blocked:{blocked['reason']}")
    if not latest_seq:
        requirements.append("missing_seq_promotion_snapshot")
        return requirements

    combined = latest_seq.get("combined_E")
    required = latest_seq.get("required_E")
    if isinstance(combined, (int, float)) and isinstance(required, (int, float)):
        if combined < required:
            requirements.append("combined_E_below_required")
    elif required is not None:
        requirements.append("combined_E_unavailable")

    if latest_seq.get("fresh_eval") is not True:
        requirements.append("fresh_promotion_eval_required")
    if latest_seq.get("baseline_reference_state") != "fresh":
        requirements.append("fresh_baseline_reference_required")
    if latest_seq.get("confirmed") is not True:
        requirements.append("seq_confirmation_required")
    return requirements


def _latest_w8_seq_snapshot(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return the latest journal seq row relevant to W8 promotion progress."""
    latest: Mapping[str, Any] | None = None
    for row in rows:
        seq = row.get("seq") if isinstance(row, Mapping) else None
        if not isinstance(seq, Mapping):
            continue
        if not (
            "baseline_promotion_combined_E" in seq
            or "baseline_reference_state" in seq
            or "baseline_promotion_fresh_eval" in seq
        ):
            continue
        if latest is None or _latest_trial_id(row) >= _latest_trial_id(latest):
            latest = row
    if latest is None:
        return {}
    seq = _safe_mapping(latest.get("seq"))
    return {
        "trial_id": _trial_id(latest),
        "candidate": seq.get("candidate"),
        "combined_E": seq.get("baseline_promotion_combined_E"),
        "required_E": seq.get("baseline_promotion_required_E"),
        "confirmed": seq.get("confirmed"),
        "state": seq.get("state"),
        "baseline_reference_state": seq.get("baseline_reference_state"),
        "fresh_eval": seq.get("baseline_promotion_fresh_eval"),
    }


def _cutover_blockers(
    *,
    trusted_vector_trials: int,
    seq_shadow_rows: int,
    flip_rate: float | None,
    pairwise_count: int,
    min_trusted_vector_trials: int,
    min_seq_shadow_rows: int,
    min_flip_rate: float,
) -> list[str]:
    blockers: list[str] = []
    if trusted_vector_trials < min_trusted_vector_trials:
        blockers.append(
            "trusted vector history too small: "
            f"{trusted_vector_trials} < {min_trusted_vector_trials}"
        )
    if pairwise_count == 0:
        blockers.append("no multi-fingerprint paired replay with enough shared qids")
    if seq_shadow_rows < min_seq_shadow_rows:
        blockers.append(
            "sequential shadow history too small: "
            f"{seq_shadow_rows} < {min_seq_shadow_rows}"
        )
    if flip_rate is None:
        blockers.append("no seq-vs-legacy flip-rate denominator yet")
    elif flip_rate < min_flip_rate:
        blockers.append(
            f"observed seq-vs-legacy flip rate {flip_rate:.1%} < {min_flip_rate:.0%}"
        )
    return blockers


def _recommendation(
    *,
    cutover_ready: bool,
    trusted_vector_trials: int,
    seq_shadow_rows: int,
    flip_rate: float | None,
    hold_flip_rate: float,
) -> str:
    if cutover_ready:
        return "review report and cut over only with an explicit restart window"
    if seq_shadow_rows == 0:
        return (
            "do not wire sequential verdicts as authority yet; collect trusted "
            "vectors and dual-log seq shadow verdicts first"
        )
    if flip_rate is not None and flip_rate < hold_flip_rate:
        return "hold cutover; observed flip rate is below the handoff hold band"
    if trusted_vector_trials == 0:
        return "collect clean vector-bearing trials before any W4/W6 decision"
    return "continue shadow collection; cutover blockers remain"


def _legacy_effective_verdict(row: Mapping[str, Any]) -> str:
    by = _learning_exclusion_by(row)
    if by in LEGACY_MAD_EXCLUSIONS:
        return "benign_no_learning"
    if by:
        return f"excluded:{by}"
    return "clean_pass"


def _seq_effective_verdict(row: Mapping[str, Any]) -> str:
    state = _seq_state(row)
    if state == STATE_CONFIRMED:
        return "clean_pass"
    if state == STATE_ACCUMULATING:
        return "benign_no_learning"
    if state == STATE_REFUTED:
        return "failed_experiment"
    return "missing_seq"


def _seq_state(row: Mapping[str, Any]) -> str | None:
    seq = row.get("seq")
    if not isinstance(seq, Mapping):
        return None
    state = seq.get("state")
    return str(state) if state else None


def _learning_exclusion_by(row: Mapping[str, Any]) -> str | None:
    eval_details = row.get("eval_details") or {}
    if not isinstance(eval_details, Mapping):
        eval_details = {}
    raw = eval_details.get("learning_exclusion") or row.get("learning_exclusion")
    if isinstance(raw, Mapping):
        by = raw.get("by")
        return str(by) if by else None
    return None


def _trial_id(row: Mapping[str, Any]) -> int | None:
    try:
        return int(row.get("trial_id"))
    except (TypeError, ValueError):
        return None


def _trial_ids(rows: Iterable[Mapping[str, Any]]) -> list[int]:
    return sorted(trial_id for row in rows if (trial_id := _trial_id(row)) is not None)


def _latest_trial_id(rows_or_row: Iterable[Mapping[str, Any]] | Mapping[str, Any]) -> int:
    if isinstance(rows_or_row, Mapping):
        return _trial_id(rows_or_row) or -1
    return max((_trial_id(row) or -1 for row in rows_or_row), default=-1)


def _compact_trials(trial_ids: list[int]) -> str:
    if not trial_ids:
        return "[]"
    if len(trial_ids) <= 6:
        return "[" + ",".join(str(trial_id) for trial_id in trial_ids) + "]"
    head = ",".join(str(trial_id) for trial_id in trial_ids[:3])
    tail = ",".join(str(trial_id) for trial_id in trial_ids[-2:])
    return f"[{head},...,{tail}]"


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _fmt_rate(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.1%}"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report read-only W4/W6 sequential-verdict cutover readiness."
    )
    parser.add_argument("--journal", type=Path, default=DEFAULT_JOURNAL_DIR)
    parser.add_argument(
        "--state",
        type=Path,
        default=DEFAULT_STATE_PATH,
        help="Optional AutoPilot state JSON path for W8 promotion-eval evidence.",
    )
    parser.add_argument("--json", action="store_true", help="Emit structured JSON.")
    parser.add_argument(
        "--out-json",
        type=Path,
        help="Write the structured report JSON to this path while preserving stdout behavior.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        help="Write the rendered Markdown report to this path while preserving stdout behavior.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when sequential-verdict cutover is not ready.",
    )
    parser.add_argument(
        "--min-trusted-vector-trials",
        type=int,
        default=DEFAULT_MIN_TRUSTED_VECTOR_TRIALS,
    )
    parser.add_argument("--min-seq-shadow-rows", type=int, default=DEFAULT_MIN_SEQ_SHADOW_ROWS)
    parser.add_argument("--min-flip-rate", type=float, default=DEFAULT_MIN_FLIP_RATE)
    parser.add_argument("--hold-flip-rate", type=float, default=DEFAULT_HOLD_FLIP_RATE)
    parser.add_argument("--min-shared-qids", type=int, default=DEFAULT_MIN_SHARED_QIDS)
    return parser.parse_args(argv)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rows = list(iter_journal_rows(args.journal))
    state = _read_optional_state(args.state)
    report = build_seq_readiness_report(
        rows,
        state=state,
        min_trusted_vector_trials=max(0, args.min_trusted_vector_trials),
        min_seq_shadow_rows=max(0, args.min_seq_shadow_rows),
        min_flip_rate=max(0.0, args.min_flip_rate),
        hold_flip_rate=max(0.0, args.hold_flip_rate),
        min_shared_qids=max(0, args.min_shared_qids),
    )
    json_text = json.dumps(report, sort_keys=True, default=str)
    markdown_text = render_markdown(report)
    if args.out_json:
        _write_text(args.out_json, json_text + "\n")
    if args.out_md:
        _write_text(args.out_md, markdown_text + "\n")
    if args.json:
        print(json_text)
    else:
        print(markdown_text)
    if args.strict and not report["cutover_ready"]:
        return 1
    return 0


def _read_optional_state(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
