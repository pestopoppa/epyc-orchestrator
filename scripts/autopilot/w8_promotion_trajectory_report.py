#!/usr/bin/env python3
"""Read-only W8 promotion-evidence trajectory report.

The sequential readiness report summarizes the latest W8 state. This report
answers a narrower operational question: are candidate replays actually adding
repeated evidence toward confirmation/refutation, or are accumulating
candidates becoming one-off rows that never receive another observation?
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ORCH_ROOT))

try:
    from scripts.autopilot.paired_stats import DEFAULT_JOURNAL_DIR, iter_journal_rows
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from paired_stats import DEFAULT_JOURNAL_DIR, iter_journal_rows  # type: ignore[no-redef]

try:
    from scripts.autopilot import controller_io
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    import controller_io  # type: ignore[no-redef]


DEFAULT_MAX_REPLAY_ATTEMPTS = 12
DEFAULT_STALE_TRIALS = 12


@dataclass(frozen=True)
class W8Snapshot:
    trial_id: int
    candidate: str
    action_type: str | None
    config_snapshot: Mapping[str, Any] | None
    quality: float | None
    state: str | None
    confirmed: bool | None
    combined_E: float | None
    required_E: float | None
    E_quality: float | None
    E_rate_noninf: float | None
    k: int | None
    r_eff: int | None
    fresh_eval: bool | None
    baseline_reference_state: str | None
    finalized: bool | None
    keep_revert_decision: str | None
    failure_first_violation: str | None


@dataclass(frozen=True)
class CandidateTrajectory:
    candidate: str
    status: str
    trials: list[int]
    attempts: int
    first_trial_id: int
    latest_trial_id: int
    latest_state: str | None
    latest_keep_revert_decision: str | None
    latest_confirmed: bool | None
    latest_combined_E: float | None
    required_E: float | None
    max_combined_E: float | None
    combined_E_delta: float | None
    latest_k: int | None
    max_k: int | None
    replay_capacity_remaining: int | None
    fresh_eval_count: int
    stale_reference_count: int
    latest_failure_first_violation: str | None
    latest_failure_violation_details: dict[str, Any]
    recent: bool
    stale_accumulating: bool
    replay_eligible: bool
    replay_blocker: str | None


def build_w8_trajectory_report(
    rows: Iterable[Mapping[str, Any]],
    *,
    max_replay_attempts: int = DEFAULT_MAX_REPLAY_ATTEMPTS,
    stale_trials: int = DEFAULT_STALE_TRIALS,
) -> dict[str, Any]:
    """Build a no-write W8 replay trajectory report from folded journal rows."""
    snapshots = sorted(
        (_snapshot_from_row(row) for row in rows),
        key=lambda snapshot: snapshot.trial_id,
    )
    snapshots = [snapshot for snapshot in snapshots if snapshot.candidate]
    latest_trial_id = max((snapshot.trial_id for snapshot in snapshots), default=None)
    grouped: dict[str, list[W8Snapshot]] = defaultdict(list)
    for snapshot in snapshots:
        grouped[snapshot.candidate].append(snapshot)

    trajectories = [
        _candidate_trajectory(
            candidate,
            group,
            latest_trial_id=latest_trial_id,
            max_replay_attempts=max_replay_attempts,
            stale_trials=stale_trials,
        )
        for candidate, group in grouped.items()
    ]
    trajectories.sort(
        key=lambda item: (
            item.status != "active_recent_replay",
            -(item.latest_trial_id or -1),
            item.candidate,
        )
    )

    status_counts: dict[str, int] = {}
    for trajectory in trajectories:
        status_counts[trajectory.status] = status_counts.get(trajectory.status, 0) + 1
    terminal_reason = _terminal_reason_summary(trajectories)

    recent_active = [
        item for item in trajectories if item.status == "active_recent_replay"
    ]
    replay_eligible = [item for item in trajectories if item.replay_eligible]
    recent_replay_eligible = [item for item in replay_eligible if item.recent]
    stale_accumulating = [
        item for item in trajectories if item.status == "stale_accumulating"
    ]
    concentration = _replay_concentration(
        trajectories,
        recent_active=recent_replay_eligible,
        stale_accumulating=stale_accumulating,
    )
    blocked = _open_requirements(
        trajectories,
        recent_active=recent_active,
        replay_eligible=replay_eligible,
        recent_replay_eligible=recent_replay_eligible,
        stale_accumulating=stale_accumulating,
        concentration=concentration,
    )
    if not snapshots:
        status = "no_w8_snapshots"
    elif recent_replay_eligible:
        status = "progressing"
    elif stale_accumulating:
        status = "stale_accumulating"
    else:
        status = "evidence_bound"

    return {
        "ok": status == "progressing",
        "status": status,
        "latest_trial_id": latest_trial_id,
        "snapshot_count": len(snapshots),
        "candidate_count": len(trajectories),
        "max_replay_attempts": max_replay_attempts,
        "stale_trials": stale_trials,
        "status_counts": dict(sorted(status_counts.items())),
        "terminal_reason_counts": terminal_reason["counts"],
        "dominant_terminal_reason": terminal_reason["dominant"],
        "replay_concentration": concentration,
        "open_requirements": blocked,
        "recent_active_candidates": [item.candidate for item in recent_active],
        "replay_eligible_candidates": [item.candidate for item in replay_eligible],
        "recent_replay_eligible_candidates": [
            item.candidate for item in recent_replay_eligible
        ],
        "replay_blockers": {
            item.candidate: item.replay_blocker
            for item in trajectories
            if item.replay_blocker
        },
        "stale_accumulating_candidates": [item.candidate for item in stale_accumulating],
        "trajectories": [asdict(item) for item in trajectories],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a concise operator-facing Markdown report."""
    lines = [
        "# AutoPilot W8 Promotion Trajectory Report",
        "",
        f"- Status: {report.get('status')}",
        f"- Latest trial: {report.get('latest_trial_id')}",
        (
            "- Evidence: "
            f"snapshots={report.get('snapshot_count')}, "
            f"candidates={report.get('candidate_count')}, "
            f"status_counts={report.get('status_counts')}"
        ),
        (
            "- Replay policy: "
            f"max_attempts={report.get('max_replay_attempts')}, "
            f"stale_trials={report.get('stale_trials')}"
        ),
        (
            "- Replay eligibility: "
            f"eligible={report.get('replay_eligible_candidates') or []}, "
            f"recent={report.get('recent_replay_eligible_candidates') or []}, "
            f"blocked={len(report.get('replay_blockers') or {})}"
        ),
    ]
    concentration = dict(report.get("replay_concentration") or {})
    if concentration:
        lines.extend(
            [
                "",
                "## Replay Concentration",
                "",
                (
                    "- active_recent={active}, stale_accumulating={stale}, "
                    "single_observation={single}, top_active_share={share}, "
                    "warning={warning}".format(
                        active=concentration.get("active_recent_candidate_count"),
                        stale=concentration.get("stale_accumulating_count"),
                        single=concentration.get("single_observation_count"),
                        share=_fmt_float(concentration.get("top_active_attempt_share")),
                        warning=concentration.get("warning"),
                    )
                ),
            ]
        )
        reason = concentration.get("warning_reason")
        if reason:
            lines.append(f"- reason: {reason}")
    dominant_reason = report.get("dominant_terminal_reason") or {}
    if dominant_reason:
        lines.extend(
            [
                "",
                "## Terminal Candidate Reasons",
                "",
                (
                    "- dominant: {reason} ({count} candidate(s), status={status})".format(
                        reason=dominant_reason.get("reason") or "unknown",
                        count=dominant_reason.get("count"),
                        status=dominant_reason.get("status") or "unknown",
                    )
                ),
            ]
        )
        if dominant_reason.get("baseline_sample_warning"):
            lines.append(
                "- warning: dominant per-suite regression compares against a very small baseline sample"
            )
    requirements = list(report.get("open_requirements") or [])
    if requirements:
        lines.extend(["", "## Open Requirements", ""])
        lines.extend(f"- {requirement}" for requirement in requirements)

    trajectories = list(report.get("trajectories") or [])
    lines.extend(["", "## Candidate Trajectories", ""])
    if not trajectories:
        lines.append("- no W8 promotion snapshots found")
        return "\n".join(lines)

    lines.append(
        "| Candidate | Status | Trials | latest E / required | latest state | k | fresh evals | Replay |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for item in trajectories[:25]:
        lines.append(
            "| {candidate} | {status} | {trials} | {combined} / {required} | "
            "{state} | {k} | {fresh} | {replay} |".format(
                candidate=item.get("candidate"),
                status=item.get("status"),
                trials=_compact_trials(list(item.get("trials") or [])),
                combined=_fmt_float(item.get("latest_combined_E")),
                required=_fmt_float(item.get("required_E")),
                state=item.get("latest_state") or "none",
                k=item.get("latest_k"),
                fresh=item.get("fresh_eval_count"),
                replay=(
                    "eligible"
                    if item.get("replay_eligible")
                    else item.get("replay_blocker") or ""
                ),
            )
        )
    if len(trajectories) > 25:
        lines.append(f"\n_Only the latest 25 of {len(trajectories)} candidates are shown._")
    return "\n".join(lines)


def _candidate_trajectory(
    candidate: str,
    snapshots: list[W8Snapshot],
    *,
    latest_trial_id: int | None,
    max_replay_attempts: int,
    stale_trials: int,
) -> CandidateTrajectory:
    group = sorted(snapshots, key=lambda snapshot: snapshot.trial_id)
    first = group[0]
    latest = group[-1]
    combined_values = [
        snapshot.combined_E for snapshot in group if snapshot.combined_E is not None
    ]
    latest_age = None if latest_trial_id is None else latest_trial_id - latest.trial_id
    recent = latest_age is not None and latest_age <= stale_trials
    capacity_remaining = (
        max(0, max_replay_attempts - latest.k) if latest.k is not None else None
    )
    replay_blocker = _replay_blocker(latest, max_replay_attempts=max_replay_attempts)
    replay_eligible = replay_blocker is None
    latest_ap24_ineligible = _terminal_keep_revert_decision(latest)
    stale_accumulating = (
        latest.state == "accumulating"
        and not latest.confirmed
        and latest.finalized is not True
        and not latest_ap24_ineligible
        and replay_eligible
        and not recent
        and (capacity_remaining is None or capacity_remaining > 0)
    )
    status = _candidate_status(
        latest,
        attempts=len(group),
        recent=recent,
        stale_accumulating=stale_accumulating,
        max_replay_attempts=max_replay_attempts,
    )
    return CandidateTrajectory(
        candidate=candidate,
        status=status,
        trials=[snapshot.trial_id for snapshot in group],
        attempts=len(group),
        first_trial_id=first.trial_id,
        latest_trial_id=latest.trial_id,
        latest_state=latest.state,
        latest_keep_revert_decision=latest.keep_revert_decision,
        latest_confirmed=latest.confirmed,
        latest_combined_E=latest.combined_E,
        required_E=latest.required_E,
        max_combined_E=max(combined_values) if combined_values else None,
        combined_E_delta=_delta(first.combined_E, latest.combined_E),
        latest_k=latest.k,
        max_k=max((snapshot.k or 0 for snapshot in group), default=0),
        replay_capacity_remaining=capacity_remaining,
        fresh_eval_count=sum(1 for snapshot in group if snapshot.fresh_eval is True),
        stale_reference_count=sum(
            1 for snapshot in group if snapshot.baseline_reference_state == "stale"
        ),
        latest_failure_first_violation=latest.failure_first_violation,
        latest_failure_violation_details=_parse_suite_regression(
            latest.failure_first_violation
        ),
        recent=recent,
        stale_accumulating=stale_accumulating,
        replay_eligible=replay_eligible,
        replay_blocker=replay_blocker,
    )


def _candidate_status(
    latest: W8Snapshot,
    *,
    attempts: int,
    recent: bool,
    stale_accumulating: bool,
    max_replay_attempts: int,
) -> str:
    keep_revert = str(latest.keep_revert_decision or "").strip()
    if keep_revert == "revert":
        return "reverted"
    if latest.finalized:
        return "finalized"
    if latest.state == "refuted":
        return "refuted"
    if latest.confirmed:
        return "confirmed_waiting_fresh_eval"
    if keep_revert == "excluded" and _terminal_keep_revert_decision(latest):
        return "excluded"
    if stale_accumulating:
        return "stale_accumulating"
    if latest.k is not None and latest.k >= max_replay_attempts:
        return "attempt_cap_reached"
    if latest.state == "accumulating" and recent and attempts > 1:
        return "active_recent_replay"
    if latest.state == "accumulating":
        return "single_observation"
    return "unclassified"


def _open_requirements(
    trajectories: list[CandidateTrajectory],
    *,
    recent_active: list[CandidateTrajectory],
    replay_eligible: list[CandidateTrajectory],
    recent_replay_eligible: list[CandidateTrajectory],
    stale_accumulating: list[CandidateTrajectory],
    concentration: Mapping[str, Any],
) -> list[str]:
    if not trajectories:
        return ["missing_w8_promotion_snapshots"]
    requirements = ["combined_E_below_required", "fresh_promotion_eval_required"]
    if not recent_active:
        requirements.append("no_recent_multi_observation_accumulating_candidate")
    if not replay_eligible:
        requirements.append("no_replay_eligible_accumulating_candidate")
    elif not recent_replay_eligible:
        requirements.append("no_recent_replay_eligible_accumulating_candidate")
    if stale_accumulating:
        requirements.append("stale_accumulating_candidates_present")
    if concentration.get("warning"):
        requirements.append("replay_concentration_warning")
    if not any(item.latest_confirmed for item in trajectories):
        requirements.append("seq_confirmation_required")
    return requirements


def _replay_concentration(
    trajectories: list[CandidateTrajectory],
    *,
    recent_active: list[CandidateTrajectory],
    stale_accumulating: list[CandidateTrajectory],
) -> dict[str, Any]:
    active_attempts = {
        item.candidate: item.attempts
        for item in recent_active
        if item.attempts > 0
    }
    total_active_attempts = sum(active_attempts.values())
    top_candidate = None
    top_attempts = 0
    if active_attempts:
        top_candidate, top_attempts = max(
            active_attempts.items(),
            key=lambda item: (item[1], item[0]),
        )
    top_share = (
        round(top_attempts / total_active_attempts, 6)
        if total_active_attempts
        else None
    )
    stale_count = len(stale_accumulating)
    single_count = sum(1 for item in trajectories if item.status == "single_observation")
    warning = bool(
        recent_active
        and stale_count > 0
        and (
            len(recent_active) == 1
            or (top_share is not None and top_share >= 0.75)
        )
    )
    reason = None
    if warning:
        reason = (
            "recent replay evidence is concentrated in "
            f"{top_candidate or 'one candidate'} while "
            f"{stale_count} accumulating candidate(s) are stale"
        )
    return {
        "warning": warning,
        "warning_reason": reason,
        "active_recent_candidate_count": len(recent_active),
        "active_recent_attempts": active_attempts,
        "total_active_recent_attempts": total_active_attempts,
        "top_active_candidate": top_candidate,
        "top_active_attempts": top_attempts,
        "top_active_attempt_share": top_share,
        "stale_accumulating_count": stale_count,
        "single_observation_count": single_count,
    }


def _terminal_reason_summary(
    trajectories: Iterable[CandidateTrajectory],
) -> dict[str, Any]:
    terminal = {"reverted", "excluded", "refuted"}
    counts: Counter[str] = Counter()
    examples: dict[str, CandidateTrajectory] = {}
    for trajectory in trajectories:
        if trajectory.status not in terminal:
            continue
        reason = (
            trajectory.latest_failure_first_violation
            or f"candidate status {trajectory.status}"
        )
        counts[reason] += 1
        examples.setdefault(reason, trajectory)
    if not counts:
        return {"counts": {}, "dominant": None}
    reason, count = counts.most_common(1)[0]
    example = examples[reason]
    details = dict(example.latest_failure_violation_details or {})
    baseline_n = details.get("n_baseline")
    baseline_sample_warning = (
        details.get("kind") == "suite_regression"
        and isinstance(baseline_n, int)
        and baseline_n <= 2
    )
    dominant = {
        "reason": reason,
        "count": count,
        "status": example.status,
        "candidate": example.candidate,
        "latest_trial_id": example.latest_trial_id,
        "details": details,
        "baseline_sample_warning": baseline_sample_warning,
    }
    return {"counts": dict(counts), "dominant": dominant}


_SUITE_REGRESSION_RE = re.compile(
    r"^Suite '(?P<suite>[^']+)' regression: (?P<delta>[+-]?\d+(?:\.\d+)?) "
    r"\(threshold: (?P<threshold>[+-]?\d+(?:\.\d+)?); "
    r"n_result=(?P<n_result>[^,]+), n_baseline=(?P<n_baseline>[^)]+)\)"
)


def _parse_suite_regression(reason: str | None) -> dict[str, Any]:
    if not reason:
        return {}
    match = _SUITE_REGRESSION_RE.match(reason.strip())
    if not match:
        return {}

    def parse_int(value: str) -> int | None:
        value = value.strip()
        if value in {"", "None", "null"}:
            return None
        try:
            return int(value)
        except ValueError:
            return None

    return {
        "kind": "suite_regression",
        "suite": match.group("suite"),
        "delta": float(match.group("delta")),
        "threshold": float(match.group("threshold")),
        "n_result": parse_int(match.group("n_result")),
        "n_baseline": parse_int(match.group("n_baseline")),
    }


def _first_failure_violation(failure_analysis: Any) -> str | None:
    if not isinstance(failure_analysis, str):
        return None
    for line in failure_analysis.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            return stripped[2:].strip() or None
    return None


def _snapshot_from_row(row: Mapping[str, Any]) -> W8Snapshot:
    seq = row.get("seq") if isinstance(row, Mapping) else None
    if not isinstance(seq, Mapping):
        return W8Snapshot(
            trial_id=-1,
            candidate="",
            action_type=None,
            config_snapshot=None,
            quality=None,
            state=None,
            confirmed=None,
            combined_E=None,
            required_E=None,
            E_quality=None,
            E_rate_noninf=None,
            k=None,
            r_eff=None,
            fresh_eval=None,
            baseline_reference_state=None,
            finalized=None,
            keep_revert_decision=None,
            failure_first_violation=_first_failure_violation(
                row.get("failure_analysis")
            ),
        )
    if not (
        "baseline_promotion_combined_E" in seq
        or "baseline_reference_state" in seq
        or "baseline_promotion_fresh_eval" in seq
    ):
        return W8Snapshot(
            trial_id=-1,
            candidate="",
            action_type=None,
            config_snapshot=None,
            quality=None,
            state=None,
            confirmed=None,
            combined_E=None,
            required_E=None,
            E_quality=None,
            E_rate_noninf=None,
            k=None,
            r_eff=None,
            fresh_eval=None,
            baseline_reference_state=None,
            finalized=None,
            keep_revert_decision=None,
            failure_first_violation=_first_failure_violation(
                row.get("failure_analysis")
            ),
        )
    config_snapshot = _config_snapshot(row.get("config_snapshot"))
    return W8Snapshot(
        trial_id=_int(row.get("trial_id"), default=-1),
        candidate=str(seq.get("candidate") or ""),
        action_type=_optional_str(
            row.get("action_type") or (config_snapshot or {}).get("type")
        ),
        config_snapshot=config_snapshot,
        quality=_optional_float(row.get("quality")),
        state=_optional_str(seq.get("state")),
        confirmed=_optional_bool(seq.get("confirmed")),
        combined_E=_optional_float(seq.get("baseline_promotion_combined_E")),
        required_E=_optional_float(seq.get("baseline_promotion_required_E")),
        E_quality=_optional_float(seq.get("E_quality")),
        E_rate_noninf=_optional_float(seq.get("E_rate_noninf")),
        k=_optional_int(seq.get("k")),
        r_eff=_optional_int(seq.get("r_eff")),
        fresh_eval=_optional_bool(seq.get("baseline_promotion_fresh_eval")),
        baseline_reference_state=_optional_str(seq.get("baseline_reference_state")),
        finalized=_optional_bool(seq.get("baseline_promotion_finalized")),
        keep_revert_decision=_optional_str(row.get("keep_revert_decision")),
        failure_first_violation=_first_failure_violation(row.get("failure_analysis")),
    )


def _delta(first: float | None, latest: float | None) -> float | None:
    if first is None or latest is None:
        return None
    return round(latest - first, 6)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _terminal_keep_revert_decision(snapshot: W8Snapshot) -> bool:
    decision = str(snapshot.keep_revert_decision or "").strip()
    if decision == "revert":
        return True
    if decision != "excluded":
        return False
    if snapshot.state != "accumulating":
        return True
    return bool(snapshot.failure_first_violation)


def _config_snapshot(value: Any) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    return dict(value)


def _replay_action_blocker(snapshot: W8Snapshot) -> str | None:
    action = _config_snapshot(snapshot.config_snapshot)
    action_type = str((action or {}).get("type") or snapshot.action_type or "")
    if action is None:
        action = {"type": action_type} if action_type else {}
    if action_type == "numeric_trial":
        params = action.get("params")
        if not isinstance(params, Mapping) or not params:
            return "candidate numeric_trial lacks replayable applied params"
    elif action_type == "structural_experiment":
        flags = action.get("flags")
        if not isinstance(flags, Mapping) or not flags:
            return "candidate structural_experiment lacks replayable flags"
    else:
        return f"unreplayable_action={action_type or 'unknown'}"

    scope_err = controller_io.validate_single_variable(dict(action))
    if scope_err:
        return f"candidate action violates AP-9: {scope_err}"
    return None


def _replay_blocker(
    snapshot: W8Snapshot,
    *,
    max_replay_attempts: int,
) -> str | None:
    """Return why the report-level W8 replay path cannot use this candidate."""
    if snapshot.state != "accumulating":
        return f"state={snapshot.state or 'none'}"
    if snapshot.confirmed:
        return "already_confirmed_waiting_fresh_eval"
    if _terminal_keep_revert_decision(snapshot):
        return f"AP-24={snapshot.keep_revert_decision or 'terminal'}"
    action_blocker = _replay_action_blocker(snapshot)
    if action_blocker:
        return action_blocker
    if snapshot.combined_E is None:
        return "combined_E_unavailable"
    if snapshot.combined_E < 0.9:
        return "combined_E_below_replay_floor"
    if snapshot.E_quality is None:
        return "E_quality_unavailable"
    if snapshot.E_quality < 1.0:
        return "E_quality_below_replay_floor"
    if snapshot.k is not None and snapshot.k >= max_replay_attempts:
        return "attempt_cap_reached"
    return None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return bool(value)


def _fmt_float(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6g}"


def _compact_trials(trial_ids: list[int]) -> str:
    if not trial_ids:
        return "[]"
    if len(trial_ids) <= 5:
        return ",".join(str(trial_id) for trial_id in trial_ids)
    head = ",".join(str(trial_id) for trial_id in trial_ids[:2])
    tail = ",".join(str(trial_id) for trial_id in trial_ids[-2:])
    return f"{head},...,{tail}"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report W8 promotion replay trajectory from AutoPilot journals."
    )
    parser.add_argument("--journal", type=Path, default=DEFAULT_JOURNAL_DIR)
    parser.add_argument("--json", action="store_true", help="Emit structured JSON.")
    parser.add_argument(
        "--out-json",
        type=Path,
        help="Write structured report JSON while preserving stdout behavior.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        help="Write rendered Markdown while preserving stdout behavior.",
    )
    parser.add_argument(
        "--max-replay-attempts",
        type=int,
        default=DEFAULT_MAX_REPLAY_ATTEMPTS,
    )
    parser.add_argument("--stale-trials", type=int, default=DEFAULT_STALE_TRIALS)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero unless recent multi-observation replay is visible.",
    )
    return parser.parse_args(argv)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rows = list(iter_journal_rows(args.journal))
    report = build_w8_trajectory_report(
        rows,
        max_replay_attempts=max(1, args.max_replay_attempts),
        stale_trials=max(0, args.stale_trials),
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
    if args.strict and not report["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
