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
from collections import defaultdict
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


DEFAULT_MAX_REPLAY_ATTEMPTS = 12
DEFAULT_STALE_TRIALS = 12


@dataclass(frozen=True)
class W8Snapshot:
    trial_id: int
    candidate: str
    action_type: str | None
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


@dataclass(frozen=True)
class CandidateTrajectory:
    candidate: str
    status: str
    trials: list[int]
    attempts: int
    first_trial_id: int
    latest_trial_id: int
    latest_state: str | None
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
    recent: bool
    stale_accumulating: bool


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

    recent_active = [
        item for item in trajectories if item.status == "active_recent_replay"
    ]
    stale_accumulating = [
        item for item in trajectories if item.status == "stale_accumulating"
    ]
    blocked = _open_requirements(
        trajectories,
        recent_active=recent_active,
        stale_accumulating=stale_accumulating,
    )
    if not snapshots:
        status = "no_w8_snapshots"
    elif recent_active:
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
        "open_requirements": blocked,
        "recent_active_candidates": [item.candidate for item in recent_active],
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
    ]
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
        "| Candidate | Status | Trials | latest E / required | latest state | k | fresh evals |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for item in trajectories[:25]:
        lines.append(
            "| {candidate} | {status} | {trials} | {combined} / {required} | "
            "{state} | {k} | {fresh} |".format(
                candidate=item.get("candidate"),
                status=item.get("status"),
                trials=_compact_trials(list(item.get("trials") or [])),
                combined=_fmt_float(item.get("latest_combined_E")),
                required=_fmt_float(item.get("required_E")),
                state=item.get("latest_state") or "none",
                k=item.get("latest_k"),
                fresh=item.get("fresh_eval_count"),
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
    stale_accumulating = (
        latest.state == "accumulating"
        and not latest.confirmed
        and latest.finalized is not True
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
        recent=recent,
        stale_accumulating=stale_accumulating,
    )


def _candidate_status(
    latest: W8Snapshot,
    *,
    attempts: int,
    recent: bool,
    stale_accumulating: bool,
    max_replay_attempts: int,
) -> str:
    if latest.finalized:
        return "finalized"
    if latest.state == "refuted":
        return "refuted"
    if latest.confirmed:
        return "confirmed_waiting_fresh_eval"
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
    stale_accumulating: list[CandidateTrajectory],
) -> list[str]:
    if not trajectories:
        return ["missing_w8_promotion_snapshots"]
    requirements = ["combined_E_below_required", "fresh_promotion_eval_required"]
    if not recent_active:
        requirements.append("no_recent_multi_observation_accumulating_candidate")
    if stale_accumulating:
        requirements.append("stale_accumulating_candidates_present")
    if not any(item.latest_confirmed for item in trajectories):
        requirements.append("seq_confirmation_required")
    return requirements


def _snapshot_from_row(row: Mapping[str, Any]) -> W8Snapshot:
    seq = row.get("seq") if isinstance(row, Mapping) else None
    if not isinstance(seq, Mapping):
        return W8Snapshot(
            trial_id=-1,
            candidate="",
            action_type=None,
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
        )
    return W8Snapshot(
        trial_id=_int(row.get("trial_id"), default=-1),
        candidate=str(seq.get("candidate") or ""),
        action_type=_optional_str(row.get("action_type")),
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
