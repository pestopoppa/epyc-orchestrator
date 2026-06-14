"""Pure planner-facing evidence summaries for AutoPilot.

This module formats existing journal evidence only. It does not write seq
blocks, touch safety gates, or change baseline/archive authority.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from statistics import median
from typing import Any

from src.autopilot_core.action_identity import config_fingerprint_from_row
from src.autopilot_core.sequential_verdict import (
    DEFAULT_POLICY,
    rebuild_candidate_view,
)
from src.autopilot_core.tier_specs import goodput_qph_from_row, task_rate_qph_from_row


DEFAULT_EVIDENCE_CORE_ID = "core_v1"


def format_planner_evidence_section(
    rows: Iterable[Mapping[str, Any] | Any],
    *,
    limit: int = 6,
    core_id: str = DEFAULT_EVIDENCE_CORE_ID,
) -> str:
    """Return controller prompt text for evidence power and candidate status."""
    normalized = [_normalize_row(row) for row in rows]
    trusted = [_row for _row in normalized if _is_trusted_eval_row(_row)]
    vector_rows = [_row for _row in trusted if _question_results(_row)]
    seq_rows = _seq_observation_rows(trusted, core_id=core_id)
    lines = [
        _instrument_power_line(vector_rows, seq_rows, core_id=core_id),
        "",
        "Candidate evidence blocks:",
    ]
    blocks = _candidate_evidence_blocks(
        vector_rows,
        seq_rows,
        limit=limit,
        core_id=core_id,
    )
    if not blocks:
        lines.append("- no trusted vector-bearing candidates yet")
    else:
        lines.extend(blocks)
    return "\n".join(lines)


def _normalize_row(row: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return dict(row)
    if is_dataclass(row) and not isinstance(row, type):
        return asdict(row)
    try:
        return dict(row)
    except Exception:
        return {}


def _is_trusted_eval_row(row: Mapping[str, Any]) -> bool:
    if not row.get("trial_id"):
        return False
    if row.get("bug_corrupted_by"):
        return False
    try:
        if int(row.get("tier", 1)) < 1:
            return False
    except (TypeError, ValueError):
        return False
    status = str(row.get("outcome_status") or "ok")
    return status not in {"invalid", "skipped"}


def _question_results(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    eval_details = row.get("eval_details") or {}
    if not isinstance(eval_details, Mapping):
        eval_details = {}
    details = eval_details.get("details") or {}
    if not isinstance(details, Mapping):
        details = {}
    raw = (
        eval_details.get("question_results")
        or details.get("question_results")
        or row.get("question_results")
        or []
    )
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def _seq_observation_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    core_id: str,
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for row in rows:
        seq = row.get("seq")
        if not isinstance(seq, Mapping):
            continue
        if str(seq.get("core_id") or core_id) != core_id:
            continue
        if not seq.get("candidate"):
            continue
        try:
            float(seq["z"])
        except (KeyError, TypeError, ValueError):
            continue
        observations.append(dict(row))
    return observations


def _instrument_power_line(
    vector_rows: list[dict[str, Any]],
    seq_rows: list[dict[str, Any]],
    *,
    core_id: str,
) -> str:
    if not vector_rows:
        return (
            f"Evidence power: core={core_id} policy={DEFAULT_POLICY.version} "
            f"alpha={DEFAULT_POLICY.alpha:g} confirm_E>={DEFAULT_POLICY.confirm_e:g} "
            f"budget<={DEFAULT_POLICY.budget}; no trusted per-question vectors yet."
        )
    q_counts = [len(_question_results(row)) for row in vector_rows]
    median_q = int(median(q_counts)) if q_counts else 0
    quality_quantum = (3.0 / median_q) if median_q else 0.0
    candidates = {config_fingerprint_from_row(row) for row in vector_rows}
    seq_candidates = {
        str(row["seq"]["candidate"])
        for row in seq_rows
        if isinstance(row.get("seq"), Mapping)
    }
    return (
        f"Evidence power: core={core_id} policy={DEFAULT_POLICY.version} "
        f"alpha={DEFAULT_POLICY.alpha:g} confirm_E>={DEFAULT_POLICY.confirm_e:g} "
        f"futility_E<={DEFAULT_POLICY.futility_e:g} budget<={DEFAULT_POLICY.budget}; "
        f"vector_trials={len(vector_rows)} candidates={len(candidates)} "
        f"seq_candidates={len(seq_candidates)} median_questions={median_q} "
        f"quality_quantum~{quality_quantum:.3f}. "
        "Below-quantum deltas need paired/reproduced evidence before acting."
    )


def _candidate_evidence_blocks(
    vector_rows: list[dict[str, Any]],
    seq_rows: list[dict[str, Any]],
    *,
    limit: int,
    core_id: str,
) -> list[str]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in vector_rows:
        grouped[config_fingerprint_from_row(row)].append(row)

    seq_by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in seq_rows:
        seq = row.get("seq")
        if isinstance(seq, Mapping):
            seq_by_candidate[str(seq["candidate"])].append(row)

    all_candidates = sorted(
        set(grouped) | set(seq_by_candidate),
        key=lambda fp: (_latest_trial_id(grouped.get(fp, []) + seq_by_candidate.get(fp, [])), fp),
        reverse=True,
    )
    blocks: list[str] = []
    for fingerprint in all_candidates[: max(0, limit)]:
        rows = grouped.get(fingerprint, [])
        seq_observations = seq_by_candidate.get(fingerprint, [])
        latest = max(rows + seq_observations, key=_latest_trial_id)
        trial_ids = sorted(
            {
                int(row["trial_id"])
                for row in rows + seq_observations
                if _trial_id(row) is not None
            }
        )
        vector_note = _vector_note(rows)
        seq_note = _seq_note(fingerprint, seq_observations, core_id=core_id)
        task_rate = task_rate_qph_from_row(latest)
        goodput = goodput_qph_from_row(latest)
        blocks.append(
            "- fp={fp} trials={trials} q={q:.3f} r={r:.2f} "
            "task_rate={task_rate:.1f} goodput={goodput:.1f}; {vector}; {seq}".format(
                fp=fingerprint,
                trials=_compact_trials(trial_ids),
                q=_float(latest.get("quality")),
                r=_float(latest.get("reliability")),
                task_rate=task_rate,
                goodput=goodput,
                vector=vector_note,
                seq=seq_note,
            )
        )
    return blocks


def _vector_note(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "no vector trials"
    q_counts = [len(_question_results(row)) for row in rows]
    return f"vector-ready n={len(rows)} median_questions={int(median(q_counts))}"


def _seq_note(
    fingerprint: str,
    seq_observations: list[dict[str, Any]],
    *,
    core_id: str,
) -> str:
    if not seq_observations:
        return "seq=not_logged_yet"
    view = rebuild_candidate_view(
        candidate=fingerprint,
        core_id=core_id,
        observations=seq_observations,
    )
    return (
        f"seq={view.state} k={view.quality_state.k} "
        f"E_quality={view.quality_state.wealth:.3f}"
    )


def _latest_trial_id(row_or_rows: Mapping[str, Any] | list[dict[str, Any]]) -> int:
    if isinstance(row_or_rows, list):
        if not row_or_rows:
            return -1
        return max(_latest_trial_id(row) for row in row_or_rows)
    return _trial_id(row_or_rows) or -1


def _trial_id(row: Mapping[str, Any]) -> int | None:
    try:
        return int(row.get("trial_id"))
    except (TypeError, ValueError):
        return None


def _compact_trials(trial_ids: list[int]) -> str:
    if not trial_ids:
        return "[]"
    if len(trial_ids) <= 5:
        return "[" + ",".join(str(trial_id) for trial_id in trial_ids) + "]"
    head = ",".join(str(trial_id) for trial_id in trial_ids[:2])
    tail = ",".join(str(trial_id) for trial_id in trial_ids[-2:])
    return f"[{head},...,{tail}]"


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
