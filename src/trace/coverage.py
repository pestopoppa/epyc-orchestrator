"""TM-8 / RD-12 — review-plane trace coverage + replay-metric helpers.

Pure READ helpers over the trace store. NO inference, no writes. They answer the
two control-plane gates:

  * **TM-8 coverage gate** — % of review *invocations* (identified by their
    ``session_id``) that produced at least one trace row over a replay set
    (must be ~100% before H4 starts), plus verification that the rows carry the
    per-step ``phase`` tags, ``executor_model_id``, and PLAN_REMINDER events
    plan-compliance metrics need (intake-835).
  * **RD-12 baseline helpers** — per-decision ``latency_ms`` / token accounting
    (prompt + completion) and parse-failure counts aggregated from the emitted
    REVIEW_DECISION rows; the numbers that feed H-LB.

The 50-question replay harness (``scripts/review/review_replay_50.py``) is the
intended consumer: it runs the shadow reviewer with one distinct ``session_id``
per question, then folds this module over the resulting DB.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping

# Categories a review invocation is expected to produce (TM-2 review-plane set).
REVIEW_CATEGORIES = (
    "review_decision",
    "review_escalation",
    "plan_reminder",
    "candidate_package",
    "verification_report",
)

# Statuses/details that would indicate ENFORCEMENT (must never appear while
# ``review_decision_enforce`` is OFF — the shadow plane emits/records only).
_ENFORCEMENT_STATUSES = frozenset({"enforced", "enforcement", "blocked_by_review"})
_ENFORCEMENT_DETAIL_MARKERS = ('"enforced": true', '"enforcement": "acted"')


def _rows(db_path: Path | str) -> list[Mapping[str, Any]]:
    path = Path(db_path)
    if not path.exists():
        return []
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM event").fetchall()]
    finally:
        conn.close()


def _detail(row: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(row.get("detail_json") or "{}")
    except json.JSONDecodeError:
        return {}


def review_trace_coverage(
    db_path: Path | str,
    session_ids: Iterable[str],
    *,
    categories: tuple[str, ...] = REVIEW_CATEGORIES,
) -> dict[str, Any]:
    """% of review invocations (``session_ids``) that produced a trace row.

    Returns::

        {
          "total_invocations": N,
          "traced": K,                      # sessions with >=1 row in `categories`
          "coverage_pct": 100*K/N,          # 0.0 when N == 0 or DB missing
          "missing_session_ids": [...],     # the untraced invocations (gate losers)
          "traced_session_ids": [...],
        }

    A missing/invalid DB yields coverage 0 with every session listed as missing —
    the gate FAILS LOUD rather than reporting a false 100%.
    """
    ids = list(session_ids)
    rows = _rows(db_path)
    traced = {
        str(r.get("session_id"))
        for r in rows
        if r.get("session_id") is not None and r.get("category") in categories
    }
    traced_ids = [sid for sid in ids if sid in traced]
    missing = [sid for sid in ids if sid not in traced]
    total = len(ids)
    return {
        "total_invocations": total,
        "traced": len(traced_ids),
        "coverage_pct": round(100.0 * len(traced_ids) / total, 2) if total else 0.0,
        "missing_session_ids": missing,
        "traced_session_ids": traced_ids,
    }


def aggregate_decision_metrics(
    db_path: Path | str,
    session_ids: Iterable[str],
) -> dict[str, Any]:
    """Per-decision latency_ms + token + parse-failure aggregates (RD-12 → H-LB).

    Folds every ``review_decision`` row under ``session_ids`` (detail_json's
    ``latency_ms``, ``tokens.tokens_in/tokens_out``, ``parse_ok``,
    ``model_call_failed``, ``parse_failure``). Returns::

        {
          "n_decisions": N,
          "n_parse_failures": k,          # rows with parse_ok False, model ok
          "n_model_call_failures": m,     # rows where the model call raised
          "latency_ms": {"mean","median","p95","min","max","total"},
          "tokens_in":  {"sum","mean"},
          "tokens_out": {"sum","mean"},
          "per_decision": [{session_id, decision, parse_ok, latency_ms,
                            tokens_in, tokens_out, status}],
        }

    Rows whose detail lacks ``latency_ms`` are excluded from latency stats but
    still counted in ``n_decisions`` (a missing metric is a coverage defect, not
    evidence of zero cost).
    """
    ids = set(session_ids)
    per: list[dict[str, Any]] = []
    for row in _rows(db_path):
        if row.get("category") != "review_decision":
            continue
        sid = str(row.get("session_id") or "")
        if sid not in ids:
            continue
        detail = _detail(row)
        tokens = detail.get("tokens") or {}
        per.append(
            {
                "session_id": sid,
                "status": row.get("status"),
                "decision": detail.get("decision"),
                "parse_ok": detail.get("parse_ok", True),
                "model_call_failed": bool(detail.get("model_call_failed", False)),
                "parse_failure": detail.get("parse_failure"),
                "latency_ms": detail.get("latency_ms"),
                "tokens_in": (tokens.get("tokens_in") or 0),
                "tokens_out": (tokens.get("tokens_out") or 0),
            }
        )

    latencies = [d["latency_ms"] for d in per if isinstance(d["latency_ms"], (int, float))]
    n_parse = sum(1 for d in per if not d["parse_ok"] and not d["model_call_failed"])
    n_model = sum(1 for d in per if d["model_call_failed"])

    def _pct(sorted_vals: list[float], p: float) -> float | None:
        if not sorted_vals:
            return None
        idx = min(len(sorted_vals) - 1, int(p * len(sorted_vals)))
        return round(float(sorted_vals[idx]), 2)

    return {
        "n_decisions": len(per),
        "n_parse_failures": n_parse,
        "n_model_call_failures": n_model,
        "latency_ms": {
            "mean": round(sum(latencies) / len(latencies), 2) if latencies else None,
            "median": _pct(sorted(latencies), 0.5),
            "p95": _pct(sorted(latencies), 0.95),
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
            "total": round(sum(latencies), 2) if latencies else None,
        },
        "tokens_in": {
            "sum": sum(d["tokens_in"] for d in per),
            "mean": round(sum(d["tokens_in"] for d in per) / len(per), 2) if per else None,
        },
        "tokens_out": {
            "sum": sum(d["tokens_out"] for d in per),
            "mean": round(sum(d["tokens_out"] for d in per) / len(per), 2) if per else None,
        },
        "per_decision": per,
    }


def enforcement_side_effects(
    db_path: Path | str,
    session_ids: Iterable[str],
) -> list[dict[str, Any]]:
    """Rows under ``session_ids`` that look like enforcement — MUST be empty in shadow.

    A shadow plane emits and records but never acts. This is a tripwire scan for
    (a) statuses such as ``enforced`` / ``blocked_by_review`` and (b) details that
    declare an enforcement action. Any hit means the shadow contract broke; the
    RD-12 replay asserts ``len(...) == 0``.
    """
    ids = set(session_ids)
    hits: list[dict[str, Any]] = []
    for row in _rows(db_path):
        sid = str(row.get("session_id") or "")
        if sid not in ids:
            continue
        status = str(row.get("status") or "").lower()
        detail_json = row.get("detail_json") or ""
        if status in _ENFORCEMENT_STATUSES or any(
            m in detail_json for m in _ENFORCEMENT_DETAIL_MARKERS
        ):
            hits.append(
                {"session_id": sid, "category": row.get("category"), "status": row.get("status")}
            )
    return hits


def verify_phase_metadata(
    db_path: Path | str,
    session_ids: Iterable[str],
) -> dict[str, Any]:
    """TM-8 verification half: are phase tags + executor-model-id + reminders recorded?

    Over every review-plane row under ``session_ids``, reports::

        {
          "n_rows": N,
          "phase_tagged": {n, pct},          # detail has a non-empty "phase"
          "executor_model_id_present": {n, pct},  # detail has a non-null executor_model_id
          "reminder_events": k,              # plan_reminder rows (phase=reminder)
          "phases_seen": sorted set,
          "untagged_session_ids": [...],     # sessions with >=1 row lacking a phase tag
        }

    Plan-compliance metrics (intake-835) consume the phase + executor attribution
    and the reminder cadence events; this gate fails LOUD when they are absent.
    """
    ids = set(session_ids)
    n = phase_tagged = executor_present = 0
    reminders = 0
    phases: set[str] = set()
    untagged_sessions: set[str] = set()
    for row in _rows(db_path):
        sid = str(row.get("session_id") or "")
        if sid not in ids:
            continue
        if row.get("category") not in REVIEW_CATEGORIES:
            continue
        n += 1
        detail = _detail(row)
        phase = detail.get("phase")
        if phase:
            phase_tagged += 1
            phases.add(str(phase))
        else:
            untagged_sessions.add(sid)
        if detail.get("executor_model_id") is not None:
            executor_present += 1
        if row.get("category") == "plan_reminder":
            reminders += 1

    def _pct(k: int) -> float:
        return round(100.0 * k / n, 2) if n else 0.0

    return {
        "n_rows": n,
        "phase_tagged": {"n": phase_tagged, "pct": _pct(phase_tagged)},
        "executor_model_id_present": {"n": executor_present, "pct": _pct(executor_present)},
        "reminder_events": reminders,
        "phases_seen": sorted(phases),
        "untagged_session_ids": sorted(untagged_sessions),
    }
