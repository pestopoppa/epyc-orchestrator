#!/usr/bin/env python3
"""TM-8 reviewer-trace coverage counter (pure sqlite analysis; NO inference).

What it computes
----------------
Reviewer-trace *coverage* answers a single governance question: **are review
invocations actually being traced?** ``src/proactive_delegation/review_service.py``
documents the invariant "every invocation emits a REVIEW_DECISION trace event"
(TM-3, review_service.py:412 / :517). TM-8 is the *counter* that verifies that
invariant holds against the durable store:

    coverage = covered_invocations / review_invocations

where the numerator/denominator are drawn from the ``event`` table of
``data/trace/events.sqlite`` (schema: ``src/trace/store.py``; read path:
``src/trace/query.py::query``). A ``REVIEW_DECISION`` event (category
``review_decision``, emitted via ``src/trace/emit.py``) is the *trace row*; a
``review_invocation`` marker event is the independent ground truth for "the
reviewer was called".

Ground-truth modes (why coverage can be < 100%)
-----------------------------------------------
* **markers** — if the store contains ``review_invocation`` marker events, the
  denominator is the set of distinct invocation ids from those markers, and the
  numerator is how many of those ids appear on >=1 ``review_decision`` event
  (joined on ``detail_json.invocation_id``). An invocation with no decision
  drops coverage below 1.0; a decision whose id matches no marker is an *orphan*
  (reported separately, never inflates coverage past 1.0).
* **self_attested** — if the store has NO ``review_invocation`` markers (the
  current historical reality: the 32 live ``review_decision`` rows carry no
  paired marker), each ``review_decision`` attests its own invocation, so
  coverage is 1.0 over N decisions. The report FLAGS this mode explicitly: with
  no independent marker you cannot *see* an un-traced invocation, so 1.0 here
  means "every traced decision is counted", not "nothing was missed".

Per-event-type presence breakdown
---------------------------------
Alongside the fraction, a presence breakdown over the ``review_decision``
population reports how many rows carry each trace-completeness signal:
  * ``phase_tag``          — a review-phase tag (``detail_json.mode`` / ``.phase``).
  * ``executor_model_id``  — an executor MODEL/QUANT id (``detail_json``
    ``executor_model_id`` / ``executor_model_quant`` / ``model_quant``). NOTE:
    ``assigned_role`` is deliberately NOT counted here — a role is not a model
    id, and TM-8 results are model/quant-indexed, never role-indexed.
  * ``reminder_events``    — count of ``plan_reminder`` events in the store.

Execution model (harness contract)
----------------------------------
Default is a pure **dry-run**: validate the db is reachable, resolve+count the
relevant categories, print the plan, exit 0 — NO model, no full compute. The
full coverage compute runs ONLY when BOTH ``--execute`` is passed AND
``TM8_COVERAGE_EXECUTE=1`` is set. This counter issues ZERO inference (it is
pure sqlite), but it declares the placement-queue transport contract
(``workload_class=eval_batch``, ``uses_chat_endpoint=False``) so any future
model-assisted extension routes through the eval-batch placement queue and NEVER
a foreground ``/chat`` call. It never mutates events.sqlite or any serving-path
module.

Two-step pin (dry-run first to eyeball the plan; add --execute + env to run):
  1. .venv/bin/python scripts/analysis/tm8_coverage.py
  2. TM8_COVERAGE_EXECUTE=1 .venv/bin/python scripts/analysis/tm8_coverage.py --execute

The parse / plan / coverage-compute core is PURE and fixture-tested without
inference (``tests/test_tm8_coverage.py``).
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

from src.trace.query import query  # canonical read path (src/trace/query.py)
from src.trace.store import DEFAULT_DB_PATH, EventCategory

# ── Category + detail-key config (single source of truth) ────────────────────
CAT_REVIEW_DECISION = EventCategory.REVIEW_DECISION        # "review_decision"
CAT_REVIEW_INVOCATION = "review_invocation"               # independent invocation marker
CAT_PLAN_REMINDER = EventCategory.PLAN_REMINDER            # "plan_reminder"

# A review-phase tag: plan_review vs review (review_service uses detail.mode),
# or an explicit detail.phase.
PHASE_TAG_KEYS: tuple[str, ...] = ("mode", "phase")
# An executor MODEL/QUANT id. Deliberately excludes ``assigned_role`` (role !=
# model): TM-8 results are model/quant-indexed, never role-indexed.
EXECUTOR_MODEL_KEYS: tuple[str, ...] = (
    "executor_model_id",
    "executor_model_quant",
    "model_quant",
)
INVOCATION_ID_KEY = "invocation_id"
UNATTRIBUTED = "unattributed"

# ── Execution gate + placement-queue transport contract ──────────────────────
# Both required to leave dry-run. Mirrors the reviewer_corpus_ledger_run gate.
TM8_EXECUTE_ENV = "TM8_COVERAGE_EXECUTE"
# Mirror of scripts/autopilot/screening_tier_runner.PLACEMENT_* — TM-8 issues no
# inference itself, but declares the contract so any model-assisted extension
# routes via the eval_batch placement queue, NEVER a foreground /chat call.
PLACEMENT_QUEUE_TRANSPORT = "placement_queue"
PLACEMENT_REQUEST_PRIORITY = "background"
PLACEMENT_WORKLOAD_CLASS = "eval_batch"
INFERENCE_REQUIRED = False

_FETCH_LIMIT = 10_000_000  # effectively "all rows" for query()'s LIMIT clause


def _env_flag_enabled(name: str) -> bool:
    """True iff env var ``name`` is a truthy flag (matches screening_tier_runner)."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _transport_block() -> dict[str, Any]:
    return {
        "transport": PLACEMENT_QUEUE_TRANSPORT,
        "request_priority": PLACEMENT_REQUEST_PRIORITY,
        "workload_class": PLACEMENT_WORKLOAD_CLASS,
        "uses_chat_endpoint": False,
        "inference_required": INFERENCE_REQUIRED,
    }


# ── PURE helpers ─────────────────────────────────────────────────────────────
def _detail(row: dict[str, Any]) -> dict[str, Any]:
    """Safely decode a row's ``detail_json`` to a dict ({} on any failure)."""
    raw = row.get("detail_json")
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return obj if isinstance(obj, dict) else {}


def _first_present(detail: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    """First non-null value among ``keys`` in ``detail`` (else None)."""
    for k in keys:
        v = detail.get(k)
        if v is not None and v != "":
            return v
    return None


def _invocation_id_of(row: dict[str, Any]) -> str | None:
    """The invocation id a marker/decision row links on (detail.invocation_id)."""
    v = _detail(row).get(INVOCATION_ID_KEY)
    return str(v) if v is not None and v != "" else None


def _executor_model_quant_of(row: dict[str, Any]) -> str:
    """Executor model/quant key for a decision row (never a role; else UNATTRIBUTED)."""
    v = _first_present(_detail(row), EXECUTOR_MODEL_KEYS)
    return str(v) if v is not None else UNATTRIBUTED


# ── PURE core: the coverage computation (fixture-tested, no I/O) ──────────────
def compute_coverage(
    decisions: list[dict[str, Any]],
    invocations: list[dict[str, Any]],
    reminders: list[dict[str, Any]],
    *,
    phase_tag_keys: tuple[str, ...] = PHASE_TAG_KEYS,
    executor_model_keys: tuple[str, ...] = EXECUTOR_MODEL_KEYS,
) -> dict[str, Any]:
    """Compute TM-8 coverage + presence breakdown from already-fetched row dicts.

    ``decisions``    — rows with category ``review_decision`` (the trace rows).
    ``invocations``  — rows with category ``review_invocation`` (marker ground truth).
    ``reminders``    — rows with category ``plan_reminder``.

    Returns the coverage dict (see module docstring for field semantics). Pure:
    no db, no filesystem, no model.
    """
    traced_decisions = len(decisions)

    # Ground-truth denominator: markers if present, else self-attested decisions.
    marker_ids = {mid for r in invocations if (mid := _invocation_id_of(r)) is not None}
    # A marker with no invocation_id still counts as one invocation (keyed by row id).
    marker_fallback = sum(1 for r in invocations if _invocation_id_of(r) is None)

    decision_ids = {did for r in decisions if (did := _invocation_id_of(r)) is not None}

    if invocations:
        ground_truth = "markers"
        review_invocations = len(marker_ids) + marker_fallback
        covered_invocations = len(marker_ids & decision_ids)
        # Orphan decisions: traced decisions whose invocation_id matches no marker
        # (or that carry no invocation_id at all).
        orphan_decisions = sum(
            1
            for r in decisions
            if (_invocation_id_of(r) is None) or (_invocation_id_of(r) not in marker_ids)
        )
    else:
        ground_truth = "self_attested"
        review_invocations = traced_decisions
        covered_invocations = traced_decisions
        orphan_decisions = 0

    coverage = (
        covered_invocations / review_invocations if review_invocations > 0 else None
    )

    # Presence breakdown over the decision population.
    phase_present = sum(
        1 for r in decisions if _first_present(_detail(r), phase_tag_keys) is not None
    )
    exec_present = sum(
        1 for r in decisions if _first_present(_detail(r), executor_model_keys) is not None
    )

    def _frac(n: int) -> float | None:
        return (n / traced_decisions) if traced_decisions > 0 else None

    # Result emission is MODEL/QUANT-indexed (never role-indexed).
    by_executor: dict[str, int] = {}
    for r in decisions:
        key = _executor_model_quant_of(r)
        by_executor[key] = by_executor.get(key, 0) + 1

    return {
        "kind": "tm8_trace_coverage",
        "ground_truth": ground_truth,
        "review_invocations": review_invocations,
        "traced_decisions": traced_decisions,
        "covered_invocations": covered_invocations,
        "orphan_decisions": orphan_decisions,
        "coverage": coverage,
        "coverage_pct": round(coverage * 100.0, 2) if coverage is not None else None,
        "presence": {
            "phase_tag": {
                "present": phase_present,
                "total": traced_decisions,
                "fraction": _frac(phase_present),
            },
            "executor_model_id": {
                "present": exec_present,
                "total": traced_decisions,
                "fraction": _frac(exec_present),
            },
            "reminder_events": {"count": len(reminders)},
        },
        "by_executor_model_quant": dict(sorted(by_executor.items())),
    }


# ── Read path (reuses src/trace/query.py::query) ─────────────────────────────
def fetch_review_events(db_path: Path | str) -> dict[str, list[dict[str, Any]]]:
    """Fetch the three review-plane categories via the canonical query() read path."""
    return {
        "decisions": query(db_path=db_path, category=CAT_REVIEW_DECISION, limit=_FETCH_LIMIT),
        "invocations": query(db_path=db_path, category=CAT_REVIEW_INVOCATION, limit=_FETCH_LIMIT),
        "reminders": query(db_path=db_path, category=CAT_PLAN_REMINDER, limit=_FETCH_LIMIT),
    }


def _category_counts(db_path: Path) -> dict[str, int] | None:
    """Cheap read-only COUNT(*) per relevant category. None if table absent."""
    conn = sqlite3.connect(str(db_path))
    try:
        try:
            rows = conn.execute(
                "SELECT category, COUNT(*) c FROM event "
                "WHERE category IN (?, ?, ?) GROUP BY category",
                (CAT_REVIEW_DECISION, CAT_REVIEW_INVOCATION, CAT_PLAN_REMINDER),
            ).fetchall()
        except sqlite3.OperationalError:
            return None  # no `event` table
    finally:
        conn.close()
    got = {cat: c for cat, c in rows}
    return {
        CAT_REVIEW_DECISION: got.get(CAT_REVIEW_DECISION, 0),
        CAT_REVIEW_INVOCATION: got.get(CAT_REVIEW_INVOCATION, 0),
        CAT_PLAN_REMINDER: got.get(CAT_PLAN_REMINDER, 0),
    }


# ── Dry-run plan resolution (validate + resolve + count; NO full compute) ─────
def resolve_plan(db_path: Path, *, execute_requested: bool, env_ok: bool) -> dict[str, Any]:
    """Validate the db, resolve+count the relevant categories, describe the run."""
    counts = _category_counts(db_path) if db_path.exists() else None
    valid = counts is not None
    predicted_ground_truth = None
    if valid:
        predicted_ground_truth = (
            "markers" if counts[CAT_REVIEW_INVOCATION] > 0 else "self_attested"
        )
    plan: dict[str, Any] = {
        "kind": "tm8_trace_coverage_plan",
        "mode": "dry_run",
        "inference_ran": False,
        "db_path": str(db_path),
        "db_exists": db_path.exists(),
        "valid": valid,
        "resolved_counts": counts,
        "predicted_ground_truth": predicted_ground_truth,
        "categories": {
            "decision": CAT_REVIEW_DECISION,
            "invocation_marker": CAT_REVIEW_INVOCATION,
            "reminder": CAT_PLAN_REMINDER,
        },
        "presence_probes": {
            "phase_tag_keys": list(PHASE_TAG_KEYS),
            "executor_model_keys": list(EXECUTOR_MODEL_KEYS),
            "reminder_category": CAT_PLAN_REMINDER,
        },
        "result_indexing": "model_quant",  # never role-indexed
        "transport": _transport_block(),
        "will_compute": (
            "coverage = covered_invocations / review_invocations, plus per-event-type "
            "presence breakdown (phase_tag / executor_model_id / reminder_events)"
        ),
        "execute_gate": {
            "execute_flag": execute_requested,
            "env_var": TM8_EXECUTE_ENV,
            "env_ok": env_ok,
            "would_execute": bool(execute_requested and env_ok),
        },
    }
    if execute_requested and not env_ok:
        plan["note"] = (
            f"--execute passed but {TM8_EXECUTE_ENV} not set; staying in dry-run "
            "(both gates required to compute)."
        )
    return plan


# ── Execute: full compute (env + --execute gated) ────────────────────────────
def run_coverage(db_path: Path, *, output: Path | None = None) -> dict[str, Any]:
    """Fetch review-plane events and compute the TM-8 coverage report."""
    ev = fetch_review_events(db_path)
    result = compute_coverage(ev["decisions"], ev["invocations"], ev["reminders"])
    result.update({
        "mode": "execute",
        "inference_ran": False,
        "db_path": str(db_path),
        "transport": _transport_block(),
    })
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
        result["output"] = str(output)
    return result


# ── CLI ──────────────────────────────────────────────────────────────────────
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "TM-8 reviewer-trace coverage counter over data/trace/events.sqlite. "
            "Default is a pure dry-run (validate + resolve + count + plan, NO model). "
            f"Full compute requires --execute AND {TM8_EXECUTE_ENV}=1."
        )
    )
    p.add_argument(
        "--db", default=str(DEFAULT_DB_PATH),
        help="path to the trace events.sqlite (default: data/trace/events.sqlite)",
    )
    p.add_argument(
        "--output", default=None,
        help="optional path to write the model/quant-indexed JSON result (execute mode)",
    )
    p.add_argument(
        "--execute", action="store_true",
        help=f"run the full coverage compute (ALSO gated by {TM8_EXECUTE_ENV}=1); "
        "default is a pure dry-run plan with no compute",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    db_path = Path(args.db)
    env_ok = _env_flag_enabled(TM8_EXECUTE_ENV)
    execute = bool(args.execute) and env_ok

    if not execute:
        plan = resolve_plan(db_path, execute_requested=bool(args.execute), env_ok=env_ok)
        print(json.dumps(plan, indent=2, sort_keys=True, default=str))
        # Missing/invalid db in dry-run is a validation FAILURE reported in the plan.
        return 0 if plan["valid"] else 2

    if not db_path.exists():
        print(json.dumps({"error": f"db not found: {db_path}"}, indent=2))
        return 2

    result = run_coverage(db_path, output=Path(args.output) if args.output else None)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
