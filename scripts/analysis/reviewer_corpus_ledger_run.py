#!/usr/bin/env python3
"""Mechanism B — reviewer-corpus → review-ledger bridge (RCP-W3 / RC-8 / RM-6).

This is the ~80-LOC glue that gives ``reviewer_calibration_report.py`` a LIVE
producer. Today that report is real and correct but its INPUT (a ``review_ledger``
SQLite table OR a decisions-JSONL of ledger-shaped rows) has no live source:
``screening_tier_runner.py`` runs the reviewer over the near-miss corpus but emits
its own per-*pairing* FA/FR/CR summaries, NOT per-*decision* ledger rows, and never
connects to the report. This driver closes that gap: run one reviewer role over a
fixed corpus slice and emit one ledger-shaped row PER DECISION, in exactly the shape
``reviewer_calibration_report.py`` consumes (both ``--decisions`` JSONL mode and
``--ledger`` SQLite mode).

Reuse, NOT reimplementation (all seams come from
``scripts/autopilot/screening_tier_runner.py`` — nothing here is a fresh copy):
  * ``iter_judgeable_rows`` — lazy, domain-filtered judgeable-row iterator.
  * ``select_rows_for_job`` — deterministic (seed_key) row sampling.
  * ``TrialJobSpec`` + ``_default_reviewer_probe`` — the placement-queue reviewer
    probe seam (``request_priority=background`` + ``workload_class=eval_batch`` +
    ``force_role=<reviewer>``; NEVER a foreground ``/chat`` call — RM-3 discipline).
  * ``_default_tower`` — deferred EvalTower construction for the probe.
The row schema is imported from ``src.trace.review_ledger`` (``ReviewLedgerRow`` +
``insert_review_ledger_row``) — the single source of truth for the ledger columns.

Execution is env-gated EXACTLY like screening_tier_runner: default is a pure
dry-run (validate config, resolve+count corpus rows, print the plan, exit 0 — NO
model). Inference happens ONLY when ``--execute`` is passed OR
``AUTOPILOT_SCREENING_TIER_INFERENCE=1`` is set. The row-mapping / emit / plan
logic is PURE and unit-tested without inference
(``tests/test_reviewer_corpus_ledger_run.py``).

KNOWN CAVEAT (populated vs null fields — pre-P-REV-1):
  The reviewer probe currently returns ONLY ``{decision, gate, latency_ms}`` — it
  captures NO per-decision confidence / logprob / token count. So downstream the
  report computes FA rate, FR rate, FA/FR ratio, acceptance rate, Consistency Rate
  and parse-failure rate cleanly, but **ECE / AUC / Brier are null** (they need a
  confidence signal). Every number produced through this bridge is
  OBSERVATION-grade (MEASUREMENT.md; MEASUREMENT protocol P-REV-1 is still a draft,
  RC-6a) — it MUST NOT gate any keep/revert/deploy/promote of a reviewer config.
  Populated: decision_id, reviewer_model_quant, candidate_id, domain, corpus_id,
  decision, gold_label, gold_source, gold_instrument_version, latency_ms.
  Null (until a confidence signal is captured): confidence, tokens.

Two-step RCP-W3 pin (dry-run first to eyeball the plan; drop --execute to plan):
  1. .venv/bin/python scripts/analysis/reviewer_corpus_ledger_run.py \
       --corpus /mnt/raid0/llm/datasets/nearmiss-corpus-v1/rows.jsonl \
       --reviewer architect_general --n 200 --domain code \
       --output runs/rcp_w3 --emit decisions-jsonl --execute
  2. .venv/bin/python scripts/analysis/reviewer_calibration_report.py \
       --decisions runs/rcp_w3/decisions.jsonl \
       --corpus /mnt/raid0/llm/datasets/nearmiss-corpus-v1/rows.jsonl --k 2 --print
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable

SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

DEFAULT_CORPUS = "/mnt/raid0/llm/datasets/nearmiss-corpus-v1/rows.jsonl"
DEFAULT_REVIEWER = "architect_general"

_RUNNER = None  # cached screening_tier_runner module (loaded by path)


def load_runner():
    """Load ``screening_tier_runner`` by path (robust; no scripts.* package needed)."""
    global _RUNNER
    if _RUNNER is None:
        mp = ORCH_ROOT / "scripts" / "autopilot" / "screening_tier_runner.py"
        spec = importlib.util.spec_from_file_location("screening_tier_runner", mp)
        mod = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("screening_tier_runner", mod)
        spec.loader.exec_module(mod)
        _RUNNER = mod
    return _RUNNER


# ── PURE: per-decision row mapping (the fixture-tested core) ──────────────────
def decision_id_for(reviewer: str, corpus_id: str | None, row_id: str, attempt: int = 0) -> str:
    """Deterministic, unique-per-(reviewer,corpus,row,attempt) decision id.

    Stable across re-runs (so a re-emit is an INSERT OR IGNORE no-op in the
    ledger) yet distinct per test-retest ``attempt`` (so repeated scoring of one
    candidate yields the >=2 terminal runs the report's Consistency Rate needs).
    """
    key = f"{reviewer}\x00{corpus_id or ''}\x00{row_id}\x00{attempt}"
    return "rev-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:24]


def map_decision_to_ledger_row(
    reviewer: str,
    corpus_row: dict[str, Any],
    probe_result: dict[str, Any],
    *,
    attempt: int = 0,
    corpus_id: str | None = None,
    rubric_version: str | None = None,
    grading_model: str | None = None,
    era: str | None = None,
) -> dict[str, Any]:
    """Map one (corpus row, reviewer decision) → a ledger-shaped dict.

    The returned dict uses ONLY ``ReviewLedgerRow`` field names, so it round-trips
    through both consumers: dumped verbatim to decisions.jsonl (``--decisions``
    report mode) OR splatted into ``ReviewLedgerRow(**row)`` for the SQLite ledger
    (``--ledger`` report mode). ``confidence`` / ``tokens`` are null — the probe
    captures no confidence signal yet (see module caveat).
    """
    row_id = str(corpus_row.get("row_id") or corpus_row.get("candidate_id") or "")
    cid = corpus_id if corpus_id is not None else corpus_row.get("corpus_id")
    return {
        "decision_id": decision_id_for(reviewer, cid, row_id, attempt),
        "reviewer_model_quant": reviewer,
        "grading_model": grading_model,
        "rubric_version": rubric_version,
        "corpus_id": cid,
        "candidate_id": row_id,
        "domain": corpus_row.get("domain"),
        "decision": probe_result.get("decision"),
        "confidence": None,  # probe returns no confidence -> ECE/AUC/Brier null
        "gold_label": corpus_row.get("gold_label"),
        "gold_source": corpus_row.get("gold_source"),
        "gold_instrument_version": corpus_row.get("gold_instrument_version"),
        "latency_ms": probe_result.get("latency_ms"),
        "tokens": None,  # probe returns no token count yet
        "era": era,
    }


# ── PURE: emit sinks ─────────────────────────────────────────────────────────
def emit_decisions_jsonl(rows: list[dict[str, Any]], out_path: Path) -> int:
    """Write ledger-shaped rows to a decisions JSONL (report ``--decisions`` mode)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True, default=str) + "\n")
    return len(rows)


def emit_ledger_sqlite(rows: list[dict[str, Any]], db_path: Path) -> tuple[int, int]:
    """Insert ledger-shaped rows into a ``review_ledger`` SQLite table (report
    ``--ledger`` mode). Returns ``(inserted, skipped_dup)``."""
    import sqlite3

    from src.trace.review_ledger import ReviewLedgerRow, insert_review_ledger_row

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        inserted = skipped = 0
        for r in rows:
            i, s = insert_review_ledger_row(conn, ReviewLedgerRow(**r))
            inserted += i
            skipped += s
        return inserted, skipped
    finally:
        conn.close()


# ── PURE: plan resolution (dry-run; no inference) ────────────────────────────
def resolve_plan(
    corpus_path: Path,
    *,
    reviewer: str,
    n: int,
    seed: int,
    domain: str | None,
    emit: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Count judgeable corpus rows for the slice + describe the run (no model)."""
    runner = load_runner()
    n_available = sum(1 for _ in runner.iter_judgeable_rows(corpus_path, domain=domain))
    out_name = "decisions.jsonl" if emit == "decisions-jsonl" else "events.sqlite"
    return {
        "kind": "reviewer_corpus_ledger_plan",
        "mode": "dry_run",
        "inference_ran": False,
        "reviewer_model_quant": reviewer,
        "corpus_path": str(corpus_path),
        "domain_filter": domain or "all",
        "seed": seed,
        "n_requested": n,
        "n_judgeable_available": n_available,
        "n_selected": min(n, n_available),
        "emit": emit,
        "output": str(output_dir / out_name),
        "transport": {
            "transport": runner.PLACEMENT_QUEUE_TRANSPORT,
            "request_priority": runner.PLACEMENT_REQUEST_PRIORITY,
            "workload_class": runner.PLACEMENT_WORKLOAD_CLASS,
            "force_role": reviewer,
            "uses_chat_endpoint": False,
        },
        "populated_fields": [
            "decision_id", "reviewer_model_quant", "candidate_id", "domain",
            "corpus_id", "decision", "gold_label", "gold_source",
            "gold_instrument_version", "latency_ms",
        ],
        "null_fields": ["confidence", "tokens"],
        "observation_only": True,
        "measurement_note": (
            "pre-P-REV-1 observation; FA/FR/ratio/CR/parse compute, ECE/AUC/Brier "
            "null until a confidence signal is captured. Non-decision-gating."
        ),
    }


# ── Execution bridge (env/--execute gated; reuses the probe seam) ────────────
def run_corpus_ledger(
    corpus_path: Path,
    *,
    reviewer: str,
    n: int,
    seed: int,
    domain: str | None,
    emit: str,
    output_dir: Path,
    reviewer_probe: Callable[[Any, dict[str, Any], Any], dict[str, Any]] | None = None,
    tower: Any | None = None,
) -> dict[str, Any]:
    """Score ``n`` corpus rows with the reviewer over the placement queue, map each
    decision to a ledger row, and emit. Tests inject ``reviewer_probe`` (no model)."""
    runner = load_runner()
    probe = reviewer_probe or runner._default_reviewer_probe
    rows = list(runner.iter_judgeable_rows(corpus_path, domain=domain))
    corpus_id = rows[0].get("corpus_id") if rows else None
    sample = runner.select_rows_for_job(rows, n=n, seed_key=f"{seed}:{reviewer}")

    job = runner.TrialJobSpec(
        pairing_id=f"corpus-ledger::{reviewer}",
        architect=None, reviewer=reviewer, grader=None, anchor_arm=None,
        self_review=False, cross_family=False, staged_involved=False,
        n=n, eval_tier="T0", corpus_id=str(corpus_id) if corpus_id else "unknown",
        domain=domain or "all", corpus_content_sha256="",
        corpus_n_rows=len(rows), coresidency_fits=None, priority_rank=0,
    )
    # Only build a real EvalTower when we actually use the default (inference) probe.
    if probe is runner._default_reviewer_probe and tower is None:
        tower = runner._default_tower()

    ledger_rows = [
        map_decision_to_ledger_row(reviewer, row, probe(job, row, tower), corpus_id=job.corpus_id)
        for row in sample
    ]

    out_name = "decisions.jsonl" if emit == "decisions-jsonl" else "events.sqlite"
    out_path = output_dir / out_name
    if emit == "decisions-jsonl":
        emit_decisions_jsonl(ledger_rows, out_path)
        emit_summary: Any = {"decisions_written": len(ledger_rows)}
    else:
        inserted, skipped = emit_ledger_sqlite(ledger_rows, out_path)
        emit_summary = {"inserted": inserted, "skipped_dup": skipped}

    # Sidecar run manifest (observation stamp + provenance; keeps the ledger clean).
    manifest = {
        "kind": "reviewer_corpus_ledger_run",
        "reviewer_model_quant": reviewer, "corpus_path": str(corpus_path),
        "corpus_id": job.corpus_id, "domain_filter": domain or "all",
        "seed": seed, "n_requested": n, "n_scored": len(ledger_rows),
        "emit": emit, "output": str(out_path), "observation_only": True,
        "transport": job.transport, "uses_chat_endpoint": False,
        "null_fields": ["confidence", "tokens"],
    }
    (output_dir).mkdir(parents=True, exist_ok=True)
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    return {
        "mode": "execute", "inference_ran": True, "n_scored": len(ledger_rows),
        "output": str(out_path), "emit_summary": emit_summary, "rows": ledger_rows,
        "manifest": manifest,
    }


# ── CLI ──────────────────────────────────────────────────────────────────────
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Mechanism B: run one reviewer role over a corpus slice and emit "
            "per-decision review-ledger rows for reviewer_calibration_report.py. "
            "Default is a pure dry-run (plan + count, NO inference)."
        )
    )
    p.add_argument("--corpus", default=DEFAULT_CORPUS, help="near-miss corpus rows.jsonl")
    p.add_argument("--reviewer", default=DEFAULT_REVIEWER, help="reviewer role (force_role)")
    p.add_argument("--n", type=int, default=200, help="number of decisions to produce")
    p.add_argument("--seed", type=int, default=42, help="deterministic sample seed")
    p.add_argument(
        "--slice", "--domain", dest="domain", default=None,
        help="restrict to one corpus domain slice (code/general/thinking/hotpotqa/simpleqa)",
    )
    p.add_argument("--output", default="runs/reviewer_corpus_ledger", help="output DIRECTORY")
    p.add_argument(
        "--emit", choices=["decisions-jsonl", "ledger-sqlite"], default="decisions-jsonl",
        help="decisions.jsonl (default; no sqlite handle) or events.sqlite review_ledger",
    )
    p.add_argument(
        "--execute", action="store_true",
        help="run inference (ALSO gated by AUTOPILOT_SCREENING_TIER_INFERENCE=1); "
        "default is a pure dry-run plan with no model",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    runner = load_runner()
    corpus_path = Path(args.corpus)
    output_dir = Path(args.output)

    if not corpus_path.exists():
        print(json.dumps({"error": f"corpus not found: {corpus_path}"}, indent=2))
        return 2

    execute = bool(args.execute) or runner._env_flag_enabled(
        runner.SCREENING_TIER_INFERENCE_ENV
    )

    if not execute:
        plan = resolve_plan(
            corpus_path, reviewer=args.reviewer, n=args.n, seed=args.seed,
            domain=args.domain, emit=args.emit, output_dir=output_dir,
        )
        print(json.dumps(plan, indent=2, sort_keys=True, default=str))
        return 0

    result = run_corpus_ledger(
        corpus_path, reviewer=args.reviewer, n=args.n, seed=args.seed,
        domain=args.domain, emit=args.emit, output_dir=output_dir,
    )
    # Drop the (potentially large) per-row payload from the printed summary.
    printed = {k: v for k, v in result.items() if k != "rows"}
    print(json.dumps(printed, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
