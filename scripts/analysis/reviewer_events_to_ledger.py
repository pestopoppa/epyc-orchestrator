#!/usr/bin/env python3
"""Materialize `review_ledger`-shaped rows from emitted REVIEW_DECISION trace events.

MECHANISM A (RCP-W2 / RD-12 workaround) — offline, NON-STACK, no inference.
==========================================================================
The reviewer shadow plane fires IN-PROCESS during live task execution
(``src/proactive_delegation/review_service.py`` ~L344-374 / L479-499) and writes a
trace EVENT into the ``event`` table of ``events.sqlite`` via ``src/trace/emit.py``
(category ``review_decision``, source ``review_plane``). It does NOT write
``review_ledger`` rows — but ``scripts/analysis/reviewer_calibration_report.py``
needs ledger/decision rows to compute FA/FR. There is no events->ledger
materializer in the serving path. THIS is that materializer, built as a
best-effort OFFLINE post-processing script.

It reads the already-emitted REVIEW_DECISION events and re-shapes each into a
``review_ledger`` row, emitted as either a decisions-JSONL or a ``review_ledger``
SQLite table — both directly consumable by ``reviewer_calibration_report.py``
(``--decisions`` and ``--ledger`` modes respectively).

  ┌── live run ──┐        ┌── THIS SCRIPT (offline) ──┐        ┌── report ──┐
  review_service ─emit()─▶ event table  ──materialize──▶ review_ledger ──▶ FA/FR
    (shadow plane)          (events.sqlite)                (rows/jsonl)

*** FIRING CONTINGENCY — READ THIS ***
--------------------------------------
This script only YIELDS rows if the shadow plane actually EMITTED
REVIEW_DECISION events during the workload. Whether the plane fires depends on
the DELEGATION path being exercised: EvalTower's rubric-judge HARD-DISABLES
delegation, whereas the shadow plane fires on the ``delegator.py`` /
``parallel_step_executor.py`` path. So it is an EXPECTED, NON-ERROR outcome for
this script to find ZERO events until a live run confirms the plane fired. Zero
events => exit 0 with a loud "0 events materialized" message, NOT a failure.

Field mapping (grounded in the emit schema — file:line):
  event.id                              -> decision_id      (stable: "revevt-<id>")
  event.ts_utc                          -> ts
  event.role (== architect_role)        -> reviewer_model_quant  (best-effort: a
                                           ROLE id, NOT a resolved model+quant; the
                                           event carries no resolved model — override
                                           with --reviewer-model-quant)
  detail["decision"]                    -> decision  (falls back to event.status;
                                           "error"/absent -> None => parse-failure)
  detail["subtask_id"]                  -> candidate_id  (enables --corpus gold join)
  detail["confidence"]                  -> confidence
  detail["tripwire"]                    -> tripwire
  detail["latency_ms"]                  -> latency_ms
  detail["tokens"]["tokens_out"]        -> tokens  (flattened to INT)
  event.source_path / event.id          -> event_source_path / event_id (provenance)

NULL unless supplied by --corpus / stamps: gold_label, gold_source,
gold_instrument_version, domain, corpus_id, grading_model, rubric_version, era,
rationale_cause_match, family_match_flag. Gold labels never exist on an event;
supply them with ``--corpus rows.jsonl`` (joined by candidate_id == row_id) so the
downstream report can compute FA/FR. NOTE: ``session_id``/``trial_id`` have no
``review_ledger`` column — they survive only via the event provenance link.

NO inference. NO server. NO stack mutation. Reads events.sqlite, writes a ledger.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Repo root on path so `src.*` imports whether run as a CLI or imported by tests.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.trace.query import query  # noqa: E402
from src.trace.review_ledger import (  # noqa: E402
    ReviewLedgerRow,
    insert_review_ledger_row,
)
from src.trace.store import EventCategory, ensure_schema  # noqa: E402

# The event category the shadow plane emits its decisions under (store.py L70).
REVIEW_DECISION_CATEGORY = EventCategory.REVIEW_DECISION

# An "error"-status emission (e.g. plan_review failure, review_service L576) is a
# reviewer format/parse failure, mapped to a null decision so the downstream report
# counts it toward parse-failure rate rather than any FA/FR denominator.
_ERROR_SENTINEL = "error"

EMPTY_MESSAGE = (
    "0 events materialized — shadow plane did not fire (or no events in range).\n"
    "This is the EXPECTED-EMPTY path, not a bug: REVIEW_DECISION events only exist "
    "if the reviewer shadow plane fired during a live run (delegator.py / "
    "parallel_step_executor.py path). EvalTower's rubric-judge hard-disables "
    "delegation, so an eval-only workload legitimately yields zero events. "
    "Re-run after a live delegation workload to confirm the plane fired."
)


# --------------------------------------------------------------------------- #
# Event -> ledger-row mapping
# --------------------------------------------------------------------------- #
def _parse_detail(detail_json: Any) -> dict[str, Any]:
    """Decode the event's detail_json column into a dict (never raises)."""
    if isinstance(detail_json, dict):
        return detail_json
    if not detail_json:
        return {}
    try:
        parsed = json.loads(detail_json)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _decision_id(event: dict[str, Any]) -> str:
    """Stable, idempotent decision_id derived from the event id.

    Within one ``events.sqlite`` the integer ``event.id`` is a 1:1 stable handle
    for a content-addressed emit:// row, so ``revevt-<id>`` re-derives identically
    on every run => ``INSERT OR IGNORE`` on ``UNIQUE(decision_id)`` makes re-runs
    no-ops. Falls back to a hash of the (already content-addressed) source_path if
    an id is somehow absent.
    """
    eid = event.get("id")
    if eid is not None:
        return f"revevt-{eid}"
    import hashlib

    sp = str(event.get("source_path") or "")
    return "revevt-" + hashlib.sha1(sp.encode("utf-8", "replace")).hexdigest()[:16]


def _tokens_int(detail: dict[str, Any]) -> int | None:
    """Flatten the nested per-decision token accounting to a single INT.

    ``review_service._response_tokens`` emits ``tokens`` as
    ``{"tokens_out": int, "chars_out": int}``; the ledger ``tokens`` column is a
    scalar INTEGER, so extract ``tokens_out``. Tolerates a bare int as well.
    """
    tok = detail.get("tokens")
    if isinstance(tok, dict):
        val = tok.get("tokens_out")
        return int(val) if isinstance(val, (int, float)) else None
    if isinstance(tok, (int, float)):
        return int(tok)
    return None


def _decision_value(event: dict[str, Any], detail: dict[str, Any]) -> str | None:
    """Resolve the reviewer verdict string.

    Prefers the authoritative ``detail["decision"]`` (identical to ``event.status``
    on the normal path). Falls back to ``event.status``. An ``"error"`` sentinel or
    a fully absent decision maps to ``None`` so the downstream report classifies it
    as a parse-failure (``review_ledger.is_parse_failure``: null decision).
    """
    raw = detail.get("decision")
    if raw in (None, ""):
        raw = event.get("status")
    if raw in (None, "", _ERROR_SENTINEL):
        return None
    return str(raw)


def event_to_ledger_row(
    event: dict[str, Any],
    *,
    reviewer_model_quant: str | None = None,
    grading_model: str | None = None,
    rubric_version: str | None = None,
    corpus_id: str | None = None,
    era: str | None = None,
    domain: str | None = None,
) -> ReviewLedgerRow:
    """Map one REVIEW_DECISION event dict (query() row) to a ReviewLedgerRow.

    Static stamps (``reviewer_model_quant``/``grading_model``/``rubric_version``/
    ``corpus_id``/``era``/``domain``) fill ledger group-fields the event cannot
    carry; when a stamp is None the event's own value (or None) is used.
    """
    detail = _parse_detail(event.get("detail_json"))
    return ReviewLedgerRow(
        decision_id=_decision_id(event),
        ts=event.get("ts_utc"),
        # event.role is the reviewer ROLE (architect_role); best-effort stand-in for
        # reviewer_model_quant, overridable via the stamp.
        reviewer_model_quant=reviewer_model_quant or event.get("role"),
        grading_model=grading_model,
        rubric_version=rubric_version,
        corpus_id=corpus_id,
        candidate_id=detail.get("subtask_id"),
        domain=domain,
        decision=_decision_value(event, detail),
        tripwire=detail.get("tripwire"),
        confidence=detail.get("confidence"),
        gold_label=None,  # events carry no gold; filled by --corpus join if any
        gold_source=None,
        gold_instrument_version=None,
        rationale_cause_match=None,
        latency_ms=detail.get("latency_ms"),
        tokens=_tokens_int(detail),
        family_match_flag=None,
        era=era,
        event_source_path=event.get("source_path"),
        event_id=event.get("id"),
    )


# --------------------------------------------------------------------------- #
# Optional gold-label corpus join (by candidate_id)
# --------------------------------------------------------------------------- #
def _load_corpus(corpus_path: str | Path) -> dict[str, dict[str, Any]]:
    """Index a near-miss corpus rows.jsonl by its join key (row_id / candidate_id).

    Mirrors ``reviewer_calibration_report.join_corpus_gold`` key semantics so the
    two paths agree on how gold is attached.
    """
    corpus: dict[str, dict[str, Any]] = {}
    with open(corpus_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            cr = json.loads(line)
            key = cr.get("row_id") or cr.get("candidate_id")
            if key is not None:
                corpus[str(key)] = cr
    return corpus


def apply_corpus_gold(row: ReviewLedgerRow, corpus: dict[str, dict[str, Any]]) -> ReviewLedgerRow:
    """Fill missing gold/domain/corpus fields on a row from the corpus (in place).

    Join key: ``row.candidate_id`` == corpus ``row_id``/``candidate_id``. Only fills
    fields currently empty — never overwrites an already-resolved value.
    """
    cid = str(row.candidate_id or "")
    cr = corpus.get(cid)
    if not cr:
        return row
    if not row.gold_label:
        row.gold_label = cr.get("gold_label")
    if not row.gold_source:
        row.gold_source = cr.get("gold_source")
    if not row.gold_instrument_version:
        row.gold_instrument_version = cr.get("gold_instrument_version")
    if not row.domain:
        row.domain = cr.get("domain")
    if not row.corpus_id:
        row.corpus_id = cr.get("corpus_id")
    return row


# --------------------------------------------------------------------------- #
# Read + materialize
# --------------------------------------------------------------------------- #
def read_review_events(
    events_db: str | Path,
    *,
    session_id: str | None = None,
    trial_id: int | None = None,
    since: str | None = None,
    limit: int = 1_000_000,
) -> list[dict[str, Any]]:
    """Fetch REVIEW_DECISION events, ascending by (ts_utc, id) for stable order."""
    rows = query(
        db_path=events_db,
        category=REVIEW_DECISION_CATEGORY,
        session_id=session_id,
        trial_id=trial_id,
        from_ts=since,
        limit=limit,
    )
    rows.sort(key=lambda r: (str(r.get("ts_utc") or ""), r.get("id") or 0))
    return rows


def materialize_rows(
    events: list[dict[str, Any]],
    *,
    corpus: dict[str, dict[str, Any]] | None = None,
    reviewer_model_quant: str | None = None,
    grading_model: str | None = None,
    rubric_version: str | None = None,
    corpus_id: str | None = None,
    era: str | None = None,
    domain: str | None = None,
) -> list[ReviewLedgerRow]:
    """Map every event to a ledger row, applying the optional corpus gold join."""
    out: list[ReviewLedgerRow] = []
    for ev in events:
        row = event_to_ledger_row(
            ev,
            reviewer_model_quant=reviewer_model_quant,
            grading_model=grading_model,
            rubric_version=rubric_version,
            corpus_id=corpus_id,
            era=era,
            domain=domain,
        )
        if corpus:
            apply_corpus_gold(row, corpus)
        out.append(row)
    return out


# --------------------------------------------------------------------------- #
# Emit
# --------------------------------------------------------------------------- #
def write_decisions_jsonl(rows: list[ReviewLedgerRow], out_dir: Path) -> Path:
    """Write ledger-row-shaped dicts, one per line (reviewer_calibration_report --decisions)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "reviewer_decisions.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(asdict(row), sort_keys=True, default=str) + "\n")
    return path


def write_ledger_sqlite(rows: list[ReviewLedgerRow], out_dir: Path) -> tuple[Path, int, int]:
    """Insert rows into a ``review_ledger`` table (reviewer_calibration_report --ledger).

    Idempotent: ``insert_review_ledger_row`` is ``INSERT OR IGNORE`` on
    ``UNIQUE(decision_id)`` so re-running against the same output DB skips dups.
    Returns ``(db_path, inserted, skipped)``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "review_ledger.sqlite"
    conn = ensure_schema(path)
    inserted = skipped = 0
    try:
        for row in rows:
            i, s = insert_review_ledger_row(conn, row)
            inserted += i
            skipped += s
    finally:
        conn.close()
    return path, inserted, skipped


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #
def summarize(rows: list[ReviewLedgerRow]) -> dict[str, Any]:
    """A small breakdown for the dry-run / completion report."""
    by_decision: dict[str, int] = {}
    with_candidate = 0
    with_gold = 0
    for r in rows:
        key = r.decision if r.decision is not None else "<null/parse_failure>"
        by_decision[key] = by_decision.get(key, 0) + 1
        if r.candidate_id:
            with_candidate += 1
        if r.gold_label:
            with_gold += 1
    return {
        "n_events": len(rows),
        "n_would_map": len(rows),  # every event maps to exactly one row
        "with_candidate_id": with_candidate,
        "with_gold_label": with_gold,
        "by_decision": by_decision,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--events", required=True, help="Path to events.sqlite (the trace store).")
    ap.add_argument("--session", help="Filter: only events with this session_id.")
    ap.add_argument("--trial", type=int, help="Filter: only events with this trial_id.")
    ap.add_argument("--since", help="Filter: only events with ts_utc >= this ISO8601 ts.")
    ap.add_argument("--limit", type=int, default=1_000_000, help="Max events to read (default 1e6).")
    ap.add_argument("--corpus", help="Optional near-miss corpus rows.jsonl to join gold_label by candidate_id.")
    ap.add_argument("--output", help="Output DIRECTORY for the materialized ledger (required to write).")
    ap.add_argument(
        "--emit",
        choices=("decisions-jsonl", "ledger-sqlite"),
        default="decisions-jsonl",
        help="Output format (default decisions-jsonl).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Report counts only, write nothing. Also implied when --output is absent.",
    )
    # Optional static stamps for ledger group-fields the event cannot carry.
    ap.add_argument("--reviewer-model-quant", help="Stamp reviewer_model_quant (else event.role).")
    ap.add_argument("--grading-model", help="Stamp grading_model (else NULL).")
    ap.add_argument("--rubric-version", help="Stamp rubric_version (else NULL).")
    ap.add_argument("--corpus-id", help="Stamp corpus_id (else from --corpus rows / NULL).")
    ap.add_argument("--era", help="Stamp era (else NULL).")
    ap.add_argument("--domain", help="Stamp domain (else from --corpus rows / NULL).")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    events_db = Path(args.events)
    if not events_db.exists():
        print(f"[reviewer_events_to_ledger] events DB not found: {events_db}", file=sys.stderr)
        # No DB == no events == the expected-empty path (loud, non-error).
        print("=" * 72)
        print(EMPTY_MESSAGE)
        print("=" * 72)
        return 0

    events = read_review_events(
        events_db,
        session_id=args.session,
        trial_id=args.trial,
        since=args.since,
        limit=args.limit,
    )

    # ── Expected-empty path: loud, non-error. ────────────────────────────────
    if not events:
        print("=" * 72)
        print(EMPTY_MESSAGE)
        print(
            f"(scanned {events_db} for category={REVIEW_DECISION_CATEGORY!r}"
            + (f", session={args.session}" if args.session else "")
            + (f", trial={args.trial}" if args.trial is not None else "")
            + (f", since={args.since}" if args.since else "")
            + ")"
        )
        print("=" * 72)
        return 0

    corpus = _load_corpus(args.corpus) if args.corpus else None
    rows = materialize_rows(
        events,
        corpus=corpus,
        reviewer_model_quant=args.reviewer_model_quant,
        grading_model=args.grading_model,
        rubric_version=args.rubric_version,
        corpus_id=args.corpus_id,
        era=args.era,
        domain=args.domain,
    )
    summary = summarize(rows)

    dry = args.dry_run or not args.output
    print(f"[reviewer_events_to_ledger] {summary['n_events']} REVIEW_DECISION events found; "
          f"{summary['n_would_map']} would map "
          f"({summary['with_candidate_id']} with candidate_id, "
          f"{summary['with_gold_label']} with gold_label after corpus join).")
    print(f"[reviewer_events_to_ledger] by decision: {json.dumps(summary['by_decision'], sort_keys=True)}")

    if dry:
        why = "--dry-run" if args.dry_run else "no --output given"
        print(f"[reviewer_events_to_ledger] DRY-RUN ({why}) — writing nothing.")
        return 0

    out_dir = Path(args.output)
    if args.emit == "decisions-jsonl":
        path = write_decisions_jsonl(rows, out_dir)
        print(f"[reviewer_events_to_ledger] wrote {len(rows)} rows -> {path}")
        print(f"  chain: reviewer_calibration_report.py --decisions {path}"
              + (f" --corpus {args.corpus}" if args.corpus else ""))
    else:
        path, inserted, skipped = write_ledger_sqlite(rows, out_dir)
        print(f"[reviewer_events_to_ledger] review_ledger @ {path}: "
              f"{inserted} inserted, {skipped} skipped (dup).")
        print(f"  chain: reviewer_calibration_report.py --ledger {path}"
              + (f" --corpus {args.corpus}" if args.corpus else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
