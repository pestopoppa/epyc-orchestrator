"""VB-AP53-RATE: durable per-window AutoPilot re-proposal rate rows.

WHY
---
AP-53 measured once, with a scratch script, that 133 of 1372 journal trials
(9.7%) re-ran a config an earlier trial had already hard-rejected. That number is
an OBSERVATION with no durable record: no row pins its window, its key
definition or its rejection classes, so the belief kernel cannot read it. This
module is the write side. At every trial boundary, AutoPilot calls
``record_closed_windows(journal)``. When a trial-id window has closed, the call
appends one producer-authored line to
``<journal_dir>/autopilot_reproposal_rates.jsonl``.

WHAT A WINDOW LINE SAYS
-----------------------
For the window ``[start, start + size)`` of trial ids:

* ``all_trials``: journal trials whose id falls in the window, counted once each.
* ``keyed_trials``: those whose action has a concrete config identity.
  The key is ``rejected_mutation_ledger._concrete_action``, the planner fold's
  own rule: observational, ``numeric_trial`` and deliberate-replay rows have no
  key. The fingerprint is ``action_identity.config_fingerprint``, which drops
  narrative fields.
* ``reproposals``: keyed trials whose fingerprint was a STILL-STANDING hard
  rejection when the trial ran. The rejection classes and the clearing rule are
  ``rejected_mutation_ledger._hard_rejection`` and the ``frontier`` clear used by
  ``rejected_configs_from_entries``. A test pins the two folds equal.
* From the AP-53 mutation ledger: the window's rejection records, and how many
  of them repeat a ``diff_sha256`` already rejected earlier.

Each line carries producer-authored ``belief_measurements`` rows, one per rate,
each with its numerator and denominator stated. The rows cite no protocol, so
the belief kernel grades them as observations. Grading happens only in the root
``claim_tuple.grade()``; this module never grades. A rate whose denominator is
zero is not written, because absence is not a 0% rate.

PRE-HOOK DATA
-------------
The first call ARMS the file and writes no window. Its ``armed`` record names
the first window that starts after the call. Windows before that point never get
a row here: the pilot spec (section 4.7) says "a row that predates a producer's
provenance hook is skipped rather than back-filled". ``backfill`` recomputes those
windows into a SEPARATE ``*.retrospective.jsonl`` file, labelled
``retrospective: true``, with ``belief_measurements: []``. It reproduces the
one-off analysis durably and asserts nothing the kernel can grade.

FAIL-OPEN
---------
``record_closed_windows`` never raises and never touches the journal. A write
failure is logged and the trial proceeds unchanged. Set
``AUTOPILOT_REPROPOSAL_RATE_WRITER=0`` to disable the hook.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import inspect
import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

if __package__ in (None, ""):
    _HERE = Path(__file__).resolve().parent
    for _p in (_HERE, _HERE.parents[1]):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

import rejected_mutation_ledger as rml  # noqa: E402

log = logging.getLogger("autopilot")

SCHEMA = "epyc.autopilot.reproposal_rate.v1"
BELIEF_SCHEMA = "epyc.autopilot.reproposal_rate_belief.v1"
WRITER_ID = "epyc-orchestrator/scripts/autopilot/reproposal_rate.py@v1"
RATE_FILENAME = "autopilot_reproposal_rates.jsonl"
RETROSPECTIVE_FILENAME = "autopilot_reproposal_rates.retrospective.jsonl"
DEFAULT_WINDOW = 100
ENV_DISABLE = "AUTOPILOT_REPROPOSAL_RATE_WRITER"

METRIC_ALL = "autopilot.reproposal_rate.all_trials"
METRIC_KEYED = "autopilot.reproposal_rate.keyed_trials"
METRIC_DIFF_REPEAT = "autopilot.rejected_mutation.diff_repeat_rate"
METRICS = (METRIC_ALL, METRIC_KEYED, METRIC_DIFF_REPEAT)

NO_WARRANT_REASON = (
    "retrospective: computed after the fact from journal rows that were written "
    "before the re-proposal rate hook existed. docs/design/vidya-pilot-spec.md section 4.7: "
    "'a row that predates a producer's provenance hook is skipped rather than "
    "back-filled, because a tuple invented on read claims warrant the original run "
    "never captured'. The key definition, the rejection classes and supersessions "
    "in force are TODAY's, not the ones in force at those trials. Zero belief rows."
)


def rate_path(journal_dir: Path) -> Path:
    return Path(journal_dir) / RATE_FILENAME


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)


def _sha(text: str | bytes) -> str:
    data = text.encode("utf-8") if isinstance(text, str) else text
    return hashlib.sha256(data).hexdigest()


def row_digest(row: dict[str, Any]) -> str:
    """sha256 over the row with ``extra.row_sha256`` removed (the reader re-derives it)."""
    body = json.loads(_canon(row))
    body.get("extra", {}).pop("row_sha256", None)
    return _sha(_canon(body))


def definition_sha256() -> str:
    """Digest of the source of every function that defines key, classes and clearing."""
    from src.autopilot_core import action_identity as ai

    parts = [inspect.getsource(fn) for fn in (
        rml._concrete_action, rml._hard_rejection, rml.rejected_configs_from_entries,
        ai.canonical_action, ai.config_fingerprint, ai.action_signature,
    )]
    parts.append(_canon(sorted(ai.EPHEMERAL_ACTION_KEYS)))
    return _sha("\n".join(parts))


DEFINITIONS = {
    "unit": "one journal trial (trial_id counted once; the last journaled row for an id wins)",
    "fold": "ExperimentJournal.entries_with_supersessions() in journal order",
    "key": ("src.autopilot_core.action_identity.config_fingerprint over "
            "rejected_mutation_ledger._concrete_action (observational, numeric_trial and "
            "deliberate-replay rows carry no key)"),
    "rejection_classes": ["safety_gate", "invalid", "sequential_refuted"],
    "standing": ("a key is standing-rejected after a hard rejection and until a later "
                 "frontier trial of the same key clears it (rejected_configs_from_entries)"),
    "reproposal": "a keyed trial whose key was standing-rejected before the trial was folded",
    "diff_repeat": ("an AP-53 ledger record whose diff_sha256 appeared on an EARLIER ledger "
                    "line; window membership by the record's trial_id"),
}


# ── the fold ──────────────────────────────────────────────────────────────────

def _dedupe(entries: Iterable[Any]) -> list[Any]:
    last: dict[int, Any] = {}
    for entry in entries:
        tid = getattr(entry, "trial_id", None)
        if isinstance(tid, int):
            last[tid] = entry
    return [last[t] for t in sorted(last)]


def fold_windows(entries: Iterable[Any], *, window: int, upto: int | None = None) -> dict[str, Any]:
    """Walk the journal once. Returns per-window counts and the final standing set.

    ``upto`` (exclusive) limits which windows are reported; the fold itself always
    runs over every earlier trial so standing rejections accumulate from trial 0.
    """
    from src.autopilot_core.action_identity import config_fingerprint

    if window < 1:
        raise ValueError("window must be >= 1")
    standing: dict[str, str] = {}
    windows: dict[int, dict[str, Any]] = {}
    fold_lines: dict[int, list[str]] = {}
    for entry in _dedupe(entries):
        tid = entry.trial_id
        if upto is not None and tid >= upto:
            break
        start = (tid // window) * window
        w = windows.setdefault(start, {
            "all_trials": 0, "keyed_trials": 0, "reproposals": 0,
            "reproposals_by_action_type": Counter(),
            "reproposals_by_standing_class": Counter(),
            "trial_ids_present": 0, "first_trial_id": tid, "last_trial_id": tid,
        })
        w["all_trials"] += 1
        w["trial_ids_present"] += 1
        w["last_trial_id"] = tid
        action = rml._concrete_action(entry)
        key, verdict = "", ""
        if action is not None:
            key = config_fingerprint(action)
            w["keyed_trials"] += 1
            if key in standing:
                w["reproposals"] += 1
                w["reproposals_by_action_type"][str(action.get("type"))] += 1
                w["reproposals_by_standing_class"][standing[key]] += 1
                verdict = "reproposal"
            reason = rml._hard_rejection(entry)
            if reason:
                standing[key] = reason
            elif str(getattr(entry, "pareto_status", "") or "") == "frontier":
                standing.pop(key, None)
        fold_lines.setdefault(start, []).append(f"{tid}|{key}|{verdict}|{standing.get(key, '')}")
    for start, w in windows.items():
        w["reproposals_by_action_type"] = dict(sorted(w["reproposals_by_action_type"].items()))
        w["reproposals_by_standing_class"] = dict(sorted(w["reproposals_by_standing_class"].items()))
        w["fold_sha256"] = _sha("\n".join(fold_lines[start]))
    return {"windows": windows, "standing": standing}


def ledger_window_counts(records: list[dict[str, Any]], start: int, end: int) -> dict[str, Any]:
    seen: set[str] = set()
    n = repeats = 0
    by_gate: Counter = Counter()
    for rec in records:
        digest = str(rec.get("diff_sha256") or "")
        tid = rec.get("trial_id")
        if isinstance(tid, int) and start <= tid < end:
            n += 1
            by_gate[str(rec.get("rejecting_gate") or "")] += 1
            if digest and digest in seen:
                repeats += 1
        if digest:
            seen.add(digest)
    return {"records": n, "diff_repeats": repeats, "by_rejecting_gate": dict(sorted(by_gate.items()))}


# ── source identity ──────────────────────────────────────────────────────────

def _prefix_identity(path: Path) -> dict[str, Any]:
    try:
        size = path.stat().st_size
        h = hashlib.sha256()
        with open(path, "rb") as f:
            remaining = size
            while remaining > 0:
                chunk = f.read(min(1 << 20, remaining))
                if not chunk:
                    break
                h.update(chunk)
                remaining -= len(chunk)
        return {"path": str(path), "bytes": size, "prefix_sha256": h.hexdigest()}
    except OSError:
        return {"path": str(path), "bytes": 0, "prefix_sha256": "", "absent": True}


def source_identity(journal_dir: Path) -> dict[str, Any]:
    from journal_shards import journal_shards

    shards = []
    try:
        shards = [_prefix_identity(Path(p)) for p in journal_shards(Path(journal_dir))]
    except Exception as exc:  # noqa: BLE001 - identity capture must not fail the writer
        shards = [{"error": f"{type(exc).__name__}: {exc}"[:200]}]
    return {
        "journal_dir": str(journal_dir),
        "journal_shards": shards,
        "mutation_ledger": _prefix_identity(rml.ledger_path(journal_dir)),
    }


# ── rows ─────────────────────────────────────────────────────────────────────

def _rate_row(*, metric: str, key: str, numerator: int, denominator: int, basis: str,
              claim: str, start: int, end: int, window: int, date: str, fold_sha: str,
              path: Path, extra: dict[str, Any]) -> dict[str, Any]:
    row = {
        "measurement_id": f"ap53-rate-w{window}-{start:07d}-{key}-{fold_sha[:12]}",
        "metric": metric,
        "value": numerator / denominator,
        "unit": "fraction",
        "date": date,
        "category": "BASELINE",
        "claim": claim,
        "metric_direction": "lower_better",
        "protocol_id": "",
        "reps": denominator,
        "reps_basis": basis,
        "attestation_path": str(path),
        "attestation_locator": f"{path}#window={start}-{end}",
        "source_kind": "measurement",
        "extra": {
            "belief_schema": BELIEF_SCHEMA,
            "producer": WRITER_ID,
            "window": {"size": window, "start": start, "end_exclusive": end},
            "numerator": numerator,
            "denominator": denominator,
            "fold_sha256": fold_sha,
            "direction_basis": "lower = fewer wasted trials on already-rejected configs",
            **extra,
        },
    }
    row["extra"]["row_sha256"] = row_digest(row)
    return row


def belief_rows(counts: dict[str, Any], ledger_counts: dict[str, Any], *, start: int,
                window: int, path: Path, definition_sha: str) -> list[dict[str, Any]]:
    end = start + window
    date = counts.get("window_closed_at", "")
    fold_sha = counts["fold_sha256"]
    common = {"definition_sha256": definition_sha}
    rows: list[dict[str, Any]] = []
    span = f"AutoPilot trials {start}..{end - 1}"
    if counts["all_trials"] > 0:
        rows.append(_rate_row(
            metric=METRIC_ALL, key="all", numerator=counts["reproposals"],
            denominator=counts["all_trials"],
            basis="scored: every journal trial in the window (all action types)",
            claim=(f"{counts['reproposals']} of {counts['all_trials']} {span} re-proposed a "
                   "config that was still standing-rejected"),
            start=start, end=end, window=window, date=date, fold_sha=fold_sha, path=path,
            extra={**common, "by_action_type": counts["reproposals_by_action_type"],
                   "by_standing_class": counts["reproposals_by_standing_class"]}))
    if counts["keyed_trials"] > 0:
        rows.append(_rate_row(
            metric=METRIC_KEYED, key="keyed", numerator=counts["reproposals"],
            denominator=counts["keyed_trials"],
            basis="scored: journal trials in the window with a concrete config key",
            claim=(f"{counts['reproposals']} of {counts['keyed_trials']} keyed {span} "
                   "re-proposed a config that was still standing-rejected"),
            start=start, end=end, window=window, date=date, fold_sha=fold_sha, path=path,
            extra={**common, "by_action_type": counts["reproposals_by_action_type"],
                   "by_standing_class": counts["reproposals_by_standing_class"]}))
    if ledger_counts["records"] > 0:
        rows.append(_rate_row(
            metric=METRIC_DIFF_REPEAT, key="diffrepeat", numerator=ledger_counts["diff_repeats"],
            denominator=ledger_counts["records"],
            basis="scored: AP-53 ledger rejection records whose trial_id is in the window",
            claim=(f"{ledger_counts['diff_repeats']} of {ledger_counts['records']} rejected "
                   f"mutations in {span} repeated an exact diff already rejected"),
            start=start, end=end, window=window, date=date, fold_sha=fold_sha, path=path,
            extra={**common, "by_rejecting_gate": ledger_counts["by_rejecting_gate"]}))
    return rows


def window_line(entries: list[Any], journal_dir: Path, *, start: int, window: int,
                armed_from: int, retrospective: bool = False,
                fold: dict[str, Any] | None = None,
                ledger_records: list[dict[str, Any]] | None = None,
                sources: dict[str, Any] | None = None,
                definition_sha: str | None = None) -> dict[str, Any] | None:
    end = start + window
    fold = fold if fold is not None else fold_windows(entries, window=window, upto=end)
    counts = fold["windows"].get(start)
    if counts is None:
        return None
    counts = dict(counts)
    counts["trial_ids_missing"] = window - counts["trial_ids_present"]
    counts["window_closed_at"] = _now()
    records = ledger_records if ledger_records is not None else rml.load_records(journal_dir)
    lcounts = ledger_window_counts(records, start, end)
    dsha = definition_sha or definition_sha256()
    path = Path(journal_dir) / (RETROSPECTIVE_FILENAME if retrospective else RATE_FILENAME)
    line = {
        "schema": SCHEMA,
        "record": "retrospective_window" if retrospective else "window",
        "retrospective": retrospective,
        "writer": WRITER_ID,
        "written_at": counts["window_closed_at"],
        "armed_from_trial": armed_from,
        "window": {"size": window, "start": start, "end_exclusive": end},
        "counts": counts,
        "ledger_counts": lcounts,
        "definitions": DEFINITIONS,
        "definition_sha256": dsha,
        "sources": sources if sources is not None else source_identity(journal_dir),
    }
    if retrospective:
        line["no_warrant_reason"] = NO_WARRANT_REASON
        line["belief_measurements"] = []
    else:
        line["belief_measurements"] = belief_rows(
            counts, lcounts, start=start, window=window, path=path, definition_sha=dsha)
    line["line_sha256"] = _sha(_canon(line))
    return line


# ── durable append ───────────────────────────────────────────────────────────

def _append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _canon(record) + "\n"
    with open(path, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def read_lines(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        with open(path) as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and row.get("schema") == SCHEMA:
                    out.append(row)
    except OSError:
        return []
    return out


def _state(path: Path) -> tuple[int | None, set[int], int | None]:
    armed_from = None
    armed_window = None
    emitted: set[int] = set()
    for row in read_lines(path):
        if row.get("record") == "armed" and armed_from is None:
            armed_from = int(row["armed_from_trial"])
            armed_window = int(row["window_size"])
        elif row.get("record") == "window":
            emitted.add(int(row["window"]["start"]))
    return armed_from, emitted, armed_window


def _entries(journal: Any) -> list[Any]:
    if hasattr(journal, "entries_with_supersessions"):
        return list(journal.entries_with_supersessions())
    return list(journal.all_entries())


def record_closed_windows(journal: Any, *, window: int = DEFAULT_WINDOW) -> list[int]:
    """Trial-boundary hook. Returns the window starts written. Never raises."""
    try:
        if os.environ.get(ENV_DISABLE, "1").strip() == "0":
            return []
        journal_dir = getattr(journal, "journal_dir", None)
        if journal_dir is None:
            return []
        path = rate_path(journal_dir)
        armed_from, emitted, armed_window = _state(path)
        raw = journal.all_entries()
        tids = [e.trial_id for e in raw if isinstance(getattr(e, "trial_id", None), int)]
        max_tid = max(tids) if tids else -1
        if armed_from is None:
            # The window holding the trial just recorded is part pre-hook: arm the next one.
            armed_from = (max_tid // window + 1) * window if max_tid >= 0 else 0
            _append(path, {
                "schema": SCHEMA, "record": "armed", "writer": WRITER_ID,
                "armed_at": _now(), "armed_from_trial": armed_from, "window_size": window,
                "last_trial_at_arming": max_tid,
                "reason": ("windows starting before armed_from_trial predate this hook and "
                           "never get a row here (vidya-pilot-spec section 4.7)"),
            })
            return []
        window = armed_window or window
        due = [s for s in range(armed_from, max_tid + 1, window)
               if s + window - 1 <= max_tid and s not in emitted]
        if not due:
            return []
        entries = _entries(journal)
        fold = fold_windows(entries, window=window, upto=max(due) + window)
        records = rml.load_records(journal_dir)
        sources = source_identity(journal_dir)
        dsha = definition_sha256()
        written = []
        for start in due:
            line = window_line(entries, journal_dir, start=start, window=window,
                               armed_from=armed_from, fold=fold, ledger_records=records,
                               sources=sources, definition_sha=dsha)
            if line is None:
                continue
            _append(path, line)
            written.append(start)
        return written
    except Exception as exc:  # noqa: BLE001 - the rate writer must never fail a trial
        log.warning("VB-AP53-RATE window writer failed: %s: %s", type(exc).__name__, exc)
        return []


# ── offline: backfill + report ───────────────────────────────────────────────

def backfill(journal_dir: Path, out: Path, *, window: int = DEFAULT_WINDOW,
             before: int | None = None) -> dict[str, Any]:
    """Recompute pre-hook windows into a RETROSPECTIVE file (zero belief rows).

    ``before`` defaults to the prospective file's ``armed_from_trial`` (or every
    closed window when the hook has never armed). Refuses to write into the
    prospective file and refuses to overwrite an existing output.
    """
    from experiment_journal import ExperimentJournal

    out = Path(out)
    if out.name == RATE_FILENAME:
        raise ValueError("backfill never writes the prospective rate file")
    if out.exists():
        raise FileExistsError(f"{out} exists; retrospective output is written once")
    journal = ExperimentJournal(Path(journal_dir), segment_snapshots=False)
    entries = _entries(journal)
    tids = [e.trial_id for e in entries]
    max_tid = max(tids) if tids else -1
    armed_from, _, _ = _state(rate_path(journal_dir))
    limit = before if before is not None else (armed_from if armed_from is not None else max_tid + 1)
    starts = [s for s in range(0, max_tid + 1, window) if s + window <= limit
              and s + window - 1 <= max_tid]
    fold = fold_windows(entries, window=window)
    records = rml.load_records(journal_dir)
    sources = source_identity(journal_dir)
    dsha = definition_sha256()
    totals = Counter()
    by_type: Counter = Counter()
    lines = 0
    for start in starts:
        line = window_line(entries, journal_dir, start=start, window=window,
                           armed_from=limit, retrospective=True, fold=fold,
                           ledger_records=records, sources=sources, definition_sha=dsha)
        if line is None:
            continue
        _append(out, line)
        lines += 1
        c = line["counts"]
        totals.update({k: c[k] for k in ("all_trials", "keyed_trials", "reproposals")})
        by_type.update(c["reproposals_by_action_type"])
    return {
        "out": str(out), "retrospective_lines": lines, "belief_rows": 0,
        "window": window, "windows_before_trial": limit, "max_trial_id": max_tid,
        "totals": dict(totals), "reproposals_by_action_type": dict(sorted(by_type.items())),
        "rate_all": (totals["reproposals"] / totals["all_trials"]) if totals["all_trials"] else None,
        "rate_keyed": (totals["reproposals"] / totals["keyed_trials"]) if totals["keyed_trials"] else None,
        "no_warrant_reason": NO_WARRANT_REASON,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backfill", help="retrospective windows (zero belief rows)")
    b.add_argument("--journal-dir", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    b.add_argument("--before", type=int, default=None)
    args = ap.parse_args(argv)
    if args.cmd == "backfill":
        report = backfill(args.journal_dir, args.out, window=args.window, before=args.before)
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
