#!/usr/bin/env python3
"""AP-55-ARM review: how often would ``enforce`` have held a promotion in shadow?

The operator pre-approved arming the AP-55 promotion gate (``enforce`` plus seed
re-runs) after ONE AutoPilot run in shadow mode and a report of this number. In
shadow mode the recorded ``hold`` is False by construction, so this reads the
counterfactual the gate records per trial (``would_hold_enforce``) and, for rows
written before that field existed, re-derives it with the gate's own rule
(``ap55_promotion_gate.would_hold_from_summary``; the basis is reported).

Denominators, from widest to the one that matters:

  gated        trial rows in the window carrying ``eval_details.ap55_promotion_gate``
  attempted    gated rows that reached ``update_baseline`` (``promotion_status`` set)
  committed    attempted rows confirmed by a ``baseline_promotion`` ledger event —
               the promotions enforce would actually have blocked

Read-only by construction: shards are parsed line by line (``ExperimentJournal`` is
not instantiated, because its loader may repair a torn tail), nothing is written.
Exit 0 = report produced; 3 = no shadow-mode gate verdict in the window, so the
AP-55-ARM precondition ("one shadow run") is not met and nothing should be flipped.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _p in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from journal_shards import journal_shards  # noqa: E402
from src.autopilot_core.ap55_promotion_gate import would_hold_from_summary  # noqa: E402

SCHEMA = "epyc.autopilot.ap55_shadow_review.v1"
DEFAULT_JOURNAL_DIR = REPO_ROOT / "orchestration"
NO_SHADOW_RUN_EXIT = 3
ARM_SWITCH = {
    "file": "scripts/autopilot/start_authority_daemon.py (AUTHORITY_ENV)",
    "AUTOPILOT_AP55_PROMOTION_GATE": "shadow -> enforce",
    "AUTOPILOT_AP55_SEED_RERUN": "0 -> 1",
    "takes_effect": "next AutoPilot restart through start_authority_daemon.py",
}


def _parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def _read(journal_dir: Path) -> tuple[dict[int, dict[str, Any]], set[int], int]:
    trials: dict[int, dict[str, Any]] = {}
    committed: set[int] = set()
    bad = 0
    for path in journal_shards(journal_dir):
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                if not isinstance(obj, dict):
                    bad += 1
                    continue
                if obj.get("type") == "baseline_promotion":
                    try:
                        committed.add(int(obj.get("source_trial_id")))
                    except (TypeError, ValueError):
                        bad += 1
                    continue
                if obj.get("type") is not None:
                    continue
                try:
                    trials[int(obj["trial_id"])] = obj  # later shard / line wins
                except (KeyError, TypeError, ValueError):
                    bad += 1
    return trials, committed, bad


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def review(
    journal_dir: Path,
    *,
    since: datetime | None = None,
    since_trial: int | None = None,
) -> dict[str, Any]:
    trials, committed_ids, bad_lines = _read(Path(journal_dir))
    window: list[dict[str, Any]] = []
    for tid in sorted(trials):
        row = trials[tid]
        if since_trial is not None and tid < since_trial:
            continue
        if since is not None:
            ts = _parse_ts(row.get("timestamp"))
            if ts is None or ts < since:
                continue
        window.append(row)

    modes: collections.Counter[str] = collections.Counter()
    basis: collections.Counter[str] = collections.Counter()
    reasons: collections.Counter[str] = collections.Counter()
    excluded = collections.Counter()
    buckets = {k: {"n": 0, "would_hold": 0, "trial_ids": []} for k in ("gated", "attempted", "committed")}
    for row in window:
        if row.get("bug_corrupted_by"):
            excluded["bug_corrupted"] += 1
            continue
        details = row.get("eval_details") if isinstance(row.get("eval_details"), dict) else {}
        summary = details.get("ap55_promotion_gate")
        if not isinstance(summary, dict) or not summary:
            excluded["no_gate_verdict"] += 1
            continue
        modes[str(summary.get("mode") or "")] += 1
        cf = would_hold_from_summary(summary, "enforce")
        basis[cf["basis"]] += 1
        held = bool(cf["hold"])
        if held:
            reasons.update(cf["hold_reasons"])
        tid = int(row["trial_id"])
        tags = ["gated"]
        if details.get("promotion_status"):
            tags.append("attempted")
            if tid in committed_ids:
                tags.append("committed")
        for tag in tags:
            buckets[tag]["n"] += 1
            if held:
                buckets[tag]["would_hold"] += 1
                buckets[tag]["trial_ids"].append(tid)
    for bucket in buckets.values():
        bucket["rate"] = _rate(bucket["would_hold"], bucket["n"])

    shadow_seen = modes.get("shadow", 0) > 0
    out: dict[str, Any] = {
        "schema": SCHEMA,
        "journal_dir": str(journal_dir),
        "window": {
            "since": since.isoformat() if since else None,
            "since_trial": since_trial,
            "trial_rows": len(window),
            "first_trial_id": window[0]["trial_id"] if window else None,
            "last_trial_id": window[-1]["trial_id"] if window else None,
        },
        "bad_lines": bad_lines,
        "excluded": dict(excluded),
        "gate_modes": dict(modes),
        "counterfactual_basis": dict(basis),
        "enforce_would_hold": buckets,
        "hold_reasons": dict(reasons.most_common()),
        "shadow_run_observed": shadow_seen,
        "arm_switch": ARM_SWITCH,
    }
    if not shadow_seen:
        out["verdict"] = (
            "NOT READY: no shadow-mode AP-55 verdict in this window. AP-55-ARM needs one "
            "AutoPilot run in shadow first; flip nothing."
        )
    elif modes.keys() - {"shadow"}:
        out["verdict"] = (
            "MIXED WINDOW: rows from a binding mode are present; narrow --since to the "
            "shadow run before reading the rates."
        )
    else:
        c = buckets["committed"]
        out["verdict"] = (
            f"READY FOR REVIEW: enforce would have held {c['would_hold']} of {c['n']} committed "
            f"promotions ({buckets['attempted']['would_hold']} of {buckets['attempted']['n']} "
            f"attempts, {buckets['gated']['would_hold']} of {buckets['gated']['n']} gated trials). "
            "Report these, then apply arm_switch (operator pre-approved)."
        )
    return out


def _format(report: dict[str, Any]) -> str:
    w = report["window"]
    lines = [
        f"AP-55 shadow review  window: trials {w['first_trial_id']}..{w['last_trial_id']} "
        f"({w['trial_rows']} rows; since={w['since']}, since_trial={w['since_trial']})",
        f"gate modes: {report['gate_modes']}  basis: {report['counterfactual_basis']}  "
        f"excluded: {report['excluded']}",
    ]
    for tag in ("gated", "attempted", "committed"):
        b = report["enforce_would_hold"][tag]
        lines.append(f"  {tag:<10} n={b['n']:<5} would_hold={b['would_hold']:<5} rate={b['rate']}  "
                     f"trials={b['trial_ids'][:20]}")
    lines.append(f"hold reasons: {report['hold_reasons']}")
    lines.append(report["verdict"])
    sw = report["arm_switch"]
    lines.append(
        f"arm switch: {sw['file']}: AUTOPILOT_AP55_PROMOTION_GATE {sw['AUTOPILOT_AP55_PROMOTION_GATE']}, "
        f"AUTOPILOT_AP55_SEED_RERUN {sw['AUTOPILOT_AP55_SEED_RERUN']} ({sw['takes_effect']})"
    )
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--journal-dir", type=Path, default=DEFAULT_JOURNAL_DIR)
    ap.add_argument("--since", help="ISO timestamp: first row of the shadow run")
    ap.add_argument("--since-trial", type=int, help="first trial id of the shadow run")
    ap.add_argument("--json", action="store_true", help="emit the JSON report")
    args = ap.parse_args(list(argv) if argv is not None else None)
    since = _parse_ts(args.since) if args.since else None
    if args.since and since is None:
        ap.error(f"--since is not an ISO timestamp: {args.since}")
    report = review(args.journal_dir, since=since, since_trial=args.since_trial)
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else _format(report))
    return 0 if report["shadow_run_observed"] else NO_SHADOW_RUN_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
