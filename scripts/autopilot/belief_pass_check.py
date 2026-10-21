#!/usr/bin/env python3
"""DP-4(A) pass-rule checker: counterfactual-confirmed reliance, from planner-archive receipts.

PASS when at least `--min` (default 3) distinct planner turns are confirmed. A turn is confirmed by

  * a `planner_coordinator` row whose `vidya_loop_beliefs` receipt has
      - a non-empty `qualifying_reliance_claim_ids` (an accepted knob/target-tier claim whose
        knob family equals the chosen action's family), AND
      - `counterfactual.status == "ok"`, `counterfactual.dispatched == false` and
        `reliance_confirmed == true` (the shadow draft with the belief section blanked proposed a
        different action, or a different value on a knob); or
  * a `belief_premise_skip` row (a premise skip that blocked a dispatch counts as 1).

Read-only. Exit 0 = PASS, 1 = not yet, 2 = unreadable archive.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

DEFAULT_ARCHIVE = Path(__file__).resolve().parents[2] / "logs" / "planner_archive.jsonl"
DEFAULT_MIN = 3


def _rows(path: Path) -> Iterable[dict[str, Any]]:
    with open(path) as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                yield row


def evaluate(rows: Iterable[dict[str, Any]], *, since: float = 0.0, minimum: int = DEFAULT_MIN
             ) -> dict[str, Any]:
    rows = list(rows)
    # Rows produced BY a shadow call (tagged at source) or whose prompt hash is a recorded shadow
    # prompt are never real drafts: ignore them, whatever else they claim.
    shadow_hashes = {
        str(((r.get("vidya_loop_beliefs") or {}).get("counterfactual") or {})
            .get("shadow_prompt_sha256_16") or "") for r in rows} - {""}
    confirmed: dict[Any, dict[str, Any]] = {}
    considered = shadow_runs = shadow_failed = 0
    for row in rows:
        if float(row.get("ts") or 0) < since:
            continue
        if (row.get("shadow") is True or row.get("purpose") == "belief_counterfactual"
                or str(row.get("prompt_sha256_16") or "") in shadow_hashes):
            continue
        kind = row.get("type")
        if kind == "belief_premise_skip" and row.get("reliance_confirmed") is True:
            confirmed.setdefault(("skip", row.get("trial_id"), row.get("ts")),
                                 {"trial_id": row.get("trial_id"), "via": "premise_skip",
                                  "claim_ids": row.get("premise_claim_ids")})
            continue
        if kind != "planner_coordinator":
            continue
        receipt = row.get("vidya_loop_beliefs") or {}
        qualifying = receipt.get("qualifying_reliance_claim_ids") or []
        if not qualifying:
            continue
        considered += 1
        cf = receipt.get("counterfactual") or {}
        if cf:
            shadow_runs += 1
        if cf.get("status") != "ok":
            shadow_failed += bool(cf)
            continue
        if cf.get("dispatched") is False and receipt.get("reliance_confirmed") is True:
            confirmed.setdefault(("shadow", row.get("trial_id"), row.get("ts")),
                                 {"trial_id": row.get("trial_id"), "via": "shadow_draft",
                                  "claim_ids": qualifying,
                                  "families": receipt.get("action_knob_families")})
    count = len(confirmed)
    return {"schema": "epyc.autopilot.belief_pass_check.v1", "rule": "DP-4(A)",
            "minimum": minimum, "confirmed_turns": count, "pass": count >= minimum,
            "qualifying_turns_seen": considered, "shadow_runs": shadow_runs,
            "shadow_not_ok": shadow_failed, "confirmed": list(confirmed.values())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    ap.add_argument("--since", type=float, default=0.0, help="epoch seconds; earlier rows ignored")
    ap.add_argument("--min", type=int, default=DEFAULT_MIN, dest="minimum")
    args = ap.parse_args(argv)
    try:
        result = evaluate(_rows(args.archive), since=args.since, minimum=args.minimum)
    except OSError as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
