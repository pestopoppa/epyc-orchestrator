#!/usr/bin/env python3
"""J2/J3 within-role KV-migration probe (bulk-inference Package J).

Verifies the WP-3 (forward) + WP-4 (reverse) migration legs that are merged + unit-tested
but never observed live (J4/J6 autopilot eval uses distinct sessions at steady concurrency,
so it triggers 0 migrations — see within-role-placement-state-machine.md). This drives the
specific traffic patterns the triggers need, and reads the counters off
`GET /dashboard/api/contention` (per-role: migrations_started [forward], reverse_migrations
[reverse — needs API restarted past 3ec218f to surface], migration_failures).

FORWARD (J2): session A lands on full; A completes (full idle); a NEW session B arrives →
the dispatcher migrates A's KV to an idle quarter (`_migrations += 1`) and gives B full.
A's next turn should then route to its migrated quarter (affinity).

REVERSE (J3): push concurrency > safe-slots so requests spill onto quarters, then drop to
1 and let the cooldown elapse → a warm quarter session migrates back to full
(`_reverse_migration_counts`).

Usage:
  python migration_probe.py --role frontdoor --leg forward --iters 6
  python migration_probe.py --role frontdoor --leg both --iters 6
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

API = "http://127.0.0.1:8000"
# A prompt that does real (non-trivial) work so the instance is genuinely occupied,
# but completes quickly. max_turns>=4 (max_turns=1 + force_role yields [Max turns reached]).
PROMPT = "Compute the sum of integers from 1 to 20. Show the value, then call FINAL with it."


def _counts(role: str) -> dict:
    """Read per-role migration counters off the contention dashboard."""
    try:
        with httpx.Client(timeout=10) as c:
            r = c.get(f"{API}/dashboard/api/contention")
            r.raise_for_status()
            data = r.json()
        per = data.get("per_role_scheduling", data.get("per_role", {})) or {}
        rec = per.get(role, {})
        return {
            "migrations_started": int(rec.get("migrations_started", 0)),
            "reverse_migrations": int(rec.get("reverse_migrations", -1)),  # -1 = not surfaced
            "migration_failures": int(rec.get("migration_failures", 0)),
        }
    except Exception as e:
        return {"error": str(e)}


def _chat(role: str, session_id: str, timeout: float = 90.0) -> dict:
    payload = {
        "prompt": PROMPT,
        "force_role": role,
        "max_turns": 4,
        "cache_prompt": False,
        "session_id": session_id,
    }
    t0 = time.time()
    try:
        with httpx.Client(timeout=timeout) as c:
            r = c.post(f"{API}/chat", json=payload)
        ans = ""
        try:
            ans = (r.json().get("answer") or r.json().get("response") or "")[:60]
        except Exception:
            ans = f"<status {r.status_code}>"
        return {"session": session_id, "elapsed_s": round(time.time() - t0, 2), "answer": ans}
    except Exception as e:
        return {"session": session_id, "elapsed_s": round(time.time() - t0, 2), "error": str(e)}


def forward(role: str, iters: int) -> dict:
    """A→(complete)→B handover; repeated to beat any concurrent full-occupancy from J6."""
    before = _counts(role)
    print(f"  [forward] baseline migrations_started={before.get('migrations_started')}")
    for i in range(iters):
        a = f"j23-fwd-A-{i}-{int(time.time()*1000)}"
        b = f"j23-fwd-B-{i}-{int(time.time()*1000)}"
        _chat(role, a)               # A lands on full, completes → full idle
        _chat(role, b)               # B (new) arrives → should migrate A → quarter
        _chat(role, a)               # A turn 2 → should route to migrated quarter
        cur = _counts(role).get("migrations_started", "?")
        print(f"    iter {i}: migrations_started={cur}")
    after = _counts(role)
    delta = after.get("migrations_started", 0) - before.get("migrations_started", 0)
    return {"leg": "forward", "before": before, "after": after, "migrations_delta": delta}


def reverse(role: str, burst: int, iters: int) -> dict:
    """Spill onto quarters (concurrency>safe-slots) then drop to 1 + wait cooldown."""
    before = _counts(role)
    print(f"  [reverse] baseline reverse_migrations={before.get('reverse_migrations')}")
    sid = f"j23-rev-{int(time.time()*1000)}"
    for i in range(iters):
        # burst: many concurrent requests → spill to quarters (one keeps `sid` warm)
        with ThreadPoolExecutor(max_workers=burst) as ex:
            futs = [ex.submit(_chat, role, sid if k == 0 else f"{sid}-bg{k}") for k in range(burst)]
            [f.result() for f in futs]
        # drop to single-flight + let reverse cooldown (default 2s) + window elapse
        time.sleep(4)
        _chat(role, sid)  # warm session, low load → should reverse-migrate back to full
        cur = _counts(role).get("reverse_migrations", "?")
        print(f"    iter {i}: reverse_migrations={cur}")
    after = _counts(role)
    rv_before = before.get("reverse_migrations", -1)
    rv_after = after.get("reverse_migrations", -1)
    delta = (rv_after - rv_before) if (rv_before >= 0 and rv_after >= 0) else None
    return {"leg": "reverse", "before": before, "after": after, "reverse_delta": delta,
            "note": "reverse_migrations=-1 means telemetry not surfaced (restart API past 3ec218f)"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="frontdoor")
    ap.add_argument("--leg", choices=["forward", "reverse", "both"], default="forward")
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--burst", type=int, default=5)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if "error" in _counts(args.role):
        print(f"[migration_probe] dashboard unreachable: {_counts(args.role)['error']}", file=sys.stderr)
        return 2

    out: dict = {"role": args.role, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if args.leg in ("forward", "both"):
        out["forward"] = forward(args.role, args.iters)
    if args.leg in ("reverse", "both"):
        out["reverse"] = reverse(args.role, args.burst, args.iters)

    print(json.dumps(out, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
