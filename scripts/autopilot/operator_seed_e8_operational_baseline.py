#!/usr/bin/env python3
"""OPERATOR ACTION — seed an OPERATIONAL E8 quality baseline and stamp the era.

WHY THIS EXISTS
---------------
AutoPilot cannot promote quality while the resident baseline's `eval_quality_era`
differs from the active era. That is a CIRCULAR dependency, not a missing
measurement: `update_baseline()` refuses promotion while the eras differ, and the
era field is only rewritten inside a *successful* promotion. Nothing inside the
loop can break it. Something outside must write a fresh baseline that is ALREADY
stamped with the active era.

`autopilot.py calibrate-baseline` produces the values (`baselines_by_tier`,
`per_suite_quality_by_tier`, `per_suite_counts_by_tier`) but
`_apply_calibrated_baseline_result()` never touches `eval_quality_era`, so on its
own it leaves the hold exactly where it was. This script closes that one gap.

WHAT THIS IS, AND IS NOT
------------------------
This produces an OPERATIONAL baseline for config search: the number AutoPilot
ratchets against so it can promote better router/harness configurations.

It is NOT the ratification-grade E8 protocol baseline. There is no operator
receipt, no `run_seal.json`, no three-repetition replay, no T2 vector. Do not
cite a number derived from this baseline in an external claim. The heavyweight
path (`run_e8_quality_baseline_reseed.py` + `apply_e8_quality_baseline_state.py`)
remains available and is unaffected by this.

WHY IT IS AN OPERATOR ACTION
----------------------------
`e8_quality_rebaseline.required_next_action` reads "human-only E8 baseline value
reseed after fresh evidence", and MEASUREMENT.md is human-amendment-only. Deciding
that a fresh measurement IS the E8 baseline is a measurement-trust decision. An
agent must not make it; this script exists so a human can make it in one command.

SAFETY GATES (all fail-closed, all before anything is written)
--------------------------------------------------------------
1.  The API must be healthy AND actually generating. On 2026-08-03 a calibration
    ran 70/100 questions at "0% correct" purely because the API was down and the
    eval scored every `Connection refused` as WRONG. `health: ok` alone is not
    evidence; this script demands a correct answer to a known question.
2.  AutoPilot must not be mid-trial. Its `structural_experiment` trials restart the
    API to apply flag changes, and a calibration racing that will measure noise.
3.  The measured result must be SANE: quality > 0, reliability >= the floor, and a
    non-empty question count. A 0.000 baseline is refused, loudly. This is the gate
    that would have caught the 2026-08-03 incident.
4.  The state file is backed up before the write, and the write is atomic
    (temp file + os.replace).

Usage:
    .venv/bin/python scripts/autopilot/operator_seed_e8_operational_baseline.py --plan
    .venv/bin/python scripts/autopilot/operator_seed_e8_operational_baseline.py --apply
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AUTOPILOT_DIR = REPO_ROOT / "scripts" / "autopilot"
for _p in (str(REPO_ROOT), str(AUTOPILOT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

STATE_PATH = REPO_ROOT / "orchestration" / "autopilot_state.json"
API = "http://127.0.0.1:8000"
MIN_RELIABILITY = 0.8


_RESUME_ON_EXIT = False


def _fail(msg: str) -> None:
    """Refuse, and never leave AutoPilot paused because we bailed out."""
    print(f"\n  REFUSED: {msg}")
    if _RESUME_ON_EXIT:
        print("  resuming AutoPilot (this script paused it)...")
        try:
            _autopilot_cmd("resume")
            print("  AutoPilot resumed.")
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: could not resume AutoPilot ({exc}) — resume it by hand.")
    sys.exit(1)


def _api_is_generating() -> tuple[bool, str]:
    """health:ok is NOT enough — demand a correct answer to a known question."""
    try:
        with urllib.request.urlopen(f"{API}/health", timeout=10) as r:
            if r.status != 200:
                return False, f"/health returned {r.status}"
    except Exception as exc:  # noqa: BLE001
        return False, f"/health unreachable: {exc}"

    payload = {
        "model": "frontdoor",
        "messages": [{"role": "user", "content": "What is 2+2? Reply with only the number."}],
        "max_tokens": 8,
        "temperature": 0,
    }
    req = urllib.request.Request(
        f"{API}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            body = json.loads(r.read())
        answer = (body["choices"][0]["message"]["content"] or "").strip()
    except Exception as exc:  # noqa: BLE001
        return False, f"generation failed: {exc}"
    if "4" not in answer:
        return False, f"API answered {answer!r} to 2+2 — not trustworthy"
    return True, answer


def _load_state() -> dict:
    with open(STATE_PATH) as fh:
        return json.load(fh)


def _autopilot_is_running() -> bool:
    """True if a live `autopilot.py start` process exists.

    Deliberately matched on the exact argv shape, never a bare name pattern: this is a
    shared host and a name pattern is a wildcard over other sessions' processes
    (INC-20260731-broad-process-pattern-kills).
    """
    import subprocess  # noqa: PLC0415

    try:
        out = subprocess.run(
            ["ps", "-eo", "args"], capture_output=True, text=True, timeout=20
        ).stdout
    except Exception:  # noqa: BLE001
        return True  # cannot prove it is down -> assume it is up and refuse. Fail closed.
    return any(
        "autopilot.py" in ln and " start" in ln and "operator_seed" not in ln
        for ln in out.splitlines()
    )


def _autopilot_cmd(sub: str) -> None:
    import subprocess  # noqa: PLC0415

    subprocess.run(
        [sys.executable, str(AUTOPILOT_DIR / "autopilot.py"), sub],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300,
    )


def _wait_for_trial_boundary(timeout_s: int = 3600) -> bool:
    """Wait until AutoPilot records paused and no in-flight trial."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        st = _load_state()
        if st.get("paused") and not st.get("in_flight_trial"):
            return True
        time.sleep(20)
    return False


def _active_era(state: dict) -> str:
    era = ((state.get("active_instrument_eras") or {}).get("eval_quality") or "").strip()
    if not era:
        _fail("state has no active_instrument_eras.eval_quality — nothing to stamp")
    return era


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true", help="read-only: show what would happen")
    mode.add_argument("--apply", action="store_true", help="measure, then persist values + era stamp")
    ap.add_argument("--tier", type=int, default=1, help="eval tier to calibrate (default 1 — AutoPilot's frontier tier)")
    ap.add_argument("--n", type=int, default=None, help="question count override")
    ap.add_argument(
        "--handle-autopilot",
        action="store_true",
        help="pause AutoPilot at a trial boundary, measure, then resume it. Required when AutoPilot is live.",
    )
    args = ap.parse_args()
    resumed = False

    state = _load_state()
    baseline_state = state.get("baseline_state") or {}
    active = _active_era(state)
    resident = (baseline_state.get("eval_quality_era") or "").strip() or "<pre-boundary>"
    hold = (state.get("e8_quality_rebaseline") or {}).get("status")

    print("=" * 68)
    print("  OPERATIONAL E8 QUALITY BASELINE SEED")
    print("=" * 68)
    print(f"  state file        : {STATE_PATH}")
    print(f"  resident era      : {resident}")
    print(f"  active era        : {active}")
    print(f"  rebaseline hold   : {hold}")
    print(f"  tier to calibrate : T{args.tier}")
    print(f"  current baselines : {baseline_state.get('baselines_by_tier')}")
    psq = baseline_state.get("per_suite_quality") or {}
    unmeasured = [k for k, v in psq.items() if v is None]
    if unmeasured:
        print(f"  per-suite NULL    : {len(unmeasured)} of {len(psq)} suites carry no baseline")

    if resident == active:
        print("\n  Nothing to do — resident era already equals the active era.")
        return 0

    ok, detail = _api_is_generating()
    print(f"  API generating    : {'YES' if ok else 'NO'} ({detail})")
    if not ok:
        _fail(
            "the API is not generating. A calibration against a dead API scores every "
            "question WRONG and would write a 0.000 baseline (this happened 2026-08-03)."
        )

    # HARD GATE — not a warning. A calibration racing AutoPilot measures under its load,
    # and AutoPilot's structural_experiment trials RESTART THE API, which scores every
    # remaining question as WRONG. This was a warning-and-proceed on the first version;
    # a gate that warns and continues is not a gate.
    # HARD GATE — the AutoPilot DAEMON must be STOPPED, not merely paused.
    #
    # 2026-08-03, learned the expensive way: a 59-minute measurement was written
    # correctly and then silently lost. `save_state(merge_control=True)` — the daemon's
    # periodic / trial-end / lifecycle save — re-reads ONLY the out-of-band control
    # fields (`_EXTERNAL_CONTROL_FIELDS`: paused, pause_reason, _in_cache_flush) from
    # disk and keeps its OWN in-memory copy of everything else. `baseline_state` is
    # daemon-owned, so the daemon's next save overwrites any external edit to it from
    # memory. `autopilot.py:6003-6005` states the discipline outright: "atomic
    # os.replace stops torn reads but NOT lost updates."
    #
    # Therefore PAUSING IS NOT ENOUGH, and neither is taking the write lock: the lock
    # serialises writes, it does not stop the daemon from later writing a stale copy.
    # The only safe window is with the daemon absent.
    running = _autopilot_is_running()
    if running:
        _fail(
            "the AutoPilot DAEMON is still running. Pausing is NOT sufficient.\n"
            "           `baseline_state` is daemon-owned: its next save(merge_control=True)\n"
            "           re-reads only the control fields and rewrites baseline_state from its\n"
            "           OWN memory, silently discarding anything written here. On 2026-08-03\n"
            "           this destroyed a 59-minute measurement that had already been applied.\n\n"
            "           Stop the daemon (supervisor FIRST, or it will restart the child):\n"
            "             kill -TERM <supervisor-pid> && kill -TERM <autopilot-pid>\n"
            "           verify both are gone, run this script, then start AutoPilot again with\n"
            "             scripts/autopilot/start_authority_daemon.py\n"
            "           A fresh daemon loads state from disk and will see the new era."
        )

    if args.plan:
        print("\n  PLAN (nothing written):")
        print(f"    1. run EvalTower T{args.tier} and compute fresh quality / per-suite / counts")
        print("    2. refuse if quality <= 0, reliability < %.2f, or n_questions == 0" % MIN_RELIABILITY)
        print(f"    3. back up {STATE_PATH.name}, then atomically write:")
        print(f"         baseline_state.baselines_by_tier[{args.tier}]      <- measured quality")
        print(f"         baseline_state.per_suite_quality_by_tier[{args.tier}] <- measured per-suite")
        print(f"         baseline_state.per_suite_counts_by_tier[{args.tier}]  <- measured counts")
        print(f"         baseline_state.eval_quality_era                    <- {active!r}")
        print("         e8_quality_rebaseline.status                       <- 'closed_operational'")
        print("\n    Re-run with --apply to execute.")
        return 0

    # ---- measure -----------------------------------------------------------
    import autopilot as ap_mod  # noqa: PLC0415

    print(f"\n  Running T{args.tier} calibration (this takes a while; no output until done)...")
    started = time.time()
    baseline, result, _migrated = ap_mod.calibrate_baseline(
        tier=args.tier, n=args.n, write=False
    )
    if result is None:
        _fail("calibration returned no result")
    elapsed = time.time() - started
    print(f"  measured in {elapsed/60:.1f} min:")
    print(f"    quality     = {result.quality:.4f}")
    print(f"    reliability = {result.reliability:.4f}")
    print(f"    n_questions = {result.n_questions}")

    # ---- sanity gate (the one that would have caught 2026-08-03) -----------
    if result.quality <= 0:
        _fail(f"measured quality {result.quality!r} is not positive — this is an instrument failure, not a baseline")
    if result.n_questions <= 0:
        _fail("measured zero questions")
    if result.reliability < MIN_RELIABILITY:
        _fail(
            f"reliability {result.reliability:.3f} < {MIN_RELIABILITY} — the eval evidence is "
            "untrustworthy (infra errors); refusing to enshrine it as a baseline"
        )

    # ---- write -------------------------------------------------------------
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = STATE_PATH.with_suffix(f".json.pre-operational-e8-{stamp}")
    shutil.copy2(STATE_PATH, backup)
    print(f"  backup written: {backup.name}")

    state = _load_state()  # re-read: minimise the window
    bs = baseline.to_state_dict()
    bs["eval_quality_era"] = active
    state["baseline_state"] = bs
    state["e8_quality_rebaseline"] = {
        **(state.get("e8_quality_rebaseline") or {}),
        "status": "closed_operational",
        "closed_at": datetime.now(timezone.utc).isoformat(),
        "closed_by": "operator via operator_seed_e8_operational_baseline.py",
        "basis": (
            f"Operational T{args.tier} calibration: quality={result.quality:.4f} "
            f"reliability={result.reliability:.4f} n={result.n_questions}. "
            "NOT the ratification-grade E8 protocol baseline — no receipt, no run_seal, "
            "no 3-repetition replay, no T2 vector. Sufficient for AutoPilot config-search "
            "promotion; not citable as an external measurement claim."
        ),
    }

    # Write under the repo's cross-process H4 lock. The daemon is already required to be
    # absent (see the gate above) — this is defence in depth against the dashboard,
    # host_health and config_applicator, which also write this file.
    from state_lock import state_write_lock  # noqa: PLC0415

    with state_write_lock(STATE_PATH):
        tmp = STATE_PATH.with_suffix(".json.tmp")
        with open(tmp, "w") as fh:
            json.dump(state, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, STATE_PATH)

    # VERIFY THE WRITE SURVIVED. The whole reason this script exists in its current form
    # is that a previous run printed APPLIED for a write that was gone moments later.
    check = _load_state()
    got = ((check.get("baseline_state") or {}).get("eval_quality_era") or "").strip()
    if got != active:
        _fail(
            f"write did not survive: eval_quality_era reads {got!r}, expected {active!r}. "
            "Something else is writing this file — do not retry until it is stopped."
        )

    print("\n  APPLIED.")
    print(f"    eval_quality_era : {resident}  ->  {active}")
    print(f"    baselines_by_tier: {baseline.baselines_by_tier}")
    print(f"    hold             : {hold}  ->  closed_operational")
    print(f"  To undo: cp {backup} {STATE_PATH}")

    if resumed:
        print("\n  resuming AutoPilot (this script paused it)...")
        _RESUME_ON_EXIT = False  # noqa: F841 — past the failure paths
        _autopilot_cmd("resume")
        st = _load_state()
        print(f"  AutoPilot paused={st.get('paused')} — update_baseline() will now promote quality.")
    else:
        print("\n  Resume AutoPilot; update_baseline() will now promote quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
