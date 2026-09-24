#!/usr/bin/env python3
"""RTG-09 — derive per-role wall-clock task-duration baselines from progress logs.

WHY THIS EXISTS
---------------
`compute_reward` (`orchestration/repl_memory/q_reward.py`) prices the speed axis
purely off tokens/sec (`baseline_tps_by_role`). Per the DAR handoff
(`handoffs/active/decision-aware-routing.md`, "The speed axis must be
wall-clock, not tokens/sec"), that is gameable through tools and measurably
blind: e.g. `worker_vision` spends ~0.4s of model compute inside ~12s of wall
clock — a tokens/sec term scores it as fast, a task-execution-speed term does
not. DAR-5 is gated on this reward carrying a real wall-clock speed axis.

This script is the derivation, not a hand-set constant. It replays the
retained `logs/progress/*.jsonl` (append-only, already-happened events — zero
inference, zero compute beyond a jsonl scan) and computes, per role, the
empirical p50/p90 wall-clock task duration:

    wall_clock_s = task_completed.timestamp - task_started.timestamp

matched by `task_id`, exactly the pairing method already used by
`rescore_rewards_from_progress.py` (which reported "20,516 task_started /
task_completed pairs, 85 days" as of 2026-07-21).

OUTPUT
------
A checked-in JSON baseline (`orchestration/derived/duration_baselines_by_role.json`)
stamped with:
  - protocol_id: an ANALYSIS id for this derivation (see note below — this is
    NOT a MEASUREMENT.md-registered protocol; it exists so the baseline is
    reproducible and versioned rather than a magic number in code).
  - generated_at_utc, source_window (file range + count), n_pairs_total,
    n_pairs_unmatched (task_completed with no matching task_started),
    n_pairs_excluded_role (matched pairs whose role failed the role filter).
  - roles: {role: {n, p50_s, p90_s, mean_s}} for every role clearing
    MIN_N_PER_ROLE observations.

Role filter: keeps only `producer_role` (falling back to `final_answer_role`)
values that look like a real role identifier (`^[a-z][a-z0-9_]*$`) and are not
a known pipeline sentinel / synthetic role (`mock`, `plan_review`, and other
non-identifier junk from the pre-validation era — see DAR handoff "Consider
splitting a separate `stage` field" note). `architect_coding` is retained in
the OUTPUT (it clears the identifier filter and has real historical rows) but
callers should treat it like `baseline_tps_by_role` already does: it is a
deprecated role absent from the live registry, so `compute_reward`'s
"role not in config.X" guard will simply skip the duration term for it if the
live ScoringConfig does not carry it forward.

MEASUREMENT NOTE (MEASUREMENT.md)
----------------------------------
This is a REPLAY over already-recorded wall-clock events, not a new bench
run, and it does not by itself gate any keep/revert/deploy/promote decision —
it sizes a reward-shaping term in an offline scorer. It is filed the same way
`dar_common.py` and `rescore_rewards_from_progress.py` file their derived
numbers: reproducible, source-window-stamped, explicitly NOT a MEASUREMENT.md
protocol claim. If a future decision wants to CITE this baseline as gating
evidence, it must go through the constitution's claim grammar at that time.

Usage:
    python scripts/analysis/derive_duration_baselines.py \
        --out orchestration/derived/duration_baselines_by_role.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_LOG_DIR = REPO_ROOT / "logs" / "progress"
DEFAULT_OUT_PATH = REPO_ROOT / "orchestration" / "derived" / "duration_baselines_by_role.json"

# Versioned analysis id for this derivation. Bump the trailing integer (v2, v3,
# ...) whenever the derivation METHOD changes (matching rule, filters,
# percentile definition); re-running against a wider log window under the same
# method keeps the same id but updates source_window/n/generated_at.
PROTOCOL_ID = "RTG09-DURATION-BASELINE-v1"

# Below this many matched pairs, a role's p50/p90 is too noisy to trust as a
# reward-shaping constant; the role is reported under `excluded_low_n` instead
# of `roles` and the caller must fall back to no-duration-penalty for it.
MIN_N_PER_ROLE = 30

# A real role identifier: lowercase snake_case, starts with a letter. Filters
# out the pipeline sentinels and pre-validation-era garbage found in
# producer_role (empty string, "mock", "plan_review", "WORKER", "SELF", and
# literal prompt text that leaked into the field before the Role.from_string
# validation fix — DAR handoff "Reward-integrity follow-up — 2026-07-21").
_ROLE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_SENTINEL_ROLES = frozenset({
    "mock", "plan_review", "plan", "stream_init", "proactive_delegation",
})


def _parse_ts(value) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _role_of(data: dict) -> str:
    role = data.get("producer_role") or data.get("final_answer_role") or ""
    return role if isinstance(role, str) else ""


def _valid_role(role: str) -> bool:
    return bool(role) and role not in _SENTINEL_ROLES and bool(_ROLE_RE.match(role))


def derive(log_dir: Path) -> dict:
    files = sorted(log_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no progress logs under {log_dir}")

    starts: Dict[str, datetime] = {}
    durations_by_role: Dict[str, List[float]] = {}
    n_pairs_total = 0
    n_unmatched = 0
    n_excluded_role = 0
    n_negative_or_zero = 0
    malformed = 0

    for path in files:
        with path.open(errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                etype = event.get("event_type")
                task_id = event.get("task_id")
                if not task_id:
                    continue
                if etype == "task_started":
                    started = _parse_ts(event.get("timestamp"))
                    if started:
                        starts[task_id] = started
                    continue
                if etype != "task_completed":
                    continue

                completed = _parse_ts(event.get("timestamp"))
                started = starts.get(task_id)
                if not completed or not started:
                    n_unmatched += 1
                    continue

                wall_s = (completed - started).total_seconds()
                n_pairs_total += 1
                if wall_s <= 0:
                    # Clock skew / mock instant-completion rows are not a real
                    # duration observation; excluded from both role bucketing
                    # and totals for percentile purposes but tallied for audit.
                    n_negative_or_zero += 1
                    continue

                data = event.get("data") or {}
                role = _role_of(data)
                if not _valid_role(role):
                    n_excluded_role += 1
                    continue

                durations_by_role.setdefault(role, []).append(wall_s)

    roles_out: Dict[str, dict] = {}
    excluded_low_n: Dict[str, int] = {}
    for role, vals in sorted(durations_by_role.items()):
        n = len(vals)
        if n < MIN_N_PER_ROLE:
            excluded_low_n[role] = n
            continue
        vals_sorted = sorted(vals)
        p50 = statistics.median(vals_sorted)
        # statistics.quantiles needs n>=2; MIN_N_PER_ROLE guards that.
        p90 = statistics.quantiles(vals_sorted, n=100, method="inclusive")[89]
        roles_out[role] = {
            "n": n,
            "p50_s": round(p50, 3),
            "p90_s": round(p90, 3),
            "mean_s": round(statistics.mean(vals_sorted), 3),
        }

    result = {
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": (
            "wall_clock_s = task_completed.timestamp - task_started.timestamp, "
            "matched by task_id; role = producer_role or final_answer_role; "
            "p50/p90 over matched, positive-duration, valid-role pairs."
        ),
        "source_window": {
            "log_dir": str(log_dir),
            "first_file": files[0].name,
            "last_file": files[-1].name,
            "n_files": len(files),
        },
        "n_pairs_total": n_pairs_total,
        "n_pairs_unmatched_no_start": n_unmatched,
        "n_pairs_excluded_nonpositive_duration": n_negative_or_zero,
        "n_pairs_excluded_invalid_role": n_excluded_role,
        "n_malformed_lines_skipped": malformed,
        "min_n_per_role": MIN_N_PER_ROLE,
        "roles": roles_out,
        "excluded_low_n_roles": excluded_low_n,
    }
    return result


def _report(result: dict) -> None:
    print(f"protocol_id             : {result['protocol_id']}")
    win = result["source_window"]
    print(f"source window            : {win['first_file']} .. {win['last_file']} "
          f"({win['n_files']} files)")
    print(f"pairs total / unmatched  : {result['n_pairs_total']:,} / "
          f"{result['n_pairs_unmatched_no_start']:,}")
    print(f"excluded (<=0s duration) : {result['n_pairs_excluded_nonpositive_duration']:,}")
    print(f"excluded (invalid role)  : {result['n_pairs_excluded_invalid_role']:,}")
    print(f"malformed lines skipped  : {result['n_malformed_lines_skipped']:,}")
    print(f"\nper-role wall-clock duration (n >= {result['min_n_per_role']}):")
    print(f"  {'role':<22} {'n':>8} {'p50_s':>10} {'p90_s':>10} {'mean_s':>10}")
    for role, stats in sorted(result["roles"].items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {role:<22} {stats['n']:>8,} {stats['p50_s']:>10.2f} "
              f"{stats['p90_s']:>10.2f} {stats['mean_s']:>10.2f}")
    if result["excluded_low_n_roles"]:
        print("\nroles below MIN_N_PER_ROLE (excluded from baseline):")
        for role, n in sorted(result["excluded_low_n_roles"].items()):
            print(f"  {role:<22} n={n}")
    print(
        "\nMEASUREMENT: this is a replay-derived OBSERVATION (dar_common.py / "
        "rescore_rewards_from_progress.py convention), not a MEASUREMENT.md "
        "protocol claim. It sizes a reward-shaping constant; it does not gate "
        "a keep/revert/deploy/promote decision by itself."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    args = ap.parse_args()

    result = derive(args.log_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        json.dump(result, fh, indent=2, sort_keys=False)
        fh.write("\n")

    _report(result)
    print(f"\nartifact: {args.out}")


if __name__ == "__main__":
    main()
