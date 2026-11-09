#!/usr/bin/env python3
"""TOC-CMP-4 option (a): point draw, seeded arm permutation, and (point, arm, repeat) runner.

Implements ``docs/design/toc-cmp-4-preregistration-20261109.md`` (root repo). Zero inference:
``draw`` and ``plan`` are pure functions of the pool and the seed; ``run_schedule`` takes an injected
``submit`` callable, so running against the live stack is a separate, later step.

Determinism: every choice is a sha256 ordering (never ``random``), so the point list and arm
permutations are identical across Python versions and re-runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

ARMS_A = ("summary_only", "pointer_only", "prune_pointer")  # pre-registered, option (a)
# Boundary-turn bands per stratum (assumption recorded in the receipt: the pre-registration fixes the
# strata "by the point list"; absolute turn bands stand in for terciles until trajectory lengths are
# measured by the pilot).
STRATA = {"early": (2, 3), "middle": (4, 5, 6), "late": (7, 8, 9)}
DEFAULT_SEED = 20261109
POINTS_PER_STRATUM = 10
REPEATS = 8
REQUIRED_PINS = (
    "model_pin", "build", "sampling_config", "region",
    "ORCHESTRATOR_CONTEXT_BUDGET_TOKENS", "ORCHESTRATOR_CONTEXT_TRIGGER_FRACTION", "prompt_hashes",
)


def _h(*parts: object) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


def draw_points(
    pool_ids: Iterable[str],
    *,
    seed: int = DEFAULT_SEED,
    per_stratum: int = POINTS_PER_STRATUM,
    strata: Mapping[str, Sequence[int]] = STRATA,
) -> list[dict[str, Any]]:
    """Draw ``per_stratum`` points per stratum: tasks without replacement within a stratum."""
    ids = sorted(set(pool_ids))
    if len(ids) < per_stratum:
        raise ValueError(f"pool of {len(ids)} tasks is smaller than per_stratum={per_stratum}")
    points: list[dict[str, Any]] = []
    for stratum, turns in strata.items():
        chosen = sorted(ids, key=lambda t: _h(seed, "task", stratum, t))[:per_stratum]
        for task in chosen:
            turn = sorted(turns)[int(_h(seed, "turn", stratum, task), 16) % len(turns)]
            points.append(
                {"point_id": f"{task}@{turn}", "task_id": task, "boundary_turn": turn, "stratum": stratum}
            )
    return points


def points_sha256(points: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(list(points), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def arm_permutation(point_id: str, arms: Sequence[str] = ARMS_A, *, seed: int = DEFAULT_SEED) -> list[str]:
    return sorted(arms, key=lambda a: _h(seed, "arm", point_id, a))


def build_schedule(
    points: Sequence[Mapping[str, Any]],
    *,
    arms: Sequence[str] = ARMS_A,
    repeats: int = REPEATS,
    seed: int = DEFAULT_SEED,
    budget_tokens: int = 16384,
    trigger_fraction: float = 0.5,
) -> list[dict[str, Any]]:
    """One row per (point, arm, repeat). Repeats differ only in sampling seed; arm order within a
    (point, repeat) is the point's seeded permutation."""
    rows: list[dict[str, Any]] = []
    for p in points:
        order = arm_permutation(p["point_id"], arms, seed=seed)
        for repeat in range(repeats):
            for pos, arm in enumerate(order):
                rows.append(
                    {
                        "run_id": f"{p['point_id']}|{arm}|r{repeat}",
                        "point_id": p["point_id"],
                        "task_id": p["task_id"],
                        "stratum": p["stratum"],
                        "arm": arm,
                        "repeat": repeat,
                        "arm_position": pos,
                        "sampling_seed": int(_h(seed, "sample", p["point_id"], repeat)[:8], 16),
                        "env": {
                            "ORCHESTRATOR_CONTEXT_ARM_AB": "1",
                            "ORCHESTRATOR_CONTEXT_ARM": arm,
                            "ORCHESTRATOR_CONTEXT_FORCE_TURNS": str(p["boundary_turn"]),
                            "ORCHESTRATOR_CONTEXT_TRIAL_INDEX": str(repeat),
                            "ORCHESTRATOR_CONTEXT_BUDGET_TOKENS": str(budget_tokens),
                            "ORCHESTRATOR_CONTEXT_TRIGGER_FRACTION": str(trigger_fraction),
                        },
                    }
                )
    return rows


def missing_pins(result: Mapping[str, Any]) -> list[str]:
    pins = result.get("pins") if isinstance(result.get("pins"), Mapping) else {}
    return [k for k in REQUIRED_PINS if pins.get(k) in (None, "", {})]


def run_schedule(
    schedule: Sequence[Mapping[str, Any]],
    submit: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    ledger_path: Path,
) -> dict[str, int]:
    """Run rows not yet in the ledger. Invalid-run policy (pre-reg section 7): a transport failure
    or missing pin is re-run once with the same seed; a second failure drops the row and is counted."""
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if ledger_path.exists():
        done = {json.loads(l)["run_id"] for l in ledger_path.read_text().splitlines() if l.strip()}
    counts = {"ok": 0, "rerun": 0, "dropped": 0, "skipped_done": 0}
    for row in schedule:
        if row["run_id"] in done:
            counts["skipped_done"] += 1
            continue
        status, result, attempts = "dropped", {}, 0
        for attempts in (1, 2):
            try:
                result = dict(submit(row))
                bad = missing_pins(result)
                if not bad:
                    status = "ok"
                    break
                result = {"invalid_reason": f"missing_pins:{bad}"}
            except Exception as exc:  # transport failure
                result = {"invalid_reason": f"{type(exc).__name__}: {exc}"}
            if attempts == 1:
                counts["rerun"] += 1
        counts["ok" if status == "ok" else "dropped"] += 1
        with ledger_path.open("a") as fh:
            fh.write(json.dumps({"run_id": row["run_id"], "status": status, "attempts": attempts,
                                 "result": result}, sort_keys=True) + "\n")
    return counts


def _pool_from_core(path: Path) -> list[str]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return [r["id"] for r in rows if "id" in r and not r.get("__core_metadata__")]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw", help="draw and freeze the 30 points + receipt")
    d.add_argument("--pool-jsonl", type=Path, required=True, help="core_v2-style JSONL with an 'id' field")
    d.add_argument("--seed", type=int, default=DEFAULT_SEED)
    d.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("plan", help="write the (point, arm, repeat) schedule; runs nothing")
    p.add_argument("--points", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=REPEATS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "draw":
        pool = _pool_from_core(a.pool_jsonl)
        points = draw_points(pool, seed=a.seed)
        receipt = {
            "preregistration": "docs/design/toc-cmp-4-preregistration-20261109.md",
            "seed": a.seed, "arms": list(ARMS_A), "strata_turn_bands": {k: list(v) for k, v in STRATA.items()},
            "pool_size": len(set(pool)), "pool_source": str(a.pool_jsonl),
            "points_sha256": points_sha256(points), "points": points,
            "arm_permutations": {q["point_id"]: arm_permutation(q["point_id"], seed=a.seed) for q in points},
        }
        a.out.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
        print(receipt["points_sha256"])
        return 0
    receipt = json.loads(a.points.read_text())
    if points_sha256(receipt["points"]) != receipt["points_sha256"]:
        print("points file does not match its frozen sha256", file=sys.stderr)
        return 3
    rows = build_schedule(receipt["points"], repeats=a.repeats, seed=receipt["seed"])
    a.out.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    print(len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
