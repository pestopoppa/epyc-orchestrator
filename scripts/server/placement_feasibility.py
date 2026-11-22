#!/usr/bin/env python3
"""Derived placement-feasibility artifact (handoff box W-40).

Pure cpuset intersection over the shape table in ``stack_numa.py`` — NEVER
measured. Two shapes may co-run iff their cpusets are disjoint (raw logical
CPUs AND SMT-folded physical cores, since an SMT sibling shares a core); any
intersection means exclusive. The artifact is deterministic (no timestamp) so
``--check`` is a byte-level comparison of the regenerated content, and it is
regenerated on every topology change:

    python scripts/server/placement_feasibility.py            # write
    python scripts/server/placement_feasibility.py --check    # exit 0 fresh / 2 stale|missing

This is what the orchestrator pre-commit gate consults
(``scripts/validate/check_contention_matrix_fresh.py``); the benchmarked
contention matrix no longer gates commits on placement-feasibility grounds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

_THIS = Path(__file__).resolve()
REPO_ROOT = _THIS.parents[2]
sys.path.insert(0, str(_THIS.parent))

ARTIFACT_PATH = REPO_ROOT / "data" / "contention_matrix" / "placement_feasibility.json"
SCHEMA = "epyc.placement_feasibility.v1"
PHYSICAL_CORES = 96  # logical cpu c and c+96 are SMT siblings of one physical core


def parse_cpus(spec: str) -> set[int]:
    out: set[int] = set()
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


def fold_physical(cpus: set[int]) -> set[int]:
    return {c % PHYSICAL_CORES for c in cpus}


def pair_verdict(spec_a: str, spec_b: str) -> dict:
    a, b = parse_cpus(spec_a), parse_cpus(spec_b)
    raw = a & b
    phys = fold_physical(a) & fold_physical(b)
    co_run = not raw and not phys
    return {
        "verdict": "disjoint" if co_run else "intersecting",
        "may_co_run": co_run,
        "shared_physical_cores": len(phys),
    }


def _shape_table(stack_numa) -> dict[str, str]:
    """shape name -> cpuset, for every shape class the module declares."""
    return {name: getattr(stack_numa, name)[0] for name in sorted(stack_numa._SHAPE_CLASSES)}


def build_artifact(stack_numa=None) -> dict:
    if stack_numa is None:
        import stack_numa  # type: ignore
    shapes = _shape_table(stack_numa)
    classes = {n: stack_numa._SHAPE_CLASSES[n] for n in shapes}
    pairs = []
    for x, y in combinations(sorted(shapes), 2):
        pairs.append({"a": x, "b": y, **pair_verdict(shapes[x], shapes[y])})
    roles = {
        role: [str(inst[0]) for inst in cfg.get("instances", [])]
        for role, cfg in sorted(stack_numa.NUMA_CONFIG.items())
    }
    source = {"shapes": shapes, "roles": roles}
    topo_hash = hashlib.sha256(
        json.dumps(source, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema": SCHEMA,
        "derived_from": "scripts/server/stack_numa.py",
        "measured": False,
        "rule": "disjoint cpusets (raw and SMT-folded) may co-run; any intersection is exclusive",
        "topology_hash": topo_hash,
        "shapes": {n: {"cpuset": shapes[n], "class": classes[n]} for n in shapes},
        "pairs": pairs,
        "role_instance_cpusets": roles,
    }


def render(artifact: dict) -> str:
    return json.dumps(artifact, indent=2, sort_keys=True) + "\n"


def check(path: Path = ARTIFACT_PATH, stack_numa=None) -> tuple[int, str]:
    """0 fresh; 2 missing/stale; 3 unparseable."""
    if not path.exists():
        return 2, f"MISSING {path}; run: python scripts/server/placement_feasibility.py"
    try:
        on_disk = json.loads(path.read_text())
    except Exception as exc:
        return 3, f"INVALID {path}: {exc}"
    want = build_artifact(stack_numa)
    if on_disk != want:
        return 2, (
            f"STALE {path} (on-disk topology_hash={str(on_disk.get('topology_hash'))[:8]}, "
            f"stack_numa.py={want['topology_hash'][:8]}); "
            "run: python scripts/server/placement_feasibility.py"
        )
    return 0, f"OK placement feasibility is current (topology_hash={want['topology_hash'][:8]})"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true")
    p.add_argument("--out", type=Path, default=ARTIFACT_PATH)
    args = p.parse_args(argv)
    if args.check:
        rc, msg = check(args.out)
        print(msg, file=sys.stderr if rc else sys.stdout)
        return rc
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(build_artifact()))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
