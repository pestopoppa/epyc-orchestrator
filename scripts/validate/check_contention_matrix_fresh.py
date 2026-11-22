#!/usr/bin/env python3
"""Pre-commit gate: is the DERIVED placement-feasibility artifact current?

DEFAULT mode (what the orchestrator pre-commit hook runs): consult
data/contention_matrix/placement_feasibility.json, a pure cpuset-intersection
artifact regenerated from stack_numa.py by scripts/server/placement_feasibility.py
(handoff W-40). The BENCHMARKED contention matrix no longer gates commits.
  0 current · 2 missing/stale (re-run placement_feasibility.py) · 3 invalid

`--benchmarked` keeps the legacy check below (matrix freshness vs NUMA_CONFIG),
used by scripts/server/preflight_gate.py for inference-batch attestation only.

Legacy check: is the persisted contention matrix fresh vs live NUMA_CONFIG?

Per Phase F of `handoffs/active/cross-role-bw-aware-routing.md`. Intended as a
pre-commit hook + a CI gate when stack_numa.py or role model paths change.

Exit codes:
  0 — matrix OK (parses, topology hash matches, age within window)
  2 — matrix MISSING or STALE (re-run scripts/server/contention_matrix.py)
  3 — matrix INVALID (file exists but unparseable)

Usage:
  python scripts/validate/check_contention_matrix_fresh.py
  python scripts/validate/check_contention_matrix_fresh.py --max-age-days 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
REPO_ROOT = _THIS.parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "server"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=30,
        help="Flag as stale when file is older than this many days (default: 30)",
    )
    parser.add_argument(
        "--matrix-path",
        type=Path,
        default=None,
        help="Override path to contention_matrix.yaml",
    )
    parser.add_argument(
        "--benchmarked",
        action="store_true",
        help="Legacy mode: check the benchmarked contention matrix instead of the "
        "derived placement-feasibility artifact",
    )
    args = parser.parse_args()

    if not args.benchmarked:
        try:
            import placement_feasibility
            rc, msg = placement_feasibility.check()
        except Exception as exc:
            print(f"ERROR: placement feasibility check failed: {exc}", file=sys.stderr)
            return 3
        print(("FAIL: " if rc else "") + msg, file=sys.stderr if rc else sys.stdout)
        return rc

    try:
        from stack_numa import NUMA_CONFIG
        from src.scheduling.contention import (
            load_contention_matrix,
            matrix_status,
            topology_fingerprint,
            topology_fingerprint_for_matrix,
            MatrixStatus,
        )
    except Exception as exc:
        print(f"ERROR: failed to import scheduling modules: {exc}", file=sys.stderr)
        return 3

    matrix = None
    try:
        matrix = load_contention_matrix(args.matrix_path)
    except FileNotFoundError:
        pass
    except Exception:
        # Let matrix_status classify invalid YAML while keeping the message path.
        pass
    current_hash = (
        topology_fingerprint_for_matrix(NUMA_CONFIG, matrix)
        if matrix is not None
        else topology_fingerprint(NUMA_CONFIG)
    )
    status = matrix_status(
        args.matrix_path,
        current_topology_hash=current_hash,
        max_age_days=args.max_age_days,
    )

    if status == MatrixStatus.OK:
        print(f"OK: contention matrix is fresh (topology_hash={current_hash[:8]})")
        return 0
    if status == MatrixStatus.MISSING:
        print(
            "FAIL: orchestration/contention_matrix.yaml is MISSING.\n"
            "      Run: python scripts/server/contention_matrix.py",
            file=sys.stderr,
        )
        return 2
    if status == MatrixStatus.STALE:
        print(
            f"FAIL: contention matrix is STALE (live topology hash={current_hash[:8]} "
            f"or file age > {args.max_age_days} days).\n"
            "      Run: python scripts/server/contention_matrix.py",
            file=sys.stderr,
        )
        return 2
    if status == MatrixStatus.INVALID:
        print(
            "FAIL: contention_matrix.yaml exists but is INVALID (parse error).",
            file=sys.stderr,
        )
        return 3
    return 1


if __name__ == "__main__":
    sys.exit(main())
