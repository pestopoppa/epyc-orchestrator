#!/usr/bin/env python3
"""Fail-loud check: is the persisted contention matrix fresh vs live NUMA_CONFIG?

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
    args = parser.parse_args()

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
