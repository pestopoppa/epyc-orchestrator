#!/usr/bin/env python3
"""Validate focused orchestrator slice coverage floors from coverage JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_THRESHOLDS: dict[str, float] = {
    "scripts/benchmark/seeding_infra.py": 100.0,
    "scripts/lib/executor.py": 100.0,
    "scripts/lib/registry.py": 100.0,
    "scripts/lib/output_parser.py": 100.0,
    "scripts/lib/onboard.py": 100.0,
    "scripts/benchmark/seeding_injection.py": 100.0,
    "scripts/benchmark/seeding_orchestrator.py": 100.0,
    "scripts/benchmark/seed_specialist_routing.py": 100.0,
    "scripts/benchmark/seed_specialist_routing_v2.py": 100.0,
}


def _find_coverage_key(files: dict[str, dict], target: str) -> str | None:
    if target in files:
        return target
    normalized_target = target.replace("\\", "/")
    for key in files:
        if key.replace("\\", "/").endswith(normalized_target):
            return key
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage-json", required=True, help="Path to coverage JSON output.")
    args = parser.parse_args()

    payload_path = Path(args.coverage_json)
    if not payload_path.exists():
        print(f"ERROR: coverage JSON not found: {payload_path}", file=sys.stderr)
        return 2

    payload = json.loads(payload_path.read_text())
    files: dict[str, dict] = payload.get("files", {})

    failures: list[str] = []
    print("Orchestrator slice coverage thresholds:")
    for target, floor in DEFAULT_THRESHOLDS.items():
        key = _find_coverage_key(files, target)
        if key is None:
            failures.append(f"{target}: missing from coverage report")
            print(f"  FAIL {target} missing from coverage report")
            continue
        summary = files[key]["summary"]
        percent = float(summary["percent_covered"])
        covered = int(summary["covered_lines"])
        total = int(summary["num_statements"])
        status = "OK" if percent >= floor else "FAIL"
        print(f"  {status:<4} {target}: {percent:.2f}% (floor {floor:.1f}%, {covered}/{total})")
        if percent < floor:
            failures.append(f"{target}: {percent:.2f}% < {floor:.1f}%")

    if failures:
        print("\nCoverage floor failures:")
        for item in failures:
            print(f"  - {item}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
