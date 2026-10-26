#!/usr/bin/env python3
"""Compare Ruff diagnostics to an explicit, source-pinned debt baseline."""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any


def _identity(item: dict[str, Any]) -> tuple[str, str, str]:
    return (item["path"], item["code"], item["message"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ruff-json", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--ruff-exit-code", required=True, type=int)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    current = json.loads(args.ruff_json.read_text(encoding="utf-8"))
    expected = collections.Counter(_identity(item) for item in baseline["known_diagnostics"])
    observed = collections.Counter(
        (
            Path(item["filename"]).resolve().relative_to(Path.cwd().resolve()).as_posix(),
            item["code"],
            item["message"],
        )
        for item in current
    )
    known = observed & expected
    new = observed - expected

    print(f"Ruff baseline source: {baseline['source_head']}")
    print(f"Known diagnostics still present: {sum(known.values())} / {sum(expected.values())}")
    print(f"New diagnostics: {sum(new.values())}")
    if args.ruff_exit_code not in (0, 1):
        print(f"Ruff exited with infrastructure/usage status {args.ruff_exit_code}")
    elif args.ruff_exit_code == 1 and not current:
        print("Ruff exited nonzero but returned no diagnostics")
    for (path, code, message), count in sorted(known.items()):
        print(f"  known x{count}: {path}: {code} {message}")
    for (path, code, message), count in sorted(new.items()):
        print(f"  NEW x{count}: {path}: {code} {message}")

    return int(
        bool(new) or args.ruff_exit_code not in (0, 1) or (args.ruff_exit_code == 1 and not current)
    )


if __name__ == "__main__":
    sys.exit(main())
