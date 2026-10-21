#!/usr/bin/env python3
"""MHS-15 calibration: whole-file read-only lint over orchestration/prompts.

Treats each file as mutated against an empty original (every line is "added"),
so per-rule counts show how noisy each rule is on the real prompts. Read-only.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "autopilot"))

from species.prompt_forge import prompt_interference_lint  # noqa: E402


def main() -> int:
    prompts = ROOT / "orchestration" / "prompts"
    totals: Counter[str] = Counter()
    per_file: dict[str, dict[str, int]] = {}
    examples: dict[str, list[str]] = {}
    files = sorted(prompts.rglob("*.md"))
    for path in files:
        findings = prompt_interference_lint(
            "", path.read_text(encoding="utf-8", errors="replace")
        )
        counts = Counter(f.split(":")[1] for f in findings)
        per_file[str(path.relative_to(ROOT))] = dict(counts)
        totals.update(counts)
        for f in findings:
            examples.setdefault(f.split(":")[1], []).append(f"{path.name}: {f[:200]}")
    print(
        json.dumps(
            {
                "files": len(files),
                "totals": dict(totals),
                "per_file": per_file,
                "samples": {k: v[:8] for k, v in examples.items()},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
