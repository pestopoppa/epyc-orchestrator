"""HF-SKIPS: parse a pytest run into outcome counts and skip reasons.

Pure functions, no flag reads: callers gate on ``features().test_result_skip_aware``.

A pytest run that exits 0 with every test skipped decided nothing about the code
under test. Before this, ``returncode == 0`` was read as "passed" in gate_runner,
tools/code/run_tests and builtin_tools. Callers must parse the FULL output
(before any truncation: run_tests keeps 16k, builtin_tools the last 2000 chars).
"""

from __future__ import annotations

import re
from typing import Any

_COUNT_RE = re.compile(
    r"(\d+)\s+(passed|failed|skipped|xfailed|xpassed|errors?|deselected)\b", re.IGNORECASE
)
# Final summary line: "=== 3 passed, 2 skipped in 0.12s ===" or "-q" form without the bars.
_SUMMARY_LINE_RE = re.compile(r"^[=\s]*(?:\d+\s+\w+(?:,\s*)?)+.*\bin\s+[\d.]+s\b.*$", re.IGNORECASE)
_SKIP_REASON_RE = re.compile(r"^SKIPPED\s*\[(\d+)\]\s*(?:\S+?:\d+:)?\s*(.*)$")
_MAX_REASONS = 20


def parse_pytest_summary(output: str) -> dict[str, Any] | None:
    """Return ``{passed, failed, skipped, xfailed, xpassed, errors, skip_reasons}``.

    ``None`` when no pytest summary line is found (not a pytest run, or output
    cut off), so callers keep the legacy returncode reading.
    """
    if not output:
        return None
    summary_line = None
    for line in reversed(output.splitlines()):
        stripped = line.strip()
        if _SUMMARY_LINE_RE.match(stripped) and _COUNT_RE.search(stripped):
            summary_line = stripped
            break
        if stripped.lower().startswith("no tests ran"):
            summary_line = stripped
            break
    if summary_line is None:
        return None
    counts = {"passed": 0, "failed": 0, "skipped": 0, "xfailed": 0, "xpassed": 0, "errors": 0}
    for num, kind in _COUNT_RE.findall(summary_line):
        kind = kind.lower()
        if kind == "deselected":
            continue
        counts["errors" if kind.startswith("error") else kind] += int(num)
    reasons: dict[str, int] = {}
    for line in output.splitlines():
        m = _SKIP_REASON_RE.match(line.strip())
        if m:
            reason = m.group(2).strip()[:200] or "(no reason)"
            reasons[reason] = reasons.get(reason, 0) + int(m.group(1))
    counts["skip_reasons"] = [
        {"reason": r, "count": c} for r, c in sorted(reasons.items(), key=lambda kv: -kv[1])[:_MAX_REASONS]
    ]
    return counts


def is_inconclusive(returncode: int, counts: dict[str, Any] | None) -> bool:
    """rc==0 but nothing passed (all skipped / xfailed / no tests ran) decided nothing."""
    if counts is None or returncode != 0:
        return False
    return counts["passed"] == 0 and counts["failed"] == 0 and counts["errors"] == 0
