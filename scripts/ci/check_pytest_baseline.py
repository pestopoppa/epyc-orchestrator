#!/usr/bin/env python3
"""Fail CI only for new pytest IDs while reporting the known baseline failures."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def _junit_identity(node_id: str) -> tuple[str, str]:
    # Pytest node IDs use ``::`` for structural separators, while parameter
    # IDs may contain the same characters inside ``[...]``.
    parts = []
    start = 0
    parameter_depth = 0
    for index, character in enumerate(node_id):
        if character == "[":
            parameter_depth += 1
        elif character == "]" and parameter_depth:
            parameter_depth -= 1
        elif character == ":" and parameter_depth == 0 and node_id[index : index + 2] == "::":
            parts.append(node_id[start:index])
            start = index + 2
    parts.append(node_id[start:])
    file_name, *tail = parts
    if not tail:
        raise ValueError(f"baseline entry is not a test node id: {node_id!r}")
    module = file_name.replace("\\", "/").removesuffix(".py").replace("/", ".")
    return ".".join((module, *tail[:-1])), tail[-1]


def _load_suites(path: Path) -> list[ET.Element]:
    root = ET.parse(path).getroot()
    if root.tag == "testsuite":
        return [root]
    if root.tag == "testsuites":
        return list(root.findall("testsuite"))
    raise ValueError(f"unexpected JUnit root element: {root.tag}")


def _report_failures(suites: list[ET.Element]) -> list[tuple[str, str]]:
    failures = []
    for suite in suites:
        for case in suite.findall("testcase"):
            if any(case.find(tag) is not None for tag in ("failure", "error")):
                failures.append((case.attrib.get("classname", ""), case.attrib.get("name", "")))
    return failures


def _suite_count(suites: list[ET.Element]) -> int:
    # Headers may include collection skips or passed-subtest counters that do
    # not represent executed parent tests. Duplicated records cannot pad a floor.
    return len({
        (case.attrib.get("classname"), case.attrib.get("name"))
        for suite in suites
        for case in suite.findall("testcase")
        if case.attrib.get("classname") and case.attrib.get("name")
    })


def _write_summary(path: Path | None, lines: list[str]) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--pytest-exit-code", required=True, type=int)
    parser.add_argument("--summary-file", type=Path)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    expected_ids = baseline["known_failure_ids"]
    expected = {_junit_identity(node_id): node_id for node_id in expected_ids}
    if len(expected) != len(expected_ids):
        raise ValueError("baseline has duplicate JUnit identities")

    suites = _load_suites(args.junit)
    collected = _suite_count(suites)
    failures = _report_failures(suites)
    known = sorted({expected[item] for item in failures if item in expected})
    new = sorted(
        f"{classname}::{name}" for classname, name in failures if (classname, name) not in expected
    )
    floor = int(baseline["minimum_collected"])
    suite_reported_failures = sum(
        int(suite.attrib.get("failures", 0)) + int(suite.attrib.get("errors", 0))
        for suite in suites
    )
    lines = [
        "## Orchestrator pytest baseline",
        "",
        f"- Baseline source: `{baseline['source_head']}`",
        f"- Collected/executed: {collected} (required floor: {floor})",
        f"- Known baseline failing/erroring IDs observed: {len(known)} / {len(expected_ids)}",
        f"- New failing/erroring IDs: {len(new)}",
    ]
    if known:
        lines.extend(
            ["", "Known baseline failures (still executed):", *[f"- `{item}`" for item in known]]
        )
    if new:
        lines.extend(["", "New failures:", *[f"- `{item}`" for item in new]])
    if collected < floor:
        lines.append(f"\nCollection floor failed: {collected} < {floor}.")
    if suite_reported_failures != len(failures):
        lines.append(
            "\nJUnit contains failure/error counts without matching test-case nodes; refusing to "
            "treat those as known baseline failures."
        )
    if args.pytest_exit_code not in (0, 1):
        lines.append(f"\npytest exited with infrastructure/usage status {args.pytest_exit_code}.")
    elif args.pytest_exit_code == 1 and not failures:
        lines.append("\npytest exited nonzero but JUnit contains no failing/erroring test IDs.")

    _write_summary(args.summary_file, lines)
    print("\n".join(lines))

    if (
        collected < floor
        or new
        or suite_reported_failures != len(failures)
        or args.pytest_exit_code not in (0, 1)
        or (args.pytest_exit_code == 1 and not failures)
    ):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
