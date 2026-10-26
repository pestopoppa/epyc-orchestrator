#!/usr/bin/env python3
"""Format-check changed Python files and identify failures inherited from a base."""

from __future__ import annotations

import argparse
import subprocess
import sys


def _changed_python_files(base: str | None) -> list[str]:
    commands = []
    if base:
        commands.append(
            ["git", "diff", "--name-only", "--diff-filter=ACMR", f"{base}...HEAD", "--", "*.py"]
        )
    else:
        commands.extend(
            [
                ["git", "diff", "--name-only", "--diff-filter=ACMR", "HEAD", "--", "*.py"],
                ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "--", "*.py"],
                ["git", "ls-files", "--others", "--exclude-standard", "--", "*.py"],
            ]
        )
    names = set()
    for command in commands:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        names.update(line for line in result.stdout.splitlines() if line.endswith(".py"))
    return sorted(names)


def _check_current(path: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["ruff", "format", "--check", path], capture_output=True, text=True)


def _check_base(path: str, base: str) -> subprocess.CompletedProcess[str] | None:
    result = subprocess.run(["git", "show", f"{base}:{path}"], capture_output=True, check=False)
    if result.returncode:
        return None
    return subprocess.run(
        ["ruff", "format", "--check", "--stdin-filename", path, "-"],
        input=result.stdout,
        capture_output=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=None, help="base commit used to classify inherited debt")
    args = parser.parse_args()

    base = args.base or "HEAD"
    paths = _changed_python_files(args.base)
    if not paths:
        print("No changed Python files.")
        return 0

    failing = []
    inherited = []
    for path in paths:
        current = _check_current(path)
        if current.returncode == 0:
            continue
        if current.returncode != 1:
            print(current.stderr or current.stdout, file=sys.stderr)
            return current.returncode
        failing.append(path)
        base_result = _check_base(path, base)
        if base_result is not None and base_result.returncode == 1:
            inherited.append(path)

    print(f"Changed Python files checked: {len(paths)}")
    print(f"Ruff format failures: {len(failing)}")
    print(f"Already failing at base: {len(inherited)}")
    for path in failing:
        classification = (
            "inherited baseline debt" if path in inherited else "introduced or unclassified"
        )
        print(f"  {classification}: {path}")
    return int(bool(failing))


if __name__ == "__main__":
    sys.exit(main())
