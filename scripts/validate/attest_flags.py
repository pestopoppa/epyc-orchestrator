#!/usr/bin/env python3
"""Poll /config/attest and fail if API workers disagree on feature flags."""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import httpx


def _parse_expect(items: list[str]) -> dict[str, bool]:
    expected: dict[str, bool] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--expect requires flag=true/false, got {item!r}")
        name, raw = item.split("=", 1)
        value = raw.strip().lower()
        if value not in {"1", "0", "true", "false", "yes", "no", "on", "off"}:
            raise SystemExit(f"invalid boolean for {name}: {raw!r}")
        expected[name.strip()] = value in {"1", "true", "yes", "on"}
    return expected


def _collect(url: str, polls: int, delay_s: float) -> dict[str, dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    endpoint = f"{url.rstrip('/')}/config/attest"
    for idx in range(polls):
        try:
            # Close the connection each time so a multi-worker uvicorn socket has
            # a chance to hand requests to different worker processes.
            resp = httpx.get(endpoint, headers={"Connection": "close"}, timeout=2.0)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            data = {"pid": f"error-{idx}", "error": str(exc), "flags": {}}
        pid = str(data.get("pid") or f"unknown-{idx}")
        seen[pid] = data
        time.sleep(delay_s)
    return seen


def _diff(seen: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    all_flags = sorted({
        name
        for data in seen.values()
        for name in (data.get("flags", {}) or {}).keys()
    })
    hetero: dict[str, dict[str, Any]] = {}
    for name in all_flags:
        values = {
            pid: (data.get("flags", {}) or {}).get(name)
            for pid, data in seen.items()
        }
        if len(set(values.values())) > 1:
            hetero[name] = values
    return hetero


def _expect_diffs(
    seen: dict[str, dict[str, Any]],
    expected: dict[str, bool],
) -> list[dict[str, Any]]:
    diffs: list[dict[str, Any]] = []
    for pid, data in seen.items():
        flags = data.get("flags", {}) or {}
        for name, value in expected.items():
            if flags.get(name) != value:
                diffs.append({
                    "pid": pid,
                    "flag": name,
                    "expected": value,
                    "actual": flags.get(name),
                })
    return diffs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--polls", type=int, default=120)
    parser.add_argument("--delay-s", type=float, default=0.05)
    parser.add_argument("--expect", action="append", default=[])
    parser.add_argument("--min-workers", type=int, default=1)
    args = parser.parse_args()

    expected = _parse_expect(args.expect)
    seen = _collect(args.url, args.polls, args.delay_s)
    hetero = _diff(seen)
    expected_diffs = _expect_diffs(seen, expected)
    errors = {
        pid: data.get("error")
        for pid, data in seen.items()
        if data.get("error")
    }
    too_few_workers = len(seen) < args.min_workers
    report = {
        "workers_seen": len(seen),
        "min_workers": args.min_workers,
        "heterogeneous": hetero,
        "expected_diffs": expected_diffs,
        "errors": errors,
        "too_few_workers": too_few_workers,
        "workers": seen,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if hetero or expected_diffs or errors or too_few_workers or not seen else 0


if __name__ == "__main__":
    raise SystemExit(main())
