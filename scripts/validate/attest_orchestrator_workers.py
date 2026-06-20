#!/usr/bin/env python3
"""Attest API worker feature flags and selected process environment values.

This is a local-only companion to ``attest_flags.py``. ``/config/attest`` proves
the worker-local ``Features`` view, but several rollout controls are plain
process environment variables. Poll the attestation endpoint to discover worker
PIDs, then read only the requested keys from ``/proc/<pid>/environ`` so reports
do not dump secrets.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import httpx


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _parse_bool_expect(items: list[str]) -> dict[str, bool]:
    expected: dict[str, bool] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"expected name=true/false, got {item!r}")
        name, raw = item.split("=", 1)
        value = raw.strip().lower()
        if value not in _TRUE_VALUES | _FALSE_VALUES:
            raise SystemExit(f"invalid boolean for {name}: {raw!r}")
        expected[name.strip()] = value in _TRUE_VALUES
    return expected


def _parse_env_expect(items: list[str]) -> dict[str, str]:
    expected: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--expect-env requires NAME=value, got {item!r}")
        name, value = item.split("=", 1)
        name = name.strip()
        if not name:
            raise SystemExit(f"--expect-env requires non-empty NAME, got {item!r}")
        expected[name] = value
    return expected


def _collect_attest(url: str, polls: int, delay_s: float) -> dict[str, dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    endpoint = f"{url.rstrip('/')}/config/attest"
    for idx in range(polls):
        try:
            # Close each connection so uvicorn can hand polls to different workers.
            resp = httpx.get(endpoint, headers={"Connection": "close"}, timeout=2.0)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            data = {"pid": f"error-{idx}", "error": str(exc), "flags": {}, "sources": {}}
        pid = str(data.get("pid") or f"unknown-{idx}")
        seen[pid] = data
        time.sleep(delay_s)
    return seen


def _read_proc_environ(pid: str, proc_root: Path = Path("/proc")) -> tuple[dict[str, str], str | None]:
    try:
        raw = (proc_root / pid / "environ").read_bytes()
    except Exception as exc:
        return {}, str(exc)

    env: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        env[key.decode(errors="replace")] = value.decode(errors="replace")
    return env, None


def _feature_heterogeneity(seen: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
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
            if not data.get("error")
        }
        if values and len(set(values.values())) > 1:
            hetero[name] = values
    return hetero


def _feature_expected_diffs(
    seen: dict[str, dict[str, Any]],
    expected: dict[str, bool],
) -> list[dict[str, Any]]:
    diffs: list[dict[str, Any]] = []
    for pid, data in seen.items():
        flags = data.get("flags", {}) or {}
        for name, value in expected.items():
            actual = flags.get(name)
            if actual != value:
                diffs.append({
                    "pid": pid,
                    "flag": name,
                    "expected": value,
                    "actual": actual,
                })
    return diffs


def _collect_env(
    pids: list[str],
    expected_keys: set[str],
    proc_root: Path = Path("/proc"),
) -> tuple[dict[str, dict[str, str | None]], dict[str, str]]:
    observed: dict[str, dict[str, str | None]] = {}
    errors: dict[str, str] = {}
    for pid in pids:
        env, error = _read_proc_environ(pid, proc_root=proc_root)
        if error:
            errors[pid] = error
        observed[pid] = {key: env.get(key) for key in sorted(expected_keys)}
    return observed, errors


def _env_expected_diffs(
    observed: dict[str, dict[str, str | None]],
    expected: dict[str, str],
) -> list[dict[str, Any]]:
    diffs: list[dict[str, Any]] = []
    for pid, env in observed.items():
        for name, value in expected.items():
            actual = env.get(name)
            if actual != value:
                diffs.append({
                    "pid": pid,
                    "env": name,
                    "expected": value,
                    "actual": actual,
                })
    return diffs


def build_report(
    *,
    seen: dict[str, dict[str, Any]],
    expected_features: dict[str, bool],
    expected_env: dict[str, str],
    min_workers: int,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    endpoint_errors = {
        pid: data.get("error")
        for pid, data in seen.items()
        if data.get("error")
    }
    worker_pids = [pid for pid, data in seen.items() if not data.get("error") and pid.isdigit()]
    env_observed, proc_errors = _collect_env(
        worker_pids,
        set(expected_env),
        proc_root=proc_root,
    )
    selected_features = sorted(expected_features)
    return {
        "workers_seen": len(worker_pids),
        "min_workers": min_workers,
        "too_few_workers": len(worker_pids) < min_workers,
        "feature_heterogeneous": _feature_heterogeneity(seen),
        "feature_expected_diffs": _feature_expected_diffs(seen, expected_features),
        "env_expected_diffs": _env_expected_diffs(env_observed, expected_env),
        "endpoint_errors": endpoint_errors,
        "proc_errors": proc_errors,
        "workers": {
            pid: {
                "flags": {
                    name: (data.get("flags", {}) or {}).get(name)
                    for name in selected_features
                },
                "sources": {
                    name: (data.get("sources", {}) or {}).get(name)
                    for name in selected_features
                },
                "env": env_observed.get(pid, {}),
            }
            for pid, data in seen.items()
            if not data.get("error")
        },
    }


def report_failed(report: dict[str, Any]) -> bool:
    return bool(
        report["too_few_workers"]
        or report["feature_heterogeneous"]
        or report["feature_expected_diffs"]
        or report["env_expected_diffs"]
        or report["endpoint_errors"]
        or report["proc_errors"]
        or not report["workers_seen"]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--polls", type=int, default=120)
    parser.add_argument("--delay-s", type=float, default=0.05)
    parser.add_argument("--min-workers", type=int, default=1)
    parser.add_argument("--expect-feature", action="append", default=[])
    parser.add_argument("--expect-env", action="append", default=[])
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"))
    args = parser.parse_args()

    report = build_report(
        seen=_collect_attest(args.url, args.polls, args.delay_s),
        expected_features=_parse_bool_expect(args.expect_feature),
        expected_env=_parse_env_expect(args.expect_env),
        min_workers=args.min_workers,
        proc_root=args.proc_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report_failed(report) else 0


if __name__ == "__main__":
    raise SystemExit(main())
