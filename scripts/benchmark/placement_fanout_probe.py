#!/usr/bin/env python3
"""Placement fan-out probe (bulk-inference Package J / J1-J3).

Fires N concurrent (or serial) /chat requests pinned to one or more roles and,
during the burst, polls /dashboard/api/region_locks to record how many
DISTINCT instances of each role are simultaneously active. Cross-references
the orchestrator log for the WP-2 `placement queued role=... reason=...`
line so the operator can confirm a queued (not overlapping) placement.

Outputs a JSON record with the metrics the bulk-inference baseline-mutation
rule requires: speed_metric_mode, eval_concurrency, median per-request t/s,
aggregate batch t/s, eval wall time, plus the placement observations.

NO benchmarking binary is launched — this drives the live orchestrator API
over HTTP only. Honors per-run inference approval (operator-launched).

Usage:
  python scripts/benchmark/placement_fanout_probe.py \
      --role frontdoor --n 4 --mode concurrent \
      --out data/bulk_inference_2026_05_26/j1_concurrent_n4.json

  python scripts/benchmark/placement_fanout_probe.py \
      --roles frontdoor,ingest_long_context --n 4 --mode concurrent \
      --out /tmp/step2_cross_role_fanout.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

_REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_API = "http://127.0.0.1:8000"
DEFAULT_LOG = str(_REPO_ROOT / "logs/orchestrator.log")
QUEUE_MARK = "placement queued role="


def parse_roles(role: str, roles: str | None) -> list[str]:
    """Return the role sequence to probe, preserving --role compatibility."""
    raw = roles if roles is not None else role
    parsed = [item.strip() for item in raw.split(",") if item.strip()]
    if not parsed:
        raise ValueError("at least one role is required")
    return parsed


def request_roles(roles: list[str], n: int) -> list[str]:
    """Assign N requests to roles round-robin."""
    if n <= 0:
        raise ValueError("--n must be positive")
    return [roles[idx % len(roles)] for idx in range(n)]


def one_request(api: str, role: str, prompt: str, idx: int, timeout: float) -> dict:
    """Single /chat call pinned to `role`. Returns timing + token record."""
    payload = {
        "prompt": prompt,
        "force_role": role,
        "allow_delegation": False,
        "max_turns": 1,
        "real_mode": True,
        "mock_mode": False,
        "cache_prompt": False,
        "session_id": f"j1-probe-{role}-{idx}-{int(time.time()*1000)}",
    }
    t0 = time.perf_counter()
    rec = {
        "idx": idx,
        "role": role,
        "ok": False,
        "status": None,
        "latency_s": None,
        "tokens_generated": 0,
        "elapsed_seconds": None,
        "req_tps": None,
    }
    try:
        with httpx.Client(timeout=timeout) as c:
            r = c.post(f"{api}/chat", json=payload)
            rec["status"] = r.status_code
            r.raise_for_status()
            data = r.json()
        rec["latency_s"] = time.perf_counter() - t0
        rec["tokens_generated"] = int(data.get("tokens_generated", 0) or 0)
        rec["elapsed_seconds"] = float(data.get("elapsed_seconds", 0) or 0)
        if rec["elapsed_seconds"] and rec["tokens_generated"]:
            rec["req_tps"] = rec["tokens_generated"] / rec["elapsed_seconds"]
        rec["ok"] = True
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["latency_s"] = time.perf_counter() - t0
    return rec


class RegionLockPoller(threading.Thread):
    """Polls /dashboard/api/region_locks; records active instance idxs by role."""

    def __init__(self, api: str, roles: list[str], interval: float):
        super().__init__(daemon=True)
        self.api, self.roles, self.interval = api, roles, interval
        self._stop = threading.Event()
        self.samples: list[dict] = []
        self.max_active_by_role: dict[str, int] = {role: 0 for role in roles}
        self.observed_idxs_by_role: dict[str, set[int]] = {role: set() for role in roles}
        self.max_roles_active_same_sample = 0
        self.enabled_flag = None

    def run(self):
        with httpx.Client(timeout=5.0) as c:
            while not self._stop.is_set():
                try:
                    r = c.get(f"{self.api}/dashboard/api/region_locks")
                    d = r.json()
                    if self.enabled_flag is None:
                        self.enabled_flag = d.get("per_region_locks_enabled")
                    by_role = d.get("by_role") or {}
                    sample_roles: dict[str, list[int]] = {}
                    active_roles = 0
                    for role in self.roles:
                        bucket = by_role.get(role, {})
                        active = bucket.get("active_instance_idxs", []) or []
                        active_ints = [int(item) for item in active]
                        sample_roles[role] = active_ints
                        self.max_active_by_role[role] = max(
                            self.max_active_by_role[role], len(active_ints)
                        )
                        self.observed_idxs_by_role[role] |= set(active_ints)
                        if active_ints:
                            active_roles += 1
                    self.max_roles_active_same_sample = max(
                        self.max_roles_active_same_sample, active_roles
                    )
                    self.samples.append({"t": time.time(), "roles": sample_roles})
                except Exception:
                    pass
                self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()


def _metrics(recs: list[dict], wall_s: float, mode: str, n: int) -> dict:
    ok = [r for r in recs if r["ok"]]
    req_tps = [r["req_tps"] for r in ok if r["req_tps"]]
    lat = [r["latency_s"] for r in ok if r["latency_s"]]
    tot_tok = sum(r["tokens_generated"] for r in ok)
    def pct(xs, p):
        if not xs:
            return None
        s = sorted(xs)
        k = min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))
        return s[k]
    return {
        "speed_metric_mode": "aggregate_batch" if mode == "concurrent" else "serial_median",
        "eval_concurrency": n if mode == "concurrent" else 1,
        "n_requests": len(recs),
        "n_ok": len(ok),
        "median_request_tps": statistics.median(req_tps) if req_tps else None,
        "aggregate_batch_tps": (tot_tok / wall_s) if wall_s > 0 else None,
        "eval_wall_s": round(wall_s, 3),
        "p50_latency_s": pct(lat, 50),
        "p99_latency_s": pct(lat, 99),
        "total_tokens_generated": tot_tok,
    }


def _placement_summary(poller: RegionLockPoller | None, primary_role: str) -> dict:
    """Return placement summary while preserving legacy single-role keys."""
    if poller is None:
        by_role = None
        return {
            "per_region_locks_enabled": None,
            "max_distinct_active_instances": None,
            "observed_active_instance_idxs": None,
            "n_poll_samples": 0,
            "max_roles_active_same_sample": None,
            "by_role": by_role,
        }
    by_role = {
        role: {
            "max_distinct_active_instances": poller.max_active_by_role.get(role, 0),
            "observed_active_instance_idxs": sorted(
                poller.observed_idxs_by_role.get(role, set())
            ),
        }
        for role in poller.roles
    }
    primary = by_role.get(primary_role, {})
    return {
        "per_region_locks_enabled": poller.enabled_flag,
        "max_distinct_active_instances": primary.get("max_distinct_active_instances"),
        "observed_active_instance_idxs": primary.get("observed_active_instance_idxs"),
        "n_poll_samples": len(poller.samples),
        "max_roles_active_same_sample": poller.max_roles_active_same_sample,
        "by_role": by_role,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--role", default="frontdoor")
    ap.add_argument(
        "--roles",
        default=None,
        help="comma-separated roles to probe round-robin; overrides --role",
    )
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--mode", choices=["serial", "concurrent"], default="concurrent")
    ap.add_argument("--api", default=DEFAULT_API)
    ap.add_argument("--poll-interval", type=float, default=0.12)
    ap.add_argument("--no-dashboard", action="store_true",
                    help="skip the HTTP region_locks poller (avoids draining the per-IP rate bucket; "
                         "rely on the SM log line + a separate file-based active_region_holders poller)")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--log-file", default=DEFAULT_LOG)
    ap.add_argument("--prompt", default="Write a concise 100-word paragraph explaining what a NUMA node is.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    try:
        roles = parse_roles(args.role, args.roles)
        request_plan = request_roles(roles, args.n)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    # Isolate this run's queue-log lines by byte offset.
    log_path = Path(args.log_file)
    log_off0 = log_path.stat().st_size if log_path.exists() else 0

    poller = None
    if args.mode == "concurrent" and not args.no_dashboard:
        poller = RegionLockPoller(args.api, roles, args.poll_interval)
        poller.start()
        time.sleep(args.poll_interval * 2)

    t0 = time.perf_counter()
    if args.mode == "serial":
        recs = [
            one_request(args.api, role, args.prompt, i, args.timeout)
            for i, role in enumerate(request_plan)
        ]
    else:
        with ThreadPoolExecutor(max_workers=args.n) as ex:
            futs = [
                ex.submit(one_request, args.api, role, args.prompt, i, args.timeout)
                for i, role in enumerate(request_plan)
            ]
            recs = [f.result() for f in futs]
    wall = time.perf_counter() - t0

    if poller:
        time.sleep(args.poll_interval * 2)
        poller.stop()
        poller.join(timeout=2)

    # Scan log tail for this run's placement-queue lines.
    queue_lines = []
    if log_path.exists():
        with open(log_path, "r", errors="replace") as f:
            f.seek(log_off0)
            for line in f:
                if QUEUE_MARK in line and any(f"role={role}" in line for role in roles):
                    queue_lines.append(line.strip())

    placement = _placement_summary(poller, roles[0])
    out = {
        "task": "placement_fanout_probe",
        "role": roles[0],
        "roles": roles,
        "request_plan": request_plan,
        "mode": args.mode,
        "n": args.n,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metrics": _metrics(recs, wall, args.mode, args.n),
        "placement": placement,
        "queue_log_lines": queue_lines,
        "queue_log_count": len(queue_lines),
        "requests": recs,
    }
    txt = json.dumps(out, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(txt + "\n")
    # concise stderr-free summary
    m = out["metrics"]
    print(json.dumps({
        "mode": args.mode, "n": args.n, "roles": roles, "n_ok": m["n_ok"],
        "max_active": out["placement"]["max_distinct_active_instances"],
        "active_idxs": out["placement"]["observed_active_instance_idxs"],
        "by_role": out["placement"]["by_role"],
        "max_roles_active_same_sample": out["placement"]["max_roles_active_same_sample"],
        "queue_log_count": out["queue_log_count"],
        "median_req_tps": m["median_request_tps"],
        "aggregate_tps": m["aggregate_batch_tps"],
        "p99_latency_s": m["p99_latency_s"],
        "wall_s": m["eval_wall_s"],
        "flag_enabled": out["placement"]["per_region_locks_enabled"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
