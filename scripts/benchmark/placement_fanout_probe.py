#!/usr/bin/env python3
"""Within-role placement fan-out probe (bulk-inference Package J / J1-J3).

Fires N concurrent (or serial) /chat requests pinned to a single role and,
during the burst, polls /dashboard/api/region_locks to record how many
DISTINCT instances of that role are simultaneously active. Cross-references
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

DEFAULT_API = "http://127.0.0.1:8000"
DEFAULT_LOG = "/mnt/raid0/llm/epyc-orchestrator/logs/orchestrator.log"
QUEUE_MARK = "placement queued role="


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
    rec = {"idx": idx, "ok": False, "status": None, "latency_s": None,
           "tokens_generated": 0, "elapsed_seconds": None, "req_tps": None}
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
    """Polls /dashboard/api/region_locks; records active instance idxs for role."""

    def __init__(self, api: str, role: str, interval: float):
        super().__init__(daemon=True)
        self.api, self.role, self.interval = api, role, interval
        self._stop = threading.Event()
        self.samples: list[dict] = []
        self.max_active = 0
        self.observed_idxs: set[int] = set()
        self.enabled_flag = None

    def run(self):
        with httpx.Client(timeout=5.0) as c:
            while not self._stop.is_set():
                try:
                    r = c.get(f"{self.api}/dashboard/api/region_locks")
                    d = r.json()
                    if self.enabled_flag is None:
                        self.enabled_flag = d.get("per_region_locks_enabled")
                    bucket = (d.get("by_role") or {}).get(self.role, {})
                    active = bucket.get("active_instance_idxs", []) or []
                    self.samples.append({"t": time.time(), "active": list(active)})
                    self.max_active = max(self.max_active, len(active))
                    self.observed_idxs |= set(active)
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--role", default="frontdoor")
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

    # Isolate this run's queue-log lines by byte offset.
    log_path = Path(args.log_file)
    log_off0 = log_path.stat().st_size if log_path.exists() else 0

    poller = None
    if args.mode == "concurrent" and not args.no_dashboard:
        poller = RegionLockPoller(args.api, args.role, args.poll_interval)
        poller.start()
        time.sleep(args.poll_interval * 2)

    t0 = time.perf_counter()
    if args.mode == "serial":
        recs = [one_request(args.api, args.role, args.prompt, i, args.timeout) for i in range(args.n)]
    else:
        with ThreadPoolExecutor(max_workers=args.n) as ex:
            futs = [ex.submit(one_request, args.api, args.role, args.prompt, i, args.timeout) for i in range(args.n)]
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
                if QUEUE_MARK in line and f"role={args.role}" in line:
                    queue_lines.append(line.strip())

    out = {
        "task": "placement_fanout_probe",
        "role": args.role,
        "mode": args.mode,
        "n": args.n,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metrics": _metrics(recs, wall, args.mode, args.n),
        "placement": {
            "per_region_locks_enabled": poller.enabled_flag if poller else None,
            "max_distinct_active_instances": poller.max_active if poller else None,
            "observed_active_instance_idxs": sorted(poller.observed_idxs) if poller else None,
            "n_poll_samples": len(poller.samples) if poller else 0,
        },
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
        "mode": args.mode, "n": args.n, "n_ok": m["n_ok"],
        "max_active": out["placement"]["max_distinct_active_instances"],
        "active_idxs": out["placement"]["observed_active_instance_idxs"],
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
