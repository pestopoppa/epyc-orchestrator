#!/usr/bin/env python3
"""Read-only L2 mechanical failure-taxonomy scan (HIL-13). Zero inference, no network.

  l2_scan.py --progress-logs 'logs/progress/*.jsonl' --out report.json
  l2_scan.py --normalised traces.jsonl --out report.json
  l2_scan.py --step-db events.sqlite [--progress-logs GLOB] --out report.json   # HIL-13a step events
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trace.l2.detectors import DetectorConfig  # noqa: E402
from src.trace.l2.loaders import apply_tool_required, tool_required_map, load_normalised_jsonl, load_progress_logs, load_step_events, merge_step_traces  # noqa: E402
from src.trace.l2.report import dumps, scan  # noqa: E402
from src.trace.l2.schemas import export_tool_schemas  # noqa: E402


from scripts.lib.epyc_root import default_epyc_root  # noqa: E402


def write_sidecar(out: str, kind: str, run_id: str, epyc_root: str) -> None:
    """VB-L2-WRITE: emit <report>.beliefs.jsonl at write time (opt-in; never backfills)."""
    import subprocess

    sys.path.insert(0, str(Path(epyc_root) / "scripts" / "vidya"))
    from adapters.l2_scan_capture import write_belief_measurements  # type: ignore

    ref = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or "unknown"
    write_belief_measurements(out, kind=kind, run_id=run_id, producer="epyc-orchestrator/scripts/trace/l2_" + kind + ".py", code_ref=ref)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--progress-logs", help="glob of logs/progress/*.jsonl")
    ap.add_argument("--normalised", help="JSONL of L2Trace dicts")
    ap.add_argument("--step-db", help="trace-store sqlite holding HIL-13a tool_step rows (flag step_event_logging); "
                    "outcomes are joined from --progress-logs by run id = task id")
    ap.add_argument("--tasks", help="driver --tasks JSON; sets tool_required per task key (explicit field, else the prompt "
                    "instructs 'use the <x> tool'). Without it zero_step_violation is UNOBSERVED, never inferred.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--belief-sidecar", action="store_true", help="also write <out>.beliefs.jsonl via the root belief-kernel writer")
    ap.add_argument("--run-id", default="l2-run")
    ap.add_argument("--epyc-root", default=default_epyc_root("scripts/vidya/adapters/l2_scan_capture.py"))
    a = ap.parse_args(argv)
    traces = []
    if a.progress_logs:
        traces += list(load_progress_logs(sorted(glob.glob(a.progress_logs))))
    if a.step_db:
        steps = list(load_step_events(a.step_db, {t.trace_id: t for t in traces}))
        joined = sum(1 for t in steps if t.outcome != "unknown")
        traces = merge_step_traces(traces, steps)
        print(f"step-bearing runs: {len(steps)} ({joined} with a joined outcome)", file=sys.stderr)
    if a.tasks:
        import json as _json
        n_set = apply_tool_required(traces, tool_required_map(_json.loads(Path(a.tasks).read_text())))
        print(f"tool_required set on {n_set} traces from {a.tasks}", file=sys.stderr)
    if a.normalised:
        traces += list(load_normalised_jsonl(a.normalised))
    if not traces:
        print("no traces loaded", file=sys.stderr)
        return 2
    cfg = DetectorConfig(schemas=export_tool_schemas())
    Path(a.out).write_text(dumps(scan(traces, cfg, seed=a.seed, B=a.boot)) + "\n")
    print(f"wrote {a.out} ({len(traces)} traces)")
    if a.belief_sidecar:
        write_sidecar(a.out, "scan", a.run_id, a.epyc_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
