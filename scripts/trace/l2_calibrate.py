#!/usr/bin/env python3
"""HIL-14 calibration tables (read-only, zero inference). Paths are arguments; absent sets are reported, not faked."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trace.l2 import calibrate as C  # noqa: E402
from src.trace.l2.detectors import DetectorConfig  # noqa: E402
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
    ap.add_argument("--whowhen-pro")
    ap.add_argument("--traceelephant", help="dir holding <system>/<run>/{trace_metadata,step_records}.json")
    ap.add_argument("--trail", help="dir holding processed_annotations_gaia/ and processed_annotations_swe_bench/")
    ap.add_argument("--kbh", help="dir of KernelBench-Hard *.jsonl; selection.json picks the files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--belief-sidecar", action="store_true", help="also write <out>.beliefs.jsonl via the root belief-kernel writer")
    ap.add_argument("--run-id", default="l2-run")
    ap.add_argument("--epyc-root", default=default_epyc_root("scripts/vidya/adapters/l2_scan_capture.py"))
    a = ap.parse_args(argv)
    cfg = DetectorConfig()
    res: dict = {"detector_config": {"loop_threshold": cfg.loop_threshold, "schemas": "none for foreign tool names"}}
    if a.whowhen_pro:
        res["whowhen_pro_text"] = C.whowhen_pro_table(a.whowhen_pro, cfg)
    if a.traceelephant:
        res["traceelephant"] = C.traceelephant_table(a.traceelephant, cfg)
    if a.trail:
        d = Path(a.trail)
        has = any((d / f"processed_annotations_{s}").is_dir() for s in ("gaia", "swe_bench"))
        res["trail"] = C.trail_table(d, cfg) if has else {"status": "not available: no processed_annotations_* under --trail"}
    if a.kbh:
        sel = json.loads((Path(a.kbh) / "selection.json").read_text())
        res["kernelbench_hard"] = C.kbh_firing([Path(a.kbh) / f for f in sel], cfg)
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    if a.belief_sidecar:
        write_sidecar(a.out, "calibration", a.run_id, a.epyc_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
