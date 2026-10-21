#!/usr/bin/env python3
"""Read-only journal replays: MHS-16 drift, SEQ-6 frozen-null, SEQ-7 paired-z.

Frozen input = a sha256 digest over the journal shards read (printed in the report).
No gate is changed, no AutoPilot process is started, no inference is run.

    python3 scripts/autopilot/gate_replay_reports.py --journal <dir> \
        --state <autopilot_state.json> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ORCH_ROOT))

from journal_shards import journal_shards  # noqa: E402
from paired_stats import iter_journal_rows  # noqa: E402
from src.autopilot_core.sequential_replay import (  # noqa: E402
    cumulative_drift_report,
    frozen_null_replay,
    paired_keep_replay,
    promotion_attempt_ledger,
)


# Fitted on the 2026-10-21 journal snapshot, NOT a recorded fact: this bug tag was applied
# retroactively, and the live null used its rows at trials below ~1225 (best single cutoff of
# the journaled-z match count; 315 of 334 trials match with it, 236 without).
RETRO_TAG_CUTOFFS = {"backend_connectivity_pollution_20260708": 1225}


def journal_digest(journal_dir: Path) -> dict[str, Any]:
    h = hashlib.sha256()
    files = []
    for shard in journal_shards(journal_dir):
        data = shard.read_bytes()
        h.update(shard.name.encode() + b"\0" + hashlib.sha256(data).digest())
        files.append({"file": shard.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    return {"snapshot_sha256": h.hexdigest(), "shards": files}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--journal", type=Path, required=True, help="journal directory")
    ap.add_argument("--state", type=Path, help="autopilot_state.json (era boundary only)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--only", choices=["mhs16", "seq6", "seq7", "seq5"], action="append")
    args = ap.parse_args(argv)

    era_ts = None
    if args.state and args.state.exists():
        raw = json.loads(args.state.read_text(encoding="utf-8")).get("quality_exclude_before_ts")
        era_ts = float(raw) if isinstance(raw, (int, float)) else None
    digest = journal_digest(args.journal)
    rows = list(iter_journal_rows(args.journal))
    header = {"journal_dir": str(args.journal), "journal": digest, "rows_loaded": len(rows),
              "quality_exclude_before_ts": era_ts}
    jobs = {
        "mhs16": lambda: cumulative_drift_report(rows),
        "seq5": lambda: promotion_attempt_ledger(rows, era_before_ts=era_ts),
        "seq6": lambda: {
            "all_groups_no_retro_tag_rule": frozen_null_replay(rows, era_before_ts=None),
            "all_groups_with_retro_tag_rule": frozen_null_replay(
                rows, era_before_ts=None, retro_tag_cutoffs=RETRO_TAG_CUTOFFS),
            "matching_subset_with_retro_tag_rule": frozen_null_replay(
                rows, era_before_ts=None, retro_tag_cutoffs=RETRO_TAG_CUTOFFS,
                matching_subset_only=True),
        },
        "seq7": lambda: paired_keep_replay(rows),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.only or list(jobs):
        report = {"report": name, **header, "result": jobs[name]()}
        if name == "seq6" and era_ts is not None:
            report["result_with_era_fence"] = frozen_null_replay(rows, era_before_ts=era_ts)
        if name == "seq6":
            for sub in report["result"].values():
                sub.pop("groups", None)
        path = args.out_dir / f"{name}.json"
        path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
