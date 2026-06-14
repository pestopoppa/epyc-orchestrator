#!/usr/bin/env python3
"""Run repeated T1 calibration passes for core_v2 selection.

The selector needs repeated outcomes for the same questions. This runner keeps
that contract explicit: each repeat uses the same T1 ``n`` and ``seed`` and
writes standalone JSONL rows for ``core_v2_select.py``. It does not write the
AutoPilot journal, Pareto archive, baseline state, or short-term memory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ORCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ORCH_ROOT))
sys.path.insert(0, str(ORCH_ROOT / "scripts" / "autopilot"))

from eval_tower import EvalTower  # noqa: E402
from safety_gate import EvalResult  # noqa: E402

DEFAULT_OUTPUT_ROOT = Path("/mnt/raid0/llm/tmp/core_v2_calibration")


def utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_output_path(calibration_id: str) -> Path:
    return DEFAULT_OUTPUT_ROOT / f"{calibration_id}.jsonl"


def configure_tool_sentinels(include_tool_sentinels: bool) -> str | None:
    """Disable tool sentinels for calibration unless explicitly requested."""
    prior = os.environ.get("AUTOPILOT_TOOL_SENTINELS")
    if include_tool_sentinels:
        os.environ["AUTOPILOT_TOOL_SENTINELS"] = "1"
    else:
        os.environ.pop("AUTOPILOT_TOOL_SENTINELS", None)
    return prior


def restore_tool_sentinels(prior: str | None) -> None:
    if prior is None:
        os.environ.pop("AUTOPILOT_TOOL_SENTINELS", None)
    else:
        os.environ["AUTOPILOT_TOOL_SENTINELS"] = prior


def result_to_row(
    *,
    result: EvalResult,
    calibration_id: str,
    repeat_index: int,
    repeats: int,
    requested_n: int,
    seed: int,
    trial_id: int,
    started_at: str,
) -> dict[str, Any]:
    return {
        "event_type": "core_v2_calibration",
        "schema_version": 1,
        "calibration_id": calibration_id,
        "repeat_index": repeat_index,
        "repeats": repeats,
        "requested_n": requested_n,
        "seed": seed,
        "trial_id": trial_id,
        "started_at": started_at,
        "finished_at": utc_iso(),
        "tier": result.tier,
        "quality": result.quality,
        "speed": result.speed,
        "speed_metric_mode": result.speed_metric_mode,
        "median_request_speed": result.median_request_speed,
        "aggregate_speed": result.aggregate_speed,
        "eval_concurrency": result.eval_concurrency,
        "eval_wall_s": result.eval_wall_s,
        "cost": result.cost,
        "reliability": result.reliability,
        "n_questions": result.n_questions,
        "core_id": result.core_id,
        "per_suite_quality": dict(result.per_suite_quality),
        "per_suite_counts": dict(result.per_suite_counts),
        "routing_distribution": dict(result.routing_distribution),
        "eval_details": {
            "question_results": list(result.question_results or []),
            "details": dict(result.details or {}),
        },
    }


def write_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-id", default=f"core_v2_calibration_{utc_compact()}")
    parser.add_argument("--out-jsonl", type=Path)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--trial-id-base", type=int, default=900000)
    parser.add_argument(
        "--include-tool-sentinels",
        action="store_true",
        help="Include tool-use sentinels. Default disables them for core selection calibration.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output JSONL instead of failing closed.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.n <= 0:
        raise SystemExit("--n must be positive")
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")

    out_path = args.out_jsonl or default_output_path(args.calibration_id)
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite to replace: {out_path}")
    if out_path.exists() and args.overwrite:
        out_path.unlink()

    prior_sentinels = configure_tool_sentinels(args.include_tool_sentinels)
    try:
        tower = EvalTower()
        for repeat_index in range(args.repeats):
            trial_id = args.trial_id_base + repeat_index
            started_at = utc_iso()
            print(
                f"core_v2 calibration repeat {repeat_index + 1}/{args.repeats}: "
                f"n={args.n} seed={args.seed} trial_id={trial_id}",
                flush=True,
            )
            result = tower.eval_t1(n=args.n, seed=args.seed, trial_id=trial_id)
            row = result_to_row(
                result=result,
                calibration_id=args.calibration_id,
                repeat_index=repeat_index,
                repeats=args.repeats,
                requested_n=args.n,
                seed=args.seed,
                trial_id=trial_id,
                started_at=started_at,
            )
            write_row(out_path, row)
            print(
                f"  wrote repeat {repeat_index + 1}: q={result.quality:.3f} "
                f"r={result.reliability:.3f} n={result.n_questions} -> {out_path}",
                flush=True,
            )
    finally:
        restore_tool_sentinels(prior_sentinels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
