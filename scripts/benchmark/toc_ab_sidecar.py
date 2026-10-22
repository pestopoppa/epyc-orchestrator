#!/usr/bin/env python3
"""VB-TOC-AB-WRITE: emit ``toc_ab.beliefs.jsonl`` from trajectory artifacts at run finalize.

Read side of ``src/graph/trajectory_artifact.py`` and the call site of the root belief
carrier ``scripts/vidya/adapters/toc_ab.py::write_belief_measurements`` (same opt-in pattern
as ``scripts/trace/l2_scan.py --belief-sidecar``: the carrier lives in epyc-root, located by
``--epyc-root``).  Never backfills: a task with no ``final`` event, a task whose rolling input
tokens were not observed on every turn, or an arm without graded outcomes is a refusal,
not a guess.

Spec JSON (``--spec``)::

    {"arms": {"<label>": {"policy": {...}            # or "policy_digest": "<64 hex>"
                          "trajectory_dir": "..."    # optional per-arm dir (same task ids across arms)
                          "trigger_fraction": 0.5 | null,
                          "window_denominator": "65536 per slot on :8070",
                          "summary_placement": "tail" | "top" | "none",
                          "is_control": false,
                          "task_ids": ["t1", "t2"]}}}

Outcomes (``--outcomes``): ``{"<label>": {"<task_id>": true|false}}``, the harness's GRADED
verdicts.  ``--outcome-from-finish`` instead takes the graph's finish flag, which is not a
grade and is recorded as such in the run id suffix; use it only for plumbing checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.graph import trajectory_artifact as ta  # noqa: E402


from scripts.lib.epyc_root import default_epyc_root  # noqa: E402


def policy_digest(policy: object) -> str:
    return hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_arms(spec: dict, traj_dir: Path, outcomes: dict | None, outcome_from_finish: bool,
               ) -> tuple[dict, list[Path]]:
    arms: dict = {}
    files: list[Path] = []
    for label, a in spec["arms"].items():
        tasks = []
        for tid in a["task_ids"]:
            path = Path(a.get("trajectory_dir") or traj_dir) / (__import__("re").sub(r"[^A-Za-z0-9._-]", "_", tid) + ".trajectory.jsonl")
            if not path.is_file():
                raise SystemExit(f"arm {label!r}: no trajectory artifact for task {tid!r} ({path})")
            fin = ta.final_event(ta.read_events(path))
            if fin["rolling_input_tokens_observed_turns"] != fin["turns"]:
                raise SystemExit(
                    f"arm {label!r} task {tid!r}: prompt tokens observed on "
                    f"{fin['rolling_input_tokens_observed_turns']} of {fin['turns']} turns; "
                    "rolling input tokens would be a partial sum")
            if outcome_from_finish:
                solved = bool(fin["solved"])
            else:
                try:
                    solved = bool(outcomes[label][tid])
                except (KeyError, TypeError):
                    raise SystemExit(f"arm {label!r} task {tid!r}: no graded outcome in --outcomes")
            tasks.append({
                "task_id": tid, "terminal_state": fin["terminal_state"], "solved": solved,
                "turns": fin["turns"], "rolling_input_tokens": fin["rolling_input_tokens"],
                "prefix_cache_invalidations": fin["prefix_cache_invalidations"],
                "compressor_wall_s": fin["compressor_wall_s"],
            })
            files.append(path)
        arms[label] = {
            "policy_digest": a.get("policy_digest") or policy_digest(a["policy"]),
            "trigger_fraction": a.get("trigger_fraction"),
            "window_denominator": a["window_denominator"],
            "summary_placement": a["summary_placement"], "is_control": a["is_control"],
            "tasks": tasks,
        }
    return arms, files


def consolidate(files: list[Path], out: Path) -> Path:
    """One chronological file per run (per-task files concatenated in arm/task order);
    the carrier re-hashes this file on read."""
    out.write_text("".join(f.read_text(encoding="utf-8") for f in files), encoding="utf-8")
    return out


def write_sidecar(*, spec: dict, traj_dir: Path, out_dir: Path, run_id: str, epyc_root: str,
                  outcomes: dict | None = None, outcome_from_finish: bool = False) -> Path:
    arms, files = build_arms(spec, traj_dir, outcomes, outcome_from_finish)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_traj = consolidate(files, out_dir / f"{run_id}.trajectory.jsonl")
    sys.path.insert(0, str(Path(epyc_root) / "scripts" / "vidya"))
    from adapters.toc_ab import write_belief_measurements  # type: ignore

    return write_belief_measurements(
        out_dir, run_id=run_id, producer="epyc-orchestrator/scripts/benchmark/toc_ab_sidecar.py",
        trajectory_path=run_traj, arms=arms)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trajectory-dir", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--outcomes")
    ap.add_argument("--outcome-from-finish", action="store_true")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epyc-root", default=default_epyc_root("scripts/vidya/adapters/toc_ab.py"))
    a = ap.parse_args(argv)
    if not a.outcomes and not a.outcome_from_finish:
        ap.error("pass --outcomes (graded verdicts) or --outcome-from-finish")
    out = write_sidecar(
        spec=json.loads(Path(a.spec).read_text()), traj_dir=Path(a.trajectory_dir),
        out_dir=Path(a.out_dir), run_id=a.run_id, epyc_root=a.epyc_root,
        outcomes=json.loads(Path(a.outcomes).read_text()) if a.outcomes else None,
        outcome_from_finish=a.outcome_from_finish)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
