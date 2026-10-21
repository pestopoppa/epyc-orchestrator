"""HIL-18 wrapper: replay one task N times, label with the verifier, rank step-type keys.

Inference is injected (``runner``), so everything here is testable without a model. The runner for a
live run (scripts/trace/spectrum_replay.py) refuses unless ``WINDOW_CONFIRMED=1``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from src.trace.l2.detectors import DetectorConfig
from src.trace.l2.loaders import load_step_events
from src.trace.l2.model import L2Trace
from src.trace.spectrum import (eligible_for_spectrum, l2_agreement, overlap_with_human, rank_task,
                                ranking_to_dict)

SCHEMA = "spectrum_report/v1"


@dataclass
class RunOutcome:
    final_answer: str | None = None
    error: str | None = None      # runner failed (HTTP error, timeout): the replay is counted, not labelled


Runner = Callable[[dict, str, int], RunOutcome]       # (task, run_id, replay_index)
Verifier = Callable[[dict, str | None], bool]


def replay_task(task: dict, n: int, runner: Runner, verifier: Verifier, manifest: Path | str,
                id_prefix: str | None = None) -> list[dict]:
    """N replays; every one is appended to ``manifest`` (jsonl) as it finishes. Returns the entries."""
    prefix = id_prefix or f"spec-{task['task_key']}"
    entries = []
    with open(manifest, "a") as fh:
        for i in range(n):
            rid = f"{prefix}-{i:02d}"
            out = runner(task, rid, i)
            outcome = "unknown" if out.error else ("pass" if verifier(task, out.final_answer) else "fail")
            e = {"task_key": task["task_key"], "run_id": rid, "replay": i, "outcome": outcome, "error": out.error}
            fh.write(json.dumps(e, sort_keys=True) + "\n")
            fh.flush()
            entries.append(e)
    return entries


def load_manifest(path: Path | str) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def traces_for_manifest(db_path: Path | str, entries: Iterable[dict]) -> tuple[dict[str, list[L2Trace]], dict[str, Any]]:
    """task_key -> traces. Returns also the join accounting: replays started / with steps / unlabelled."""
    entries = list(entries)
    by_rid = {e["run_id"]: e for e in entries}
    base = {rid: L2Trace(trace_id=rid, task_key=e["task_key"], outcome=e["outcome"]) for rid, e in by_rid.items()}
    got = {t.trace_id: t for t in load_step_events(db_path, outcomes=base) if t.trace_id in by_rid}
    per_task: dict[str, list[L2Trace]] = {}
    for rid, t in got.items():
        per_task.setdefault(by_rid[rid]["task_key"], []).append(t)
    acct = {"replays_started": len(entries), "with_steps": len(got),
            "without_steps": sorted(set(by_rid) - set(got)),
            "outcome_unknown": sum(e["outcome"] == "unknown" for e in entries)}
    return per_task, acct


def build_report(per_task: dict[str, list[L2Trace]], acct: dict, *, n_tasks: int = 10, min_replays: int = 20,
                 top_m: int = 3, human: dict[str, list[list[str]]] | None = None,
                 cfg: DetectorConfig | None = None, label: str = "measurement") -> dict[str, Any]:
    tasks, skipped = [], []
    for tk in sorted(per_task):
        traces = per_task[tk]
        rk = rank_task(traces, tk)
        if not eligible_for_spectrum(rk.n_pass, rk.n_fail, min_replays):
            skipped.append({"task_key": tk, "n_pass": rk.n_pass, "n_fail": rk.n_fail,
                            "reason": "pass-rate not strictly in (0,1) or fewer than %d labelled replays" % min_replays})
            continue
        row = {"ranking": ranking_to_dict(rk), "l2_agreement": l2_agreement(traces, rk, top_m, cfg)}
        if human and tk in human:
            row["human_overlap"] = overlap_with_human(rk, [tuple(k) for k in human[tk]], top_m)
        tasks.append(row)
    tasks = tasks[:n_tasks]
    agg: dict[str, dict[str, int]] = {}
    for t in tasks:
        for cat, v in t["l2_agreement"]["by_category"].items():
            a = agg.setdefault(cat, {"failed_runs_flagged": 0, "top_key_at_flagged_step": 0})
            for f in a:
                a[f] += v[f]
    return {"schema": SCHEMA, "label": label, "target_tasks": n_tasks, "tasks_reported": len(tasks),
            "tasks_skipped": skipped, "join": acct, "top_m": top_m, "tasks": tasks,
            "l2_agreement_pooled": agg,
            "ambiguous_key_note": "keys with status 'unknown' are marked ambiguous per row; no local model was consulted",
            "role_ablation_shapley": "not attempted: gated on a usable ranking and the dependency-edge derivation"}


def dumps(report: dict) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"
