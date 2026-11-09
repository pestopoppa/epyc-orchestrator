#!/usr/bin/env python3
"""AP-68 (S3-AP-06): are planner-authored numeric trials the worst proposer arm?

Per surface and parameter, compares PLANNER-sourced numeric trials (explicit ``params`` on the
journaled action) with SAMPLER-sourced ones (Optuna suggestions, recovered from the Optuna study
DB because the journal stores ``"params": {}`` for them). For each arm it reports

* ``n`` trials,
* ``spread``: population std of the value normalized to the parameter's bounds (0..1),
* ``mean_step``: mean absolute change in normalized value between consecutive trials of the same
  stream (a planner stream = a surface's planner trials in journal order; a sampler stream = one
  Optuna study in trial order).

This decides AP-67's priority. The exact-repeat rate of (surface, param, value) proposals is NOT
computed here: AP-53's zero-compute re-proposal count owns it (measured once, not twice).

Read-only: journal shards through ``journal_shards.py`` and the Optuna DB opened ``mode=ro``;
no inference, no network, no optuna import.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.autopilot.journal_shards import journal_shards  # noqa: E402


def _load_surfaces() -> dict[str, list[Any]]:
    """Load SURFACES from numeric_swarm.py by path: the species package __init__ pulls the whole
    seeding stack (numpy, httpx), which a read-only analysis must not need."""
    import importlib.util

    path = ROOT / "scripts" / "autopilot" / "species" / "numeric_swarm.py"
    spec = importlib.util.spec_from_file_location("_numeric_swarm_surfaces", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("_numeric_swarm_surfaces", module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.SURFACES


SURFACES = _load_surfaces()

DEFAULT_ORCH = Path("/mnt/raid0/llm/epyc-orchestrator/orchestration")


def _bounds() -> dict[str, dict[str, tuple[float, float]]]:
    return {s: {p.name: (float(p.low), float(p.high)) for p in specs} for s, specs in SURFACES.items()}


def normalize(bounds: dict[str, dict[str, tuple[float, float]]], surface: str, name: str,
              value: Any) -> float | None:
    """Value normalized to the parameter's bounds; None when unknown/non-numeric/degenerate."""
    span = bounds.get(surface, {}).get(name)
    if span is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    low, high = span
    if not math.isfinite(value) or high <= low:
        return None
    return (float(value) - low) / (high - low)


def planner_trials(journal_dir: Path) -> list[dict[str, Any]]:
    """Journaled numeric trials whose action carried explicit params (planner-authored)."""
    out: list[dict[str, Any]] = []
    for shard in journal_shards(journal_dir):
        with shard.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict) or row.get("action_type") != "numeric_trial":
                    continue
                try:
                    action = json.loads(row.get("reasoning") or "{}")
                except json.JSONDecodeError:
                    continue
                params = action.get("params") if isinstance(action, dict) else None
                if isinstance(params, dict) and params and action.get("surface"):
                    out.append({"trial_id": row.get("trial_id"), "surface": str(action["surface"]),
                                "params": params, "outcome_status": row.get("outcome_status", "")})
    out.sort(key=lambda t: (t["trial_id"] is None, t["trial_id"]))
    return out


def _surface_of(study_name: str) -> str | None:
    if not study_name.startswith("autopilot_"):
        return None
    rest = study_name[len("autopilot_"):]
    for surface in sorted(SURFACES, key=len, reverse=True):
        if rest == surface or rest.startswith(surface + "_"):
            return surface
    return None


def sampler_streams(db_path: Path) -> list[dict[str, Any]]:
    """One stream per Optuna study: COMPLETE trials in trial order with their params."""
    if not db_path.is_file():
        return []
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        streams = []
        for study_id, name in con.execute("select study_id, study_name from studies"):
            surface = _surface_of(name)
            if surface is None:
                continue
            trials: list[dict[str, float]] = []
            for trial_id, in con.execute(
                "select trial_id from trials where study_id=? and state='COMPLETE' order by number",
                (study_id,),
            ):
                params = {n: v for n, v in con.execute(
                    "select param_name, param_value from trial_params where trial_id=?", (trial_id,))}
                if params:
                    trials.append(params)
            if trials:
                streams.append({"study": name, "surface": surface, "trials": trials})
        return streams
    finally:
        con.close()


def _stats(streams: list[list[float]]) -> dict[str, Any]:
    values = [v for s in streams for v in s]
    if not values:
        return {"n": 0, "spread": None, "mean_step": None}
    mean = sum(values) / len(values)
    spread = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
    steps = [abs(b - a) for s in streams for a, b in zip(s, s[1:])]
    return {"n": len(values), "spread": round(spread, 6),
            "mean_step": round(sum(steps) / len(steps), 6) if steps else None}


def build_report(journal_dir: Path, optuna_db: Path) -> dict[str, Any]:
    bounds = _bounds()
    planner = planner_trials(journal_dir)
    sampler = sampler_streams(optuna_db)
    cells: dict[tuple[str, str], dict[str, list[list[float]]]] = {}
    planner_streams: dict[tuple[str, str], list[float]] = {}
    for t in planner:
        for name, raw in t["params"].items():
            norm = normalize(bounds, t["surface"], name, raw)
            if norm is not None:
                planner_streams.setdefault((t["surface"], name), []).append(norm)
    for key, stream in planner_streams.items():
        cells.setdefault(key, {"planner": [], "sampler": []})["planner"].append(stream)
    for s in sampler:
        per_param: dict[str, list[float]] = {}
        for params in s["trials"]:
            for name, raw in params.items():
                norm = normalize(bounds, s["surface"], name, raw)
                if norm is not None:
                    per_param.setdefault(name, []).append(norm)
        for name, stream in per_param.items():
            cells.setdefault((s["surface"], name), {"planner": [], "sampler": []})["sampler"].append(stream)
    rows = []
    for (surface, name), arms in sorted(cells.items()):
        p, q = _stats(arms["planner"]), _stats(arms["sampler"])
        rows.append({"surface": surface, "param": name, "planner": p, "sampler": q})
    comparable = [r for r in rows if r["planner"]["n"] >= 3 and r["sampler"]["n"] >= 3]
    worse_spread = [r for r in comparable if r["planner"]["spread"] < r["sampler"]["spread"]]
    worse_step = [r for r in comparable if (r["planner"]["mean_step"] or 0) < (r["sampler"]["mean_step"] or 0)]
    return {
        "schema_version": 1,
        "planner_trials": len(planner),
        "sampler_studies": len(sampler),
        "sampler_trials": sum(len(s["trials"]) for s in sampler),
        "cells": rows,
        "summary": {
            "cells_with_both_arms_n_ge_3": len(comparable),
            "planner_narrower_spread": len(worse_spread),
            "planner_smaller_step": len(worse_step),
            "note": ("planner arm = explicit params on the journaled action; sampler arm = Optuna "
                     "study trials (journal rows store params={} for sampler trials). Values are "
                     "normalized to parameter bounds, linear scale. Exact-repeat rate is AP-53's."),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = ["# Numeric trial arms: planner vs sampler", "",
             f"Planner trials: {report['planner_trials']}; sampler: {report['sampler_trials']} trials "
             f"in {report['sampler_studies']} studies. Cells with both arms (n>=3): "
             f"{s['cells_with_both_arms_n_ge_3']}; planner narrower spread in {s['planner_narrower_spread']}, "
             f"smaller step in {s['planner_smaller_step']}.", "",
             "| surface | param | planner n/spread/step | sampler n/spread/step |", "|---|---|---|---|"]
    for r in report["cells"]:
        a, b = r["planner"], r["sampler"]
        lines.append(f"| {r['surface']} | {r['param']} | {a['n']}/{a['spread']}/{a['mean_step']} "
                     f"| {b['n']}/{b['spread']}/{b['mean_step']} |")
    lines += ["", s["note"], ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--journal-dir", type=Path, default=DEFAULT_ORCH)
    ap.add_argument("--optuna-db", type=Path, default=DEFAULT_ORCH / "optuna_study.db")
    ap.add_argument("--format", choices=("md", "json"), default="md")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    report = build_report(args.journal_dir, args.optuna_db)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n" if args.format == "json" else render_markdown(report)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
