"""HIL-15 loader and controls: do observed-dependency features add AUC over run size?

Expected result: NULL (the GRADE authors found structural gains vanish under run-size and
composition baselines, intake-2329#record). Controls: run size, tool composition, and a
degree-matched counterfeit (edges re-paired at random, same writer/reader degree sequences).
Edge definition compared here is ``observed``; ``intervention-validated`` needs the replay tier (HIL-16).
Intervals resample TASKS, not runs.
"""
from __future__ import annotations

import json
import random
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load_runs(db_path: Path | str) -> dict[str, list[dict]]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    runs: dict[str, list[dict]] = defaultdict(list)
    for (dj,) in conn.execute("select detail_json from event where category='tool_dependency' order by id"):
        rec = json.loads(dj)
        runs[rec["run_id"]].append(rec)
    conn.close()
    return dict(runs)


def _depth(edges: list[tuple[str, str]]) -> int:
    """Longest writer-chain length over (writer_step, reader_step) edges (steps are ordered)."""
    best: dict[str, int] = {}
    for w, r in sorted(edges, key=lambda e: int(e[1].rsplit(":", 1)[1])):
        best[r] = max(best.get(r, 0), best.get(w, 0) + 1)
    return max(best.values(), default=0)


def edges_of(recs: list[dict]) -> list[tuple[str, str]]:
    return [(rd["last_writer"], r["step_id"]) for r in recs for rd in r["reads"] if rd.get("last_writer")]


def dep_features(edges: list[tuple[str, str]]) -> list[float]:
    fan_in = Counter(r for _, r in edges)
    return [float(len(edges)), float(_depth(edges)), float(max(fan_in.values(), default=0))]


def counterfeit_edges(edges: list[tuple[str, str]], rng: random.Random) -> list[tuple[str, str]]:
    """Degree-matched control: shuffle writers across reader slots, keeping only order-valid pairs."""
    writers = [w for w, _ in edges]
    readers = [r for _, r in edges]
    for _ in range(20):
        rng.shuffle(writers)
        pairs = list(zip(writers, readers))
        if all(int(w.rsplit(":", 1)[1]) < int(r.rsplit(":", 1)[1]) for w, r in pairs):
            return pairs
    return [(w, r) for w, r in zip(writers, readers) if int(w.rsplit(":", 1)[1]) < int(r.rsplit(":", 1)[1])]


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def _oof(X: np.ndarray, y: np.ndarray, groups: list[str], k: int = 5, seed: int = 0) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    out = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=min(k, len(set(groups)))).split(X, y, groups):
        if len(set(y[tr])) < 2:   # a training fold with one class cannot fit; predict its base rate (uninformative)
            out[te] = float(y[tr].mean())
            continue
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr], y[tr])
        out[te] = m.predict_proba(X[te])[:, 1]
    return out


def auc_table(runs: dict[str, list[dict]], labels: dict[str, tuple[str, bool]], seed: int = 0, B: int = 500) -> dict:
    """labels: run_id -> (task_key, passed). Outcome modelled: failed (1) vs passed (0)."""
    rng = random.Random(seed)
    ids = [r for r in runs if r in labels]
    tools = sorted({rec["tool"] for r in ids for rec in runs[r]})
    y = np.array([0 if labels[r][1] else 1 for r in ids])
    groups = [labels[r][0] for r in ids]
    if len(set(y)) < 2 or len(set(groups)) < 5:
        return {"status": "insufficient", "n_runs": len(ids), "n_tasks": len(set(groups))}
    size = np.array([[float(len(runs[r]))] for r in ids])
    comp = np.array([[sum(1 for rec in runs[r] if rec["tool"] == t) / max(1, len(runs[r])) for t in tools] for r in ids])
    dep = np.array([dep_features(edges_of(runs[r])) for r in ids])
    fake = np.array([dep_features(counterfeit_edges(edges_of(runs[r]), rng)) for r in ids])
    sets = {
        "size": size, "size+composition": np.hstack([size, comp]),
        "size+composition+observed_dependency": np.hstack([size, comp, dep]),
        "size+composition+degree_matched_counterfeit": np.hstack([size, comp, fake]),
    }
    oof = {k: _oof(X, y, groups, seed=seed) for k, X in sets.items()}
    aucs = {k: round(_auc(y, v), 4) for k, v in oof.items()}
    by_task: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        by_task[g].append(i)
    tasks = list(by_task)
    deltas = {"observed_dependency_minus_composition": ("size+composition+observed_dependency", "size+composition"),
              "observed_dependency_minus_size": ("size+composition+observed_dependency", "size"),
              "counterfeit_minus_composition": ("size+composition+degree_matched_counterfeit", "size+composition")}
    out_d = {}
    for name, (a, b) in deltas.items():
        vals = []
        for _ in range(B):
            idx = [i for g in (rng.choice(tasks) for _ in tasks) for i in by_task[g]]
            yy = y[idx]
            if len(set(yy)) < 2:
                continue
            vals.append(_auc(yy, oof[a][idx]) - _auc(yy, oof[b][idx]))
        vals.sort()
        out_d[name] = {"delta_auc": round(aucs[a] - aucs[b], 4),
                       "ci95_task_resampled": [round(vals[int(0.025 * len(vals))], 4), round(vals[int(0.975 * len(vals)) - 1], 4)] if len(vals) > 20 else None}
    return {"n_runs": len(ids), "n_tasks": len(tasks), "n_failed": int(y.sum()), "auc_out_of_fold": aucs,
            "deltas": out_d, "edge_definition": "observed (intervention-validated needs HIL-16 replay)"}
