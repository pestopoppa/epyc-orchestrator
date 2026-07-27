#!/usr/bin/env python3
"""escalation_prediction_probe.py — "will this role fail?" probe for the routing controller.

Implements the **Escalation prediction** surface listed as status *Ready* and never built in
`handoffs/active/learned-routing-controller.md` (Future phases table). master-handoff-index N17
names this predictor as the gate a conditional-depth surface would need: depth should be gated on
*predicted failure*, not pinned.

    Task:   P(outcome == 'failure' | e(x), action = r)
    Input:  the 1024-d BGE vector ALREADY stored in sessions/embeddings.faiss
    Labels: episodic.db `outcome` for action_type='routing' rows

NO INFERENCE IS PERFORMED. This reads a read-only snapshot of the live stores; the vectors already
exist. It is therefore runnable without a clean inference window.

Relationship to the COMP_r probe (scripts/analysis/comp_region_probe.py, 2026-07-22)
-----------------------------------------------------------------------------------
COMP_r returned a clean NULL (pooled AUC 0.497). It asked a *nearest-neighbour similarity* question:
"is this objective close to past successes for role r?" This probe asks a different one: "is failure
**supervised-decodable** from the vector at all?" A linear/MLP probe can find structure a max-cosine
feature cannot, so the null does not settle this — but it is a low prior and the report says so.

Methodology guards (the ones that decide whether the number means anything)
---------------------------------------------------------------------------
1. **Grouped split by objective is MANDATORY.** COMP_r documents that with ~2.4k distinct objectives
   an in-sample neighbour finds a near-duplicate at cos~=1.0 and scores AUC~=1.0 for free. The
   supervised analogue is objective leakage across the train/test boundary, so every split is a
   GroupKFold on the objective string. An ungrouped run is emitted too, but ONLY as a leakage anchor.
2. **Shuffled-label control.** Same pipeline, labels permuted within group structure. Must land at
   ~0.5; if it does not, the harness itself is leaking and no other number is trustworthy.
3. **Base-rate and AP reported alongside AUC.** Failure rates here are ~22-33%, so AUC alone
   flatters; average precision against the base rate is the honest read.

Decision gate (pre-declared, mirroring the COMP_r contract)
------------------------------------------------------------
    grouped AUC <= 0.55  -> record a clean NULL; escalation prediction is not learnable from BGE
                            vectors on current data. That also removes the cheapest route to a
                            conditional-depth surface, so N17 should be re-scoped or closed.
    grouped AUC >= 0.65  -> first real spread; RECOMMEND wiring as a feature. RECOMMEND ONLY —
                            do NOT enable live. `learned-routing-controller.md` is FROZEN for
                            expansion per fable5-findings-02; building and measuring a probe is
                            allowed (COMP_r was completed 2026-07-22 under the same freeze),
                            PROMOTING one is not.
    0.55 < AUC < 0.65    -> inconclusive; report and let the operator decide.

All numbers emitted are OBSERVATIONS per MEASUREMENT.md (no protocol-id, no attestation): usable to
gate THIS research line's keep/close decision, never a production deploy/promote.

Usage:
    # snapshot first (never read the live faiss index while the API is writing it)
    python escalation_prediction_probe.py --make-snapshot --snapshot-dir /mnt/raid0/llm/tmp/esc_snapshot
    python escalation_prediction_probe.py --snapshot-dir /mnt/raid0/llm/tmp/esc_snapshot \
        --out-json orchestration/reports/escalation_prediction_probe/escalation_prediction_probe.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Roles with enough rows for a grouped split to mean anything.
MIN_ROWS_PER_ROLE = 1000
# A class with a handful of members cannot be cross-validated; worker_vision has 4 failures.
MIN_POSITIVES = 50
# Pre-declared decision gate.
NULL_AT = 0.55
SIGNAL_AT = 0.65

LIVE_SESSIONS = Path("orchestration/repl_memory/sessions")


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Snapshot
# --------------------------------------------------------------------------- #
def make_snapshot(snap_dir: Path, live: Path) -> None:
    """Copy the stores so we never read an index the API is concurrently writing."""
    snap_dir.mkdir(parents=True, exist_ok=True)
    for name in ("episodic.db", "embeddings.faiss"):
        src = live / name
        if not src.exists():
            raise SystemExit(f"missing live store: {src}")
        dst = snap_dir / name
        t = time.time()
        shutil.copy2(src, dst)
        log(f"[snap] {name}: {src.stat().st_size / 1e9:.2f} GB in {time.time() - t:.1f}s")


def snapshot_provenance(snap_dir: Path) -> dict:
    prov = {}
    for name in ("episodic.db", "embeddings.faiss"):
        p = snap_dir / name
        st = p.stat()
        prov[name] = {"bytes": st.st_size, "mtime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime))}
    return prov


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load_rows(snap_dir: Path) -> list[tuple]:
    con = sqlite3.connect(f"file:{snap_dir / 'episodic.db'}?mode=ro", uri=True)
    rows = con.execute(
        """
        SELECT embedding_idx, action, outcome, context
        FROM memories
        WHERE action_type='routing'
          AND embedding_idx IS NOT NULL
          AND outcome IN ('success','failure')
        """
    ).fetchall()
    con.close()
    log(f"[load] routing rows with an outcome + embedding: {len(rows)}")
    return rows


def load_vectors(snap_dir: Path, needed_idx: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
    """Reconstruct ONLY the vectors we need.

    COMP_r reconstructs the whole ~2.8 GB matrix. We need ~27k rows, so selective reconstruction
    keeps the memory-bandwidth footprint near zero — this matters because CPU decode on this host is
    bandwidth-bound, and a bench may be running.
    """
    import faiss

    t = time.time()
    index = faiss.read_index(str(snap_dir / "embeddings.faiss"))
    log(f"[load] faiss ntotal={index.ntotal} dim={index.d} ({time.time() - t:.1f}s)")

    in_range = needed_idx[(needed_idx >= 0) & (needed_idx < index.ntotal)]
    uniq = np.unique(in_range)
    log(f"[load] reconstructing {len(uniq)} of {index.ntotal} vectors ({100.0 * len(uniq) / index.ntotal:.2f}%)")

    t = time.time()
    vecs = np.vstack([index.reconstruct(int(i)) for i in uniq]).astype(np.float32)
    log(f"[load] reconstructed {vecs.shape} in {time.time() - t:.1f}s ({vecs.nbytes / 1e6:.0f} MB)")
    return {int(v): i for i, v in enumerate(uniq)}, vecs


# --------------------------------------------------------------------------- #
# Probe
# --------------------------------------------------------------------------- #
def _vector_key(vec: np.ndarray) -> str:
    """Group key = the EMBEDDING itself, not the context string.

    CRITICAL, and it inverted this probe's first result. Grouping by a hash of the `context` column
    looked right and was wrong: measured on live data, 26,995 frontdoor rows carry only **2,384
    distinct vectors** (~11x reuse), and 497 vector hashes span more than one context-derived group.
    The context field holds material beyond the embedded objective, so hashing it splits identical
    vectors into different groups — the same point then lands in train AND test and the model scores
    by lookup. The first run reported grouped AUC 0.770 for frontdoor under context grouping; it is
    an artifact.

    2,384 is the same figure comp_region_probe.py flags in its own docstring. The embedding is the
    only defensible unit of identity here.
    """
    return hashlib.sha1(np.ascontiguousarray(vec, dtype=np.float32).tobytes()).hexdigest()[:16]


def run_probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int, grouped: bool) -> dict:
    """Cross-validated linear probe.

    Two metric views, because they answer different questions and disagree here:

    * **row-weighted** — pooled over every out-of-fold row. Flattered by group-size imbalance: the
      top-10 objectives hold ~33% of frontdoor rows while the MEDIAN group is size 1, so a handful
      of repeated boilerplate prompts can carry the number.
    * **group-weighted** — one prediction per distinct objective (mean score, mean label). This is
      the honest read of "can we rank an UNSEEN objective's failure propensity".

    `grouped=False` is a leakage anchor only, never a result.

    Uses GroupShuffleSplit, not GroupKFold: GroupKFold is deterministic, so varying the seed changed
    nothing and the first stability check silently reported a spread of exactly 0.0000.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

    n_splits = 5
    if grouped:
        splitter = GroupShuffleSplit(n_splits=n_splits, test_size=0.25, random_state=seed).split(X, y, groups)
    else:
        splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.25, random_state=seed).split(X, y)

    row_auc, row_ap, grp_auc, grp_ap = [], [], [], []
    for train_idx, test_idx in splitter:
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=seed)
        clf.fit(X[train_idx], y[train_idx])
        p = clf.predict_proba(X[test_idx])[:, 1]
        yt, gt = y[test_idx], groups[test_idx]

        row_auc.append(float(roc_auc_score(yt, p)))
        row_ap.append(float(average_precision_score(yt, p)))

        # collapse to one point per distinct objective
        order = {g: i for i, g in enumerate(np.unique(gt))}
        gi = np.array([order[g] for g in gt])
        n_g = len(order)
        ps = np.bincount(gi, weights=p, minlength=n_g) / np.bincount(gi, minlength=n_g)
        ys = np.bincount(gi, weights=yt, minlength=n_g) / np.bincount(gi, minlength=n_g)
        yb = (ys >= 0.5).astype(int)
        if len(np.unique(yb)) >= 2:
            grp_auc.append(float(roc_auc_score(yb, ps)))
            grp_ap.append(float(average_precision_score(yb, ps)))

    def summarize(vals):
        if not vals:
            return None
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                "min": float(min(vals)), "max": float(max(vals)), "folds": len(vals)}

    if not row_auc:
        return {"auc": None, "note": "no evaluable folds"}
    return {
        "auc": float(np.mean(row_auc)),                     # headline == row-weighted mean
        "row_weighted": {"auc": summarize(row_auc), "ap": summarize(row_ap)},
        "group_weighted": {"auc": summarize(grp_auc), "ap": summarize(grp_ap)},
        "base_rate": float(y.mean()),
        "n": int(len(y)),
        "n_groups": int(len(np.unique(groups))),
    }


def verdict_for(auc: Optional[float]) -> str:
    if auc is None:
        return "INSUFFICIENT-DATA"
    if auc <= NULL_AT:
        return f"NULL — AUC<={NULL_AT}: failure is not decodable from the BGE vector."
    if auc >= SIGNAL_AT:
        return f"SIGNAL — AUC>={SIGNAL_AT}: RECOMMEND as a feature (do NOT enable live; handoff is frozen for expansion)."
    return f"INCONCLUSIVE — {NULL_AT} < AUC < {SIGNAL_AT}: operator decision."


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot-dir", default="/mnt/raid0/llm/tmp/esc_snapshot")
    ap.add_argument("--live-dir", default=str(LIVE_SESSIONS))
    ap.add_argument("--make-snapshot", action="store_true", help="copy live stores into --snapshot-dir first")
    ap.add_argument("--out-json", default="orchestration/reports/escalation_prediction_probe/escalation_prediction_probe.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    snap = Path(args.snapshot_dir)
    if args.make_snapshot:
        make_snapshot(snap, Path(args.live_dir))
    if not (snap / "embeddings.faiss").exists():
        raise SystemExit(f"no snapshot at {snap}; re-run with --make-snapshot")

    rows = load_rows(snap)
    idx = np.array([r[0] for r in rows], dtype=np.int64)
    action = np.array([r[1] or "" for r in rows], dtype=object)
    label = np.array([1 if r[2] == "failure" else 0 for r in rows], dtype=np.int64)
    pos, vecs = load_vectors(snap, idx)
    keep = np.array([i in pos for i in idx])
    log(f"[probe] rows with a reconstructable vector: {keep.sum()} / {len(rows)}")
    idx, action, label = idx[keep], action[keep], label[keep]
    X_all = vecs[np.array([pos[int(i)] for i in idx])]
    # Group by the EMBEDDING, not the context string — see _vector_key.
    group = np.array([_vector_key(X_all[i]) for i in range(len(X_all))], dtype=object)
    log(f"[probe] distinct vectors: {len(np.unique(group))} across {len(group)} rows "
        f"({len(group) / max(1, len(np.unique(group))):.1f}x reuse)")

    results: dict[str, dict] = {}
    roles = sorted({a for a in action}, key=lambda a: -int((action == a).sum()))
    for role in roles:
        sel = action == role
        n = int(sel.sum())
        if n < MIN_ROWS_PER_ROLE:
            continue
        y = label[sel]
        if len(np.unique(y)) < 2:
            results[role] = {"n": n, "skipped": "single-class outcome"}
            continue
        if int(y.sum()) < MIN_POSITIVES or int((1 - y).sum()) < MIN_POSITIVES:
            results[role] = {"n": n, "positives": int(y.sum()),
                             "skipped": f"fewer than {MIN_POSITIVES} in a class — not evaluable"}
            log(f"[probe] {role}: SKIPPED (positives={int(y.sum())})")
            continue
        X, g = X_all[sel], group[sel]
        log(f"[probe] {role}: n={n} failures={int(y.sum())} ({100.0 * y.mean():.1f}%) groups={len(np.unique(g))}")

        grouped = run_probe(X, y, g, args.seed, grouped=True)
        # How concentrated is the data? If a few objectives carry most rows, the row-weighted
        # number is about them, not about generalization.
        _, counts = np.unique(g, return_counts=True)
        counts = np.sort(counts)[::-1]
        grouped["group_size_concentration"] = {
            "top10_row_share": float(counts[:10].sum() / counts.sum()),
            "median_group_size": int(np.median(counts)),
            "max_group_size": int(counts[0]),
        }
        ungrouped = run_probe(X, y, g, args.seed, grouped=False)
        rng = np.random.default_rng(args.seed)
        shuffled = run_probe(X, rng.permutation(y), g, args.seed, grouped=True)

        results[role] = {
            "n": n,
            "grouped": grouped,
            "ungrouped_leakage_anchor": ungrouped,
            "shuffled_label_control": shuffled,
            "verdict": verdict_for(((grouped.get("group_weighted") or {}).get("auc") or {}).get("mean")),
            "verdict_basis": "group-weighted AUC (one point per distinct objective)",
        }
        gw = (grouped.get("group_weighted") or {}).get("auc") or {}
        rw = (grouped.get("row_weighted") or {}).get("auc") or {}
        log(
            f"[probe] {role}: GROUP-weighted AUC={gw.get('mean', float('nan')):.4f} "
            f"+/-{gw.get('std', float('nan')):.4f} (min {gw.get('min', float('nan')):.3f} "
            f"max {gw.get('max', float('nan')):.3f}) | row-weighted={rw.get('mean', float('nan')):.4f} "
            f"| ungrouped-anchor={ungrouped.get('auc')} | shuffled={shuffled.get('auc')}"
        )

    def _gw(r):
        return ((r.get("grouped") or {}).get("group_weighted") or {}).get("auc", {}) or {}
    evaluable = [r for r in results.values() if _gw(r).get("mean") is not None]
    best = max((_gw(r)["mean"] for r in evaluable), default=None)
    payload = {
        "probe": "escalation_prediction",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inference_performed": False,
        "measurement_grade": "observation (no protocol-id, no attestation) per MEASUREMENT.md",
        "decision_gate": {"null_at": NULL_AT, "signal_at": SIGNAL_AT},
        "snapshot_provenance": snapshot_provenance(snap),
        "per_role": results,
        "best_grouped_auc": best,
        "verdict": verdict_for(best),
        "prior_context": (
            "COMP_r (comp_region_probe.py, 2026-07-22) returned a clean NULL at pooled AUC 0.497 on a "
            "nearest-neighbour similarity formulation. This probe tests supervised decodability, a "
            "different question, but the prior is low."
        ),
    }

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    log(f"\n[out] {out}")
    log(f"[verdict] {payload['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
