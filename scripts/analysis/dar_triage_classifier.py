#!/usr/bin/env python3
"""DAR handoff L624 — DAR-3 (rescoped) triage-gate classifier (zero inference).

Rescoped DAR-3 is NOT a global argmax policy; it is a TRIAGE GATE that separates
  - cost-dominated objectives (any role clears quality; choose cheapest/fastest)
  - quality-decisive objectives (role choice changes the outcome)
so a policy can be applied only where it matters (~22% of matched objectives; a
floor at low tiers). This trains that gate over the counterfactuals ALREADY in
the store (no 10% epsilon-greedy exploration -- 386K counterfactual decisions
exist for free) and reports LIFT on the decisive subset, never over all traffic.

Label: decisive (within-objective best-worst role success gap > gap threshold).
Features (PROMPT-SIDE only, no outcome leakage): BGE objective embedding (PCA)
+ task_class band + objective length + priority. A leakage upper bound using the
objective's overall difficulty is reported separately and clearly labelled.

OBSERVATION only (MEASUREMENT.md).

Usage:
    python scripts/analysis/dar_triage_classifier.py [--gap 0.05] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import dar_common as dc


def _precision_at_k_lift(y_true, scores, base_rate):
    """Precision among the top-(base_rate) fraction ranked by score, / base_rate."""
    n = len(y_true)
    k = max(1, int(round(base_rate * n)))
    order = np.argsort(-scores)
    top = order[:k]
    prec = float(np.mean(y_true[top]))
    return prec, prec / base_rate if base_rate > 0 else 0.0


def run(gap_thresh: float, out_dir: Path | None) -> dict:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    rows = dc.load_rows("routing")
    _, matched = dc.matched_set(rows, min_obs=5, min_roles=2)
    gaps = {obj: dc.objective_role_gap(stats) for obj, stats in matched.items()}

    # one representative row (embedding + features) per matched objective
    obj_emb_idx = {}
    obj_len = {}
    obj_band = {}
    obj_prio = {}
    obj_succ = {}
    for r in rows:
        if r.objective in matched and r.action_type == "routing":
            obj_emb_idx.setdefault(r.objective, r.emb_idx)
            obj_len.setdefault(r.objective, len(r.objective or ""))
            obj_band.setdefault(r.objective, dc.task_class_band(r.task_type, r.objective))
            obj_prio.setdefault(r.objective, r.priority or "")
    for obj, stats in matched.items():
        tot_n = sum(n for (n, s) in stats.values())
        tot_s = sum(s for (n, s) in stats.values())
        obj_succ[obj] = tot_s / tot_n if tot_n else 0.0

    objs = list(matched.keys())
    y = np.array([1 if gaps[o] > gap_thresh else 0 for o in objs])
    base_rate = float(y.mean())

    # embeddings
    emb = dc.Embeddings()
    X_emb, mask = emb.matrix([obj_emb_idx[o] for o in objs])
    # keep only objectives with a real embedding
    keep = np.where(mask)[0]
    objs = [objs[i] for i in keep]
    y = y[keep]
    X_emb = X_emb[keep]
    base_rate = float(y.mean())

    bands = sorted(set(obj_band[o] for o in objs))
    band_oh = np.array([[1.0 if obj_band[o] == b else 0.0 for b in bands] for o in objs])
    lens = np.array([[obj_len[o]] for o in objs], dtype=float)

    # PCA the embedding to control 1024-d overfitting on ~600 samples
    n_comp = min(30, X_emb.shape[0] - 1, X_emb.shape[1])
    Xp = PCA(n_components=n_comp, random_state=0).fit_transform(X_emb)
    X_prompt = np.hstack([Xp, band_oh, lens])
    X_prompt = StandardScaler().fit_transform(X_prompt)

    # leakage upper-bound feature set: add overall difficulty (=1-success)
    diff = np.array([[1.0 - obj_succ[o]] for o in objs])
    X_leak = StandardScaler().fit_transform(np.hstack([X_prompt, diff]))

    def cv_eval(X):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        oof = np.zeros(len(y))
        for tr, te in skf.split(X, y):
            clf = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")
            clf.fit(X[tr], y[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
        auc = roc_auc_score(y, oof)
        prec, lift = _precision_at_k_lift(y, oof, base_rate)
        return auc, prec, lift, oof

    auc_p, prec_p, lift_p, oof_p = cv_eval(X_prompt)
    auc_l, prec_l, lift_l, _ = cv_eval(X_leak)

    result = {
        "task": "DAR-L624 triage-gate classifier (cost-dominated vs quality-decisive)",
        "snapshot": dc.snapshot_meta(),
        "gap_threshold_pp": gap_thresh * 100,
        "n_objectives": len(objs),
        "decisive_base_rate": base_rate,
        "features_prompt_side": {
            "pca_components": n_comp,
            "bands": bands,
            "cv_auc": auc_p,
            "precision_at_base_rate_k": prec_p,
            "precision_lift_vs_base": lift_p,
        },
        "features_with_difficulty_LEAKAGE_upper_bound": {
            "cv_auc": auc_l,
            "precision_at_base_rate_k": prec_l,
            "precision_lift_vs_base": lift_l,
            "note": "includes outcome-derived difficulty; upper bound, NOT deployable pre-decision",
        },
        "interpretation": (
            "prompt-side AUC measures whether decisiveness is predictable BEFORE "
            "running any model. Precision-lift is how much a top-k triage gate "
            "concentrates the decisive minority vs random selection."),
    }

    print("[DAR-L624] triage-gate classifier (decisive vs cost-dominated)")
    print(f"snapshot {result['snapshot']['snapshot_ts_utc']}")
    print(f"objectives: {len(objs):,}   decisive base rate: {base_rate*100:.1f}%")
    print("\nPROMPT-SIDE features (BGE-PCA + task_class band + length):")
    print(f"  5-fold CV AUC            : {auc_p:.3f}")
    print(f"  precision @ top-{base_rate*100:.0f}%      : {prec_p*100:.1f}%  "
          f"(lift {lift_p:.2f}x vs {base_rate*100:.1f}% base)")
    print("\nWITH-DIFFICULTY (leakage upper bound, NOT deployable):")
    print(f"  5-fold CV AUC            : {auc_l:.3f}")
    print(f"  precision @ top-{base_rate*100:.0f}%      : {prec_l*100:.1f}%  (lift {lift_l:.2f}x)")

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "dar_triage_classifier.json").write_text(json.dumps(result, indent=2))
        print(f"\nartifact: {out_dir/'dar_triage_classifier.json'}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gap", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    run(args.gap, args.out)


if __name__ == "__main__":
    main()
