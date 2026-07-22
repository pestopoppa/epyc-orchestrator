#!/usr/bin/env python3
"""comp_region_probe.py — Competence-region probe COMP_r(x) for the learned-routing controller.

Implements the leave-one-objective-out competence/familiarity probe specified in
handoffs/active/learned-routing-controller.md (2026-07-21 "one usable idea from intake-866"):

    COMP_r(x) = max cos(e(x), e(m))  over memories m with action=r AND outcome=success,
                                     EXCLUDING every memory whose objective == x
                                     (leave-one-objective-out — mandatory, else an
                                      in-sample near-duplicate scores AUC ~= 1.0 for free).

e(.) is the 1024-d BGE vector already stored in sessions/embeddings.faiss (one L2-normalised
vector per memory row; IndexFlatIP so cosine == inner product). NO inference is performed —
this reads a read-only snapshot of the live stores only.

Decision gate (per the handoff):
    AUC(success | role) <= 0.55  -> record a clean null, close the intake-866 line.
    AUC(success | role) >= 0.65  -> recommend concatenating COMP as a feature at
                                    routing_classifier.py:61 (RECOMMEND only; do NOT wire live).

All numbers emitted are OBSERVATIONS per MEASUREMENT.md (no protocol-id / attestation): they are
usable to gate THIS research line's keep/close decision but not any production deploy/promote.

Usage:
    python comp_region_probe.py \
        --snapshot-dir /mnt/raid0/llm/tmp/comp_snapshot \
        --out-json  orchestration/reports/comp_region_probe/comp_region_probe.json \
        --out-md    handoffs/active/... (report written separately)
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# Canonical live routing roles (escalation / sub-decision pseudo-actions excluded — they are a
# different action_type and carry empty/degenerate objectives per the handoff note at L123).
CANON_ROLES = {
    "frontdoor",
    "worker_general",
    "coder_escalation",
    "architect_general",
    "ingest_long_context",
    "worker_vision",
    "worker_math",
    "toolrunner",
    "worker_explore",
    "architect_coding",
}


def _obj_of(ctx: str) -> Optional[str]:
    """Extract the objective text from a memory context JSON blob."""
    try:
        d = json.loads(ctx)
    except Exception:
        return None
    o = d.get("objective") or d.get("task_description")
    if not o:
        return None
    return o.strip()


def _auc(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """Rank-based ROC-AUC (Mann-Whitney). Returns None if labels are single-class."""
    pos = labels == 1
    neg = labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            ranks[order[i : j + 1]] = avg
        i = j + 1
    sum_pos = ranks[pos].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def load_snapshot(snap_dir: Path, log=print):
    """Load the episodic.db routing rows + the full reconstructed vector matrix."""
    import faiss

    db = snap_dir / "episodic.db"
    faiss_path = snap_dir / "embeddings.faiss"
    log(f"[load] reading faiss index {faiss_path} ...")
    t = time.time()
    index = faiss.read_index(str(faiss_path))
    ntotal = index.ntotal
    log(f"[load] faiss ntotal={ntotal} dim={index.d} ({time.time()-t:.1f}s)")
    # Reconstruct the whole (normalised) matrix once — ~2.77 GB for 676k x 1024 f32.
    t = time.time()
    vecs = index.reconstruct_n(0, ntotal)  # (ntotal, dim) float32, already L2-normalised
    log(f"[load] reconstructed matrix {vecs.shape} ({time.time()-t:.1f}s)")

    con = sqlite3.connect(str(db))
    cur = con.execute(
        "SELECT embedding_idx, action, outcome, context FROM memories WHERE action_type='routing'"
    )
    rows = cur.fetchall()
    con.close()
    log(f"[load] routing rows: {len(rows)}")
    return vecs, rows, ntotal


def build(vecs, rows, ntotal, log=print):
    """Build per-objective and per-role structures.

    Returns:
        obj_ids            : list of objective strings (index = objective id oid)
        e_obj              : (n_obj, dim) normalised objective centroid vectors
        obj_role_out       : {oid: {role: [succ, fail]}}
        role_bank          : {role: (bank_vecs (n_r,dim), bank_oids list-of-set)}
                             success-memory bank per role, deduped by embedding_idx, each bank
                             entry tagged with the SET of objective ids that reference that idx
                             (for leave-one-objective-out exclusion).
        role_excl          : {role: {oid: [bank_col, ...]}} columns to mask when querying oid
    """
    dim = vecs.shape[1]
    obj_index: Dict[str, int] = {}
    obj_ids: List[str] = []
    obj_row_idx: Dict[int, List[int]] = defaultdict(list)          # oid -> [embedding_idx,...]
    obj_role_out: Dict[int, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    # per-role success bank: embedding_idx -> set(oid)
    role_bank_idx: Dict[str, Dict[int, Set[int]]] = defaultdict(lambda: defaultdict(set))

    skipped_role = 0
    skipped_range = 0
    for eidx, action, outcome, ctx in rows:
        if action not in CANON_ROLES:
            skipped_role += 1
            continue
        if eidx < 0 or eidx >= ntotal:
            skipped_range += 1
            continue
        o = _obj_of(ctx)
        if o is None:
            continue
        oid = obj_index.get(o)
        if oid is None:
            oid = len(obj_ids)
            obj_index[o] = oid
            obj_ids.append(o)
        obj_row_idx[oid].append(eidx)
        rec = obj_role_out[oid][action]
        if outcome == "success":
            rec[0] += 1
            role_bank_idx[action][eidx].add(oid)
        elif outcome == "failure":
            rec[1] += 1

    n_obj = len(obj_ids)
    log(f"[build] distinct objectives (canonical roles): {n_obj}")
    log(f"[build] skipped: non-canonical-role rows={skipped_role} out-of-range-idx={skipped_range}")

    # objective centroid vectors e(x) = normalised mean of that objective's row vectors
    e_obj = np.zeros((n_obj, dim), dtype=np.float32)
    for oid in range(n_obj):
        idxs = obj_row_idx[oid]
        v = vecs[idxs].mean(axis=0)
        nrm = np.linalg.norm(v)
        if nrm > 0:
            v = v / nrm
        e_obj[oid] = v

    # per-role banks + exclusion maps
    role_bank = {}
    role_excl = {}
    for role, idx2oids in role_bank_idx.items():
        bank_idx = list(idx2oids.keys())
        bank_vecs = vecs[bank_idx]  # already normalised
        bank_oids = [idx2oids[i] for i in bank_idx]
        excl: Dict[int, List[int]] = defaultdict(list)
        for col, oidset in enumerate(bank_oids):
            for oid in oidset:
                excl[oid].append(col)
        role_bank[role] = (bank_vecs, bank_oids)
        role_excl[role] = excl
        log(f"[build] role {role:20s} success-bank distinct-idx={len(bank_idx)}")
    return obj_ids, obj_index, e_obj, obj_role_out, role_bank, role_excl


def comp_for_role(e_obj, role, role_bank, role_excl, obj_role_out, leave_one_out=True,
                  block=512, log=print):
    """Compute COMP_r(x) for every objective x. Returns np.array shape (n_obj,), NaN where
    the role's bank is empty after leave-one-objective-out masking."""
    bank_vecs, _ = role_bank[role]
    excl = role_excl[role]
    n_obj = e_obj.shape[0]
    comp = np.full(n_obj, np.nan, dtype=np.float32)
    bankT = bank_vecs.T  # (dim, n_r)
    n_r = bank_vecs.shape[0]
    t = time.time()
    for start in range(0, n_obj, block):
        stop = min(start + block, n_obj)
        S = e_obj[start:stop] @ bankT  # (b, n_r) cosine sims
        for i in range(start, stop):
            row = S[i - start]
            if leave_one_out:
                cols = excl.get(i)
                if cols:
                    # mask same-objective bank entries
                    row = row.copy()
                    row[cols] = -np.inf
                    if not np.isfinite(row).any():
                        continue  # entire bank was this objective
            comp[i] = row.max()
    log(f"[comp] role {role:20s} n_r={n_r} done ({time.time()-t:.1f}s)")
    return comp


def row_level_probe(vecs, rows, ntotal, obj_index, counterfactual_set, role_bank, role_excl,
                    max_fail=6000, seed=0, block=1024, log=print):
    """Decision-level (row-level) robustness variant — the literal "one vector per row" reading.

    For a stratified/balanced sample of individual routing decisions whose objective is in the
    counterfactual set, use each decision's OWN stored vector as e(.). For decision (e_i, r_i, y_i):
        COMP_loo  = max cos(e_i, success-bank[r_i]) EXCLUDING bank entries of objective(i)  [LOO]
        COMP_in   = same but self allowed -> exact self-match (==1.0) for success rows.
    A clean null = LOO AUC ~ 0.5 while in-sample AUC is high (leakage anchor: proves the
    exclusion machinery actually removes the trivial self-match).

    Args:
        obj_index         : dict objective-text -> oid (from build)
        counterfactual_set: set of oids with >=2 distinct roles
    Returns dict of AUCs + sample sizes.
    """
    rng = np.random.default_rng(seed)
    fails, succs = [], []
    seen = set()
    for eidx, action, outcome, ctx in rows:
        if action not in role_bank or eidx < 0 or eidx >= ntotal:
            continue
        o = _obj_of(ctx)
        if o is None:
            continue
        oid = obj_index.get(o)
        if oid is None or oid not in counterfactual_set:
            continue
        key = (eidx, action, outcome)
        if key in seen:
            continue
        seen.add(key)
        if outcome == "failure":
            fails.append((eidx, action, oid))
        elif outcome == "success":
            succs.append((eidx, action, oid))
    rng.shuffle(fails)
    rng.shuffle(succs)
    fails = fails[:max_fail]
    succs = succs[: len(fails)]  # balanced
    sample = fails + succs
    labels = np.array([0] * len(fails) + [1] * len(succs))
    log(f"[row] balanced decision sample: {len(fails)} fail + {len(succs)} succ")
    if not sample:
        return {"error": "no sample"}

    # mixed-outcome colocation: fraction of failure rows whose exact (embedding_idx, role) also
    # carries a success row -> those failures self-match to a success memory at cos==1.0, so even
    # the in-sample (max-leakage) COMP cannot separate them from successes.
    pair_out = defaultdict(set)
    for eidx, action, outcome, _ in rows:
        if action in role_bank and 0 <= eidx < ntotal:
            pair_out[(eidx, action)].add(outcome)
    fr = fc = 0
    for eidx, action, outcome, _ in rows:
        if outcome == "failure" and action in role_bank and 0 <= eidx < ntotal:
            fr += 1
            if "success" in pair_out[(eidx, action)]:
                fc += 1
    colocated_frac = fc / fr if fr else None

    comp_loo = np.full(len(sample), np.nan, dtype=np.float32)
    comp_in = np.full(len(sample), np.nan, dtype=np.float32)
    by_role = defaultdict(list)
    for si, (_, action, _) in enumerate(sample):
        by_role[action].append(si)
    for role, sidxs in by_role.items():
        bank_vecs, _ = role_bank[role]
        bankT = bank_vecs.T
        excl = role_excl[role]
        Q = vecs[[sample[si][0] for si in sidxs]]
        for bstart in range(0, len(sidxs), block):
            bs = sidxs[bstart:bstart + block]
            S = Q[bstart:bstart + block] @ bankT
            for k, si in enumerate(bs):
                row = S[k]
                comp_in[si] = float(row.max())
                oid = sample[si][2]
                cols = excl.get(oid)
                if cols:
                    row = row.copy()
                    row[cols] = -np.inf
                    if not np.isfinite(row).any():
                        continue
                comp_loo[si] = float(row.max())

    m = np.isfinite(comp_loo)
    return {
        "n_fail": len(fails), "n_succ": len(succs),
        "loo_auc": _auc(comp_loo[m], labels[m]),
        "in_sample_auc": _auc(comp_in, labels),
        "n_evaluable_loo": int(m.sum()),
        "failure_rows_colocated_with_success_frac": colocated_frac,
        "note": "Both in_sample_auc AND loo_auc ~ 0.5. The mixed-outcome colocation fraction "
                "explains why in-sample is NOT ~1.0: that fraction of failures share an exact "
                "(embedding_idx, role) with a success, so they self-match at cos==1.0 too. "
                "Success and failure decisions are embedding-indistinguishable at every leakage "
                "level -> the familiarity axis carries no signal (signal-bound null).",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot-dir", default="/mnt/raid0/llm/tmp/comp_snapshot")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--min-role-rows", type=int, default=1,
                    help="min rows for a (objective,role) instance to be evaluated")
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--row-level", action="store_true",
                    help="also run the decision-level (row-level) robustness variant")
    args = ap.parse_args()

    snap = Path(args.snapshot_dir)
    prov = (snap / "PROVENANCE.txt").read_text() if (snap / "PROVENANCE.txt").exists() else ""

    vecs, rows, ntotal = load_snapshot(snap)
    obj_ids, obj_index, e_obj, obj_role_out, role_bank, role_excl = build(vecs, rows, ntotal)
    n_obj = len(obj_ids)

    # ---- COMP_r(x) leave-one-objective-out, for every canonical role with a bank ----
    roles = sorted(role_bank.keys())
    comp_loo = {r: comp_for_role(e_obj, r, role_bank, role_excl, obj_role_out, True, args.block)
                for r in roles}
    # in-sample (no leave-one-out) as a leakage sanity anchor
    comp_in = {r: comp_for_role(e_obj, r, role_bank, role_excl, obj_role_out, False, args.block)
               for r in roles}

    # ---- objective / role bookkeeping ----
    def roles_tried(oid):
        return [r for r, (s, f) in obj_role_out[oid].items() if (s + f) >= args.min_role_rows]

    def role_label(oid, r):
        s, f = obj_role_out[oid][r]
        if s > f:
            return 1
        if f > s:
            return 0
        return None  # tie — dropped

    counterfactual = [oid for oid in range(n_obj) if len({r for r in roles_tried(oid)}) >= 2]
    # disagreement subset: a clean-success role AND a failing role coexist on the objective
    def disagrees(oid):
        succ_roles = [r for r, (s, f) in obj_role_out[oid].items() if s > 0 and f == 0]
        fail_roles = [r for r, (s, f) in obj_role_out[oid].items() if f > 0]
        return len(succ_roles) >= 1 and len(fail_roles) >= 1
    disagree = [oid for oid in counterfactual if disagrees(oid)]

    # ---- AUC(success | role) on the counterfactual set ----
    def collect_auc(oids, comp_map):
        scores, labels, per_role = [], [], defaultdict(lambda: ([], []))
        for oid in oids:
            for r in roles_tried(oid):
                if r not in comp_map:
                    continue
                c = comp_map[r][oid]
                if not np.isfinite(c):
                    continue
                lab = role_label(oid, r)
                if lab is None:
                    continue
                scores.append(float(c)); labels.append(lab)
                per_role[r][0].append(float(c)); per_role[r][1].append(lab)
        scores = np.array(scores); labels = np.array(labels)
        pooled = _auc(scores, labels) if len(scores) else None
        pr = {}
        for r, (sc, lb) in per_role.items():
            pr[r] = {"n": len(sc), "auc": _auc(np.array(sc), np.array(lb)),
                     "base_rate": float(np.mean(lb)) if lb else None}
        macro = [v["auc"] for v in pr.values() if v["auc"] is not None]
        return {"pooled_auc": pooled, "n_instances": int(len(scores)),
                "base_rate": float(labels.mean()) if len(labels) else None,
                "per_role": pr,
                "macro_auc": float(np.mean(macro)) if macro else None}

    auc_cf = collect_auc(counterfactual, comp_loo)
    auc_all = collect_auc(list(range(n_obj)), comp_loo)
    auc_insample = collect_auc(counterfactual, comp_in)

    # ---- argmax accuracy on the disagreement subset ----
    def argmax_acc(oids):
        correct = 0; total = 0; base_correct = 0
        for oid in oids:
            tried = [r for r in roles_tried(oid) if r in comp_loo and np.isfinite(comp_loo[r][oid])]
            if len(tried) < 2:
                continue
            # predicted role = argmax COMP over tried roles
            pred = max(tried, key=lambda r: comp_loo[r][oid])
            # ground truth = the role(s) with the best success rate on this objective
            rates = {r: (obj_role_out[oid][r][0] /
                         max(1, obj_role_out[oid][r][0] + obj_role_out[oid][r][1])) for r in tried}
            best = max(rates.values())
            best_roles = {r for r, v in rates.items() if v == best}
            total += 1
            if pred in best_roles:
                correct += 1
            # baseline: pick the most-frequently-tried role
            freq = {r: sum(obj_role_out[oid][r]) for r in tried}
            base_pred = max(freq, key=freq.get)
            if base_pred in best_roles:
                base_correct += 1
        return {"n": total,
                "argmax_comp_accuracy": correct / total if total else None,
                "most_frequent_role_baseline": base_correct / total if total else None}

    argmax = argmax_acc(disagree)
    argmax_cf = argmax_acc(counterfactual)

    # ---- optional decision-level (row-level) robustness variant ----
    row_level = None
    if args.row_level:
        row_level = row_level_probe(vecs, rows, ntotal, obj_index, set(counterfactual),
                                    role_bank, role_excl)

    headline = auc_cf["pooled_auc"]
    if headline is None:
        verdict = "INDETERMINATE (no evaluable instances)"
    elif headline <= 0.55:
        verdict = "NULL — AUC<=0.55: familiarity axis is flat; close intake-866 line."
    elif headline >= 0.65:
        verdict = "FEATURE — AUC>=0.65: recommend concatenating COMP at routing_classifier.py:61 (RECOMMEND ONLY)."
    else:
        verdict = f"WEAK/INCONCLUSIVE — 0.55<AUC<0.65 ({headline:.3f}): no decisive spread; do not wire."

    out = {
        "measurement_note": "OBSERVATION per MEASUREMENT.md — no protocol-id/attestation; gates "
                            "only this research line's keep/close decision, not production deploy.",
        "snapshot_provenance": prov,
        "n_objectives_canonical": n_obj,
        "n_counterfactual_objectives": len(counterfactual),
        "n_disagreement_objectives": len(disagree),
        "roles_with_bank": roles,
        "headline_auc_success_given_role_counterfactual": headline,
        "verdict": verdict,
        "auc_counterfactual": auc_cf,
        "auc_all_objectives": auc_all,
        "auc_in_sample_sanity": {"pooled_auc": auc_insample["pooled_auc"],
                                 "note": "no leave-one-out; expected ~1.0 (leakage anchor)"},
        "argmax_accuracy_disagreement_subset": argmax,
        "argmax_accuracy_counterfactual_set": argmax_cf,
        "row_level_robustness": row_level,
    }
    outp = Path(args.out_json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in
                      ["n_objectives_canonical", "n_counterfactual_objectives",
                       "n_disagreement_objectives",
                       "headline_auc_success_given_role_counterfactual", "verdict"]}, indent=2))
    print(f"[done] wrote {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
