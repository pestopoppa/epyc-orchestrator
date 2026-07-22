#!/usr/bin/env python3
"""DAR handoff L181 (DAR-5.1) — Offline IRT prompt-difficulty scorer (zero inference).

Fits a 2-parameter logistic (2PL) Item Response Theory model over the routing
store's observed outcomes, treating:
  - items      = objectives (prompts) seen under >= 2 roles at >= 5 obs
  - responders = roles/models
  - responses  = per-(item,role) success counts

    P(correct | item i, role j) = sigmoid( a_i * (theta_j - b_i) )

yielding per-prompt (b_i = latent difficulty, a_i = latent discrimination) and
per-role theta_j (latent ability). Fit by penalized MLE (analytic gradient,
L-BFGS). Calibrated by Platt scaling on held-out cells. A ridge regressor from
the BGE prompt embedding -> fitted difficulty gives a difficulty SCORER for new
prompts (the DAR-5.1 deliverable: difficulty predictable from embedding alone).

Held-out eval compares IRT vs a role-marginal baseline (predict each cell by the
role's overall train success rate) on AUC / log-loss / ECE.

OBSERVATION only (MEASUREMENT.md); pre-fix reward era; non-randomised assignment.

Usage:
    python scripts/analysis/dar_irt_scorer.py [--out DIR]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import dar_common as dc


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_2pl(n_mat, s_mat, l2_theta=1.0, l2_b=1.0, l2_loga=2.5, iters=2000):
    """Penalized MLE for a 2PL IRT model over dense (item x role) count matrices.

    n_mat[i,j] = observations, s_mat[i,j] = successes. Returns theta(J), b(I), a(I).
    """
    from scipy.optimize import minimize
    I, J = n_mat.shape
    obs = n_mat > 0

    def unpack(x):
        theta = x[:J]
        b = x[J:J + I]
        loga = x[J + I:]
        a = np.exp(np.clip(loga, -4, 4))
        return theta, b, a, loga

    def nll_grad(x):
        theta, b, a, loga = unpack(x)
        z = a[:, None] * (theta[None, :] - b[:, None])          # (I,J)
        p = _sigmoid(z)
        # penalized negative log-likelihood
        eps = 1e-9
        ll = s_mat * np.log(p + eps) + (n_mat - s_mat) * np.log(1 - p + eps)
        nll = -np.sum(ll[obs])
        nll += 0.5 * l2_theta * np.sum(theta ** 2)
        nll += 0.5 * l2_b * np.sum(b ** 2)
        nll += 0.5 * l2_loga * np.sum(loga ** 2)
        # gradients
        g = np.where(obs, n_mat * p - s_mat, 0.0)               # (I,J)
        g_theta = np.sum(a[:, None] * g, axis=0) + l2_theta * theta
        g_b = -a * np.sum(g, axis=1) + l2_b * b
        g_loga = a * np.sum((theta[None, :] - b[:, None]) * g, axis=1) + l2_loga * loga
        return nll, np.concatenate([g_theta, g_b, g_loga])

    x0 = np.concatenate([np.zeros(J), np.zeros(I), np.zeros(I)])
    res = minimize(nll_grad, x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": iters, "maxfun": iters * 20,
                            "ftol": 1e-10, "gtol": 1e-7})
    theta, b, a, _ = unpack(res.x)
    # identifiability: center role abilities (difficulty absorbs the shift)
    shift = theta.mean()
    theta = theta - shift
    b = b - shift
    return theta, b, a, res


def _ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for k in range(bins):
        m = (p >= edges[k]) & (p < edges[k + 1] if k < bins - 1 else p <= edges[k + 1])
        if m.sum():
            e += (m.mean()) * abs(y[m].mean() - p[m].mean())
    return e


def run(out_dir: Path | None) -> dict:
    from sklearn.metrics import roc_auc_score, log_loss
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import KFold

    rows = dc.load_rows("routing")
    _, matched = dc.matched_set(rows, min_obs=5, min_roles=2)

    # per-row observations for matched items over their eligible roles
    obs_rows = []  # (item_idx, role_idx, success, emb_idx)
    items = list(matched.keys())
    item_idx = {o: i for i, o in enumerate(items)}
    roles = sorted({role for stats in matched.values() for role in stats})
    role_idx = {r: j for j, r in enumerate(roles)}
    item_emb = {}
    rng = np.random.default_rng(0)
    for r in rows:
        if r.action_type != "routing" or r.objective not in matched:
            continue
        if r.role not in matched[r.objective]:  # eligible role only
            continue
        obs_rows.append((item_idx[r.objective], role_idx[r.role], r.success))
        item_emb.setdefault(r.objective, r.emb_idx)

    obs_rows = np.array(obs_rows)
    # train/test split at the observation level
    perm = rng.permutation(len(obs_rows))
    cut = int(0.8 * len(obs_rows))
    tr, te = obs_rows[perm[:cut]], obs_rows[perm[cut:]]

    I, J = len(items), len(roles)

    def cell_mats(split):
        n = np.zeros((I, J)); s = np.zeros((I, J))
        for it, jr, su in split:
            n[it, jr] += 1; s[it, jr] += su
        return n, s

    n_tr, s_tr = cell_mats(tr)
    theta, b, a, res = fit_2pl(n_tr, s_tr)

    # held-out predictions
    role_rate = np.divide(s_tr.sum(0), np.maximum(n_tr.sum(0), 1))  # baseline per role
    p_irt, p_base, y_te = [], [], []
    for it, jr, su in te:
        z = a[it] * (theta[jr] - b[it])
        p_irt.append(float(_sigmoid(z)))
        p_base.append(float(role_rate[jr]))
        y_te.append(int(su))
    p_irt = np.array(p_irt); p_base = np.array(p_base); y_te = np.array(y_te)

    def _safe_auc(y, p):
        return roc_auc_score(y, p) if len(set(y)) > 1 else float("nan")

    auc_irt = _safe_auc(y_te, p_irt)
    auc_base = _safe_auc(y_te, p_base)
    ll_irt = log_loss(y_te, np.clip(p_irt, 1e-6, 1 - 1e-6), labels=[0, 1])
    ll_base = log_loss(y_te, np.clip(p_base, 1e-6, 1 - 1e-6), labels=[0, 1])
    ece_irt = _ece(y_te, p_irt)

    # Platt scaling on held-out logits (fit on half of test, eval on other half)
    half = len(y_te) // 2
    logit = np.log(np.clip(p_irt, 1e-6, 1 - 1e-6) / np.clip(1 - p_irt, 1e-6, 1 - 1e-6))
    platt = LogisticRegression()
    platt.fit(logit[:half].reshape(-1, 1), y_te[:half])
    p_cal = platt.predict_proba(logit[half:].reshape(-1, 1))[:, 1]
    ece_cal = _ece(y_te[half:], p_cal)

    # ridge: BGE embedding -> fitted difficulty (the "scorer for new prompts")
    emb = dc.Embeddings()
    X_emb, mask = emb.matrix([item_emb.get(o, -1) for o in items])
    idx = np.where(mask)[0]
    Xe, be = X_emb[idx], b[idx]
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    r2s = []
    for trk, tek in kf.split(Xe):
        rr = Ridge(alpha=10.0).fit(Xe[trk], be[trk])
        pred = rr.predict(Xe[tek])
        ss_res = np.sum((be[tek] - pred) ** 2)
        ss_tot = np.sum((be[tek] - be[trk].mean()) ** 2)
        r2s.append(1 - ss_res / ss_tot if ss_tot > 0 else float("nan"))
    emb_r2 = float(np.nanmean(r2s))

    role_ability = {roles[j]: float(theta[j]) for j in range(J)}
    result = {
        "task": "DAR-L181 / DAR-5.1 offline IRT difficulty scorer",
        "snapshot": dc.snapshot_meta(),
        "n_items_objectives": I,
        "n_roles": J,
        "n_observations": int(len(obs_rows)),
        "fit_converged": bool(res.success),
        "final_nll": float(res.fun),
        "role_ability_theta": dict(sorted(role_ability.items(), key=lambda kv: -kv[1])),
        "difficulty_b": {"mean": float(b.mean()), "sd": float(b.std()),
                         "p10": float(np.percentile(b, 10)),
                         "p50": float(np.percentile(b, 50)),
                         "p90": float(np.percentile(b, 90))},
        "discrimination_a": {"mean": float(a.mean()), "sd": float(a.std()),
                             "p50": float(np.percentile(a, 50)),
                             "p90": float(np.percentile(a, 90))},
        "heldout_eval": {
            "auc_irt": auc_irt, "auc_role_marginal_baseline": auc_base,
            "logloss_irt": ll_irt, "logloss_baseline": ll_base,
            "ece_irt": ece_irt, "ece_after_platt": ece_cal,
            "n_test": int(len(y_te)),
        },
        "embedding_to_difficulty_scorer": {
            "cv_r2": emb_r2, "n_items_with_embedding": int(len(idx)),
            "note": ("ridge BGE_embedding -> fitted IRT difficulty; positive R2 => "
                     "difficulty is predictable for a NEW prompt from its embedding "
                     "alone (DAR-5.1 cold-start scorer)."),
        },
    }

    print("[DAR-L181 / DAR-5.1] offline 2PL IRT difficulty scorer")
    print(f"snapshot {result['snapshot']['snapshot_ts_utc']}")
    print(f"items(objectives)={I}  roles={J}  observations={len(obs_rows):,}  "
          f"converged={res.success}")
    print("\nrole ability theta (higher = stronger):")
    for role, t in sorted(role_ability.items(), key=lambda kv: -kv[1]):
        print(f"  {role:<22} theta={t:+.3f}")
    print(f"\ndifficulty b : mean {b.mean():+.3f}  sd {b.std():.3f}  "
          f"[p10 {np.percentile(b,10):+.2f}, p90 {np.percentile(b,90):+.2f}]")
    print(f"discrimination a: mean {a.mean():.3f}  sd {a.std():.3f}  "
          f"p90 {np.percentile(a,90):.2f}")
    print("\nheld-out (20% obs):")
    print(f"  AUC  IRT {auc_irt:.3f}  vs role-marginal baseline {auc_base:.3f}")
    print(f"  logloss IRT {ll_irt:.3f} vs baseline {ll_base:.3f}")
    print(f"  ECE  IRT {ece_irt:.3f}  -> after Platt {ece_cal:.3f}")
    print(f"\nembedding->difficulty ridge scorer: CV R2 = {emb_r2:.3f}  "
          f"({len(idx)} items)")

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "dar_irt_scorer.json").write_text(json.dumps(result, indent=2))
        print(f"\nartifact: {out_dir/'dar_irt_scorer.json'}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    run(ap.parse_args().out)


if __name__ == "__main__":
    main()
