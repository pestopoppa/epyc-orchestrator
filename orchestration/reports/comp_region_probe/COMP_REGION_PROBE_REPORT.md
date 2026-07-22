# Competence-Region Probe COMP_r(x) — Result Report

**Task**: `handoffs/active/learned-routing-controller.md` L1444 (the one usable idea from intake-866).
**Date**: 2026-07-22 · **Script**: `scripts/analysis/comp_region_probe.py` · **Raw**: `comp_region_probe.json`
**Status**: COMPLETE — decisive **NULL**.

> All numbers below are **OBSERVATIONS** per MEASUREMENT.md (no protocol-id / attestation). They
> gate only this research line's keep/close decision — not any production deploy/promote.

---

## Verdict

**AUC(success | role) = 0.497** (leave-one-objective-out, counterfactual set, n=28,608 (obj,role)
instances). **≤ 0.55 → NULL.** Close the intake-866 competence-region line. Both the *difficulty*
axis (prior nulls) and the *familiarity* axis (this probe) are flat.

**Do NOT concatenate COMP at `routing_classifier.py:61`.** The AUX≥0.65 branch was not reached.

## Data snapshot (read-only, live stores were being written)

- Snapshot: `/mnt/raid0/llm/tmp/comp_snapshot/` taken 2026-07-22T12:55:34Z (`sqlite3 .backup` for
  `episodic.db`; plain copy of `embeddings.faiss` + `id_map.npy`, not the `.tmp` files).
- Source `episodic.db` mtime at copy: 2026-07-22 12:55:35Z. FAISS `ntotal` = **676,195** vectors,
  dim 1024, `IndexFlatIP` (vectors L2-normalised at insert, so cosine == inner product).
- Routing rows: 671,776. Distinct objectives (canonical roles): **7,037**. The handoff's "~2,384
  distinct / 622-objective counterfactual set" was a snapshot-time count; the store roughly tripled
  overnight. Counterfactual set here (objectives with ≥2 distinct canonical roles tried) = **5,320**;
  role-disagreement subset (a clean-success role AND a failing role coexist) = **1,435**.
- Row-level outcome base rate 0.905 success — matches the handoff's stated 90.3%.

## Method

`COMP_r(x) = max cos(e(x), e(m))` over success-memories `m` with `action=r`, **excluding every
memory whose objective == x** (leave-one-objective-out — mandatory). `e(·)` are the stored BGE
vectors, reconstructed once into a 676,195×1024 matrix. Per role, the success bank is deduped to
distinct `embedding_idx`, each tagged with the set of objectives referencing it; the query is masked
against its own objective's bank columns. No inference performed. Reuses the `retriever._retrieve()`
FAISS-lookup pattern read-only. Two variants:

1. **Objective-centroid** (headline): `e(x)` = normalised mean of x's row vectors; eval unit =
   (objective, role) with majority-vote success label.
2. **Row-level** (robustness): each decision's own stored vector is the query; balanced 6,000-fail /
   6,000-success decision sample.

## Results

| Metric | Value | Read |
|---|---:|---|
| **Headline AUC(success\|role), counterfactual, LOO** | **0.497** | chance → NULL |
| AUC over all 7,037 objectives, LOO | 0.507 | chance |
| Macro-avg per-role AUC | 0.538 | only lifted by tiny-n roles |
| In-sample (no LOO) AUC, centroid | 0.643 | *not* ~1.0 — see mechanism |
| Row-level LOO AUC | 0.464 | chance |
| Row-level in-sample AUC | 0.523 | chance |

Per-role AUC (counterfactual, LOO) — every high-volume role sits at chance:

| role | n | AUC | base rate |
|---|---:|---:|---:|
| frontdoor | 4,938 | 0.500 | 0.716 |
| coder_escalation | 4,845 | 0.498 | 0.717 |
| architect_general | 4,750 | 0.485 | 0.745 |
| ingest_long_context | 4,784 | 0.520 | 0.730 |
| worker_general | 4,291 | 0.481 | 0.645 |
| worker_vision | 57 | 0.617 | 0.947 |
| worker_math | 4 | 0.667 | 0.750 |

**Argmax accuracy** (predict role = argmax_r COMP_r(x)) is *worse than a trivial baseline*:

| set | argmax-COMP acc | most-frequent-role baseline |
|---|---:|---:|
| disagreement (n=1,435) | 0.779 | **0.905** |
| counterfactual (n=5,320) | 0.935 | **0.973** |

## Mechanism — why in-sample is NOT ~1.0 (a stronger null)

The handoff feared that without LOO an in-sample near-duplicate at cos≈1.0 would score AUC≈1.0 "for
free". It doesn't: **in-sample AUC is also ≈0.5.** The reason is **mixed-outcome colocation** —
**61.7%** of failure rows share their exact `(embedding_idx, role)` with a success row. So a failure
decision self-matches to a success memory at cos==1.0 exactly as a success decision does; the two are
**embedding-indistinguishable at every leakage level**. The outcome of a given (objective, role) is
sampling luck, not anything the BGE embedding — or any feature derived from it — can see.

This corroborates the 2026-07-21 "signal-bound, not policy-bound" finding: 88.8% of memories at
`q=1.0`/`update_count=0`, 90.3% success. No feature (difficulty or familiarity) can exceed the mutual
information between the embedding and a saturated, luck-driven outcome.

## Recommendation

Record the null and close the intake-866 competence-region line. For reference, `gitnexus impact
RoutingClassifier --repo epyc-orchestrator --direction upstream` reports **MEDIUM** (43 upstream / 12
direct) — a wiring change would have been permissible risk-wise, but the AUC did not warrant it. No
existing symbol was modified by this probe (additive script only).
