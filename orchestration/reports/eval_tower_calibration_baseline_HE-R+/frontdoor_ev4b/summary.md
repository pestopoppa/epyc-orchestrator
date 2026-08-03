# Eval-Batch Serving Verifier-Mode Window

- mode: `calibration`
- status: `complete`
- decision_grade: `False`
- applied: `True`
- suite/split: `scoring_verifiers / HE-R+`
- roles: `frontdoor`
- scoring: `math_verify`  full: `True`  n: `None`  seed: `42`
- eval_concurrency: resolved=`4` min=`4`
- autopilot_active: `False`

## Pin Command

```bash
.venv/bin/python scripts/benchmark/eval_batch_serving_evaltower_window.py --mode calibration --suite scoring_verifiers --split HE-R+ --roles frontdoor --full --min-eval-concurrency 4 --seed 42 --api-url http://localhost:8000 --apply --confirm-clean-window
```

## Decision-Grade Demotions

- arm frontdoor calibration is not decision-grade (confidence_is_real=False)

## Result

- dataset_sha256: `87eaabbdfd4adbbd12b1b6200d099aaa2dd4af83170b18d1f9909205d14059f0`
- `frontdoor`: `{"accuracy": 0.7085365853658536, "auroc": null, "bottom1_accuracy": null, "confidence_is_real": false, "confidence_source_counts": {"binary_correctness_proxy": 820}, "ece": null, "mae": null, "n_questions": 820, "n_scored": 820, "reliability": 1.0, "role": "frontdoor", "spearman_rho": null, "top1_accuracy": null}`
