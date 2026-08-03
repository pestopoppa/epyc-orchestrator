# Eval-Batch Serving Verifier-Mode Window

- mode: `calibration`
- status: `complete`
- decision_grade: `True`
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

## Result

- dataset_sha256: `87eaabbdfd4adbbd12b1b6200d099aaa2dd4af83170b18d1f9909205d14059f0`
- `frontdoor`: `{"accuracy": 0.7085365853658536, "auroc": 0.6337255777443306, "bottom1_accuracy": 0.0, "confidence_is_real": true, "confidence_source_counts": {"completion_probabilities_geomean": 820}, "ece": 0.253185442996694, "mae": 0.29486868847220055, "n_questions": 820, "n_scored": 820, "reliability": 1.0, "role": "frontdoor", "spearman_rho": 0.21051284595951225, "top1_accuracy": 1.0}`
