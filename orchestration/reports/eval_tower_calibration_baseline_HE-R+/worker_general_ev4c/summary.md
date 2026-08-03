# Eval-Batch Serving Verifier-Mode Window

- mode: `calibration`
- status: `complete`
- decision_grade: `True`
- applied: `True`
- suite/split: `scoring_verifiers / HE-R+`
- roles: `worker_general`
- scoring: `math_verify`  full: `True`  n: `None`  seed: `42`
- eval_concurrency: resolved=`4` min=`4`
- autopilot_active: `False`

## Pin Command

```bash
.venv/bin/python scripts/benchmark/eval_batch_serving_evaltower_window.py --mode calibration --suite scoring_verifiers --split HE-R+ --roles worker_general --full --min-eval-concurrency 4 --seed 42 --api-url http://localhost:8000 --apply --confirm-clean-window
```

## Result

- dataset_sha256: `87eaabbdfd4adbbd12b1b6200d099aaa2dd4af83170b18d1f9909205d14059f0`
- `worker_general`: `{"accuracy": 0.6585067319461444, "auroc": 0.5751155880667812, "bottom1_accuracy": 0.0, "confidence_is_real": true, "confidence_source_counts": {"completion_probabilities_geomean": 817}, "ece": 0.321643058178994, "mae": 0.3453870095268573, "n_questions": 820, "n_scored": 817, "reliability": 0.9963414634146341, "role": "worker_general", "spearman_rho": 0.1233934948975394, "top1_accuracy": 1.0}`
