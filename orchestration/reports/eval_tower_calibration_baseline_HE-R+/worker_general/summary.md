# Eval-Batch Serving Verifier-Mode Window

- mode: `calibration`
- status: `complete`
- decision_grade: `True`
- applied: `True`
- suite/split: `scoring_verifiers / HE-R+`
- roles: `worker_general`
- scoring: `math_verify`  full: `True`  n: `None`  seed: `42`
- eval_concurrency: resolved=`1` min=`1`
- autopilot_active: `False`

## Pin Command

```bash
.venv/bin/python scripts/benchmark/eval_batch_serving_evaltower_window.py --mode calibration --suite scoring_verifiers --split HE-R+ --roles worker_general --full --allow-serial --seed 42 --api-url http://localhost:8000 --apply --confirm-clean-window
```

## Result

- dataset_sha256: `87eaabbdfd4adbbd12b1b6200d099aaa2dd4af83170b18d1f9909205d14059f0`
- `worker_general`: `{"accuracy": 0.6572481572481572, "auroc": null, "bottom1_accuracy": 0.0, "ece": 0.0, "mae": 0.0, "n_questions": 820, "n_scored": 814, "role": "worker_general", "spearman_rho": 1.0, "top1_accuracy": 1.0}`
