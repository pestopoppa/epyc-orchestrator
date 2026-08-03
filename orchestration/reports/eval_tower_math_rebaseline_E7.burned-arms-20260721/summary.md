# Eval-Batch Serving Verifier-Mode Window

- mode: `math_rebaseline`
- status: `eval_failed`
- decision_grade: `False`
- applied: `True`
- suite/split: `None / None`
- roles: `worker_general, worker_math`
- scoring: `math_verify`  full: `True`  n: `None`  seed: `42`
- eval_concurrency: resolved=`1` min=`1`
- autopilot_active: `False`

## Pin Command

```bash
.venv/bin/python scripts/benchmark/eval_batch_serving_evaltower_window.py --mode math_rebaseline --roles worker_general,worker_math --scoring math_verify --full --allow-serial --seed 42 --api-url http://localhost:8000 --apply --confirm-clean-window
```

## Blockers

- verifier-mode eval failed: math re-baseline drew 0 questions — GSM8K/MATH-500 datasets unavailable (MODEL-DOWNLOAD/data required)
