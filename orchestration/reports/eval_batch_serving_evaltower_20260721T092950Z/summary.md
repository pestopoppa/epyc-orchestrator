# Eval-Batch Serving Verifier-Mode Window

- mode: `calibration`
- status: `blocked`
- decision_grade: `False`
- applied: `True`
- suite/split: `scoring_verifiers / HE-R+`
- roles: `worker_general, frontdoor`
- scoring: `math_verify`  full: `True`  n: `None`  seed: `42`
- eval_concurrency: resolved=`1` min=`3`
- autopilot_active: `False`

## Pin Command

```bash
.venv/bin/python scripts/benchmark/eval_batch_serving_evaltower_window.py --mode calibration --suite scoring_verifiers --split HE-R+ --roles worker_general,frontdoor --full --min-eval-concurrency 3 --seed 42 --api-url http://localhost:8000 --apply --confirm-clean-window
```

## Blockers

- resolved EvalTower concurrency 1 is below --min-eval-concurrency 3; refresh the contention matrix/topology certification or intentionally run a serial entry
