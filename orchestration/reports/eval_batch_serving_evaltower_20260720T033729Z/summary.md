# Eval-Batch Serving Verifier-Mode Window

- mode: `math_rebaseline`
- status: `plan_only`
- decision_grade: `False`
- applied: `False`
- suite/split: `None / None`
- roles: `worker_general`
- scoring: `math_verify`  full: `False`  n: `50`  seed: `42`
- autopilot_active: `False`

## Pin Command

```bash
.venv/bin/python scripts/benchmark/eval_batch_serving_evaltower_window.py --mode math_rebaseline --roles worker_general --scoring math_verify --n 50 --seed 42 --api-url http://localhost:8000 --apply --confirm-clean-window
```
