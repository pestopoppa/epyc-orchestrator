# Eval-Batch Serving Verifier-Mode Window

- mode: `calibration`
- status: `plan_only`
- decision_grade: `False`
- applied: `False`
- suite/split: `scoring_verifiers / HE-R+`
- roles: `worker_general, frontdoor`
- scoring: `math_verify`  full: `True`  n: `None`  seed: `42`
- autopilot_active: `False`

## Pin Command

```bash
.venv/bin/python scripts/benchmark/eval_batch_serving_evaltower_window.py --mode calibration --suite scoring_verifiers --split HE-R+ --roles worker_general,frontdoor --full --seed 42 --api-url http://localhost:8000 --apply --confirm-clean-window
```
