# Eval-Batch Serving Verifier-Mode Window

- mode: `math_rebaseline`
- status: `complete`
- decision_grade: `True`
- applied: `True`
- suite/split: `None / None`
- roles: `worker_general, worker_math`
- scoring: `math_verify`  full: `True`  n: `None`  seed: `42`
- eval_concurrency: resolved=`4` min=`4`
- autopilot_active: `False`

## Pin Command

```bash
.venv/bin/python scripts/benchmark/eval_batch_serving_evaltower_window.py --mode math_rebaseline --roles worker_general,worker_math --scoring math_verify --full --min-eval-concurrency 4 --seed 42 --api-url http://localhost:8000 --apply --confirm-clean-window
```

## Result

- dataset_sha256: `38e582cf652f1ecfea8e8f9704d7b08120920da41d789c0478c2dbfc2ab6fa91`
- `worker_general`: `{"accuracy": 0.7897862232779097, "arm": "ev11-math-rebaseline::worker_general::seed42::38e582cf652f", "auroc": 0.0, "correct": 1330, "ece": 0.0, "n_questions": 1819, "n_scored": 1684, "quality": 2.369358669833729, "reliability": 0.925783397471138, "role": "worker_general", "test_profile": {"dataset_sha256": "38e582cf652f1ecfea8e8f9704d7b08120920da41d789c0478c2dbfc2ab6fa91", "n_gsm8k": 1319, "n_math500": 500, "n_questions": 1819, "production_sampling": true, "sampling_profile": "production_temp_seed42", "scoring": "math_verify", "seed": 42, "version": "ev11-math-rebaseline-v1"}}`
- `worker_math`: `{"accuracy": 0.7599067599067599, "arm": "ev11-math-rebaseline::worker_math::seed42::38e582cf652f", "auroc": 0.0, "correct": 978, "ece": 0.0, "n_questions": 1819, "n_scored": 1287, "quality": 2.27972027972028, "reliability": 0.7075316107751511, "role": "worker_math", "test_profile": {"dataset_sha256": "38e582cf652f1ecfea8e8f9704d7b08120920da41d789c0478c2dbfc2ab6fa91", "n_gsm8k": 1319, "n_math500": 500, "n_questions": 1819, "production_sampling": true, "sampling_profile": "production_temp_seed42", "scoring": "math_verify", "seed": 42, "version": "ev11-math-rebaseline-v1"}}`
