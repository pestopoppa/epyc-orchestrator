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
- `worker_general`: `{"accuracy": 0.7885985748218527, "arm": "ev11-math-rebaseline::worker_general::seed42::38e582cf652f", "auroc": 0.4013025416271829, "confidence_is_real": true, "confidence_source_counts": {"completion_probabilities_geomean": 1684}, "correct": 1328, "ece": 0.2113974416427804, "n_questions": 1819, "n_scored": 1684, "quality": 2.365795724465558, "reliability": 0.925783397471138, "role": "worker_general", "test_profile": {"dataset_sha256": "38e582cf652f1ecfea8e8f9704d7b08120920da41d789c0478c2dbfc2ab6fa91", "n_gsm8k": 1319, "n_math500": 500, "n_questions": 1819, "production_sampling": true, "sampling_profile": "production_temp_seed42", "scoring": "math_verify", "seed": 42, "version": "ev11-math-rebaseline-v1"}}`
- `worker_math`: `{"accuracy": 0.7800982800982801, "arm": "ev11-math-rebaseline::worker_math::seed42::38e582cf652f", "auroc": 0.4114393172920424, "confidence_is_real": true, "confidence_source_counts": {"completion_probabilities_geomean": 1628}, "correct": 1270, "ece": 0.21989848853602278, "n_questions": 1819, "n_scored": 1628, "quality": 2.34029484029484, "reliability": 0.8949972512369434, "role": "worker_math", "test_profile": {"dataset_sha256": "38e582cf652f1ecfea8e8f9704d7b08120920da41d789c0478c2dbfc2ab6fa91", "n_gsm8k": 1319, "n_math500": 500, "n_questions": 1819, "production_sampling": true, "sampling_profile": "production_temp_seed42", "scoring": "math_verify", "seed": 42, "version": "ev11-math-rebaseline-v1"}}`
