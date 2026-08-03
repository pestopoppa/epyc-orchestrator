# Eval-Batch Serving EvalTower Window

- status: `current_eval_degenerate`
- decision_grade: `False`
- applied: `True`
- tier/n/seed: `T1 / 50 / 42`
- skip_current_arm: `False`
- keep_enabled: `False`
- eval_batch_url: `http://localhost:18070`
- eval_concurrency: resolved=`4` min=`4`
- autopilot_active: `False`

## Blockers

- current EvalTower arm scored 26/50 non-error questions

## Planned Activation

```bash
/mnt/raid0/llm/epyc-orchestrator/.venv/bin/python scripts/server/orchestrator_stack.py start --only eval_batch_frontdoor
```
```bash
ORCHESTRATOR_FEATURE_EVAL_BATCH_SERVING=1 ORCHESTRATOR_EVAL_BATCH_FRONTDOOR_URL=http://localhost:18070 /mnt/raid0/llm/epyc-orchestrator/.venv/bin/python scripts/server/orchestrator_stack.py reload orchestrator
```
```bash
/mnt/raid0/llm/epyc-orchestrator/.venv/bin/python scripts/benchmark/eval_batch_serving_probe.py --smoke --confirm-clean-window --require-enabled --api-url http://localhost:8000 --eval-batch-url http://localhost:18070 --output-dir orchestration/reports/ev_baseline_e7_tier1_diag/activation/smoke_probe --summary-only
```

## Planned Rollback

```bash
ORCHESTRATOR_FEATURE_EVAL_BATCH_SERVING=0 /mnt/raid0/llm/epyc-orchestrator/.venv/bin/python scripts/server/orchestrator_stack.py reload orchestrator
```
```bash
/mnt/raid0/llm/epyc-orchestrator/.venv/bin/python scripts/server/orchestrator_stack.py stop eval_batch_frontdoor
```

## Current Arm

- ok: `True`
- quality: `1.9615384615384617`
- speed: `27.469461869562075`
- reliability: `0.52`
- wall_s: `479.359715705039`
- n_questions: `50`
- n_scored: `26`
- errors: `24`
