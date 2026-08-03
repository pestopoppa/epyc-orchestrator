# Eval-Batch Serving EvalTower Window

- status: `current_arm_complete`
- decision_grade: `False`
- applied: `True`
- tier/n/seed: `T1 / 50 / 42`
- skip_current_arm: `False`
- keep_enabled: `False`
- eval_batch_url: `http://localhost:18070`
- eval_concurrency: resolved=`3` min=`3`
- autopilot_active: `False`

## Planned Activation

```bash
/mnt/raid0/llm/epyc-orchestrator/.venv/bin/python scripts/server/orchestrator_stack.py start --only eval_batch_frontdoor
```
```bash
ORCHESTRATOR_FEATURE_EVAL_BATCH_SERVING=1 ORCHESTRATOR_EVAL_BATCH_FRONTDOOR_URL=http://localhost:18070 /mnt/raid0/llm/epyc-orchestrator/.venv/bin/python scripts/server/orchestrator_stack.py reload orchestrator
```
```bash
/mnt/raid0/llm/epyc-orchestrator/.venv/bin/python scripts/benchmark/eval_batch_serving_probe.py --smoke --confirm-clean-window --require-enabled --api-url http://localhost:8000 --eval-batch-url http://localhost:18070 --output-dir orchestration/reports/r3_current_arm_rebaseline_20260723/activation/smoke_probe --summary-only
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
- quality: `1.723404255319149`
- speed: `30.600509734025653`
- reliability: `0.94`
- wall_s: `1312.4653012759518`
- n_questions: `50`
- n_scored: `47`
- errors: `3`
