# Eval-Batch Serving Activation Window

- status: `activation_failed`
- decision_grade: `False`
- applied: `True`
- keep_enabled: `False`
- eval_batch_url: `http://localhost:18070`
- autopilot_active: `False`

## Blockers

- smoke_probe failed with rc=75

## Activation Plan

```bash
/mnt/raid0/llm/epyc-orchestrator/.venv/bin/python3 scripts/server/orchestrator_stack.py start --only eval_batch_frontdoor
```
```bash
ORCHESTRATOR_FEATURE_EVAL_BATCH_SERVING=1 ORCHESTRATOR_EVAL_BATCH_FRONTDOOR_URL=http://localhost:18070 /mnt/raid0/llm/epyc-orchestrator/.venv/bin/python3 scripts/server/orchestrator_stack.py reload orchestrator
```
```bash
/mnt/raid0/llm/epyc-orchestrator/.venv/bin/python3 scripts/benchmark/eval_batch_serving_probe.py --smoke --confirm-clean-window --require-enabled --api-url http://localhost:8000 --eval-batch-url http://localhost:18070 --output-dir /mnt/raid0/llm/epyc-orchestrator/orchestration/reports/eval_batch_serving_activation_20260705T004631Z/smoke_probe --summary-only
```

## Rollback Plan

```bash
ORCHESTRATOR_FEATURE_EVAL_BATCH_SERVING=0 /mnt/raid0/llm/epyc-orchestrator/.venv/bin/python3 scripts/server/orchestrator_stack.py reload orchestrator
```
```bash
/mnt/raid0/llm/epyc-orchestrator/.venv/bin/python3 scripts/server/orchestrator_stack.py stop eval_batch_frontdoor
```

## Steps

- `start_eval_batch_frontdoor` rc=`0` elapsed_s=`33.178`
- `reload_orchestrator_eval_batch_enabled` rc=`0` elapsed_s=`12.885`
- `smoke_probe` rc=`75` elapsed_s=`0.790`

## Smoke Probe

- status: `blocked`
- decision_grade: `False`
- blockers: `['eval_batch_serving is not enabled on every sampled API worker']`
