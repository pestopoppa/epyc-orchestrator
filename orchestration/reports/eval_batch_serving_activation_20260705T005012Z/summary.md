# Eval-Batch Serving Activation Window

- status: `smoke_passed_rolled_back`
- decision_grade: `True`
- applied: `True`
- keep_enabled: `False`
- eval_batch_url: `http://localhost:18070`
- autopilot_active: `False`

## Activation Plan

```bash
/mnt/raid0/llm/epyc-orchestrator/.venv/bin/python3 scripts/server/orchestrator_stack.py start --only eval_batch_frontdoor
```
```bash
ORCHESTRATOR_FEATURE_EVAL_BATCH_SERVING=1 ORCHESTRATOR_EVAL_BATCH_FRONTDOOR_URL=http://localhost:18070 /mnt/raid0/llm/epyc-orchestrator/.venv/bin/python3 scripts/server/orchestrator_stack.py reload orchestrator
```
```bash
/mnt/raid0/llm/epyc-orchestrator/.venv/bin/python3 scripts/benchmark/eval_batch_serving_probe.py --smoke --confirm-clean-window --require-enabled --api-url http://localhost:8000 --eval-batch-url http://localhost:18070 --output-dir /mnt/raid0/llm/epyc-orchestrator/orchestration/reports/eval_batch_serving_activation_20260705T005012Z/smoke_probe --summary-only
```

## Rollback Plan

```bash
ORCHESTRATOR_FEATURE_EVAL_BATCH_SERVING=0 /mnt/raid0/llm/epyc-orchestrator/.venv/bin/python3 scripts/server/orchestrator_stack.py reload orchestrator
```
```bash
/mnt/raid0/llm/epyc-orchestrator/.venv/bin/python3 scripts/server/orchestrator_stack.py stop eval_batch_frontdoor
```

## Steps

- `start_eval_batch_frontdoor` rc=`0` elapsed_s=`32.982`
- `reload_orchestrator_eval_batch_enabled` rc=`0` elapsed_s=`15.894`
- `smoke_probe` rc=`0` elapsed_s=`3.304`

## Smoke Probe

- status: `smoke_passed`
- decision_grade: `True`
- blockers: `[]`
