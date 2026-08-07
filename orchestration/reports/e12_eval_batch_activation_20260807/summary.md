# Eval-Batch Serving Activation Window

- status: `smoke_passed_enabled_left_on`
- decision_grade: `True`
- applied: `True`
- keep_enabled: `True`
- eval_batch_url: `http://127.0.0.1:18070`
- autopilot_active: `False`

## Activation Plan

```bash
/mnt/raid0/llm/epyc-orchestrator/.venv/bin/python scripts/server/orchestrator_stack.py start --only eval_batch_frontdoor
```
```bash
ORCHESTRATOR_FEATURE_EVAL_BATCH_SERVING=1 ORCHESTRATOR_EVAL_BATCH_FRONTDOOR_URL=http://127.0.0.1:18070 /mnt/raid0/llm/epyc-orchestrator/.venv/bin/python scripts/server/orchestrator_stack.py reload orchestrator
```
```bash
/mnt/raid0/llm/epyc-orchestrator/.venv/bin/python scripts/benchmark/eval_batch_serving_probe.py --smoke --confirm-clean-window --require-enabled --api-url http://127.0.0.1:8000 --eval-batch-url http://127.0.0.1:18070 --output-dir orchestration/reports/e12_eval_batch_activation_20260807/smoke_probe --summary-only
```

## Rollback Plan

```bash
ORCHESTRATOR_FEATURE_EVAL_BATCH_SERVING=0 /mnt/raid0/llm/epyc-orchestrator/.venv/bin/python scripts/server/orchestrator_stack.py reload orchestrator
```
```bash
/mnt/raid0/llm/epyc-orchestrator/.venv/bin/python scripts/server/orchestrator_stack.py stop eval_batch_frontdoor
```

## Steps

- `start_eval_batch_frontdoor` rc=`0` elapsed_s=`52.848`
- `reload_orchestrator_eval_batch_enabled` rc=`0` elapsed_s=`15.557`
- `smoke_probe` rc=`0` elapsed_s=`3.929`

## Smoke Probe

- status: `smoke_passed`
- decision_grade: `True`
- blockers: `[]`
