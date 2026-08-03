# Eval-Batch Serving EvalTower Window

- status: `activation_failed`
- decision_grade: `False`
- applied: `True`
- tier/n/seed: `T2 / 50 / 42`
- skip_current_arm: `False`
- keep_enabled: `False`
- eval_batch_url: `http://localhost:18070`
- eval_concurrency: resolved=`4` min=`4`
- autopilot_active: `False`

## Blockers

- start_eval_batch_frontdoor failed with rc=2

## Planned Activation

```bash
/mnt/raid0/llm/epyc-orchestrator/.venv/bin/python scripts/server/orchestrator_stack.py start --only eval_batch_frontdoor
```
```bash
ORCHESTRATOR_FEATURE_EVAL_BATCH_SERVING=1 ORCHESTRATOR_EVAL_BATCH_FRONTDOOR_URL=http://localhost:18070 /mnt/raid0/llm/epyc-orchestrator/.venv/bin/python scripts/server/orchestrator_stack.py reload orchestrator
```
```bash
/mnt/raid0/llm/epyc-orchestrator/.venv/bin/python scripts/benchmark/eval_batch_serving_probe.py --smoke --confirm-clean-window --require-enabled --api-url http://localhost:8000 --eval-batch-url http://localhost:18070 --output-dir orchestration/reports/ev_baseline_corev2_tier2/activation/smoke_probe --summary-only
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
- quality: `1.891304347826087`
- speed: `33.31071190477097`
- reliability: `0.92`
- wall_s: `1276.8367036860436`
- n_questions: `50`
- n_scored: `46`
- errors: `4`

## Rollback Steps

- `rollback_reload_orchestrator_eval_batch_disabled` rc=`0` elapsed_s=`15.044`
- `rollback_stop_eval_batch_frontdoor` rc=`0` elapsed_s=`0.372`
