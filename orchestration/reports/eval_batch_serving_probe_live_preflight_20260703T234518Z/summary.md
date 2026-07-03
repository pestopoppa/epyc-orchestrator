# Eval-Batch Serving Probe

- status: `blocked`
- decision_grade: `False`
- api: `http://localhost:8000`
- eval_batch_frontdoor: `http://localhost:18070`
- autopilot_active: `True`
- api_health_ok: `True`
- eval_frontdoor_health_ok: `False`
- sampled_workers_enabled: `False`

## Blockers

- eval_batch_frontdoor health is not OK

## Activation Commands

```bash
cd /mnt/raid0/llm/epyc-orchestrator && uv run python scripts/server/orchestrator_stack.py start --include-warm eval_batch_frontdoor
```
```bash
cd /mnt/raid0/llm/epyc-orchestrator && ORCHESTRATOR_FEATURE_EVAL_BATCH_SERVING=1 ORCHESTRATOR_EVAL_BATCH_FRONTDOOR_URL=http://localhost:18070 uv run python scripts/server/orchestrator_stack.py reload orchestrator
```
```bash
cd /mnt/raid0/llm/epyc-orchestrator && uv run python scripts/benchmark/eval_batch_serving_probe.py --smoke --confirm-clean-window --require-enabled
```
