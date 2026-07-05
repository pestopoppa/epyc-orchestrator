# Eval-Batch Serving Probe

- status: `smoke_passed`
- decision_grade: `True`
- api: `http://localhost:8000`
- eval_batch_frontdoor: `http://localhost:18070`
- autopilot_active: `False`
- api_health_ok: `True`
- eval_frontdoor_health_ok: `True`
- sampled_workers_enabled: `True`

## Smoke

- response_ok: `True`
- status_code: `200`
- batch_id: `eval-batch-probe-20260705T005102Z`
- tap_events_path: `/mnt/raid0/llm/tmp/inference_tap_events.jsonl`
- tap_hit_expected_port: `True`
- tap_ports: `[18070]`
- median_tps: `1.2258415797889333`

## Activation Commands

```bash
cd /mnt/raid0/llm/epyc-orchestrator && uv run python scripts/server/orchestrator_stack.py start --only eval_batch_frontdoor
```
```bash
cd /mnt/raid0/llm/epyc-orchestrator && ORCHESTRATOR_FEATURE_EVAL_BATCH_SERVING=1 ORCHESTRATOR_EVAL_BATCH_FRONTDOOR_URL=http://localhost:18070 uv run python scripts/server/orchestrator_stack.py reload orchestrator
```
```bash
cd /mnt/raid0/llm/epyc-orchestrator && uv run python scripts/benchmark/eval_batch_serving_probe.py --smoke --confirm-clean-window --require-enabled
```
