# Routing Classifier Rollout Window

- status: `plan_only`
- applied: `False`
- keep_enabled: `False`
- rollout_attested: `False`
- decision_grade: `False`
- autopilot_active: `True`
- weights_present: `True`

## Activation Plan

```bash
/mnt/raid0/llm/epyc-orchestrator/.venv/bin/python3 scripts/maintenance/verify_routing_wiring.py
```
```bash
ORCHESTRATOR_FEATURE_ROUTING_CLASSIFIER=1 ROUTING_CLASSIFIER_WEIGHTS=/mnt/raid0/llm/epyc-orchestrator/orchestration/repl_memory/routing_classifier_weights.npz /mnt/raid0/llm/epyc-orchestrator/.venv/bin/python3 scripts/server/orchestrator_stack.py reload orchestrator
```
```bash
/mnt/raid0/llm/epyc-orchestrator/.venv/bin/python3 scripts/validate/attest_flags.py --url http://localhost:8000 --polls 120 --delay-s 0.05 --min-workers 1 --expect routing_classifier=true
```

## Rollback Plan

```bash
ORCHESTRATOR_FEATURE_ROUTING_CLASSIFIER=0 ROUTING_CLASSIFIER_WEIGHTS=/mnt/raid0/llm/epyc-orchestrator/orchestration/repl_memory/routing_classifier_weights.npz /mnt/raid0/llm/epyc-orchestrator/.venv/bin/python3 scripts/server/orchestrator_stack.py reload orchestrator
```
```bash
/mnt/raid0/llm/epyc-orchestrator/.venv/bin/python3 scripts/validate/attest_flags.py --url http://localhost:8000 --polls 120 --delay-s 0.05 --min-workers 1 --expect routing_classifier=false
```
