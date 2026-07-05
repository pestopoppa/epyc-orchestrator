# Routing Classifier Rollout Window

- status: `attestation_passed_rolled_back`
- applied: `True`
- keep_enabled: `False`
- rollout_attested: `True`
- decision_grade: `False`
- autopilot_active: `False`
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

## Steps

- `verify_routing_wiring` rc=`0` elapsed_s=`0.220`
- `reload_orchestrator_routing_classifier_enabled` rc=`0` elapsed_s=`10.898`
- `attest_routing_classifier_enabled` rc=`0` elapsed_s=`6.807`

## Rollback Steps

- `reload_orchestrator_routing_classifier_disabled` rc=`0` elapsed_s=`10.877`
- `attest_routing_classifier_disabled` rc=`0` elapsed_s=`6.832`
