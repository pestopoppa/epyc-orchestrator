# A9 Pairwise Expanded-Gap Priority Collection Runbook

Generated from `offline_reward_pairwise_expanded_gap_plan_summary.{json,md}`
after orchestrator `10e5133b`.

Do not run these commands during active AutoPilot or another live measurement
window. `--dry-run` prevents reward injection, but it still performs live model
evaluation and consumes model slots.

## Priority 0 — Source-Family Blockers

Set a shared run id:

```bash
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
```

Collect `source_family:orchestrator_live_seed:architect_general>frontdoor`:

```bash
uv run python scripts/benchmark/seed_specialist_routing.py \
  --suites all \
  --roles architect_general frontdoor \
  --modes direct \
  --sample-size 20 \
  --dry-run \
  --output "/mnt/raid0/llm/epyc-inference-research/benchmarks/results/orchestrator/seeding_live_a9_source_family_orchestrator_live_seed_architect_general_frontdoor_${RUN_ID}.json"
```

Collect `source_family:seeding_eval:architect_general>coder_escalation`:

```bash
uv run python scripts/benchmark/seed_specialist_routing.py \
  --suites all \
  --roles architect_general coder_escalation \
  --modes direct \
  --sample-size 20 \
  --dry-run \
  --output "/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_a9_source_family_seeding_eval_architect_general_coder_escalation_${RUN_ID}.json"
```

Collect `source_family:seeding_eval:architect_general>frontdoor`:

```bash
uv run python scripts/benchmark/seed_specialist_routing.py \
  --suites all \
  --roles architect_general frontdoor \
  --modes direct \
  --sample-size 20 \
  --dry-run \
  --output "/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_a9_source_family_seeding_eval_architect_general_frontdoor_${RUN_ID}.json"
```

## Priority 1 — Suite Blocker

Collect `suite:general:architect_general>coder_escalation` after priority 0:

```bash
uv run python scripts/benchmark/seed_specialist_routing.py \
  --suites general \
  --roles architect_general coder_escalation \
  --modes direct \
  --sample-size 20 \
  --dry-run \
  --output "/mnt/raid0/llm/epyc-inference-research/benchmarks/results/eval/seeding_a9_suite_general_architect_general_coder_escalation_${RUN_ID}.json"
```

## Post-Collection Rebuild

After collection, rerun the planner with the new result files included, then
feed the selected candidates through the existing post-collection pipeline in
`offline_reward_pairwise_expanded_gap_plan_summary.json`.

Expected decision gate: runtime gate changes remain disallowed until the rebuilt
pairwise contract passes the independent holdouts for `orchestrator_live_seed`,
`seeding_eval`, and `suite:general`.
