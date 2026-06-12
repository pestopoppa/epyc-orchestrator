# Economic Ledger

Window: 2026-06-06 through 2026-06-12 UTC

## Cloud spend

- planner archive billable calls: 106 / 324
- planner archive spend: $94.4643
- manual cloud spend: $0.0000 (no `cloud_costs.yaml` found)
- total cloud spend: $94.4643
- planner wall time: 4.31h

### Planner spend by provider
  - `claude`: $94.4643

### Planner spend by purpose
  - `planner:cloud_session`: $94.4643

### Manual spend by purpose
  - none

## Local inference

- autopilot eval trials with wall time: 99
- eval wall time: 22.36h

### Local eval wall time by consumer
  - `seed_batch:seeder`: 10.72h
  - `structural_experiment:structural_lab`: 10.12h
  - `prompt_mutation:prompt_forge`: 0.64h
  - `code_mutation:prompt_forge`: 0.44h
  - `gepa_optimize:prompt_forge`: 0.22h
  - `train_routing_models:structural_lab`: 0.22h

## Decision throughput proxy

This section is a proxy. The repo does not yet expose a canonical operator-decision event stream.
- root progress files scanned: 3
- progress decision markers: 111
- halt/resume/restart mentions: 29
- automated routing decisions: 7674
- completed interactive tasks: 6907
- median task duration from progress JSONL: 19.80s

## Parse health

- planner malformed rows: 0
- journal malformed rows: 0
- progress malformed rows: 0
