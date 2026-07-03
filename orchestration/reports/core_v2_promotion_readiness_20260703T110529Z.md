# core_v2 Promotion Readiness Report

- Status: blocked
- Recommendation: candidate artifact is inspected, but activation remains blocked until the operator appends a matching autopilot_quality instrument-era row
- Core ID: `core_v2_ledger_20260703_min5`
- Core artifact: `/mnt/raid0/llm/epyc-orchestrator/benchmarks/prompts/core_v2_ledger_20260703_min5.jsonl`
- Core rows: 40 question row(s)
- Selection evidence: selected=40, eligible=79, observed=923, source_rows=77, unresolved=0
- Ledger provenance: trusted_rows=77, untrusted_rows=25, era_excluded_rows=849, exclude_before_ts=1782511631.0
- Instrument-era guard: status=missing_core_era, ok=False, path=`/mnt/raid0/llm/epyc-orchestrator/orchestration/instrument_eras.yaml`

## Blockers

- instrument era: no active autopilot_quality instrument-era row declares a core_id; append the human-owned E4/core row before enabling AUTOPILOT_T1_CORE_ID

## Activation Env

- `AUTOPILOT_T1_CORE_ID=core_v2_ledger_20260703_min5`
- `AUTOPILOT_T1_CORE_PATH=/mnt/raid0/llm/epyc-orchestrator/benchmarks/prompts/core_v2_ledger_20260703_min5.jsonl`
