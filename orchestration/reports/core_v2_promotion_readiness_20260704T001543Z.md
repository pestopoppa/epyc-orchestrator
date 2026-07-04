# core_v2 Promotion Readiness Report

- Status: blocked
- Recommendation: candidate artifact is inspected, but activation remains blocked until the operator appends a matching autopilot_quality instrument-era row
- Core ID: `core_v2_ledger_20260703_min5`
- Core artifact: `/mnt/raid0/llm/epyc-orchestrator/benchmarks/prompts/core_v2_ledger_20260703_min5.jsonl`
- Core rows: 40 question row(s)
- Selection evidence: selected=40, eligible=79, observed=923, source_rows=77, unresolved=0
- Ledger provenance: trusted_rows=77, untrusted_rows=25, era_excluded_rows=849, exclude_before_ts=1782511631.0
- Instrument-era guard: status=missing_core_era, ok=False, path=`orchestration/instrument_eras.yaml`

## Blockers

- instrument era: no active autopilot_quality instrument-era row declares a core_id; append the human-owned E4/core row before enabling AUTOPILOT_T1_CORE_ID

## Activation Env

- `AUTOPILOT_T1_CORE_ID=core_v2_ledger_20260703_min5`
- `AUTOPILOT_T1_CORE_PATH=/mnt/raid0/llm/epyc-orchestrator/benchmarks/prompts/core_v2_ledger_20260703_min5.jsonl`

## Operator Era Row Draft

- Status: draft-only; append requires human approval under `/workspace/MEASUREMENT.md`.
- Target path: `orchestration/instrument_eras.yaml`
- Post-append validation: `uv run python scripts/autopilot/core_v2_promotion_report.py --core-id core_v2_ledger_20260703_min5 --core-path /mnt/raid0/llm/epyc-orchestrator/benchmarks/prompts/core_v2_ledger_20260703_min5.jsonl --eras-path orchestration/instrument_eras.yaml --json`

```yaml
# Append this row under the existing top-level `eras:` list.
  - id: "E4-core-core-v2-ledger-20260703-min5"
    from: "2026-07-04T00:15:43Z"
    scope: "autopilot_quality"
    core_id: "core_v2_ledger_20260703_min5"
    policy_version: "core-v2-ledger-20260703"
    note: "Human-owned designed T1 core activation draft. Core artifact selected 40 rows from current-era ledger evidence; trusted_rows=77, untrusted_rows=25, era_excluded_rows=849. Agents may generate this draft but must not append it to instrument_eras.yaml."
```
