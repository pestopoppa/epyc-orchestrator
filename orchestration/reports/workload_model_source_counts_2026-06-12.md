# Workload Model Source Counts

Generated: `2026-06-12T20:56:32Z`

Purpose: evidence record for `orchestration/workload_model.yaml` W1.

## Sources

| source | files | measured unit | count | use |
|---|---:|---|---:|---|
| `/mnt/raid0/llm/epyc-root/progress` | 33 | top-level `##` sections | 314 | real-workload proxy |
| `/mnt/raid0/llm/epyc-orchestrator/logs/progress` | 24 | `task_started` events | 49,297 | traffic pressure, not demand |
| `/mnt/raid0/llm/hermes-agent` | 0 | durable task logs | 0 | no observed Hermes task ledger |

## Progress-Section Proxy

Generic/admin headings were counted but excluded from demand shares.

| class | sections | share of 247 classified sections |
|---|---:|---:|
| benchmark_eval_measurement | 46 | 0.186 |
| ops_deploy_process | 45 | 0.182 |
| code_change_implementation | 44 | 0.178 |
| debug_root_cause | 36 | 0.146 |
| governance_docs_handoff | 35 | 0.142 |
| research_intake_deep_dive | 26 | 0.105 |
| planning_architecture_review | 15 | 0.061 |
| other_session_admin | 67 | excluded |

## Structured Task Log

The structured task log is dominated by repeated benchmark/eval prompts. It should not be treated as the human demand distribution until W2 lands explicit `task_record` capture.

| bucket | events | unique objectives |
|---|---:|---:|
| uncategorized_chat_mostly_synthetic | 20,275 | 1,241 |
| synthetic_qa_knowledge | 18,180 | 1,940 |
| synthetic_code_eval | 3,950 | 304 |
| synthetic_math_reasoning | 3,259 | 134 |
| synthetic_instruction_following | 2,338 | 53 |
| synthetic_tool_use | 1,295 | 17 |

## Caveats

- Progress headings are a coarse proxy: one section can contain multiple tasks, and some headings are generic.
- Runtime task logs are valuable for load and routing pressure, but synthetic eval traffic swamps real operator tasks.
- No durable Hermes task ledger was found; this is an absence of evidence, not evidence of zero Hermes usage.
- W2 must replace this proxy with explicit class/outcome labels before promotion or routing uses these shares as ground truth.
