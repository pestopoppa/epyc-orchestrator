# Running-State Attestation

Generated: `2026-06-19T10:52:39Z`
Trigger: `dcp6a-deploy-boundary-check`
Scope: `W1_W2_W3_W4_process_flags_serving_eval_drift_cadence_consumers`
Processes: `12`
Issues: `15`

## Process Summary

| kind | count |
|---|---:|
| llama_server | 10 |
| mcp_server | 1 |
| orchestrator_api | 1 |

## Processes

| pid | kind | port | registry | exe | sha256[:12] | link status |
|---:|---|---:|---|---|---|---|
| 2395669 | mcp_server |  |  | `/home/node/.local/share/uv/python/cpython-3.14.5-linux-x86_64-gnu/bin/python3.14` | a1512f9a0702 | ok |
| 2681646 | llama_server | 8070 | server_mode.frontdoor, server_mode.coder_escalation | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server (deleted)` |  | warn |
| 2684681 | llama_server | 8083 | server_mode.architect_general | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server (deleted)` |  | warn |
| 2685060 | llama_server | 8085 | server_mode.ingest_long_context | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server (deleted)` |  | warn |
| 2691469 | llama_server | 8086 | roles.worker_vision | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server (deleted)` |  | warn |
| 2691756 | llama_server | 8087 | roles.vision_escalation | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server (deleted)` |  | warn |
| 2692432 | llama_server | 8187 |  | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server (deleted)` |  | warn |
| 2692786 | llama_server | 8287 |  | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server (deleted)` |  | warn |
| 2693057 | llama_server | 8387 |  | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server (deleted)` |  | warn |
| 2693375 | llama_server | 8487 |  | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server (deleted)` |  | warn |
| 3145515 | llama_server | 8072 | server_mode.worker | `/mnt/raid0/llm/ik_llama.cpp/build/bin/llama-server` | 3a49fb4a781c | ok |
| 3218897 | orchestrator_api | 8000 |  | `/home/node/.local/share/uv/python/cpython-3.14.5-linux-x86_64-gnu/bin/python3.14` | a1512f9a0702 | ok |

## Feature Flags

Status: `warn`
Endpoint: `http://127.0.0.1:8000/config/attest`
Workers seen: `6`
Heterogeneous flags: `0`
Intent diffs: `36`
Env diffs: `0`

| pid | error | enabled flags |
|---:|---|---:|
| 3218899 |  | 39 |
| 3218900 |  | 39 |
| 3218901 |  | 39 |
| 3218902 |  | 39 |
| 3218903 |  | 39 |
| 3218904 |  | 39 |

## Serving Config

| pid | port | registry | model | draft | ctx | threads | proc cpus | task union | cpu intent | numa |
|---:|---:|---|---|---|---:|---:|---|---|---|---|
| 2681646 | 8070 | server_mode.frontdoor, server_mode.coder_escalation | `/mnt/raid0/llm/models/Qwen_Qwen3.6-35B-A3B-Q8_0.gguf` | `` | 32768 | 96 | 0,96 | 0-47,96-143 | 0-47,96-143 | ok |
| 2684681 | 8083 | server_mode.architect_general | `/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3.5-122B-A10B-GGUF/Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf` | `` | 16384 | 96 | 0 | 0-95 | 0-95 | ok |
| 2685060 | 8085 | server_mode.ingest_long_context | `/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf` | `` | 32768 | 96 | 0,96 | 0-47,96-143 | 0-47,96-143 | ok |
| 2691469 | 8086 | roles.worker_vision | `/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-VL-7B-Instruct-GGUF/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` | `` | 8192 | 24 | 24,120 | 24-47,120-143 | 24-47,120-143 | ok |
| 2691756 | 8087 | roles.vision_escalation | `/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-VL-30B-A3B-Instruct-GGUF/Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf` | `` | 16384 | 96 | 72,168 | 48-95,144-191 | 48-95,144-191 | ok |
| 2692432 | 8187 |  | `/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-VL-30B-A3B-Instruct-GGUF/Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf` | `` | 16384 | 48 | 0,96 | 0-23,96-119 | 0-23,96-119 | ok |
| 2692786 | 8287 |  | `/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-VL-30B-A3B-Instruct-GGUF/Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf` | `` | 16384 | 48 | 24,120 | 24-47,120-143 | 24-47,120-143 | ok |
| 2693057 | 8387 |  | `/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-VL-30B-A3B-Instruct-GGUF/Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf` | `` | 16384 | 48 | 48,144 | 48-71,144-167 | 48-71,144-167 | ok |
| 2693375 | 8487 |  | `/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-VL-30B-A3B-Instruct-GGUF/Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf` | `` | 16384 | 48 | 72,168 | 72-95,168-191 | 72-95,168-191 | ok |
| 3145515 | 8072 | server_mode.worker | `/mnt/raid0/llm/models/gemma-4-26B-A4B-it-Q4_K_M.gguf` | `/mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-Q8_0.gguf` | 16384 | 96 | 0 | 0-95 | 0-95 | ok |

## Eval Instrument

Status: `warn`

| file | exists | sha256[:12] | mtime |
|---|---|---|---|
| `orchestration/instrument_eras.yaml` | True | 4e612860c32d | 2026-06-12T15:09:07Z |
| `scripts/autopilot/sentinel_questions.yaml` | True | f31b2f07f1c8 | 2026-04-09T15:58:02Z |
| `scripts/autopilot/tool_sentinels.yaml` | True | b8b82b6d17b0 | 2026-06-04T13:06:39Z |
| `orchestration/deep_research_sentinel.yaml` | True | 2b1cfe9bc3e2 | 2026-04-22T23:24:20Z |

| pid | kind | AUTOPILOT_TOOL_SENTINELS |
|---:|---|---|
| 3218897 | orchestrator_api | `` |

## Drift

Status: `ok`

| repo | indexed | current | stale | status |
|---|---|---|---|---|
| `/mnt/raid0/llm/epyc-orchestrator` | 27e09a1 | 27e09a1 | False | ✅ up-to-date |
| `/mnt/raid0/llm/epyc-inference-research` | 4f1f6d9 | 4f1f6d9 | False | ✅ up-to-date |
| `/mnt/raid0/llm/epyc-root` | e6afbfc | e6afbfc | False | ✅ up-to-date |

## Issues

| pid | kind | port | issues |
|---:|---|---:|---|
| 2681646 | llama_server | 8070 | readelf_failed_or_not_elf |
| 2684681 | llama_server | 8083 | readelf_failed_or_not_elf |
| 2685060 | llama_server | 8085 | readelf_failed_or_not_elf |
| 2691469 | llama_server | 8086 | readelf_failed_or_not_elf |
| 2691756 | llama_server | 8087 | readelf_failed_or_not_elf |
| 2692432 | llama_server | 8187 | readelf_failed_or_not_elf |
| 2692432 | llama_server | 8187 | port_not_found_in_registry |
| 2692786 | llama_server | 8287 | readelf_failed_or_not_elf |
| 2692786 | llama_server | 8287 | port_not_found_in_registry |
| 2693057 | llama_server | 8387 | readelf_failed_or_not_elf |
| 2693057 | llama_server | 8387 | port_not_found_in_registry |
| 2693375 | llama_server | 8487 | readelf_failed_or_not_elf |
| 2693375 | llama_server | 8487 | port_not_found_in_registry |
|  | feature_flags |  | flag_intent_diffs=36 |
|  | eval_instrument |  | missing_AUTOPILOT_TOOL_SENTINELS=1 |

## Pending Sections

backup_w3
