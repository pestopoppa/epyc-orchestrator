# Running-State Attestation

Generated: `2026-06-28T04:46:52Z`
Trigger: `v6-iqk-eval-parity-armC-full-on`
Scope: `W1_W2_W3_W4_process_flags_serving_eval_drift_cadence_consumers`
Processes: `33`
Issues: `16`

## Process Summary

| kind | count |
|---|---:|
| lightonocr | 1 |
| llama_server | 28 |
| mcp_server | 3 |
| orchestrator_api | 1 |

## Processes

| pid | kind | port | registry | exe | sha256[:12] | link status |
|---:|---|---:|---|---|---|---|
| 1554 | mcp_server |  |  | `/home/node/.local/share/uv/python/cpython-3.14.5-linux-x86_64-gnu/bin/python3.14` | a1512f9a0702 | ok |
| 1328126 | orchestrator_api | 8000 |  | `/home/node/.local/share/uv/python/cpython-3.14.5-linux-x86_64-gnu/bin/python3.14` | a1512f9a0702 | ok |
| 1373518 | llama_server | 8072 | server_mode.worker | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | c670826a4654 | ok |
| 2395669 | mcp_server |  |  | `/home/node/.local/share/uv/python/cpython-3.14.5-linux-x86_64-gnu/bin/python3.14` | a1512f9a0702 | ok |
| 3750433 | llama_server | 8082 | worker_pool.workers.explore, server_mode.worker | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | c670826a4654 | ok |
| 3750708 | llama_server | 8182 | server_mode.worker | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | c670826a4654 | ok |
| 3750987 | llama_server | 8282 | server_mode.worker | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | c670826a4654 | ok |
| 3751257 | llama_server | 8382 | server_mode.worker | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | c670826a4654 | ok |
| 3761909 | llama_server | 8080 | runtime_defaults.server_defaults, server_mode.frontdoor | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | c670826a4654 | ok |
| 3762213 | llama_server | 8180 | server_mode.frontdoor | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | c670826a4654 | ok |
| 3762471 | llama_server | 8280 | server_mode.frontdoor | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | c670826a4654 | ok |
| 3762730 | llama_server | 8380 | server_mode.frontdoor | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | c670826a4654 | ok |
| 3763323 | llama_server | 8185 |  | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | c670826a4654 | ok |
| 3763591 | llama_server | 8285 |  | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | c670826a4654 | ok |
| 3763867 | llama_server | 8385 |  | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | c670826a4654 | ok |
| 3764125 | llama_server | 8485 |  | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | c670826a4654 | ok |
| 3764382 | llama_server | 8086 | roles.worker_vision | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | c670826a4654 | ok |
| 3764905 | llama_server | 8187 |  | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | c670826a4654 | ok |
| 3765155 | llama_server | 8287 |  | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | c670826a4654 | ok |
| 3765406 | llama_server | 8387 |  | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | c670826a4654 | ok |
| 3765663 | llama_server | 8487 |  | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | c670826a4654 | ok |
| 3765914 | llama_server | 8090 |  | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | c670826a4654 | ok |
| 3766118 | llama_server | 8091 |  | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | c670826a4654 | ok |
| 3766327 | llama_server | 8092 | worker_pool.workers.code | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | c670826a4654 | ok |
| 3766546 | llama_server | 8093 |  | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | c670826a4654 | ok |
| 3766750 | llama_server | 8094 |  | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | c670826a4654 | ok |
| 3766966 | llama_server | 8095 |  | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | c670826a4654 | ok |
| 3767625 | lightonocr | 9001 |  | `/home/node/.local/share/uv/python/cpython-3.14.5-linux-x86_64-gnu/bin/python3.14` | a1512f9a0702 | ok |
| 3783433 | llama_server | 8087 | roles.vision_escalation | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | c670826a4654 | ok |
| 3792598 | llama_server | 8070 | server_mode.frontdoor, server_mode.coder_escalation | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | c670826a4654 | ok |
| 3793024 | llama_server | 8085 | server_mode.ingest_long_context | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | c670826a4654 | ok |
| 3811260 | llama_server | 8083 | server_mode.architect_general | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | c670826a4654 | ok |
| 4154658 | mcp_server |  |  | `/home/node/.local/share/uv/python/cpython-3.14.5-linux-x86_64-gnu/bin/python3.14` | a1512f9a0702 | ok |

## Feature Flags

Status: `warn`
Endpoint: `http://127.0.0.1:8000/config/attest`
Workers seen: `6`
Heterogeneous flags: `0`
Intent diffs: `72`
Env diffs: `0`

| pid | error | enabled flags |
|---:|---|---:|
| 1328129 |  | 33 |
| 1328130 |  | 33 |
| 1328131 |  | 33 |
| 1328132 |  | 33 |
| 1328133 |  | 33 |
| 1328134 |  | 33 |

## Serving Config

| pid | port | registry | model | draft | ctx | threads | proc cpus | task union | cpu intent | numa |
|---:|---:|---|---|---|---:|---:|---|---|---|---|
| 1373518 | 8072 | server_mode.worker | `/mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf` | `/mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf` | 16384 | 96 | 0 | 0-95 | 0-95 | ok |
| 3750433 | 8082 | worker_pool.workers.explore, server_mode.worker | `/mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf` | `/mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf` | 16384 | 48 | 0,96 | 0-23,96-119 | 0-23,96-119 | ok |
| 3750708 | 8182 | server_mode.worker | `/mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf` | `/mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf` | 16384 | 48 | 24,120 | 24-47,120-143 | 24-47,120-143 | ok |
| 3750987 | 8282 | server_mode.worker | `/mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf` | `/mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf` | 16384 | 48 | 48,144 | 48-71,144-167 | 48-71,144-167 | ok |
| 3751257 | 8382 | server_mode.worker | `/mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf` | `/mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf` | 16384 | 48 | 72,168 | 72-95,168-191 | 72-95,168-191 | ok |
| 3761909 | 8080 | runtime_defaults.server_defaults, server_mode.frontdoor | `/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf` | `/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf` | 32768 | 48 | 0,96 | 0-23,96-119 | 0-23,96-119 | ok |
| 3762213 | 8180 | server_mode.frontdoor | `/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf` | `/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf` | 32768 | 48 | 24,120 | 24-47,120-143 | 24-47,120-143 | ok |
| 3762471 | 8280 | server_mode.frontdoor | `/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf` | `/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf` | 32768 | 48 | 48,144 | 48-71,144-167 | 48-71,144-167 | ok |
| 3762730 | 8380 | server_mode.frontdoor | `/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf` | `/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf` | 32768 | 48 | 72,168 | 72-95,168-191 | 72-95,168-191 | ok |
| 3763323 | 8185 |  | `/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf` | `` | 32768 | 48 | 0,96 | 0-23,96-119 | 0-23,96-119 | ok |
| 3763591 | 8285 |  | `/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf` | `` | 32768 | 48 | 24,120 | 24-47,120-143 | 24-47,120-143 | ok |
| 3763867 | 8385 |  | `/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf` | `` | 32768 | 48 | 48,144 | 48-71,144-167 | 48-71,144-167 | ok |
| 3764125 | 8485 |  | `/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf` | `` | 32768 | 48 | 72,168 | 72-95,168-191 | 72-95,168-191 | ok |
| 3764382 | 8086 | roles.worker_vision | `/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-VL-7B-Instruct-GGUF/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` | `` | 8192 | 24 | 24,120 | 24-47,120-143 | 24-47,120-143 | ok |
| 3764905 | 8187 |  | `/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-VL-30B-A3B-Instruct-GGUF/Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf` | `` | 16384 | 48 | 0,96 | 0-23,96-119 | 0-23,96-119 | ok |
| 3765155 | 8287 |  | `/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-VL-30B-A3B-Instruct-GGUF/Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf` | `` | 16384 | 48 | 24,120 | 24-47,120-143 | 24-47,120-143 | ok |
| 3765406 | 8387 |  | `/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-VL-30B-A3B-Instruct-GGUF/Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf` | `` | 16384 | 48 | 48,144 | 48-71,144-167 | 48-71,144-167 | ok |
| 3765663 | 8487 |  | `/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-VL-30B-A3B-Instruct-GGUF/Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf` | `` | 16384 | 48 | 72,168 | 72-95,168-191 | 72-95,168-191 | ok |
| 3765914 | 8090 |  | `/mnt/raid0/llm/models/bge-large-en-v1.5-f16.gguf` | `` | 512 | 4 | 0,96 | 0-191 |  | n/a |
| 3766118 | 8091 |  | `/mnt/raid0/llm/models/bge-large-en-v1.5-f16.gguf` | `` | 512 | 4 | 0,96 | 0-191 |  | n/a |
| 3766327 | 8092 | worker_pool.workers.code | `/mnt/raid0/llm/models/bge-large-en-v1.5-f16.gguf` | `` | 512 | 4 | 0,96 | 0-191 |  | n/a |
| 3766546 | 8093 |  | `/mnt/raid0/llm/models/bge-large-en-v1.5-f16.gguf` | `` | 512 | 4 | 0,96 | 0-191 |  | n/a |
| 3766750 | 8094 |  | `/mnt/raid0/llm/models/bge-large-en-v1.5-f16.gguf` | `` | 512 | 4 | 0,96 | 0-191 |  | n/a |
| 3766966 | 8095 |  | `/mnt/raid0/llm/models/bge-large-en-v1.5-f16.gguf` | `` | 512 | 4 | 0,96 | 0-191 |  | n/a |
| 3783433 | 8087 | roles.vision_escalation | `/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-VL-30B-A3B-Instruct-GGUF/Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf` | `` | 16384 | 96 | 72,168 | 48-95,144-191 | 48-95,144-191 | ok |
| 3792598 | 8070 | server_mode.frontdoor, server_mode.coder_escalation | `/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf` | `/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf` | 32768 | 96 | 0,96 | 0-47,96-143 | 0-47,96-143 | ok |
| 3793024 | 8085 | server_mode.ingest_long_context | `/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf` | `` | 32768 | 96 | 0,96 | 0-47,96-143 | 0-47,96-143 | ok |
| 3811260 | 8083 | server_mode.architect_general | `/mnt/raid0/llm/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00001-of-00003.gguf` | `/mnt/raid0/llm/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00001-of-00003.gguf` | 16384 | 96 | 0 | 0-95 | 0-95 | ok |

## Eval Instrument

Status: `warn`

| file | exists | sha256[:12] | mtime |
|---|---|---|---|
| `orchestration/instrument_eras.yaml` | True | e7dc46cbe458 | 2026-06-27T22:51:04Z |
| `scripts/autopilot/sentinel_questions.yaml` | True | f31b2f07f1c8 | 2026-04-09T15:58:02Z |
| `scripts/autopilot/tool_sentinels.yaml` | True | b8b82b6d17b0 | 2026-06-04T13:06:39Z |
| `orchestration/deep_research_sentinel.yaml` | True | 2b1cfe9bc3e2 | 2026-04-22T23:24:20Z |

| pid | kind | AUTOPILOT_TOOL_SENTINELS |
|---:|---|---|
| 1328126 | orchestrator_api | `` |

## Drift

Status: `warn`

| repo | indexed | current | stale | status |
|---|---|---|---|---|
| `/mnt/raid0/llm/epyc-root` | 4273d50 | 4273d50 | False | ✅ up-to-date |
| `/mnt/raid0/llm/epyc-orchestrator` | 521be3f | 521be3f | False | ✅ up-to-date |
| `/mnt/raid0/llm/epyc-inference-research` | 7638f0b | 7638f0b | False | ✅ up-to-date |
| `/mnt/raid0/llm/llama.cpp` | a6c793f | 9174561 | True | ⚠️ stale (re-run gitnexus analyze) |

## DCP/J7 Status

Status: `hold`
Latest run: `benchmarks/results/runs/dcp_j7/20260619T113143Z`
Mode: `real`
Recommendation: `keep dcp_pre_assembly default-off`
Blockers: `latency_not_improved, quality_not_scored`

| arm | n | p50 elapsed s | errors | quality scored |
|---|---:|---:|---:|---:|
| off | 3 | 20.219 | 0 | 0 |
| on | 3 | 32.628 | 0 | 0 |

Delta p50 elapsed: `-0.6137`


## Issues

| pid | kind | port | issues |
|---:|---|---:|---|
| 3763323 | llama_server | 8185 | port_not_found_in_registry |
| 3763591 | llama_server | 8285 | port_not_found_in_registry |
| 3763867 | llama_server | 8385 | port_not_found_in_registry |
| 3764125 | llama_server | 8485 | port_not_found_in_registry |
| 3764905 | llama_server | 8187 | port_not_found_in_registry |
| 3765155 | llama_server | 8287 | port_not_found_in_registry |
| 3765406 | llama_server | 8387 | port_not_found_in_registry |
| 3765663 | llama_server | 8487 | port_not_found_in_registry |
| 3765914 | llama_server | 8090 | port_not_found_in_registry |
| 3766118 | llama_server | 8091 | port_not_found_in_registry |
| 3766546 | llama_server | 8093 | port_not_found_in_registry |
| 3766750 | llama_server | 8094 | port_not_found_in_registry |
| 3766966 | llama_server | 8095 | port_not_found_in_registry |
|  | feature_flags |  | flag_intent_diffs=72 |
|  | eval_instrument |  | missing_AUTOPILOT_TOOL_SENTINELS=1 |
|  | drift |  | gitnexus_stale_or_error=1 |

## Pending Sections

backup_w3
