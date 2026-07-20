# Running-State Attestation

Generated: `2026-07-20T14:58:39Z`
Trigger: `v7_era_dashboard_final_commit`
Scope: `W1_W2_W3_W4_process_flags_serving_eval_drift_cadence_consumers`
Processes: `26`
Issues: `1`

## Process Summary

| kind | count |
|---|---:|
| lightonocr | 1 |
| llama_server | 21 |
| mcp_server | 2 |
| orchestrator_api | 1 |
| whisper | 1 |

## Processes

| pid | kind | port | registry | exe | sha256[:12] | link status |
|---:|---|---:|---|---|---|---|
| 2203664 | mcp_server |  |  | `/home/node/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12` | 9544d2a29138 | ok |
| 4047066 | llama_server | 8080 | runtime_defaults.server_defaults, server_mode.frontdoor, stack_manifest.frontdoor | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | df4806655b07 | ok |
| 4047352 | llama_server | 8180 | server_mode.frontdoor, stack_manifest.frontdoor | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | df4806655b07 | ok |
| 4047606 | llama_server | 8280 | server_mode.frontdoor, stack_manifest.frontdoor | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | df4806655b07 | ok |
| 4047878 | llama_server | 8380 | server_mode.frontdoor, stack_manifest.frontdoor | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | df4806655b07 | ok |
| 4048130 | llama_server | 8082 | worker_pool.workers.explore, server_mode.worker, stack_manifest.worker_general, stack_manifest.worker_explore, stack_manifest.worker_math, stack_manifest.toolrunner | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | df4806655b07 | ok |
| 4048408 | llama_server | 8182 | server_mode.worker, stack_manifest.worker_general | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | df4806655b07 | ok |
| 4048666 | llama_server | 8282 | server_mode.worker, stack_manifest.worker_general | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | df4806655b07 | ok |
| 4048968 | llama_server | 8382 | server_mode.worker, stack_manifest.worker_general | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | df4806655b07 | ok |
| 4049394 | llama_server | 8083 | server_mode.architect_general, stack_manifest.architect_general | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | df4806655b07 | ok |
| 4049728 | llama_server | 8185 | stack_manifest.ingest_long_context | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | df4806655b07 | ok |
| 4050010 | llama_server | 8285 | stack_manifest.ingest_long_context | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | df4806655b07 | ok |
| 4050262 | llama_server | 8385 | stack_manifest.ingest_long_context | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | df4806655b07 | ok |
| 4050533 | llama_server | 8485 | stack_manifest.ingest_long_context | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | df4806655b07 | ok |
| 4050787 | llama_server | 8086 | roles.worker_vision, stack_manifest.worker_vision | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | df4806655b07 | ok |
| 4051016 | llama_server | 8087 | roles.vision_escalation, stack_manifest.vision_escalation | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | df4806655b07 | ok |
| 4051276 | llama_server | 8090 | stack_manifest.embedder | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | df4806655b07 | ok |
| 4051477 | llama_server | 8091 | stack_manifest.embedder_1 | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | df4806655b07 | ok |
| 4051699 | llama_server | 8092 | worker_pool.workers.code, stack_manifest.embedder_2 | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | df4806655b07 | ok |
| 4051907 | llama_server | 8093 | stack_manifest.embedder_3 | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | df4806655b07 | ok |
| 4052109 | llama_server | 8094 | stack_manifest.embedder_4 | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | df4806655b07 | ok |
| 4052314 | llama_server | 8095 | stack_manifest.embedder_5 | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | df4806655b07 | ok |
| 4052971 | lightonocr | 9001 |  | `/home/node/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12` | 9544d2a29138 | ok |
| 4053492 | whisper | 9000 |  | `/usr/bin/python3.13` | efb29ce53d36 | ok |
| 4080652 | mcp_server |  |  | `/home/node/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12` | 9544d2a29138 | ok |
| 4119996 | orchestrator_api | 8000 |  | `/home/node/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12` | 9544d2a29138 | ok |

## Feature Flags

Status: `ok`
Endpoint: `http://127.0.0.1:8000/config/attest`
Workers seen: `6`
Heterogeneous flags: `0`
Intent diffs: `0`
Env diffs: `0`

| pid | error | enabled flags |
|---:|---|---:|
| 4119998 |  | 33 |
| 4119999 |  | 33 |
| 4120000 |  | 33 |
| 4120001 |  | 33 |
| 4120002 |  | 33 |
| 4120003 |  | 33 |

## Serving Config

| pid | port | registry | model | draft | ctx | threads | proc cpus | task union | cpu intent | numa |
|---:|---:|---|---|---|---:|---:|---|---|---|---|
| 4047066 | 8080 | runtime_defaults.server_defaults, server_mode.frontdoor, stack_manifest.frontdoor | `/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf` | `` | 32768 | 48 | 0,96 | 0-23,96-119 | 0-23,96-119 | ok |
| 4047352 | 8180 | server_mode.frontdoor, stack_manifest.frontdoor | `/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf` | `` | 32768 | 48 | 24,120 | 24-47,120-143 | 24-47,120-143 | ok |
| 4047606 | 8280 | server_mode.frontdoor, stack_manifest.frontdoor | `/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf` | `` | 32768 | 48 | 48,144 | 48-71,144-167 | 48-71,144-167 | ok |
| 4047878 | 8380 | server_mode.frontdoor, stack_manifest.frontdoor | `/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf` | `` | 32768 | 48 | 72,168 | 72-95,168-191 | 72-95,168-191 | ok |
| 4048130 | 8082 | worker_pool.workers.explore, server_mode.worker, stack_manifest.worker_general, stack_manifest.worker_explore, stack_manifest.worker_math, stack_manifest.toolrunner | `/mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf` | `/mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf` | 16384 | 48 | 0,96 | 0-23,96-119 | 0-23,96-119 | ok |
| 4048408 | 8182 | server_mode.worker, stack_manifest.worker_general | `/mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf` | `/mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf` | 16384 | 48 | 24,120 | 24-47,120-143 | 24-47,120-143 | ok |
| 4048666 | 8282 | server_mode.worker, stack_manifest.worker_general | `/mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf` | `/mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf` | 16384 | 48 | 48,144 | 48-71,144-167 | 48-71,144-167 | ok |
| 4048968 | 8382 | server_mode.worker, stack_manifest.worker_general | `/mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf` | `/mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf` | 16384 | 48 | 72,168 | 72-95,168-191 | 72-95,168-191 | ok |
| 4049394 | 8083 | server_mode.architect_general, stack_manifest.architect_general | `/mnt/raid0/llm/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00001-of-00003.gguf` | `` | 16384 | 96 | 0 | 0-95 | 0-95 | ok |
| 4049728 | 8185 | stack_manifest.ingest_long_context | `/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf` | `` | 32768 | 48 | 0,96 | 0-23,96-119 | 0-23,96-119 | ok |
| 4050010 | 8285 | stack_manifest.ingest_long_context | `/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf` | `` | 32768 | 48 | 24,120 | 24-47,120-143 | 24-47,120-143 | ok |
| 4050262 | 8385 | stack_manifest.ingest_long_context | `/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf` | `` | 32768 | 48 | 48,144 | 48-71,144-167 | 48-71,144-167 | ok |
| 4050533 | 8485 | stack_manifest.ingest_long_context | `/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf` | `` | 32768 | 48 | 72,168 | 72-95,168-191 | 72-95,168-191 | ok |
| 4050787 | 8086 | roles.worker_vision, stack_manifest.worker_vision | `/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-VL-7B-Instruct-GGUF/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` | `` | 8192 | 24 | 24,120 | 24-47,120-143 | 24-47,120-143 | ok |
| 4051016 | 8087 | roles.vision_escalation, stack_manifest.vision_escalation | `/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-VL-7B-Instruct-GGUF/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` | `` | 8192 | 24 | 72,168 | 72-95,168-191 | 72-95,168-191 | ok |
| 4051276 | 8090 | stack_manifest.embedder | `/mnt/raid0/llm/models/bge-large-en-v1.5-f16.gguf` | `` | 512 | 4 | 0,96 | 0-191 |  | n/a |
| 4051477 | 8091 | stack_manifest.embedder_1 | `/mnt/raid0/llm/models/bge-large-en-v1.5-f16.gguf` | `` | 512 | 4 | 0,96 | 0-191 |  | n/a |
| 4051699 | 8092 | worker_pool.workers.code, stack_manifest.embedder_2 | `/mnt/raid0/llm/models/bge-large-en-v1.5-f16.gguf` | `` | 512 | 4 | 0,96 | 0-191 |  | n/a |
| 4051907 | 8093 | stack_manifest.embedder_3 | `/mnt/raid0/llm/models/bge-large-en-v1.5-f16.gguf` | `` | 512 | 4 | 0,96 | 0-191 |  | n/a |
| 4052109 | 8094 | stack_manifest.embedder_4 | `/mnt/raid0/llm/models/bge-large-en-v1.5-f16.gguf` | `` | 512 | 4 | 0,96 | 0-191 |  | n/a |
| 4052314 | 8095 | stack_manifest.embedder_5 | `/mnt/raid0/llm/models/bge-large-en-v1.5-f16.gguf` | `` | 512 | 4 | 0,96 | 0-191 |  | n/a |

## Eval Instrument

Status: `ok`

| file | exists | sha256[:12] | mtime |
|---|---|---|---|
| `orchestration/instrument_eras.yaml` | True | 2b695f70a815 | 2026-07-20T14:47:42Z |
| `scripts/autopilot/sentinel_questions.yaml` | True | f31b2f07f1c8 | 2026-04-09T15:58:02Z |
| `scripts/autopilot/tool_sentinels.yaml` | True | aa002aa1457d | 2026-07-06T18:38:53Z |
| `orchestration/deep_research_sentinel.yaml` | True | 2b1cfe9bc3e2 | 2026-04-22T23:24:20Z |

| pid | kind | AUTOPILOT_TOOL_SENTINELS |
|---:|---|---|
| 4119996 | orchestrator_api | `1` |

## Drift

Status: `warn`

| repo | indexed | current | stale | status |
|---|---|---|---|---|
| `/mnt/raid0/llm/epyc-root` | 49486c0 | c4718cc | True | ⚠️ stale (re-run gitnexus analyze) |
| `/mnt/raid0/llm/epyc-orchestrator` | 987762a | 7dcada4 | True | ⚠️ stale (re-run gitnexus analyze) |
| `/mnt/raid0/llm/epyc-inference-research` | 2c2b94b | 0a6fd8a | True | ⚠️ stale (re-run gitnexus analyze) |
| `/mnt/raid0/llm/llama.cpp` | a6c793f | 6ad45fa | True | ⚠️ stale (re-run gitnexus analyze) |

## DCP/J7 Status

Status: `insufficient`
Latest run: `benchmarks/results/runs/dcp_j7/stub-20260706T033205Z`
Mode: `stub`
Recommendation: `rerun with enough rows and quality-scored prompts`
Blockers: `missing_latency_delta, quality_not_scored`

| arm | n | p50 elapsed s | errors | quality scored |
|---|---:|---:|---:|---:|
| off | 3 | 0.0 | 0 | 0 |
| on | 3 | 0.0 | 0 | 0 |

Delta p50 elapsed: ``


## Issues

| pid | kind | port | issues |
|---:|---|---:|---|
|  | drift |  | gitnexus_stale_or_error=4 |

## Pending Sections

backup_w3
