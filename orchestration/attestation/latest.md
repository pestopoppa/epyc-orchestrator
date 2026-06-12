# Running-State Attestation

Generated: `2026-06-12T21:11:37Z`
Scope: `W1_process_inventory`
Processes: `34`
Issues: `13`

## Process Summary

| kind | count |
|---|---:|
| autopilot | 2 |
| lightonocr | 1 |
| llama_server | 28 |
| mcp_server | 1 |
| orchestrator_api | 1 |
| whisper | 1 |

## Processes

| pid | kind | port | registry | exe | sha256[:12] | link status |
|---:|---|---:|---|---|---|---|
| 1435666 | llama_server | 8070 | server_mode.frontdoor, server_mode.coder_escalation | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | 78deb8cb0e25 | ok |
| 1436035 | llama_server | 8080 | runtime_defaults.server_defaults, server_mode.frontdoor | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | 78deb8cb0e25 | ok |
| 1436294 | llama_server | 8180 | server_mode.frontdoor | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | 78deb8cb0e25 | ok |
| 1436547 | llama_server | 8280 | server_mode.frontdoor | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | 78deb8cb0e25 | ok |
| 1436801 | llama_server | 8380 | server_mode.frontdoor | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | 78deb8cb0e25 | ok |
| 1437087 | llama_server | 8072 | server_mode.worker | `/mnt/raid0/llm/ik_llama.cpp/build/bin/llama-server` | 3a49fb4a781c | ok |
| 1437398 | llama_server | 8082 | server_mode.worker, worker_pool.workers.explore | `/mnt/raid0/llm/ik_llama.cpp/build/bin/llama-server` | 3a49fb4a781c | ok |
| 1437701 | llama_server | 8182 | server_mode.worker | `/mnt/raid0/llm/ik_llama.cpp/build/bin/llama-server` | 3a49fb4a781c | ok |
| 1439054 | llama_server | 8382 | server_mode.worker | `/mnt/raid0/llm/ik_llama.cpp/build/bin/llama-server` | 3a49fb4a781c | ok |
| 1439342 | llama_server | 8083 | server_mode.architect_general | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | 78deb8cb0e25 | ok |
| 1440472 | llama_server | 8085 | server_mode.ingest_long_context | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | 78deb8cb0e25 | ok |
| 1441532 | llama_server | 8185 |  | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | 78deb8cb0e25 | ok |
| 1442152 | llama_server | 8285 |  | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | 78deb8cb0e25 | ok |
| 1442510 | llama_server | 8385 |  | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | 78deb8cb0e25 | ok |
| 1443092 | llama_server | 8485 |  | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | 78deb8cb0e25 | ok |
| 1443581 | llama_server | 8086 | roles.worker_vision | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | 78deb8cb0e25 | ok |
| 1443828 | llama_server | 8087 | roles.vision_escalation | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | 78deb8cb0e25 | ok |
| 1444178 | llama_server | 8187 |  | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | 78deb8cb0e25 | ok |
| 1444690 | llama_server | 8287 |  | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | 78deb8cb0e25 | ok |
| 1448544 | llama_server | 8387 |  | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | 78deb8cb0e25 | ok |
| 1452208 | llama_server | 8487 |  | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | 78deb8cb0e25 | ok |
| 1457275 | whisper | 9000 | server_mode.voice_server | `/usr/bin/python3.13` | efb29ce53d36 | ok |
| 1796140 | llama_server | 8282 | server_mode.worker | `/mnt/raid0/llm/ik_llama.cpp/build/bin/llama-server` | 3a49fb4a781c | ok |
| 1895852 | llama_server | 8090 |  | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | 78deb8cb0e25 | ok |
| 1896061 | llama_server | 8091 |  | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | 78deb8cb0e25 | ok |
| 1896269 | llama_server | 8092 | worker_pool.workers.code | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | 78deb8cb0e25 | ok |
| 1896475 | llama_server | 8093 |  | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | 78deb8cb0e25 | ok |
| 1896682 | llama_server | 8094 |  | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | 78deb8cb0e25 | ok |
| 1896917 | llama_server | 8095 |  | `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` | 78deb8cb0e25 | ok |
| 2395669 | mcp_server |  |  | `/home/node/.local/share/uv/python/cpython-3.14.5-linux-x86_64-gnu/bin/python3.14` | a1512f9a0702 | ok |
| 2780307 | orchestrator_api | 8000 |  | `/home/node/.local/share/uv/python/cpython-3.14.5-linux-x86_64-gnu/bin/python3.14` | a1512f9a0702 | ok |
| 2796323 | lightonocr | 9001 | server_mode.document_formalizer | `/home/node/.local/share/uv/python/cpython-3.14.5-linux-x86_64-gnu/bin/python3.14` | a1512f9a0702 | ok |
| 2798819 | autopilot |  |  | `/usr/bin/uv` | ba56e9683b88 | ok |
| 2798824 | autopilot |  |  | `/home/node/.local/share/uv/python/cpython-3.14.5-linux-x86_64-gnu/bin/python3.14` | a1512f9a0702 | ok |

## Issues

| pid | kind | port | issues |
|---:|---|---:|---|
| 1441532 | llama_server | 8185 | port_not_found_in_registry |
| 1442152 | llama_server | 8285 | port_not_found_in_registry |
| 1442510 | llama_server | 8385 | port_not_found_in_registry |
| 1443092 | llama_server | 8485 | port_not_found_in_registry |
| 1444178 | llama_server | 8187 | port_not_found_in_registry |
| 1444690 | llama_server | 8287 | port_not_found_in_registry |
| 1448544 | llama_server | 8387 | port_not_found_in_registry |
| 1452208 | llama_server | 8487 | port_not_found_in_registry |
| 1895852 | llama_server | 8090 | port_not_found_in_registry |
| 1896061 | llama_server | 8091 | port_not_found_in_registry |
| 1896475 | llama_server | 8093 | port_not_found_in_registry |
| 1896682 | llama_server | 8094 | port_not_found_in_registry |
| 1896917 | llama_server | 8095 | port_not_found_in_registry |

## Pending Sections

feature_flags_w2, serving_config_w2, eval_instrument_w3, drift_w3, backup_w3, cadence_consumers_w4
