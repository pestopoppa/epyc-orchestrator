# Gate-3 Tool Telemetry Report

- Generated: `2026-07-04T02:35:53Z`
- Command: `AUTOPILOT_TOOL_SENTINELS=1 AUTOPILOT_EVAL_CONCURRENCY=1 .venv/bin/python scripts/autopilot/gate3_tool_telemetry.py`
- Verdict: `GATE3_HARD: PASS`
- Soft probe: `WEB_RESEARCH: PASS`

## Hard Contract

- Driver `AUTOPILOT_TOOL_SENTINELS=1`: PASS
- Orchestrator `AUTOPILOT_TOOL_SENTINELS=1`: PASS
- Orchestrator `ORCHESTRATOR_STRUCTURED_TOOL_OUTPUT=1`: PASS
- `get_eval_secret` counted `7` call(s), above the required `3`.
- All `get_eval_secret` timing rows succeeded: `7/7`.
- No-tool isolation inherited no tools: `tools_called=[]`, `tools_used=0`.

## Sentinel Batch

| Sentinel | Mode | Duration | Tools |
| --- | --- | ---: | --- |
| `tool_use_secret_alpha` | `repl` | `24.9s` | `get_eval_secret`, `get_eval_secret` |
| `tool_use_secret_bravo` | `repl` | `20.1s` | `get_eval_secret`, `get_eval_secret` |
| `tool_use_secret_charlie` | `repl` | `10.3s` | `get_eval_secret` |
| `tool_use_secret_delta` | `repl` | `10.7s` | `get_eval_secret` |
| `tool_use_secret_echo` | `repl` | `10.0s` | `get_eval_secret` |
| `isolation_no_tool` | `repl` | `9.5s` | none |

## Soft Probe

`soft_web_research` returned success telemetry and 10 result(s). The response also carried a post-tool error:

`[FAILED: Terminal role architect_general: AttributeError: 'str' object has no attribute 'get']`

Gate-3 classifies that as non-fatal soft telemetry; the hard telemetry contract is unaffected.

## Follow-Up

Fable5 `20260704T023525Z` now sees API and AutoPilot sentinel env enabled. It still reports `latest_eval_total_tool_calls_zero` because the latest journaled eval is pre-activation trial `1107`; this should clear only after a sentinel-enabled AutoPilot eval journals nonzero tool telemetry.
