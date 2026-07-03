# CPU embedded NEXTN `-md` pre-reload baseline

Generated: `2026-07-03T14:03:58Z`

Counts are unique live PIDs. `state_keys` records all aliases pointing at that PID.

## Summary

- `unique_spec_process_count`: `11`
- `unique_same_file_md_count`: `6`
- `unique_separate_md_count`: `5`
- `frontdoor_unique_same_file_md_count`: `5`
- `architect_unique_same_file_md_count`: `1`
- `worker_general_unique_separate_md_count`: `5`

## Spec Processes

| state keys | port | pid | spec | has -md | same-file -md | separate -md | RSS MiB | PSS MiB |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| `coder_escalation`, `frontdoor`, `server_8070`, `worker_summarize` | 8070 | 1114960 | `draft-mtp` | true | true | false | 80908.7 | 16020.2 |
| `server_8080` | 8080 | 1115754 | `draft-mtp` | true | true | false | 73687.3 | 8798.7 |
| `server_8180` | 8180 | 1116048 | `draft-mtp` | true | true | false | 73290.4 | 8402.2 |
| `server_8280` | 8280 | 1116363 | `draft-mtp` | true | true | false | 79054.1 | 14165.5 |
| `server_8380` | 8380 | 1116637 | `draft-mtp` | true | true | false | 76434.6 | 11546.0 |
| `server_8072`, `toolrunner`, `worker_explore`, `worker_general`, `worker_math` | 8072 | 1116996 | `draft-mtp` | true | false | true | 26142.5 | 26124.8 |
| `server_8082` | 8082 | 1117328 | `draft-mtp` | true | false | true | 19449.9 | 19432.5 |
| `server_8182` | 8182 | 1117611 | `draft-mtp` | true | false | true | 17739.0 | 17721.6 |
| `server_8282` | 8282 | 1117883 | `draft-mtp` | true | false | true | 21419.5 | 21402.1 |
| `server_8382` | 8382 | 1118216 | `draft-mtp` | true | false | true | 20804.2 | 20786.7 |
| `architect_general`, `server_8083` | 8083 | 1131429 | `draft-mtp` | true | true | false | 155350.2 | 80707.9 |
