# CPU embedded NEXTN `-md` after-reload check

Generated: `2026-07-03T14:21:52Z`

Compared to `md_self_draft_preflight_20260703T140312Z`, the six same-file CPU self-draft processes have been reloaded. Qwen embedded NEXTN processes now retain `--spec-type draft-mtp` without `-md`; Gemma worker processes keep a separate assistant-head `-md`.

## Summary

- `unique_spec_process_count`: `11`
- `unique_same_file_md_count`: `0`
- `unique_separate_md_count`: `5`
- `unique_embedded_self_draft_count`: `6`
- `frontdoor_unique_same_file_md_count`: `0`
- `frontdoor_unique_embedded_self_draft_count`: `5`
- `architect_unique_same_file_md_count`: `0`
- `architect_unique_embedded_self_draft_count`: `1`
- `worker_general_unique_separate_md_count`: `5`

## Spec Processes

| state keys | port | pid | spec | has -md | same-file -md | separate -md | RSS MiB | PSS MiB |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| `coder_escalation`, `frontdoor`, `server_8070`, `worker_summarize` | 8070 | 1661738 | `draft-mtp` | false | false | false | 37181.5 | 8332.8 |
| `server_8072`, `toolrunner`, `worker_explore`, `worker_general`, `worker_math` | 8072 | 1116996 | `draft-mtp` | true | false | true | 26152.9 | 26135.3 |
| `server_8080` | 8080 | 1665110 | `draft-mtp` | false | false | false | 37146.8 | 8298.1 |
| `server_8082` | 8082 | 1117328 | `draft-mtp` | true | false | true | 19581.0 | 19563.6 |
| `architect_general`, `server_8083` | 8083 | 1662585 | `draft-mtp` | false | false | false | 76809.8 | 76792.6 |
| `server_8180` | 8180 | 1665424 | `draft-mtp` | false | false | false | 37156.8 | 8308.1 |
| `server_8182` | 8182 | 1117611 | `draft-mtp` | true | false | true | 17739.0 | 17721.6 |
| `server_8280` | 8280 | 1665782 | `draft-mtp` | false | false | false | 37142.1 | 8293.4 |
| `server_8282` | 8282 | 1117883 | `draft-mtp` | true | false | true | 21494.5 | 21477.1 |
| `server_8380` | 8380 | 1666152 | `draft-mtp` | false | false | false | 37141.6 | 8292.8 |
| `server_8382` | 8382 | 1118216 | `draft-mtp` | true | false | true | 20804.5 | 20787.0 |
