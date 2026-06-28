# Tool Definition Token Audit — 2026-06-28

## Summary
- DEFAULT_ROOT_LM_TOOLS: **377** est. tokens (290 words)
- COMPACT_ROOT_LM_TOOLS: **209** est. tokens (161 words)
- Compression ratio (compact/default): **55.4%**
- DEFAULT_ROOT_LM_RULES: **1278** est. tokens (983 words)
- Role overlays: 10 files, **852** est. tokens (655 words)
- Total system prompt budget (tools+rules+roles): **2507** est. tokens
- Prompt-library size proxy: **10686** char-proxy tokens (not per-request AP-16 overhead)
- Runtime AP-16 scaffold, frontdoor+worker: **3021** char-proxy tokens

## Per-Tool Token Cost (DEFAULT_ROOT_LM_TOOLS)

| Tool | Section | Est. Tokens | Words | Usage Freq | Impact Score | Duplicate? |
|------|---------|-------------|-------|------------|--------------|------------|
| web_research | Ungrouped | 31 | 24 | n/a | n/a |  |
| CALL | Ungrouped | 23 | 18 | n/a | n/a |  |
| web_search | Ungrouped | 20 | 15 | n/a | n/a |  |
| context | Ungrouped | 16 | 12 | n/a | n/a |  |
| ocr_document | Ungrouped | 14 | 11 | n/a | n/a |  |
| recall | Ungrouped | 14 | 11 | n/a | n/a |  |
| context_len | Ungrouped | 14 | 11 | n/a | n/a |  |
| search_arxiv | Ungrouped | 13 | 10 | n/a | n/a |  |
| run_shell | Ungrouped | 13 | 10 | n/a | n/a |  |
| route_advice | Ungrouped | 13 | 10 | n/a | n/a |  |
| list_dir | Ungrouped | 12 | 9 | n/a | n/a |  |
| run_tests | Ungrouped | 12 | 9 | n/a | n/a |  |
| fetch_docs | Ungrouped | 10 | 8 | n/a | n/a |  |
| llm_call | Ungrouped | 10 | 8 | n/a | n/a |  |
| artifacts | Ungrouped | 9 | 7 | n/a | n/a |  |
| peek | Ungrouped | 9 | 7 | n/a | n/a |  |
| escalate | Ungrouped | 9 | 7 | n/a | n/a |  |
| get_wikipedia_article | Ungrouped | 8 | 6 | n/a | n/a |  |
| run_python_code | Ungrouped | 8 | 6 | n/a | n/a |  |
| FINAL | Ungrouped | 8 | 6 | n/a | n/a |  |
| search_wikipedia | Ungrouped | 7 | 5 | n/a | n/a |  |
| my_role | Ungrouped | 7 | 5 | n/a | n/a |  |
| fetch_report | Ungrouped | 7 | 5 | n/a | n/a |  |
| list_tools | Ungrouped | 5 | 4 | n/a | n/a |  |

## Compact Tool Definitions (COMPACT_ROOT_LM_TOOLS)

| Tool | Est. Tokens | Words |
|------|-------------|-------|
| web_research | 31 | 24 |
| web_search | 22 | 17 |
| search_wikipedia | 10 | 8 |
| run_python_code | 14 | 11 |
| context | 18 | 14 |
| artifacts | 9 | 7 |
| peek | 10 | 8 |
| grep | 9 | 7 |
| file_write_safe | 13 | 10 |
| llm_call | 10 | 8 |
| escalate | 10 | 8 |
| fetch_report | 12 | 9 |
| FINAL | 12 | 9 |
| CALL | 13 | 10 |
| list_tools | 14 | 11 |

## Duplicate Entries

No duplicate tool entries found.

## Role Overlay Costs

| File | Est. Tokens | Words |
|------|-------------|-------|
| frontdoor.md | 300 | 231 |
| worker_explore.md | 121 | 93 |
| architect_coding.md <!-- stack-change-guard: allow historical retired-role note --> | 86 | 66 |
| architect_general.md | 83 | 64 |
| coder_escalation.md | 78 | 60 |
| worker_general.md | 77 | 59 |
| coder_primary.md | 43 | 33 |
| worker_math.md | 36 | 28 |
| ingest_long_context.md | 21 | 16 |
| worker_vision.md | 7 | 5 |

## Compression Candidates (High Cost, Low/Zero Usage)

*Usage data unavailable — ranking by token cost only.*

1. **web_research** — 31 est. tokens, section: Ungrouped
2. **CALL** — 23 est. tokens, section: Ungrouped
3. **web_search** — 20 est. tokens, section: Ungrouped
4. **context** — 16 est. tokens, section: Ungrouped
5. **ocr_document** — 14 est. tokens, section: Ungrouped
6. **recall** — 14 est. tokens, section: Ungrouped
7. **context_len** — 14 est. tokens, section: Ungrouped
8. **search_arxiv** — 13 est. tokens, section: Ungrouped
9. **run_shell** — 13 est. tokens, section: Ungrouped
10. **route_advice** — 13 est. tokens, section: Ungrouped
11. **list_dir** — 12 est. tokens, section: Ungrouped
12. **run_tests** — 12 est. tokens, section: Ungrouped

## Instruction Token Ratio

- Tool definitions / total system prompt: **15.0%**
- Rules / total system prompt: **51.0%**
- Role overlays / total system prompt: **34.0%**

## Runtime AP-16 Prompt Scaffold

AP-16 runtime accounting now follows the active PromptBuilder path instead of charging every prompt-library markdown file to each request.

| Component | Char-Proxy Tokens |
|-----------|-------------------|
| root_lm_system | 143 |
| tools | 612 |
| rules | 1739 |
| role:frontdoor | 402 |
| role:worker_general | 125 |
| **frontdoor+worker total** | **3021** |
| **frontdoor+worker+architect total** | **3128** |

Recent nonzero AP-16 journal rows:

| Trial | Species | Instruction Tokens | Observed Scaffold | Instruction Ratio | Quality | Speed t/s | Reliability | Observed Routes |
|-------|---------|--------------------|-------------------|-------------------|---------|-----------|-------------|-----------------|
| #1007 | numeric_swarm | 10748 | 3322 | 97.4% | 2.100 | 55.0 | 0.98 | architect_general, coder_escalation, frontdoor, ingest_long_context, worker_general, worker_vision |
| #1008 | numeric_swarm | 10748 | 3185 | 97.4% | 2.100 | 49.5 | 0.98 | coder_escalation, frontdoor, worker_general, worker_vision |
| #1010 | numeric_swarm | 3188 | 3322 | 91.8% | 2.100 | 59.4 | 0.98 | architect_general, coder_escalation, frontdoor, ingest_long_context, worker_general, worker_vision |
| #1011 | numeric_swarm | 3295 | 3292 | 92.1% | 2.040 | 49.5 | 0.96 | architect_general, coder_escalation, frontdoor, worker_general, worker_vision |
| #1012 | numeric_swarm | 3188 | 3185 | 91.8% | 2.100 | 56.0 | 0.98 | coder_escalation, frontdoor, worker_general, worker_vision |

Interpretation: the static P3b definition compression remains real, but AP-16 frontier rows should use the active scaffold. Short EvalTower prompts still show a high instruction ratio, so P3d remains the quality gate before further prompt definition changes.

