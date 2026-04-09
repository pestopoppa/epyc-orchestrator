# REPL Pattern Analysis - 2026-04-09

## Data Summary

**autopilot.log** (/mnt/raid0/llm/epyc-orchestrator/logs/autopilot.log)
- Total REPL sessions parsed: 541
- Sessions with tool usage: 302 (55.8%)
- Sessions without tools: 239 (44.2%)

**seeding_diagnostics.jsonl** (/mnt/raid0/llm/epyc-orchestrator/logs/seeding_diagnostics.jsonl)
- Total records: 3187
- REPL-mode records: 807
- REPL with tools: 117 (14.5%)
- REPL without tools: 690 (85.5%)
- `repl_no_tools` anomaly flagged: 690

## Tool Usage Frequency

From autopilot.log per-call detail lines:

| Tool | Total Calls | % of All Tool Calls | Sessions With Tool |
|------|-------------|--------------------|--------------------|
| web_search | 6190 | 94.8% | 302 |
| search_wikipedia | 342 | 5.2% | 94 |

## Tool Latency Statistics

| Tool | Calls | Avg ms | Min ms | Max ms | P50 ms |
|------|-------|--------|--------|--------|--------|
| search_wikipedia | 342 | 44227 | 3141 | 138727 | 16757 |
| web_search | 6190 | 1593 | 1346 | 11180 | 1555 |

## Tool Co-occurrence (per session)

Pairs of different tool types appearing in the same REPL session:

| Tool A | Tool B | Sessions Together |
|--------|--------|-------------------|
| search_wikipedia | web_search | 94 |

## Multi-Tool Patterns (Bigrams)

Consecutive tool call pairs within a single REPL session:

| Pattern | Count | Est. Turn Savings | Combined Op Candidate |
|---------|-------|-------------------|-----------------------|
| web_search -> web_search | 5727 | ~1 | batch_web_search |
| web_search -> search_wikipedia | 171 | ~1 | web_search_then_search_wikipedia |
| search_wikipedia -> search_wikipedia | 171 | ~1 | batch_search_wikipedia |
| search_wikipedia -> web_search | 161 | ~1 | search_wikipedia_then_web_search |

## Multi-Tool Patterns (Trigrams)

Consecutive 3-tool sequences:

| Pattern | Count | Est. Turn Savings |
|---------|-------|-------------------|
| web_search -> web_search -> web_search | 5271 | ~2 |
| web_search -> web_search -> search_wikipedia | 171 | ~2 |
| web_search -> search_wikipedia -> search_wikipedia | 171 | ~2 |
| search_wikipedia -> search_wikipedia -> web_search | 161 | ~2 |
| search_wikipedia -> web_search -> web_search | 154 | ~2 |

## Outcome by Tool Count

| Tools Available | PASS | FAIL | INFRA | Total | Pass Rate |
|----------------|------|------|-------|-------|-----------|
| 0 | 56 | 33 | 150 | 239 | 23.4% |
| 2 | 2 | 0 | 0 | 2 | 100.0% |
| 3 | 6 | 6 | 0 | 12 | 50.0% |
| 4 | 3 | 2 | 0 | 5 | 60.0% |
| 5 | 2 | 1 | 0 | 3 | 66.7% |
| 6 | 2 | 1 | 0 | 3 | 66.7% |
| 7 | 3 | 2 | 0 | 5 | 60.0% |
| 8 | 3 | 2 | 0 | 5 | 60.0% |
| 9 | 3 | 5 | 0 | 8 | 37.5% |
| 10+ | 166 | 93 | 0 | 259 | 64.1% |

## Zero-Tool Session Analysis

Suite distribution for REPL sessions that used **no tools** (from seeding_diagnostics.jsonl):

| Suite | Count | Likely Helpful Tools |
|-------|-------|---------------------|
| usaco | 55 | peek, code_search |
| gpqa | 52 | web_search, search_wikipedia |
| livecodebench | 51 | _(pure reasoning)_ |
| debugbench | 51 | _(pure reasoning)_ |
| math | 50 | _(pure reasoning)_ |
| coder | 49 | peek, code_search, grep, list_dir |
| agentic | 44 | peek, list_dir, code_search, grep |
| long_context | 44 | peek, grep |
| mode_advantage | 43 | _(pure reasoning)_ |
| thinking | 43 | _(pure reasoning)_ |
| vl | 41 | _(pure reasoning)_ |
| hotpotqa | 40 | web_search, search_wikipedia |
| mode_advantage_hard | 36 | web_search, code_search |
| general | 35 | web_search |
| instruction_precision | 32 | _(pure reasoning)_ |
| simpleqa | 24 | web_search, search_wikipedia |

## Mode Pass Rates

| Mode | Passed | Total | Rate |
|------|--------|-------|------|
| delegated | 536 | 1532 | 35.0% |
| direct | 340 | 848 | 40.1% |
| repl | 369 | 807 | 45.7% |

## Recommended Combined Operations

1. **batch_web_search** (pattern: `web_search x N (repeated)`, count: 5727): Repeated web_search calls (5727x) could be batched into a single parallel invocation, saving ~1 turn each.
2. **web_search_then_search_wikipedia** (pattern: `web_search -> search_wikipedia`, count: 171): Sequential web_search -> search_wikipedia (171x) could be combined into a single operation that performs both steps.
3. **batch_search_wikipedia** (pattern: `search_wikipedia x N (repeated)`, count: 171): Repeated search_wikipedia calls (171x) could be batched into a single parallel invocation, saving ~1 turn each.
4. **search_wikipedia_then_web_search** (pattern: `search_wikipedia -> web_search`, count: 161): Sequential search_wikipedia -> web_search (161x) could be combined into a single operation that performs both steps.

## Instrumentation Gaps

The following gaps limit the depth of this analysis:

- Only 2 tool type(s) observed in logs: search_wikipedia, web_search. Tools like peek, grep, list_dir, code_search are defined in the REPL but never appear in logged sessions. Either they are not yet enabled in autopilot seeding, or their usage is not logged.
- `inference_tap.log` does not exist. Handoff documents reference it as a data source for raw inference traces, but it has not been created yet.
- REPL diagnostic `loops` field is always 0 for SELF:repl, even for sessions with many tool calls. The REPL loop counter may not be instrumented correctly.

## Recommendations for Additional Instrumentation

To enable deeper multi-tool pattern analysis, the following instrumentation should be added to the REPL runner:

1. **Log individual tool call names in seeding_diagnostics.jsonl** - The `tools_called` field is currently always empty even when `tools_used > 0`. Populate it with the ordered list of tool names.
2. **Log tool call ordering per REPL turn** - Currently the autopilot log emits tool names but not which REPL turn/loop iteration they belong to. Adding a turn index would enable intra-session sequencing.
3. **Log tool call arguments (hashed/summarized)** - To identify patterns like 'list_dir then peek same file', argument context is needed. A hash or truncated summary would suffice.
4. **Emit a REPL session boundary marker** - Sessions are currently inferred from `SELF:repl ->` result lines. An explicit session-start marker would simplify parsing.
5. **Track tool call dependencies** - Whether a tool's input was derived from a previous tool's output (chained vs independent calls).
