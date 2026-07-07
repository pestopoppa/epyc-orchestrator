# REPL Pattern Analysis - 2026-07-06

## Data Summary

**autopilot.log** (/mnt/raid0/llm/epyc-orchestrator/logs/autopilot.log)
- Total REPL sessions parsed: 0
- Sessions with tool usage: 0 (0.0%)
- Sessions without tools: 0 (0.0%)

**seeding_diagnostics.jsonl** (/mnt/raid0/llm/epyc-orchestrator/logs/seeding_diagnostics.jsonl)
- Total records: 3187
- REPL-mode records: 807
- REPL with tools: 117 (14.5%)
- REPL without tools: 690 (85.5%)
- REPL records with >=2 tools_called: 30 (3.7%)
- REPL records with explicit read-only tool chains: 30 (100.0% of multi-tool REPL records)
- REPL records with parallel_tools_used=True: 0 (0.0% of multi-tool REPL records)
- `repl_no_tools` anomaly flagged: 690

## Tool Usage Frequency

From autopilot.log per-call detail lines:

| Tool | Total Calls | % of All Tool Calls | Sessions With Tool |
|------|-------------|--------------------|--------------------|

## Multi-Tool Patterns (Bigrams)

Consecutive tool call pairs within a single REPL session:

| Pattern | Count | Est. Turn Savings | Combined Op Candidate |
|---------|-------|-------------------|-----------------------|
| _(no bigrams found)_ | - | - | - |

## Multi-Tool Patterns (Trigrams)

Consecutive 3-tool sequences:

| Pattern | Count | Est. Turn Savings |
|---------|-------|-------------------|
| _(no trigrams found)_ | - | - |

## Tool Chain Candidates

Exact tool chains observed in diagnostics and autopilot bigrams:

| Chain | Count | Est. Turn Savings | Sources |
|-------|-------|-------------------|---------|
| _(no chain candidates found)_ | - | - | - |

## Outcome by Tool Count

| Tools Available | PASS | FAIL | INFRA | Total | Pass Rate |
|----------------|------|------|-------|-------|-----------|

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

No multi-tool patterns with sufficient frequency found. See Instrumentation Gaps below.

## Instrumentation Gaps

The following gaps limit the depth of this analysis:

- `inference_tap.log` does not exist. Handoff documents reference it as a data source for raw inference traces, but it has not been created yet.

## Recommendations for Additional Instrumentation

To enable deeper multi-tool pattern analysis, the following instrumentation should be added to the REPL runner:

1. **Log individual tool call names in seeding_diagnostics.jsonl** - The `tools_called` field is currently always empty even when `tools_used > 0`. Populate it with the ordered list of tool names.
2. **Log tool call ordering per REPL turn** - Currently the autopilot log emits tool names but not which REPL turn/loop iteration they belong to. Adding a turn index would enable intra-session sequencing.
3. **Log tool call arguments (hashed/summarized)** - To identify patterns like 'list_dir then peek same file', argument context is needed. A hash or truncated summary would suffice.
4. **Emit a REPL session boundary marker** - Sessions are currently inferred from `SELF:repl ->` result lines. An explicit session-start marker would simplify parsing.
5. **Track tool call dependencies** - Whether a tool's input was derived from a previous tool's output (chained vs independent calls).
