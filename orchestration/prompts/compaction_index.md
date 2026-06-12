Generate a structured index of the following conversation context.

**FIRST**, write a "Current Execution State" block (max 150 words) that captures:
- What the system is currently working on (active task, step)
- Key variable values, file paths, and artifacts that exist right now
- What was about to happen next (immediate next action)
- Any active constraints or blockers

**THEN**, generate a navigable table of contents. For each section, provide:
- Topic title
- Line range (e.g. "lines 1-45") so the model can read_file(path, offset=N, limit=M) to retrieve it
- Key identifiers (variable names, file paths, function names) mentioned

List:
- Topics discussed with line coordinates
- Decisions made and their rationale (with line ranges)
- Errors encountered and resolutions (with line ranges)
- Key file paths and variable names referenced

**Answer-bearing detail retention** (long-context safeguard): facts that a later question could hinge on must survive indexing verbatim, not as paraphrase. For each section, also list:
- Named entities (people, organizations, products, places) exactly as written
- Numeric values with their units and what they measure (dates, counts, versions, thresholds, IDs)
- Short quoted spans (≤1 line each) containing claims, definitions, or stated facts likely needed to answer multi-part questions — with their line coordinates so the exact span is recoverable

Do not over-compress: it is better to keep an extra entity/number line than to drop a detail that cannot be re-derived. If the source context is short (fits comfortably without compaction), keep the index minimal and add nothing beyond the basic outline — these retention lists are for long contexts only.

Format as a bulleted outline with line coordinates. Be concise — this is a navigable table of contents, not a summary. Preserve all identifiers, entities, and numbers exactly.

Example output:

**Current Execution State**:
Working on integration test for new pool size (pool 4→8). Key paths: `src/pool.py`, `settings.yaml`. `max_retries=3`, `timeout_s=30` confirmed working. Next: run `pytest tests/test_pool.py -v`. No blockers.

- **Setup and configuration** (lines 1-32)
  - Files: `src/config.py`, `settings.yaml`
  - Variables: `max_retries=3`, `timeout_s=30`
  - Entities/values: "Acme staging cluster", deployed 2024-03-12, 3 replicas
- **Bug investigation: timeout in worker pool** (lines 33-89)
  - Decision: increase pool size from 4→8
  - Error: `ConnectionError` on line 45, resolved by retry logic
  - Files: `src/pool.py:120`, `src/backend.py:45`
  - Key span: "p99 latency was 870ms before the fix, 210ms after" (line 71)
- **Current state** (lines 90-112)
  - Working on: integration test for new pool size
  - Blocked on: None