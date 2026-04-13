# epyc-orchestrator

## Related Repositories

This repo is part of a multi-repo project:
- **epyc-root** — governance, hooks, agents, handoffs ([pestopoppa/epyc-root](https://github.com/pestopoppa/epyc-root))
- **epyc-inference-research** — benchmarks, research, full model registry ([pestopoppa/epyc-inference-research](https://github.com/pestopoppa/epyc-inference-research))
- **epyc-llama** — custom llama.cpp fork ([pestopoppa/llama.cpp](https://github.com/pestopoppa/llama.cpp))

Benchmarks, research docs, agent files, and handoffs live in their respective repos — not here.

> **Path history note**: Documentation and handoffs dated before 2026-02-25 reference
> `/mnt/raid0/llm/claude` (the pre-split monorepo). Those paths are no longer valid.
> This repo's code was extracted from that monorepo.

## Architecture

```
Request:  FastAPI(:8000) → AppState → ChatPipeline → REPLExecutor → run_task() → [graph nodes]
Graph:    orchestration_graph (pydantic-graph) → 7 node classes → LLMPrimitives → [model servers]
Memory:   EpisodicStore(SQLite) → FAISSStore → ParallelEmbedder → BGE pool
Skills:   SkillBank(skills.db+FAISS) → SkillRetriever → prompt injection | OutcomeTracker
Tools:    REPLExecutor → ToolRegistry(allowed_callers, chain telemetry) → PluginLoader
Prompts:  resolve_prompt(name) → orchestration/prompts/{name}.md (hot-swap) → fallback constant
Sessions: /chat(session_id) → latest Checkpoint restore → cross-request variable continuity
```

## Code Style

- Python 3.11+, formatted with `ruff`
- All filesystem paths via `get_config()` from `src/config/` — never hardcode paths
- Lazy imports for heavy dependencies (faiss, sklearn, etc.)
- Configuration hierarchy: env vars → .env → pydantic-settings defaults
- Registry access via `src/registry_loader.py` — typed helpers for roles, timeouts, escalation

## Key Patterns

- **TaskIR**: Intermediate representation emitted by front door, validated against `orchestration/task_ir.schema.json`
- **Escalation**: Graph nodes use EscalationPolicy rules. 1st fail → retry, 2nd → escalate tier, 3rd → architect
- **Mock mode**: `ORCHESTRATOR_MOCK_MODE=1` skips model init, returns synthetic responses. All tests use mock mode.
- **Registry modes**: `ORCHESTRATOR_REGISTRY_MODE=lean` for minimal config, `full` for complete model paths

## Testing

```bash
pytest tests/ -n 8    # Safe parallel default
pytest tests/ -v      # Verbose single-threaded
make gates            # Full verification pipeline
```

Do NOT use `pytest -n auto` — it spawns too many workers and exhausts memory.

## Verification

Run `make gates` after any changes. This runs:
1. Schema validation (`validate_ir.py`)
2. Shell lint (`shellcheck`)
3. Format check (`ruff format --check`)
4. Lint (`ruff check`)

## Directory Layout

| Directory | Purpose |
|-----------|---------|
| `src/` | Core application code |
| `src/api/` | FastAPI routes, pipeline stages |
| `src/config/` | pydantic-settings configuration |
| `src/orchestration_graph/` | pydantic-graph node definitions |
| `src/repl_environment/` | REPL executor, tool plugins |
| `src/vision/` | Vision pipeline |
| `orchestration/` | Runtime config (registry, prompts, tools, schemas) |
| `tests/` | Test suite (mock mode) |
| `scripts/server/` | Server management |

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **epyc-orchestrator** (19840 symbols, 55303 relationships, 300 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## When Debugging

1. `gitnexus_query({query: "<error or symptom>"})` — find execution flows related to the issue
2. `gitnexus_context({name: "<suspect function>"})` — see all callers, callees, and process participation
3. `READ gitnexus://repo/epyc-orchestrator/process/{processName}` — trace the full execution flow step by step
4. For regressions: `gitnexus_detect_changes({scope: "compare", base_ref: "main"})` — see what your branch changed

## When Refactoring

- **Renaming**: MUST use `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` first. Review the preview — graph edits are safe, text_search edits need manual review. Then run with `dry_run: false`.
- **Extracting/Splitting**: MUST run `gitnexus_context({name: "target"})` to see all incoming/outgoing refs, then `gitnexus_impact({target: "target", direction: "upstream"})` to find all external callers before moving code.
- After any refactor: run `gitnexus_detect_changes({scope: "all"})` to verify only expected files changed.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Tools Quick Reference

| Tool | When to use | Command |
|------|-------------|---------|
| `query` | Find code by concept | `gitnexus_query({query: "auth validation"})` |
| `context` | 360-degree view of one symbol | `gitnexus_context({name: "validateUser"})` |
| `impact` | Blast radius before editing | `gitnexus_impact({target: "X", direction: "upstream"})` |
| `detect_changes` | Pre-commit scope check | `gitnexus_detect_changes({scope: "staged"})` |
| `rename` | Safe multi-file rename | `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` |
| `cypher` | Custom graph queries | `gitnexus_cypher({query: "MATCH ..."})` |

## Impact Risk Levels

| Depth | Meaning | Action |
|-------|---------|--------|
| d=1 | WILL BREAK — direct callers/importers | MUST update these |
| d=2 | LIKELY AFFECTED — indirect deps | Should test |
| d=3 | MAY NEED TESTING — transitive | Test if critical path |

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/epyc-orchestrator/context` | Codebase overview, check index freshness |
| `gitnexus://repo/epyc-orchestrator/clusters` | All functional areas |
| `gitnexus://repo/epyc-orchestrator/processes` | All execution flows |
| `gitnexus://repo/epyc-orchestrator/process/{name}` | Step-by-step execution trace |

## Self-Check Before Finishing

Before completing any code modification task, verify:
1. `gitnexus_impact` was run for all modified symbols
2. No HIGH/CRITICAL risk warnings were ignored
3. `gitnexus_detect_changes()` confirms changes match expected scope
4. All d=1 (WILL BREAK) dependents were updated

## Keeping the Index Fresh

After committing code changes, the GitNexus index becomes stale. Re-run analyze to update it:

```bash
npx gitnexus analyze
```

If the index previously included embeddings, preserve them by adding `--embeddings`:

```bash
npx gitnexus analyze --embeddings
```

To check whether embeddings exist, inspect `.gitnexus/meta.json` — the `stats.embeddings` field shows the count (0 means no embeddings). **Running analyze without `--embeddings` will delete any previously generated embeddings.**

> Claude Code users: A PostToolUse hook handles this automatically after `git commit` and `git merge`.

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
