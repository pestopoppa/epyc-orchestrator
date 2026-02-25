# epyc-orchestrator

## Related Repositories

This repo is part of a multi-repo project:
- **epyc-root** — governance, hooks, agents, handoffs ([pestopoppa/epyc-root](https://github.com/pestopoppa/epyc-root))
- **epyc-inference-research** — benchmarks, research, full model registry ([pestopoppa/epyc-inference-research](https://github.com/pestopoppa/epyc-inference-research))
- **epyc-llama** — custom llama.cpp fork ([pestopoppa/llama.cpp](https://github.com/pestopoppa/llama.cpp))

Benchmarks, research docs, agent files, and handoffs live in their respective repos — not here.

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
