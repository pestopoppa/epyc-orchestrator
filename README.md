# epyc-orchestrator

Hierarchical multi-model orchestration for local LLM inference. Routes tasks across multiple model tiers with automatic escalation, speculative decoding, and episodic memory.

## What It Does

- **Multi-tier routing**: Assigns tasks to the right model — fast workers for simple queries, architects for complex reasoning
- **Automatic escalation**: If a model fails or times out, the task escalates to a more capable tier
- **Speculative decoding**: Draft models accelerate generation (2-12x speedup depending on task)
- **Episodic memory**: FAISS-backed session memory with skill tracking and evolution
- **Tool execution**: Sandboxed REPL with plugin system for code execution, web fetch, and more
- **Vision pipeline**: Multi-modal support with OCR and image understanding
- **MCP server**: Model Context Protocol integration for external tool providers

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/pestopoppa/epyc-orchestrator.git
cd epyc-orchestrator
pip install -e ".[dev]"

# 2. Configure
cp .env.example .env
# Edit .env — set ORCHESTRATOR_PATHS_LLM_ROOT to your models directory

# 3. Run (mock mode — no models needed)
ORCHESTRATOR_MOCK_MODE=1 uvicorn src.api:app --host 0.0.0.0 --port 8000

# 4. Run (production — requires llama.cpp model servers)
python scripts/server/orchestrator_stack.py start --dev
```

## Architecture

```
Request → FastAPI(:8000) → ChatPipeline → REPLExecutor → OrchestrationGraph

OrchestrationGraph (pydantic-graph):
  ├── Tier A: Front door (interactive, fastest)
  ├── Tier B: Specialists (coder, architect, ingest)
  ├── Tier C: Workers (explore, math, vision, summarize)
  └── Tier D: Draft models + embedders (co-loaded)

Memory:  EpisodicStore(SQLite) → FAISSStore → ParallelEmbedder
Skills:  SkillBank(SQLite+FAISS) → SkillRetriever → OutcomeTracker
Tools:   REPLExecutor → ToolRegistry → PluginLoader
```

## Documentation

- **[Architecture Reference](docs/ARCHITECTURE.md)** — living technical reference, module responsibilities, request flow
- **[Chapter Index](docs/chapters/INDEX.md)** — 17 chapters covering runtime, REPL, MemRL, escalation, tools, SkillBank, and more
- **[Setup Guide](docs/SETUP.md)** — installation and configuration

## Configuration

All paths are configurable via environment variables (see `.env.example`). The system uses `pydantic-settings` for type-safe configuration with hierarchical defaults.

### Model Registry

Edit `orchestration/model_registry.yaml` to configure:
- Model roles and tiers
- Acceleration settings (speculative decoding, MoE expert reduction)
- Escalation chains and routing hints
- Timeout policies

### Registry Modes

- **`full`** (default): Complete registry with model paths, performance data, memory profiles
- **`lean`**: Minimal registry with only routing, timeouts, and acceleration config — suitable for deployments using external API backends

Set via `ORCHESTRATOR_REGISTRY_MODE=lean`.

## API

### Chat Endpoint

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, world!", "session_id": "test"}'
```

### Streaming

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain quicksort", "session_id": "test", "stream": true}'
```

## Development

```bash
make gates          # Full verification (schema, shellcheck, format, lint)
pytest tests/ -n 8  # Run tests (parallel)
ruff check src/     # Lint only
ruff format src/    # Format only
```

## Project Structure

```
src/                    # Core application code
  api/                  # FastAPI routes and pipeline stages
  config/               # Configuration models (pydantic-settings)
  orchestration_graph/  # pydantic-graph node definitions
  repl_environment/     # REPL executor and tool plugins
  vision/               # Vision pipeline (VL models, OCR)
  mcp_server.py         # MCP integration
  registry_loader.py    # Model registry access
  dispatcher.py         # TaskIR → execution plan mapping
orchestration/          # Runtime configuration
  model_registry.yaml   # Model roles, acceleration, routing
  prompts/              # Hot-swappable prompt templates
  tools/                # Tool definitions
tests/                  # Test suite
scripts/server/         # Server management (orchestrator_stack.py)
```

## License

MIT — see [LICENSE](LICENSE).
