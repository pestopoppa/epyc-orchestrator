# epyc-orchestrator

Hierarchical multi-model orchestration for local LLM inference on AMD EPYC. Routes tasks across 29 model servers + 3 infrastructure services with automatic escalation, speculative decoding, KV cache compression, and autonomous prompt optimization.

## What It Does

- **Multi-tier routing**: Routes tasks to the optimal model — fast workers for simple queries, architects for complex reasoning
- **Automatic escalation**: Failed or timed-out tasks escalate to more capable tiers
- **Speculative decoding**: Draft models accelerate generation (2-4x speedup depending on model)
- **AM KV compaction**: Attention-score-driven KV cache compression via `POST /slots/{id}?action=compact` — 5x compression with zero quality degradation
- **EA KV compression**: Expected Attention scoring for importance-weighted cache eviction
- **Web search**: SearXNG metasearch (local Docker) with DuckDuckGo + Brave fallback
- **ColBERT retrieval**: Multi-vector code and document retrieval via NextPLAID
- **Episodic memory**: FAISS-backed session memory with skill tracking
- **Tool execution**: Sandboxed REPL with code execution, web fetch, and plugins
- **Vision pipeline**: Multi-modal support with OCR and image understanding (7B worker + 30B escalation)
- **AutoPilot**: Autonomous prompt optimization with safety gates and Pareto archive

## Production Stack

All servers run on a single AMD EPYC 9655 (96C/192T, 1.13TB DDR5) via llama.cpp `production-consolidated-v3` with KV quantization (q4_0 K / f16 V), flash attention, and Hadamard auto-rotation.

### LLM Servers (29 instances)

Each HOT role deploys as 1 full-speed instance (96 threads) + 4 quarter instances (48 threads each). The concurrency router sends single sessions to the full-speed instance for maximum per-request throughput, and distributes concurrent sessions across quarters.

| Role | Model | Instances | Speed (per inst) | Context | Acceleration |
|------|-------|:---------:|:-----------------:|:-------:|:------------|
| frontdoor | Qwen3.5-35B-A3B Q4_K_M | 1+4 | 12.7 t/s | 32K | moe6 (expert reduction) |
| coder | Qwen2.5-Coder-32B Q4_K_M | 1+4 | 10.8 t/s | 32K | spec decode (dm=32) + tree + lookup |
| worker | Qwen3-Coder-30B-A3B Q4_K_M | 1+4 | 39 t/s | 8K | spec decode (dm=8) |
| architect_general | Qwen3.5-122B-A10B Q4_K_M | 2 | 4.3 t/s | 16K | spec decode (dm=24), 2×NUMA |
| architect_coding | REAP-246B-A35B Q4_K_M | 2 | 8.0 t/s | 16K | spec decode (dm=32), 2×NUMA |
| ingest | Qwen3-Next-80B-A3B | 1 | ~12 t/s | 32K | — |
| worker_vision | Qwen2.5-VL-7B Q4_K_M | 1 | — | 8K | — |
| vision_escalation | Qwen3-VL-30B-A3B Q4_K_M | 1 | — | 16K | — |
| worker_fast | Qwen2.5-Coder-1.5B Q4_K_M | 1 | ~100 t/s | 8K | 4 parallel slots |
| embedder | BGE-large-en-v1.5 f16 | 6 | — | 512 | — |

**Memory footprint**: ~515 GB for HOT servers (46% of 1130 GB). Architect servers are WARM (started on demand).

### Infrastructure Services (3 Docker containers)

| Service | Port | Image | Purpose |
|---------|:----:|-------|---------|
| nextplaid-code | 8088 | next-plaid:cpu-1.0.4 | ColBERT multi-vector code retrieval (LateOn-Code) |
| nextplaid-docs | 8089 | next-plaid:cpu-1.0.4 | ColBERT multi-vector doc retrieval (GTE-ModernColBERT) |
| searxng | 8090 | searxng/searxng:latest | Metasearch aggregator (JSON API for web_search) |

## AutoPilot: Continuous Optimization

The orchestrator includes an autonomous optimization loop (AutoPilot) that continuously improves prompts, routing, and model configurations through controlled experiments with safety gates.

**192 trials completed** — quality stable at 2.1/3.0, speed at 64 t/s, 80% reliability. AutoPilot uses 4 species (Seeder, NumericSwarm, PromptForge, StructuralLab) to explore prompt mutations, hyperparameter tuning, and feature flag combinations.

### Diagnostic Plots

| | |
|:---:|:---:|
| ![Objectives Overview](docs/autopilot/objectives_2x2.png) | ![Pareto Frontier](docs/autopilot/pareto_frontier_2d.png) |
| **Objectives Overview** — 4-objective optimization progress | **Pareto Frontier** — quality vs speed tradeoff |
| ![Hypervolume Trend](docs/autopilot/hypervolume_trend.png) | ![Species Effectiveness](docs/autopilot/species_effectiveness.png) |
| **Hypervolume Trend** — optimization progress over trials | **Species Effectiveness** — which mutation strategies produce gains |
| ![Per-Suite Quality](docs/autopilot/per_suite_quality.png) | ![Trial Timeline](docs/autopilot/trial_timeline.png) |
| **Per-Suite Quality** — breakdown by benchmark | **Trial Timeline** — chronological trial outcomes |
| ![Memory Convergence](docs/autopilot/memory_convergence.png) | |
| **Memory Convergence** — episodic memory utilization | |

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/pestopoppa/epyc-orchestrator.git
cd epyc-orchestrator
pip install -e ".[dev]"

# 2. Launch stack (production)
python scripts/server/orchestrator_stack.py start

# 3. Pre-flight audit
python scripts/autopilot/preflight_audit.py

# 4. Launch AutoPilot
python scripts/autopilot/autopilot.py start --tui

# 5. Monitor (separate terminal)
python scripts/autopilot/autopilot.py monitor
```

## Architecture

```
Request → FastAPI(:8000) → ChatPipeline → Mode Selection
                                            ├── Direct → LLM call → Response
                                            ├── REPL → Tool loop → Response
                                            └── Delegated → Architect plan → Worker execution

Model Stack (29 servers, 2 NUMA nodes):
  Tier A: Front door (5× Qwen3.5-35B, interactive)
  Tier B: Specialists (5× Coder-32B, 2× Architect-122B, 2× REAP-246B)
  Tier C: Workers (5× 30B-A3B, 1× 80B ingest, 2× VL, 1× 1.5B fast)
  Tier D: Embedders (6× BGE-large)

Infrastructure:
  ColBERT retrieval (2× NextPLAID), SearXNG metasearch

AutoPilot: Controller → Species (Seeder/NumericSwarm/PromptForge/StructuralLab)
           → EvalTower (T0 sentinel → T1 deep → T2 full)
           → SafetyGate → ParetoArchive → Journal
```

## Eval Suites

30+ benchmark suites with automated scoring:

| Suite | Questions | Scoring | Status |
|-------|:---------:|---------|--------|
| math (GSM8K) | 1,819 | exact_match | scoring |
| coder (MBPP) | 664 | substring | scoring |
| general (MMLU) | 14,042 | multiple_choice | scoring |
| gpqa | 448 | multiple_choice | scoring |
| hotpotqa | 7,405 | f1 | scoring |
| usaco | 520 | code_execution | scoring |
| web_research | 50 | f1 | scoring |
| physics (PHYBench) | 100 | llm_judge | scoring |
| vl (OCRBench) | 2,575 | exact_match | scoring |
| + 20 more | 30K+ | various | scoring |

## Documentation

- **[Architecture Reference](docs/ARCHITECTURE.md)** — module responsibilities, request flow
- **[Chapter Index](docs/chapters/INDEX.md)** — 17 chapters: runtime, REPL, MemRL, escalation, tools, SkillBank
- **[AutoPilot Program](scripts/autopilot/program.md)** — optimization strategy and constraints

## Development

```bash
pytest tests/ -n 8       # Run tests (parallel)
ruff check src/           # Lint
python scripts/autopilot/preflight_audit.py  # 9-check diagnostic
```

## License

MIT — see [LICENSE](LICENSE).
