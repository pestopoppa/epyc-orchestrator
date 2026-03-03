# Orchestrator Documentation — Chapter Index

Documentation for the epyc-orchestrator system: routing, memory, tools, and production infrastructure.

## Chapters

| # | Title | Key Topics |
|---|-------|------------|
| [01](01-runtime-environment.md) | Runtime Environment & Configuration | Python env, feature flags, pydantic-settings, session init |
| [02](02-orchestration-architecture.md) | Orchestration Architecture | TaskIR, agent tiers, pydantic-graph, escalation loop |
| [03](03-repl-environment.md) | REPL Environment & Sandboxing | Code execution, sandbox safety, NextPLAID retrieval |
| [04](04-production-server-stack.md) | Production Server Stack | orchestrator_stack.py, port topology, health checks |
| [05](05-data-processing-pipelines.md) | Data Processing Pipelines | Document ingestion, CLIP embeddings, ChromaDB |
| [06](06-toon-encoding.md) | TOON Encoding | Compact structured output format |
| [07](07-memrl-system.md) | MemRL System | Episodic memory, QScorer, TD-learning, FAISS+SQLite |
| [08](08-graph-reasoning.md) | Graph-Based Reasoning | Hypothesis graphs, failure graphs, causal chains |
| [09](09-memory-seeding.md) | Memory Seeding & Bootstrap | 3-way seeding, mode-advantage tasks, cold-start |
| [10](10-escalation-and-routing.md) | Escalation, Routing & Delegation | Cost-aware routing, proactive delegation, failure recovery |
| [11](11-procedure-registry.md) | Procedure Registry | Declarative procedure definitions, step execution |
| [12](12-session-persistence.md) | Session Persistence | Turn history, checkpoint/restore, session logs |
| [13](13-tool-registry.md) | Tool Registry & Permissions | Cascading policy, role-based access, tool definitions |
| [14](14-security-and-monitoring.md) | Security & Monitoring | EARLY_ABORT, sandbox enforcement, audit logging |
| [15](15-skillbank-experience-distillation.md) | SkillBank & Experience Distillation | Skill extraction, recursive evolution, OutcomeTracker |
| [16](16-calibration-and-risk-control.md) | Calibration & Risk Control | Confidence calibration, risk-aware routing |
| [17](17-programmatic-tool-chaining.md) | Programmatic Tool Chaining | Multi-step tool pipelines, chain composition |

## Reading Paths

**Getting Started** — System setup through production deployment:
01 Runtime → 02 Architecture → 03 REPL → 04 Server Stack

**Intelligence** — Memory and learning subsystems:
07 MemRL → 08 Graphs → 09 Seeding → 15 SkillBank → 16 Calibration

**Routing** — Request lifecycle and task management:
10 Escalation → 11 Procedures → 12 Sessions → 17 Tool Chaining

**Security** — Permissions, sandboxing, and monitoring:
13 Tools → 14 Security → 06 TOON → 05 Pipelines

## Cross-Repository Documentation

- **Inference optimization** (speculative decoding, MoE, radix attention): epyc-inference-research docs
- **Benchmarking and evaluation** (suite construction, reward design, debugger): epyc-inference-research docs
- **llama.cpp toolchain** (worktrees, production branch, build flags): epyc-llama docs
- **Hardware and storage** (EPYC 9655 platform, RAID0 safety): epyc-root docs
