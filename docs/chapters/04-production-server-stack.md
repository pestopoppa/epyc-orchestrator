# Chapter 04: Production Server Stack

## Introduction

The production server stack runs llama-server instances plus a small set of auxiliary services, organized into HOT/WARM/COLD memory tiers. The HOT tier (~600 GB post-2026-05-09 consolidation; ~53% of 1.13 TB RAM) stays resident for immediate availability. Roles backed by the same GGUF (frontdoor and coder_escalation since 2026-05-06) share a single server process with separate admission slots, eliminating duplicate mlock. Launch mode (`--numa-mode full|quarter|both`) controls whether each role runs as a single 96-thread process or as four 48-thread NUMA-quarter instances.

Managed by `orchestrator_stack.py`, the system provides graceful start/stop, health monitoring, and granular component reload without full restart.

## Server Topology

The stack spans three tiers of servers, each mapped to a port range. The HOT tier holds the models you interact with most — the frontdoor, coders, architects, and embedders — all pinned in RAM so there is zero cold-start penalty. Auxiliary services handle retrieval and OCR on their own ports.

The current role/port/model summary is generated from `orchestration/derived/stack_priors.yaml` at [`../generated/current_stack_summary.md`](../generated/current_stack_summary.md) and checked by `scripts/registry/stack_change_pipeline.py`. Treat hand-written tables in this chapter as explanatory context, not source truth.

<details>
<summary>Server port assignments and tier breakdown</summary>

### HOT Tier (Always Resident) — NUMA-Optimized (2026-05-09 consolidation)

| Port(s) | Roles | Model | NUMA | Acceleration | Speed | RAM |
|---------|-------|-------|------|--------------|-------|-----|
| 8070 (full-mode) or 8080,8180,8280,8380 (quarter-mode) | frontdoor + coder_escalation (shared GGUF, separate slots) | Qwen3.6-35B-A3B Q8 (swapped 2026-05-04 from Qwen3.5-35B Q4_K_M) | 1×96t full or 4×48t quarters | None (Q8 MoE baseline); `enable_thinking=false` | ~12.7 t/s per instance (full); ~24.3 t/s aggregate (quarter) | 37GB shared mmap |
| 8072 (full-mode) or 8082,8182,8282,8382 (quarter-mode) | worker_general (explore, math, summarize aliases) | gemma-4-26B-A4B-it Q4_K_M + MTP drafter (swapped 2026-05-08 from Qwen2.5-7B / Qwen3-Coder-30B) | 1×96t full or 4×48t quarters | MTP spec decode (ik_llama.cpp PR #1744); `KMP_BLOCKTIME=10` | 60.7 t/s per instance | ~16GB |
| 8083 | architect_general | Qwen3.5-122B-A10B Q4_K_M (swapped 2026-03-19 from Qwen3-235B-A22B) | Node 0, 96t | MoE reduction; `enable_thinking=false` | 12.19 t/s (Probe B canonical, 2026-05-04) | ~69GB |
| 8085 | ingest_long_context | Qwen3-Next-80B-A3B Q4_K_M | Node 0, 96t | None (SSM-hybrid), mlock | 14.4–20.8 t/s @ ~12K context | ~46GB |
| 8086 | worker_vision | Qwen2.5-VL-7B Q4_K_M + mmproj | Q0B pinned | None (VL) | ~15 t/s | ~8GB |
| 8087 | vision_escalation | Qwen3-VL-30B-A3B Q4_K_M + mmproj | Node 1, 96t | MoE4 | ~10 t/s | ~20GB |
| 8090-8095 | embedder (6x) | BGE-large-en-v1.5 F16 | unpinned | probe-first | — | ~4GB |

**Notes**:
- The former port 8084 (architect_coding, Qwen3-Coder-480B-A35B) was **removed on 2026-05-06**. REAP-246B scored 70% on the coder suite — worse than worker_general (77%) and far worse than the frontdoor model (97%); the role was eliminated and its 139 GB warm-tier footprint reclaimed. Hard coding escalations now terminate at coder_escalation.
- The former port 8081 (separate coder_escalation instance) was retired on 2026-05-09. Since the 2026-05-06 swap both frontdoor and coder_escalation back onto the same Qwen3.6-35B-A3B Q8 GGUF, the dedicated server was redundant; they now share one mmap on port 8070 with separate admission slots (~36 GB of duplicate mlock + a competing 96-thread OMP team reclaimed).

**Total HOT RAM**: ~600 GB (post-2026-05-09 consolidation: architect_coding removed, frontdoor+coder_escalation share GGUF, worker swapped to 16 GB gemma-4 Q4_K_M), leaving ~460 GB for KV cache and ~70 GB for OS/buffers.

### Auxiliary Services

| Port | Service | Model | Purpose |
|------|---------|-------|---------|
| 8000 | orchestrator API | uvicorn | FastAPI HTTP entrypoint |
| 8088 | nextplaid-code | LateOn-Code (130M, ONNX INT8) | Multi-vector code retrieval (AST-chunked) |
| 8089 | nextplaid-docs | answerai-colbert-small-v1 (ONNX INT8) | Multi-vector doc retrieval |
| 9001 | document_formalizer | LightOnOCR-2-1B | PDF OCR, figure extraction |

### WARM Tier (Load on Demand)

| Port | Role | Model | Purpose |
|------|------|-------|---------|
| 8102 | worker_fast_1 | Qwen2.5-Coder-1.5B Q4_K_M | Burst capacity |
| 8112 | worker_fast_2 | Qwen2.5-Coder-1.5B Q4_K_M | Burst capacity |

**Idle Timeout**: 300 seconds (5 minutes). Automatically shut down if unused.

</details>

## Port Assignment and NUMA Modes (2026-05-09)

Default operation uses **full-mode**: one 1×96-thread instance per role on a single canonical port (frontdoor on 8070, worker on 8072, architect_general on 8083, ingest_long_context on 8085). Launch with:

```bash
python3 scripts/server/orchestrator_stack.py start --numa-mode full   # default
```

For NUMA-optimized throughput, **quarter-mode** runs 4×48-thread instances per role pinned to NUMA quarters, using the multi-port lists in the table above (frontdoor 8080/8180/8280/8380, worker 8082/8182/8282/8382):

```bash
python3 scripts/server/orchestrator_stack.py start --numa-mode quarter
```

`--numa-mode both` launches both layouts side by side for A/B benchmarking. Aggregate throughput in quarter-mode is roughly 2× full-mode per role for MoE workloads; latency is comparable.

**Consolidation**. Since 2026-05-09, frontdoor and coder_escalation share a single GGUF mmap (Qwen3.6-35B-A3B Q8) on whichever port the launch mode picks; the two roles differ only by admission slot and prompt modifier. Operators querying `/health` on `localhost:8070` see one process handling both roles. Worker_general lives on its own port (8072 full / 8082-series quarter) because the gemma-4 binary path differs from frontdoor (PR #1744 ik_llama.cpp build, separate `LD_LIBRARY_PATH`).

## Memory Architecture

About half the system RAM is pinned to HOT-tier models so they never get evicted. The remaining half is split between dynamic KV cache (which grows with concurrent requests) and OS buffers. Larger models like the 235B and 480B architects dominate the budget, but keeping them resident avoids 30-90 second reload penalties that would wreck interactive latency.

<details>
<summary>Tier allocation and model load times</summary>

### Tier Allocation

<details>
<summary>Data: RAM budget breakdown</summary>

```
Total RAM: 1130GB
├── HOT Tier: ~600GB (53%) - Always resident (post-2026-05-09 consolidation)
│   ├── Frontdoor + coder_escalation (shared Qwen3.6-35B Q8 mmap): 37GB
│   ├── Worker (gemma-4-26B-A4B Q4_K_M + MTP drafter): ~16GB
│   ├── Architect_general (Qwen3.5-122B-A10B Q4_K_M): ~69GB
│   ├── Ingest (Qwen3-Next-80B-A3B Q4_K_M): ~46GB
│   ├── Vision (Qwen2.5-VL-7B + Qwen3-VL-30B-A3B): ~28GB
│   ├── Embedder (BGE-large-en-v1.5 F16, 6×): ~4GB
│   ├── NextPLAID (2x): ~1.4GB (code: 1.2GB LateOn-Code 130M + docs: 0.2GB colbert-small)
│   └── NUMA-quarter multi-instance copies (when --numa-mode quarter): +3-4× per HOT role
├── KV Cache: ~460GB (41%) - Dynamic allocation
└── OS + Buffers: ~70GB (6%)
```

**Reclaimed since last revision**: architect_coding (~139 GB) removed 2026-05-06; coder_escalation duplicate mlock (~36 GB) removed 2026-05-09; worker swap to gemma-4 Q4_K_M dropped per-instance footprint from ~14 GB (Qwen2.5-7B f16) but is comparable per instance.

</details>

**Design Principle**: Keep specialists resident to avoid cold-start latency (15-45s model load). Only WARM tier workers are evicted.

### Model Load Times

| Model Size | Load Time | Strategy |
|------------|-----------|----------|
| 0.5B-1.5B | 2-5s | WARM tier (acceptable cold start) |
| 7B-32B | 10-20s | HOT tier (avoid reload) |
| 80B-235B | 30-60s | HOT tier (critical) |
| 480B | 60-90s | HOT tier (always resident) |

**Optimization**: Parallel tensor repack (`production-consolidated` branch) reduces load time by 2.2x vs sequential.

</details>

## Worker Pool Architecture

> **DEPRECATED 2026-05-06.** The heterogeneous worker-pool design described below (multiple small models per task type, fast 1.5B WARM workers spinning up on burst) was superseded by the unified `worker_general` role: a single gemma-4-26B-A4B-it Q4_K_M instance with MTP speculative decoding on port 8072 (full-mode) or 8082/8182/8282/8382 (quarter-mode). The Qwen2.5-7B and Qwen2.5-Coder-1.5B GGUFs referenced in this section are not on disk; `worker_pool.enabled=false` by default in the registry. The text below is retained for historical reference only — operational control runs through `roles.worker_general` in `model_registry.yaml`.

Historically, workers were not one-size-fits-all. Different models handled different task types, and the pool expanded on demand when concurrent load spiked. The original 7B coder worker was removed after benchmarks proved the 32B coder-escalation endpoint was both faster and higher quality; that 32B model was itself superseded in 2026-05-08 by gemma-4-26B-A4B-it with MTP, which now handles all worker traffic.

<details>
<summary>Worker routing, pool config, and expansion strategy</summary>

### Heterogeneous Parallelism

The worker pool uses different models for different task types:

<details>
<summary>Code: Worker pool model mapping</summary>

```python
WORKER_POOL_MODELS = {
    "explore": "/mnt/raid0/llm/models/Qwen2.5-7B-Instruct-f16.gguf",
    "fast": "/mnt/raid0/llm/lmstudio/models/.../Qwen2.5-Coder-1.5B.Q4_K_M.gguf",
}

class WorkerTier(Enum):
    HOT = "hot"    # Always resident
    WARM = "warm"  # Load on demand
```

</details>

### Task Routing

| Task Type | Worker | Model | Rationale |
|-----------|--------|-------|-----------|
| explore, summarize, understand | explore (8082) | 7B Instruct + spec decode | Quality for comprehension |
| code_impl, refactor, test_gen | worker_coder (semantic) → fast pool (8102) | Parallel coding bursts with low latency |
| boilerplate, transform | fast_1/fast_2 (8102/8112) | 1.5B WARM | High throughput, simple tasks |

**worker_coder**: Coding-worker semantics route to the fast pool on port 8102 for parallel subtask bursts; specialist `coder_escalation` remains available for heavier synthesis/debug tasks.

### Expansion Strategy

<details>
<summary>Code: Worker pool expansion config</summary>

```python
@dataclass
class WorkerPoolConfig:
    expansion_threshold: int = 4  # Concurrent tasks to trigger WARM expansion
    warm_timeout_seconds: int = 300  # 5 min idle before shutdown
```

</details>

When concurrent load exceeds 4 tasks, WARM workers spin up. After 5 minutes idle, they shut down to free RAM.

</details>

## CLI Operations

You manage the whole stack through `orchestrator_stack.py`. It supports dev mode for quick iteration with a single tiny model, production hot-only for the full resident tier, and granular reload so you can swap one component without bouncing everything.

<details>
<summary>Start, stop, reload, and state persistence commands</summary>

### Start Commands

<details>
<summary>Code: orchestrator_stack.py usage</summary>

```bash
# Development mode (single 0.5B model, fast startup)
python3 scripts/server/orchestrator_stack.py start --dev

# Production HOT tier only (~535GB RAM)
python3 scripts/server/orchestrator_stack.py start --hot-only

# Production with specific WARM tier models
python3 scripts/server/orchestrator_stack.py start --include-warm architect_general

# Check status
python3 scripts/server/orchestrator_stack.py status

# Stop all
python3 scripts/server/orchestrator_stack.py stop --all

# Reload specific component after code changes
python3 scripts/server/orchestrator_stack.py reload orchestrator
```

</details>

### Critical Environment Variables

All startup paths (`orchestrator_stack.py start`, `reset_episodic_memory.sh`, `seeding_infra.py --preflight`) set `ORCHESTRATOR_CASCADING_TOOL_POLICY=1`. Without this, the legacy tool permission path denies ALL roles ALL tools because no role has `tool_permissions` defined in `model_registry.yaml`. This was fixed on 2026-03-03 after circuit breaker cascades caused seeding stalls.

**Launch-time NUMA selection (2026-05-08/09)**: `--numa-mode {full|quarter|both}` controls instance layout (see "Port Assignment and NUMA Modes" above).

**OMP tuning (2026-05-09)**: the worker_general (gemma-4-26B-A4B with MTP) launch env keeps `OMP_WAIT_POLICY` at its **active** default. Setting it to `passive` causes a load-spike regression (decode drops from ~420 t/s back to ~9 t/s on saturated workloads) because AOCC libomp ignores `omp_pause_resource`. `KMP_BLOCKTIME=10` is set explicitly to keep idle cores from busy-spinning under MTP.

### State Persistence

<details>
<summary>Config: orchestrator_state.json schema</summary>

```json
// /mnt/raid0/llm/epyc-orchestrator/logs/orchestrator_state.json
{
  "server_8080": {
    "role": "frontdoor",
    "pid": 12345,
    "port": 8080,
    "started_at": "2026-01-28T10:30:00",
    "model_path": "/mnt/raid0/llm/models/Qwen3-Coder-30B-A3B-Q4_K_M.gguf",
    "log_file": "/mnt/raid0/llm/epyc-orchestrator/logs/llama-server-8080.log"
  }
}
```

</details>

State enables graceful shutdown and status queries without querying each server.

</details>

## Health Monitoring

Every server exposes a `/health` endpoint that the stack polls during startup and ongoing operation. The startup sequence is deliberately sequential with cooldown gaps between large models so mmap has time to settle. Vision servers get extra-long timeouts because they load both a main model and a multimodal projector.

<details>
<summary>Liveness check implementation and status output</summary>

### Liveness Checks

<details>
<summary>Code: Health polling loop</summary>

```python
def wait_for_health(port: int, timeout: int = 120) -> bool:
    """Wait for server health endpoint."""
    url = f"http://localhost:{port}/health"
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(2)
    return False
```

</details>

**Startup Sequence**: Servers start sequentially with 5s cooldown between large models (allow mmap to settle). Vision servers get 90-120s timeout (mmproj + main model).

### Status Output

<details>
<summary>Data: Example status table</summary>

```
COMPONENT                 PORT     PID        STATUS     MODEL
--------------------------------------------------------------------------------
frontdoor                 8070     12345      healthy    Qwen_Qwen3.6-35B-A3B-Q8_0
coder_escalation          8070     12345      healthy    Qwen_Qwen3.6-35B-A3B-Q8_0  (shared slot)
worker_general            8072     12346      healthy    gemma-4-26B-A4B-it-Q4_K_M
architect_general         8083     12348      healthy    Qwen3.5-122B-A10B-Q4_K_M
ingest_long_context       8085     12349      healthy    Qwen3-Next-80B-A3B-Q4_K_M
orchestrator              8000     12350      healthy    uvicorn
```

</details>

</details>

## Initialization Hooks

After all servers are healthy, the stack initializes MemRL databases, seeds the REPL with examples, warms up the embedding pool, and registers the 41 deterministic tools. This runs automatically — you do not need to trigger it manually.

<details>
<summary>MemRL and tool registry init sequence</summary>

### MemRL and Tool Registry

<details>
<summary>Code: init_memrl_and_tools()</summary>

```python
def init_memrl_and_tools() -> bool:
    """Initialize MemRL databases and tool registry."""
    # [6] REPL seed examples
    seed_loader.init()

    # Warm up embedding model with test query
    requests.post("http://localhost:8090/embedding",
                  json={"content": "test embedding warmup"})

    # [7] Tool registry (41 deterministic tools)
    from orchestration.tools.executor import get_executor
    executor = get_executor()
    tools = executor.list_tools()
    # categories: math (8), symbolic (6), numerical (4), format (7), ...

    return True
```

</details>

Called automatically after server startup. Ensures episodic memory and tools are ready.

</details>

## Checkpoint Hooks

Self-management procedures can create/restore checkpoints:

<details>
<summary>Code: Checkpoint create and restore</summary>

```python
checkpoint_create("before_model_update", include_state=True)
# ... make changes ...
checkpoint_restore("before_model_update_20260128_103000")
```

</details>

Stored in `/mnt/raid0/llm/epyc-orchestrator/orchestration/checkpoints/`.

## References

<details>
<summary>Implementation, architecture patterns, and related systems</summary>

### Implementation

1. `scripts/server/orchestrator_stack.py`: Stack launcher (1330 lines)
2. `src/services/worker_pool.py`: Worker pool manager (732 lines)
3. `src/registry_loader.py`: Model registry and role resolution

### Architecture Patterns

4. Netflix's Eureka service discovery: https://github.com/Netflix/eureka
5. Kubernetes liveness/readiness probes: https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/

### Related Systems

6. Ray Serve (model serving framework): https://docs.ray.io/en/latest/serve/
7. BentoML (ML serving): https://docs.bentoml.org/

</details>

## Concurrent Inference Sweep (2026-02-19)

Benchmarked optimal `-np`/concurrency per model tier using `scripts/benchmark/concurrent_inference_sweep.py` (asyncio + httpx.AsyncClient, 2 warmup + 5 measured batches, incremental CSV output).

**Results** (sweep run pre-2026-05 consolidation; ports noted in their then-current form):
| Role | Port (then) | Recommended `-np` | Rationale |
|------|------|--------------------|-----------|
| frontdoor (30B MoE) | 8080 | **2** (was 1) | +121% aggregate TPS, p95 multiplier 1.33 |
| coder (32B dense) | 8081 | 1 (keep) | c=2 rejected: p95 multiplier 1.98 |
| worker (7B) | 8082 | 1 (keep) | c=2+ rejected: p95 multiplier ≥1.505 |
| fast_worker (1.5B) | 8102 | — | Port unavailable during sweep |

Post-consolidation (2026-05-09), the sweep should be re-run: frontdoor now lives on 8070 (full-mode) with `-np 2`, coder_escalation shares that port via a separate slot, and worker is gemma-4-26B-A4B with MTP on 8072 — its `-np` profile is not yet re-measured under MTP.

**Action taken**: Removed `frontdoor` from `SERIAL_ROLES` in `orchestrator_stack.py` so it starts with `-np 2`.

### SERIAL_ROLES

`SERIAL_ROLES` in `orchestrator_stack.py` forces `-np 1` for roles where concurrent slot contention degrades latency: `coder_escalation`, `worker_summarize`, `architect_general`, `ingest_long_context`. (`architect_coding` was removed from the set on 2026-05-06 along with the role itself.)

---

*Previous: [Chapter 03: REPL Environment & Sandboxing](03-repl-environment.md)* | *Next: [Chapter 05: Data Processing Pipelines](05-data-processing-pipelines.md)*
