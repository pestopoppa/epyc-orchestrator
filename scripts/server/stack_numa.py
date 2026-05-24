"""NUMA CPU pinning topology + per-role pinning configuration.

Source of truth for "which CPUs and how many threads each role gets". Encodes
the validated EPYC 9655 topology + per-role NUMA wiring (Probe B 2026-05-04
canonical results, 2026-05-08 gemma4 MTP swap). `orchestrator_stack.py`
re-imports the constants and `_numa_prefix()` so the launcher and server-build
code keeps working unchanged.

Key findings (2026-03-18 benchmarks, refined through 2026-05-08):
- Models ≤65 GB: 4×48t NUMA-quarter instances give 6-7× aggregate throughput
- Models 130-250 GB: 1×96t NUMA-node pinning gives 1.2-1.5×
- Using all 192t is ANTI-OPTIMAL (46-60% cross-NUMA penalty)
- taskset alone is sufficient — numactl --membind adds no benefit (S4 result)
- mlock gives 30× latency improvement under memory pressure (S2)
- Total mlock budget: ~701 GB of 1.13 TB (62%), leaving ~429 GB for KV + OS
"""

from __future__ import annotations


# =============================================================================
# NUMA CPU Pinning — validated via benchmarks (2026-03-18)
# =============================================================================
# EPYC 9655: 192 cores, 2 NUMA nodes (~566 GB each).
# Node 0: cores 0-47, HT 96-143
# Node 1: cores 48-95, HT 144-191

# NUMA quarter definitions: (cpu_list, thread_count)
NUMA_Q0A = ("0-23,96-119", 48)
NUMA_Q0B = ("24-47,120-143", 48)
NUMA_Q1A = ("48-71,144-167", 48)
NUMA_Q1B = ("72-95,168-191", 48)
NUMA_NODE0 = ("0-47,96-143", 96)
NUMA_NODE1 = ("48-95,144-191", 96)
# Full-machine physical-cores-only (no SMT) — for canonical-recipe wiring
# (single-instance latency-optimal). 96 physical cores spanning all 4 NPS4 nodes.
# Pair with numactl_policy="interleave=all" so memory distributes across all 4 nodes
# (matches the canonical bench recipe used by Probe B 2026-05-04).
NUMA_FULL = ("0-95", 96)

# Per-role NUMA configurations.
# "instances" is a list of (cpu_list, port, threads) tuples.
# Roles with multiple instances get round-robin routing (requires orchestrator support).
NUMA_CONFIG: dict[str, dict] = {
    # Qwen3.5-35B-A3B Q4_K_M (19 GB) — pre-warm: 1×96t full-speed + 4×48t concurrent
    # Benchmark (2026-03-24): moe6 = 12.7 t/s at 48t. 96t TBD (expect higher per-request).
    # Pre-warm strategy (2026-03-29): 5 instances total, +19 GB (95 GB total for frontdoor).
    # Concurrency router: single session → full (96t), concurrent → quarter (48t) instances.
    "frontdoor": {
        "instances": [
            # 2026-05-24 REVERT: full instance returned to NUMA_NODE0 (cores
            # 0-47, SMT siblings 96-143) WITHOUT numactl_policy. April 2026-04-17
            # 3-rep head-to-head bench on Qwen3.6-35B-A3B Q8 (recorded in
            # progress/2026-05/2026-05-20.md lines 671-680):
            #   Config A: numactl --interleave=all + 96t  → 26.60 t/s
            #   Config B: NUMA_NODE0 (taskset 0-47,96-143) + 96t  → 27.06 t/s
            # B beats A by 1.7% — A3B MoE activates only ~3B/35B params per
            # token, so memory traffic is low and cache locality dominates over
            # channel parallelism. NUMA_NODE0 is correct.
            # An earlier 2026-05-24 commit (6657bbd) migrated this to
            # NUMA_FULL+interleave=all citing "consistency with worker_general /
            # architect_general"; that ignored the April head-to-head. Reverted.
            (NUMA_NODE0[0], 8070, NUMA_NODE0[1]),  # full: 1×96t on cores 0-47+SMT
            (NUMA_Q0A[0], 8080, NUMA_Q0A[1]),      # quarter 0
            (NUMA_Q0B[0], 8180, NUMA_Q0B[1]),      # quarter 1
            (NUMA_Q1A[0], 8280, NUMA_Q1A[1]),      # quarter 2
            (NUMA_Q1B[0], 8380, NUMA_Q1B[1]),      # quarter 3
        ],
        "full_instance_idx": 0,  # index of 1×96t instance in list above
        "mlock": True,   # 19 GB per instance — latency-critical (S2: 30x improvement)
        # NOTE: NPS4 single-quarter at -t 48 gives ~8.90 t/s on this model per
        # Phase 0.5 (Qwen3-Coder-30B Q4's 46.6 t/s NPS4 sweep does NOT transfer
        # to 35B Q8 — Q8 needs more BW than one quarter provides). The 4
        # quarter instances above will run noticeably slower than the full;
        # ConcurrencyAwareBackend prefers the full for solo requests anyway.
    },
    # coder_escalation NUMA_CONFIG entry REMOVED 2026-05-09 — consolidated onto
    # frontdoor's server (same Qwen3.6-35B-A3B Q8 GGUF since 2026-05-06 swap).
    # Historical: ports 8071 (full) + 8081/8181/8281/8381 (quarters), spec_overrides
    # {dm=32, ps=0.05} were Qwen2.5-Coder-32B-era params and didn't apply post-swap.
    # If a separate coder server is ever rostered again on a different model,
    # restore here with that model's tuned spec params.
    # Qwen3.5-122B-A10B Q4_K_M (69 GB) — 1×96t canonical (Probe B 2026-05-04)
    # Switched 2026-05-04 from 2× cross-NUMA (4.3 t/s/instance, 8.6 t/s agg) to
    # 1× full-machine canonical with numactl --interleave=all + GGML_NUMA_REPACK_INTERLEAVE=0
    # (c2 env block, see _ROLE_ENV_BLOCKS). Measured 12.19 t/s single-instance = +184%
    # per-request latency vs prior 2× wiring. Bundle:
    # epyc-inference-research/data/cpu_optimization/2026-05-04-qwen35-122b-arch-probe/
    # Reopen 4× per-NUMA-node wiring (16.86 t/s aggregate) ONLY if architect_general workload
    # shifts to 4+ concurrent batch eval — see findings_phase2.md.
    "architect_general": {
        "instances": [
            (NUMA_FULL[0], 8083, NUMA_FULL[1]),  # 1×96t physical cores, all 4 NUMA nodes
        ],
        "mlock": True,
        "numactl_policy": "interleave=all",  # wraps launch with `numactl --interleave=all --`
        "spec_overrides": {"draft_max": 24, "p_split": 0},  # sweep-verified
    },
    # architect_coding REMOVED 2026-05-06 — REAP-246B Q4KM scored 7/10 (70%) on coder
    # under canonical recipe, WORSE than worker_general (gemma4-26B-A4B Q4_K_M MTP at 96%)
    # AND far worse than frontdoor (Qwen3.6-35B-A3B Q8 at 97%). 139 GB warm freed.
    # Hard coding escalations now route to coder_escalation, which uses the same
    # Qwen3.6-35B-A3B Q8 model as frontdoor (shared GGUF mmap).
    # Qwen3-Next-80B-A3B Q4_K_M (45 GB) — pre-warm: 1×96t full + 4×48t quarters
    # 2026-05-24 Phase 0.5 bench: 48t single-quarter = 12.34 t/s, viable for
    # quartering despite the 45 GB GGUF being snug against per-quarter capacity.
    # Concurrency router routes solo → full (8085), concurrent → quarters
    # (8185/8285/8385/8485). Full anchor reverted to NUMA_NODE0 (same NPS4
    # half-socket pattern frontdoor uses, which the April 2026-04-17 data
    # showed beats NUMA_FULL+interleave=all by 1.7% on Qwen3.6-Q8. No direct
    # head-to-head data exists for Qwen3-Next-80B Q4, but it's an A3B MoE
    # like Qwen3.6 so cache-locality argument applies — keep parity).
    "ingest_long_context": {
        "instances": [
            (NUMA_NODE0[0], 8085, NUMA_NODE0[1]),    # full: 1×96t on cores 0-47+SMT
            (NUMA_Q0A[0], 8185, NUMA_Q0A[1]),        # quarter 0
            (NUMA_Q0B[0], 8285, NUMA_Q0B[1]),        # quarter 1
            (NUMA_Q1A[0], 8385, NUMA_Q1A[1]),        # quarter 2
            (NUMA_Q1B[0], 8485, NUMA_Q1B[1]),        # quarter 3
        ],
        "full_instance_idx": 0,
        "mlock": True,    # ~46 GB per instance — latency-critical for ingest pipeline (Stage 1 of three_stage_summarization since 2026-05-06)
    },
    # Worker: gemma4-26B-A4B Q4_K_M MTP (16 GB) — pre-warm: 1×96t + 4×48t.
    # Swapped 2026-05-08 from Qwen3-Coder-30B-A3B Q4_K_M (was 39 t/s at 48t).
    # gemma4-26B-A4B + ik_llama.cpp PR #1744 MTP: 76.5 t/s at 96t (full canonical), 95.2% draft acceptance.
    # +18pp on tool_compliance (96% vs 78%), +6pp on full suite (90% vs 84%).
    # Pre-2026-05-08: 7B f16 (until 2026-03-21), then Qwen3-Coder-30B-A3B Q4_K_M.
    # NB: full + 4 quarters share overlapping CPU sets — pick one mode at start time
    # (full instance uses 0-95; 4 quarters together also cover 0-95). See task #57.
    "worker_general": {
        "instances": [
            # 2026-05-08 swap to gemma4-26B-A4B MTP via ik_llama.cpp PR #1744:
            # full instance MUST use "0-95" (both NUMA nodes' physical cores) +
            # numactl --interleave=all to satisfy MTP's tensor-buffer NUMA expectation.
            # NUMA_NODE0's "0-47,96-143" (one-socket-with-SMT) crashed the MTP draft
            # path with "tensor buffer not set" assertion. Quarter instances retain
            # their per-quarter pinning since the full canonical recipe is incompatible
            # with the 4×concurrent design — they may need separate debugging.
            ("0-95", 8072, 96),                    # full canonical (replaces NUMA_NODE0)
            (NUMA_Q0A[0], 8082, NUMA_Q0A[1]),      # quarter 0
            (NUMA_Q0B[0], 8182, NUMA_Q0B[1]),      # quarter 1
            (NUMA_Q1A[0], 8282, NUMA_Q1A[1]),      # quarter 2
            (NUMA_Q1B[0], 8382, NUMA_Q1B[1]),      # quarter 3
        ],
        "full_instance_idx": 0,
        "mlock": True,
        "spec_overrides": {"draft_max": 2, "p_split": 0},  # gemma4 MTP recipe (was dm=8 for Qwen3-Coder)
        "numactl_policy": "interleave=all",  # 2026-05-08: required for gemma4 MTP buffer allocation
    },
    # Qwen2.5-VL-7B Q4_K_M (~4 GB) — single instance on Q0B at 24t.
    # Phase 0.5 bench (2026-05-24) showed 24t = 11.39 t/s, 48t = 11.30 t/s
    # (flat — model too small to benefit from more threads or quartering).
    # An earlier 2026-05-24 commit (6657bbd) added a 96t full instance + 4
    # quarters, all using 24t each = 16 GB unnecessary mlock budget. Reverted
    # to the original single-quarter shape: this role just doesn't have
    # enough request volume to justify concurrent-serving topology.
    "worker_vision": {
        "instances": [(NUMA_Q0B[0], 8086, 24)],
        "mlock": True,    # ~4 GB — minimal footprint
    },
    # Qwen3-VL-30B-A3B MoE (~17 GB) — pre-warm: 1×96t full + 4×48t quarters
    # Phase 0.5 bench (2026-05-24): 48t/quarter = 20.09 t/s — best quarter
    # throughput of any role (small Q4 active params + healthy BW). Quartering
    # gives ~80 t/s aggregate for concurrent vision-escalation requests.
    # Full anchor: NUMA_NODE1 (the historical pre-2026-05-24 choice, on the
    # node 1 half-socket — frees node 0 for frontdoor's heavier load). An
    # earlier 2026-05-24 commit migrated this to NUMA_FULL+interleave=all
    # without supporting head-to-head data — reverted.
    "vision_escalation": {
        "instances": [
            (NUMA_NODE1[0], 8087, NUMA_NODE1[1]),    # full: 1×96t on node 1 half-socket
            (NUMA_Q0A[0], 8187, NUMA_Q0A[1]),        # quarter 0
            (NUMA_Q0B[0], 8287, NUMA_Q0B[1]),        # quarter 1
            (NUMA_Q1A[0], 8387, NUMA_Q1A[1]),        # quarter 2
            (NUMA_Q1B[0], 8487, NUMA_Q1B[1]),        # quarter 3
        ],
        "full_instance_idx": 0,
        "mlock": True,    # ~17 GB per instance — fits in 1.13 TB budget
    },
}

# Roles that should use --mlock (requires ulimit -l unlimited in launch env)
MLOCK_ROLES = {role for role, cfg in NUMA_CONFIG.items() if cfg.get("mlock")}


def _numa_prefix(role: str, instance_idx: int = 0) -> list[str]:
    """Return CPU-pinning + memory-policy prefix for a role instance.

    Default: taskset -c <cpu_list> (S4 benchmark: numactl --membind adds no benefit
    over taskset + first-touch memory policy for per-NUMA-node-bound roles).

    If the role's NUMA_CONFIG entry has a "numactl_policy" key (e.g. "interleave=all"),
    wraps the launch with `numactl --<policy> --` ahead of taskset. Used for
    canonical-recipe roles like architect_general (Probe B 2026-05-04: numactl
    --interleave=all + taskset -c 0-95 = 12.19 t/s single-instance vs 4.3 t/s under
    legacy 2× cross-NUMA + first-touch).
    """
    cfg = NUMA_CONFIG.get(role)
    if cfg and instance_idx < len(cfg["instances"]):
        cpu_list = cfg["instances"][instance_idx][0]
        prefix: list[str] = []
        policy = cfg.get("numactl_policy")
        if policy:
            prefix.extend(["numactl", f"--{policy}", "--"])
        prefix.extend(["taskset", "-c", cpu_list])
        return prefix
    # Fallback: no pinning (embedders, fast workers, dev mode)
    return []
