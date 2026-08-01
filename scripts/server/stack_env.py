"""Launch env construction for orchestrator llama-server processes.

Owns the canonical OMP env stack, the LLVM-20 libomp path that must win over
AOCC's libomp at runtime, and the per-role GGML_* env blocks from the v5
deployment draft. `build_launch_env(role, base_env)` composes them in the
documented precedence order; `orchestrator_stack.py` re-imports the helpers
so existing call sites keep working.
"""

from __future__ import annotations

from src.roles import Role


# =============================================================================
# Per-role env blocks — applied to every llama-server launch.
# Source: handoffs/active/model-registry-v5-deployment-draft.yaml roles section.
# Universally-applied OMP env stack + per-arch-class GGML_* opt-ins.
# =============================================================================

# Always applied to every llama-server launch (the canonical OMP recipe).
# Source: cpu-kernel-env-flags-inventory.md §28-30. Without these, post-reboot
# Coder-30B drops 17 → 48.8 t/s (3-4× degraded, per feedback_omp_env_stack_required).
_CANONICAL_OMP_ENV = {
    "OMP_PROC_BIND": "spread",
    "OMP_PLACES": "cores",
    "OMP_WAIT_POLICY": "active",
    # OMP_DYNAMIC=false: prevents the runtime from quietly trimming the team
    # below OMP_NUM_THREADS. Required by canonical recipe (canonical_recipe.py:43-48
    # in epyc-inference-research). Without this, ik_llama.cpp's MTP draft path
    # asserts on "tensor buffer not set" because draft thread-team init races with
    # buffer allocation. (2026-05-08 Phase 3.)
    "OMP_DYNAMIC": "false",
    # KMP_BLOCKTIME=10 ms: tunes the idle transition under OMP_WAIT_POLICY=active.
    # Without this, AOCC libomp's worker team stays alive in busy-wait between OMP
    # regions — when a server is idle (no request), its 96 OMP threads spin and burn
    # ~half the chip in cumulative %Cpu(s) us. Originally added on the worker_pool
    # branch (2026-05-09) for ik_llama.cpp PR #1744 (gemma4 MTP) because PR #1744
    # uses bare `#pragma omp parallel` per ggml_graph_compute() with no persistent
    # threadpool — that path's idle spin was the loudest. But the same idle-spin
    # affects every llama-server launch under OMP_WAIT_POLICY=active on this libomp,
    # not just MTP. Globalising here (2026-05-21) fixes the 4 frontdoor quarters +
    # full + architect_general + ingest + visions + embedders all spinning idle
    # between requests, which had been showing up as ~50% baseline %Cpu(s) us and
    # multi-second delegation latency (other servers couldn't claim CPU against the
    # spinning idle teams). 10 ms is the sweet spot: long enough to keep workers
    # warm for back-to-back ops (no perceptible first-token regression), short
    # enough that multi-second request gaps release the cores.
    "KMP_BLOCKTIME": "10",
    # 2026-06-26 v6 cutover: GGML_IQK=1 gates ik_llama's iqk AVX-512 GEMM kernels,
    # which are compiled into the production-consolidated-v6 binary but runtime-gated
    # by this env var. Applied to the canonical OMP stack so EVERY role's
    # llama-server launch boots with the iqk kernels enabled.
    "GGML_IQK": "1",
}

# clang-20's libomp directory — prepended to LD_LIBRARY_PATH for any role that
# resolves OpenMP at runtime. The orchestrator's binaries (and the per-role
# ik_llama.cpp PR #1744 build for worker_general) would otherwise fall through
# to AOCC's libomp.so on disk; AOCC has different thread-pinning behavior that
# triggers the MTP buffer assertion. Mirrors canonical_recipe.LLVM20_LIBDIR.
_LLVM20_LIBDIR = "/usr/lib/llvm-20/lib"

# Per-role env blocks. Keyed by role name (matches NUMA_CONFIG keys + registry roles).
# Roles not listed inherit only the canonical OMP env.
# Source: model-registry-v5-deployment-draft.yaml §roles, validated under v5 audit.
_ROLE_ENV_BLOCKS: dict[str, dict[str, str]] = {
    # MoE Q4 sync-bound (CPU1 stack +1.8% on Coder-30B Q4_K_M tg32, stable).
    # NB: GGML_NUMA_WEIGHTS deliberately excluded — DEPRECATED per CPU21 P3 isolation
    # (unstable, 19-22σ at warmed state). Uses 3-flag stable stack.
    # 2026-06-26 v6 cutover: removed GGML_CCD_POOLS / GGML_CCD_WORK_DIST /
    # GGML_BARRIER_LOCAL_BETWEEN_OPS — CCD code is #ifndef GGML_USE_OPENMP, so the
    # OpenMP-ON v6 build compiles it out, making these env vars vestigial no-ops.
    "worker_general": {},
    # MoE Q8 BW-bound frontdoor — EP stack was historically +17% per the original
    # validation (drone+shard, N=2), but verified WRONG by direct A/B 2026-05-11:
    # the GGML_EP_* stack was sitting at 12.6 t/s under sustained single-instance
    # full-NUMA load. Hypothesis: the EP stack assumed the older binary / different
    # NUMA wiring. Re-verify before re-enabling.
    "frontdoor": {},
    "frontdoor_ep_stack_disabled_2026_05_11": {
        "GGML_EP_N_INSTANCES": "2",
        "GGML_EP_NUMA_PIN": "1",
        "GGML_EP_MASTER_ALL_NODES": "1",
        "GGML_EP_WORKER_DRONE": "1",
        "GGML_EP_SHARD": "1",
    },
    # architect_coding env block REMOVED 2026-05-06 — REAP-246B role eliminated.
    # MoE-Spec budget=40 plumbing (LLAMA_ARG_MOE_SPEC_BUDGET) was REAP-246B-specific
    # (validated +13-16% pp32 / +3% e2e on that model only). If a future role rosters
    # a comparable MoE-Q4 DRAM-bound model, re-add the env block AT THAT TIME using
    # benchmark data on the new model — do NOT blanket-apply the budget=40 setting.
    # architect_general (Qwen3.5-122B-A10B Q4_K_M) — Probe B closed 2026-05-04.
    # Arch class: moe_q4_bw_bound_mbind_sensitive. c2 wins at +1.28% (σ ~0.4%, z ~3)
    # vs default v5 at 96t canonical. CPU1 stack net-neutral, c3 (combined) regresses to
    # noise. Source bundle: data/cpu_optimization/2026-05-04-qwen35-122b-arch-probe/
    # 2026-08-01 W1 CUTOVER: this block is the 122B's Probe-B tuning and it MOVED
    # WITH THE MODEL to architect_critic. architect_general is now Qwen3.6-27B dense
    # Q8 on MI210 (ROCm0) — a CPU NUMA repack setting applied to a ROCm process is
    # at best inert and at worst misleading provenance.
    "architect_critic": {
        "GGML_NUMA_REPACK_INTERLEAVE": "0",
    },
    # Hybrid SSM dense (Nemotron-9B-v2-class) — c3 = CPU1 stack + mbind off.
    # Activate when a hybrid_ssm_dense model is rostered.
    # 2026-06-26 v6 cutover: removed GGML_CCD_POOLS / GGML_CCD_WORK_DIST /
    # GGML_BARRIER_LOCAL_BETWEEN_OPS — CCD code is #ifndef GGML_USE_OPENMP, so the
    # OpenMP-ON v6 build compiles it out (vestigial no-ops). GGML_NUMA_REPACK_INTERLEAVE
    # (not CCD-gated) is retained.
    "hybrid_ssm_dense": {
        "GGML_NUMA_REPACK_INTERLEAVE": "0",
    },
    # Hybrid SSM MoE (Qwen3-Next-80B-A3B-class) — default v5 (c3 +1.7% noise floor).
    "hybrid_ssm_moe": {},
    # Dense Q8 (Qwen3.6-27B Q8) — DEFAULT v5; CPU1 stack actively HURTS.
    # All probed CPU1/mbind-off configs negative (c1=-4.7%, c2=-3.3%, c3=-1.6%).
    "dense_q8": {},
    # Dense Q4 (gemma-4-31B / SuperGemma4-31B class) — default v5 within ±2% noise.
    "dense_q4": {},
}

_STACK_ENV_CANONICAL_ALIASES = {
    "worker": "worker_general",
}


def _role_env_overrides(role: str) -> dict[str, str]:
    """Return per-role env block for a given role. Empty dict if role not registered.
    Falls back through arch-class aliases (e.g. coder_escalation → worker)."""
    normalized = _STACK_ENV_CANONICAL_ALIASES.get(role, str(Role.from_string(role) or role))
    if normalized in _ROLE_ENV_BLOCKS:
        return dict(_ROLE_ENV_BLOCKS[normalized])
    # Aliases — production roles that map to v5 arch_class names.
    # 2026-05-06: coder_escalation + worker_summarize now use the SAME GGUF as frontdoor
    # (Qwen3.6-35B-A3B Q8) and should inherit frontdoor's EP-stack env block.
    # 2026-05-06: thinking_reasoning alias REMOVED (role eliminated).
    # NB: ingest_long_context (Qwen3-Next-80B-A3B hybrid SSM MoE) routes to hybrid_ssm_moe
    # (default v5 — MoE-Spec budget=40 was REAP-246B-specific, NOT validated on hybrid SSM).
    # formalizer (MathSmith-Qwen3-8B Q8 dense) routes to dense_q8 — it's not MoE at all.
    arch_aliases = {
        # 2026-08-01 W1 CUTOVER: coder_escalation no longer shares frontdoor's GGUF.
        # It is an alias on architect_general's MI210 27B, so it must inherit THAT
        # role's env block — inheriting frontdoor's CPU EP-stack env on a ROCm
        # process was the concrete risk flagged in the cutover audit.
        "coder_escalation": "architect_general",
        "vision_escalation": "worker_vision",  # one MI210 :8086 process serves both
        "worker_summarize": "frontdoor",   # Qwen3.6-35B-A3B Q8 (same model as frontdoor since 2026-05-06 swap)
        "general_gemma_3_27b_it_qat": "dense_q4",
        "ingest_long_context": "hybrid_ssm_moe",  # Qwen3-Next-80B-A3B
        "formalizer": "dense_q8",                 # MathSmith-Qwen3-8B Q8 dense; NOT MoE at all
        "toolrunner": "worker_general",           # gemma4-26B-A4B Q4_K_M MTP (shares with worker_general)
    }
    aliased = arch_aliases.get(normalized)
    if aliased and aliased in _ROLE_ENV_BLOCKS:
        return dict(_ROLE_ENV_BLOCKS[aliased])
    return {}


def build_launch_env(role: str, base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Compose the full env dict for a llama-server launch.

    Order (later overrides earlier):
        1. base_env (parent process env, typically os.environ.copy())
        2. LLVM-20 libomp prepended to LD_LIBRARY_PATH (canonical recipe)
        3. canonical OMP env stack (always applied)
        4. explicit base GGML_IQK override, if present, for v6 iqk A/B gates
        5. per-role GGML_* env block (from v5 deployment draft)

    The per-role block is allowed to override OMP if it must, though no current
    role does so.
    """
    env: dict[str, str] = dict(base_env) if base_env else {}
    # LLVM-20 libomp must win over AOCC libomp at runtime. Prepend to LD_LIBRARY_PATH
    # so the dynamic loader resolves libomp.so to clang-20's. AOCC libomp has different
    # thread-pinning + dynamic-team behavior that breaks ik_llama.cpp PR #1744's MTP
    # path (2026-05-08 Phase 3).
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    if _LLVM20_LIBDIR not in existing_ld.split(":"):
        env["LD_LIBRARY_PATH"] = (
            f"{_LLVM20_LIBDIR}:{existing_ld}" if existing_ld else _LLVM20_LIBDIR
        )
    explicit_iqk = env.get("GGML_IQK")
    env.update(_CANONICAL_OMP_ENV)
    if explicit_iqk is not None:
        env["GGML_IQK"] = explicit_iqk
    env.update(_role_env_overrides(role))
    return env
