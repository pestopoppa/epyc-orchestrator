#!/usr/bin/env python3
"""Unified orchestrator stack launcher.

Launches all models + orchestrator with granular reload support.

Usage:
    orchestrator_stack.py start [--hot-only] [--include-warm ROLE...] [--only ROLE...] [--dev]
    orchestrator_stack.py stop [--all | COMPONENT...]
    orchestrator_stack.py reload COMPONENT...
    orchestrator_stack.py status

Examples:
    # Start ONLY specific roles (skip everything else, preserve what's running)
    ./orchestrator_stack.py start --only worker_vision vision_escalation

    # Start HOT models only
    ./orchestrator_stack.py start --hot-only

    # Start with warm architect
    ./orchestrator_stack.py start --include-warm architect_general

    # Dev mode (single 0.5B model)
    ./orchestrator_stack.py start --dev

    # Reload orchestrator API after code changes
    ./orchestrator_stack.py reload orchestrator

    # Check status
    ./orchestrator_stack.py status
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import _registry_timeout
from src.registry_loader import RegistryLoader

# =============================================================================
# Configuration - loaded from src.config with fallbacks
# =============================================================================

# Health check timeouts from registry (single source of truth)
_HEALTH_SERVER_STARTUP = int(_registry_timeout("health", "server_startup", 120))
_HEALTH_VISION_SERVER = int(_registry_timeout("health", "vision_server", 120))
_HEALTH_WORKER_SERVER = int(_registry_timeout("health", "worker_server", 90))


def _get_paths() -> dict[str, Path]:
    """Get paths from config with hardcoded fallbacks for robustness."""
    try:
        from src.config import get_config

        cfg = get_config()
        return {
            "llm_root": cfg.paths.llm_root,
            "project_root": cfg.paths.project_root,
            "models_dir": cfg.paths.models_dir,
            "model_base": cfg.paths.model_base,
            "llama_cpp_bin": cfg.paths.llama_cpp_bin,
            "log_dir": cfg.paths.log_dir,
            "cache_dir": cfg.paths.cache_dir,
            "tmp_dir": cfg.paths.tmp_dir,
        }
    except Exception as e:
        # Fallback to hardcoded defaults if config unavailable
        llm_root = Path("/mnt/raid0/llm")
        project_root = llm_root / "claude"
        return {
            "llm_root": llm_root,
            "project_root": project_root,
            "models_dir": llm_root / "models",
            "model_base": llm_root / "lmstudio/models",
            "llama_cpp_bin": llm_root / "llama.cpp/build/bin",
            "log_dir": project_root / "logs",
            "cache_dir": llm_root / "cache",
            "tmp_dir": llm_root / "tmp",
        }


_PATHS = _get_paths()

STATE_FILE = _PATHS["log_dir"] / "orchestrator_state.json"
LLAMA_SERVER = _PATHS["llama_cpp_bin"] / "llama-server"
# v2 binary retained for emergency fallback only. As of 2026-05-06 stack-swap,
# all hot-tier roles use the v5 binary (production-consolidated-v5). Previously
# coder_escalation needed v2 due to a Qwen2.5 spec-decode bug, but
# coder_escalation now runs Qwen3.6-35B-A3B Q8 (same model as frontdoor) which
# is v5-compatible.
LLAMA_SERVER_V2 = _PATHS["llama_cpp_bin"].parent / "build-v2" / "bin" / "llama-server"
_V2_ROLES: frozenset[str] = frozenset()  # was {"coder_escalation"}; empty since 2026-05-06
LOG_DIR = _PATHS["log_dir"]
# DS-3: KV state save/restore directory for dynamic stack management
SLOT_SAVE_DIR = _PATHS["cache_dir"] / "kv_slots"

# Port assignments by role (primary ports — full-speed 1×96t instances)
# Pre-warm (2026-03-29): primary port is the full-speed instance.
# Quarter instances on offset ports (808x, 818x, 828x, 838x).
PORT_MAP = {
    "frontdoor": 8070,           # Full-speed 1×96t (quarters: 8080, 8180, 8280, 8380)
    "coder_escalation": 8071,    # Full-speed 1×96t (quarters: 8081, 8181, 8281, 8381)
    "worker_general": 8072,      # Full-speed 1×96t (quarters: 8082, 8182, 8282, 8382)
    "worker_explore": 8072,      # Alias -> worker_general (legacy name; pre-2026-03-19)
    "worker_math": 8072,         # Shares with worker_general
    "worker_vision": 8086,       # Dedicated VL server
    "vision_escalation": 8087,   # VL escalation (Qwen3-VL-30B MoE)
    "worker_coder": 8102,        # Fast coding worker semantic role (1.5B backend) — DEPRECATED (worker_pool)
    "worker_fast": 8102,         # Fast worker (1.5B, WARM, 4 slots) — DEPRECATED (worker_pool)
    # Specialists (no pre-warm — already multi-instance or too large for quarters)
    "architect_general": 8083,
    # architect_coding REMOVED 2026-05-06 — REAP-246B 70% coder < frontdoor 97%; role eliminated. 139 GB freed.
    "ingest_long_context": 8085,
    # Embedding servers (6 parallel instances for redundancy)
    "embedder": 8090,  # Primary embedding server
    "embedder_1": 8091,
    "embedder_2": 8092,
    "embedder_3": 8093,
    "embedder_4": 8094,
    "embedder_5": 8095,
    "orchestrator": 8000,
    "document_formalizer": 9001,
    "sd_server": 8190,         # ERNIE-Image-Turbo via stable-diffusion.cpp native (replaced ComfyUI 2026-05-07; ~1.7-3.4× CPU speedup)
    "whisper": 9000,           # faster-whisper STT (transcription service, not llama-server)
}

# NUMA_REPLICA_PORTS defined after NUMA_CONFIG below (line order dependency)

# HOT roles (always started) - NUMA-optimized
# 2026-05-06: architect_coding removed, ingest_long_context promoted (see registry).
HOT_ROLES = {
    "frontdoor", "coder_escalation", "worker_general", "embedder",
    "architect_general", "ingest_long_context",
    "worker_vision", "vision_escalation",
}

# =============================================================================
# NUMA CPU Pinning — validated via benchmarks (2026-03-18)
# =============================================================================
# EPYC 9655: 192 cores, 2 NUMA nodes (~566 GB each).
# Node 0: cores 0-47, HT 96-143
# Node 1: cores 48-95, HT 144-191
#
# Key findings:
# - Models ≤65GB: 4×48t NUMA-quarter instances give 6-7x aggregate throughput
# - Models 130-250GB: 1×96t NUMA-node pinning gives 1.2-1.5x
# - Using all 192t is ANTI-OPTIMAL (46-60% cross-NUMA penalty)
# - taskset alone is sufficient — numactl --membind adds no benefit (S4 result)
# - mlock gives 30x latency improvement under memory pressure (S2) — enabled for ALL HOT tier
# - Total mlock budget: ~701 GB of 1.13 TB (62%), leaving ~429 GB for KV caches + OS

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
            (NUMA_NODE0[0], 8070, NUMA_NODE0[1]),  # full: 1×96t (max single-session speed)
            (NUMA_Q0A[0], 8080, NUMA_Q0A[1]),      # quarter 0
            (NUMA_Q0B[0], 8180, NUMA_Q0B[1]),      # quarter 1
            (NUMA_Q1A[0], 8280, NUMA_Q1A[1]),      # quarter 2
            (NUMA_Q1B[0], 8380, NUMA_Q1B[1]),      # quarter 3
        ],
        "full_instance_idx": 0,  # index of 1×96t instance in list above
        "mlock": True,   # 19 GB per instance — latency-critical (S2: 30x improvement)
    },
    # Qwen2.5-Coder-32B Q4KM (18.5 GB) — pre-warm: 1×96t + 4×48t
    # Sweep-verified 2026-03-21: dm=32, ps=0.05, 10.8 t/s/inst at 48t
    "coder_escalation": {
        "instances": [
            (NUMA_NODE0[0], 8071, NUMA_NODE0[1]),  # full: 1×96t
            (NUMA_Q0A[0], 8081, NUMA_Q0A[1]),      # quarter 0
            (NUMA_Q0B[0], 8181, NUMA_Q0B[1]),      # quarter 1
            (NUMA_Q1A[0], 8281, NUMA_Q1A[1]),      # quarter 2
            (NUMA_Q1B[0], 8381, NUMA_Q1B[1]),      # quarter 3
        ],
        "full_instance_idx": 0,
        "mlock": True,
        "spec_overrides": {"draft_max": 32, "p_split": 0.05},  # sweep-verified
    },
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
    "ingest_long_context": {
        "instances": [
            (NUMA_NODE0[0], 8085, NUMA_NODE0[1]),
        ],
        "mlock": True,    # ~46 GB — latency-critical for ingest pipeline (Stage 1 of three_stage_summarization since 2026-05-06)
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
    # Qwen2.5-VL-7B Q4_K_M (~4 GB) — 24 threads
    "worker_vision": {
        "instances": [(NUMA_Q0B[0], 8086, 24)],
        "mlock": True,    # ~4 GB — minimal footprint
    },
    # Qwen3-VL-30B-A3B MoE (~17 GB) — 96 threads, pin to node1
    "vision_escalation": {
        "instances": [(NUMA_NODE1[0], 8087, 96)],
        "mlock": True,    # ~17 GB — fits in 1.13 TB budget
    },
}

# Roles that should use --mlock (requires ulimit -l unlimited in launch env)
MLOCK_ROLES = {role for role, cfg in NUMA_CONFIG.items() if cfg.get("mlock")}

# All NUMA replica ports (for port scanning and cleanup)
NUMA_REPLICA_PORTS = {
    port
    for cfg in NUMA_CONFIG.values()
    for _, port, _ in cfg["instances"]
    if port not in PORT_MAP.values()
}


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


# Roles that must never run concurrently (large/latency-sensitive paths).
# Note: frontdoor intentionally runs with 2 slots by default for better
# interactive responsiveness under concurrent traffic.
# 2026-05-06: removed architect_coding (role eliminated) + thinking_reasoning (role eliminated, no GGUF).
SERIAL_ROLES = {
    "coder_escalation",
    "worker_summarize",
    "architect_general",
    "ingest_long_context",
    "vision_escalation",
    "formalizer",
    "toolrunner",
}

# Servers to start (unique ports only)
# Pre-warm deployment (2026-03-29; updated 2026-05-06 stack swap):
# 1×96t full-speed + 4×48t quarter instances per role.
# Full-speed instances on 807x, quarter instances on 808x/818x/828x/838x.
#   frontdoor (37 GB Qwen3.6-35B-A3B Q8): 1×96t(8070) + 4×48t(8080-8380) = ~37 GB shared mmap
#   coder_escalation (shares frontdoor GGUF): 1×96t(8071) + 4×48t(8081-8381)
#     also hosts worker_summarize on port 8071 (same model as frontdoor since 2026-05-06)
#   worker_general (16 GB gemma4-26B-A4B Q4_K_M MTP): 1×96t(8072) + 4×48t(8082-8382) = ~16 GB (post 2026-05-08)
#   arch_gen (69 GB Qwen3.5-122B-A10B Q4): 1×96t(8083)  (Probe B 2026-05-04: 1× canonical wiring)
#   ingest (45 GB Qwen3-Next-80B-A3B Q4 SSM): 1×96t(8085)  (promoted to hot 2026-05-06: Stage 1 of three_stage_summarization)
# Total resident model footprint: ~167 GB (well under 1.13 TB host); 600+ GB free for KV caches + OS.
# architect_coding REMOVED 2026-05-06 (REAP-246B 70% coder < frontdoor 97%; 139 GB freed).
#
# =============================================================================
# Single source of truth for "which roles run where" (refactored 2026-05-06)
# =============================================================================
# Pre-2026-05-06: HOT_SERVERS + WARM_SERVERS were two parallel hand-edited lists
# of dicts that duplicated wiring data already in NUMA_CONFIG. This made it easy
# to forget one when adding/removing/renaming a role and ship a broken config.
#
# Post-2026-05-06: HOT_SERVERS / WARM_SERVERS are COMPUTED from:
#   1. ROLE_LAUNCH_META (below) — small classification dict: per-role tier +
#      launch mode + aliases + mode-specific kwargs.
#   2. NUMA_CONFIG (above) — wiring spec: per-role NUMA instances list.
#
# Adding a new role now requires editing TWO places consistently:
#   a) Add the role's NUMA wiring in NUMA_CONFIG (or set "no_numa": True in
#      ROLE_LAUNCH_META if the role doesn't need NUMA pinning, e.g. embedders).
#   b) Add a ROLE_LAUNCH_META entry with tier/mode/aliases.
# A consistency check (`_validate_role_classification()` below) catches
# common mismatches at module load time.

# Per-role launch metadata. Source of truth for tier classification + launch mode.
# Aliases (shared_with) are alternate role names that should resolve to the same
# server entries — used for --only filtering and routing fallthrough.
ROLE_LAUNCH_META: dict[str, dict] = {
    # ---- HOT tier (always started) ----
    "frontdoor":            {"tier": "hot",  "mode": "default"},
    "coder_escalation":     {"tier": "hot",  "mode": "default",
                             "shared_with_first_n": ["worker_summarize"]},  # worker_summarize co-hosts only on the full-speed instance (port 8071)
    "worker_general":       {"tier": "hot",  "mode": "worker_pool",
                             "worker_type": "explore",
                             # Aliases that share the worker_general process: worker_explore is the
                             # legacy name (pre-2026-03-19 worker pool design); worker_math + toolrunner
                             # share the GGUF mmap and process for routing fan-out.
                             "shared_with_first_n": ["worker_explore", "worker_math", "toolrunner"],
                             "shared_with_first_n_count": 2},  # aliases on full + first quarter
    "architect_general":    {"tier": "hot",  "mode": "default"},
    "ingest_long_context":  {"tier": "hot",  "mode": "default"},
    "worker_vision":        {"tier": "hot",  "mode": "vision", "vision_type": "worker"},
    "vision_escalation":    {"tier": "hot",  "mode": "vision", "vision_type": "escalation"},
    # Embedders — no NUMA pinning, fixed single port each
    "embedder":             {"tier": "hot",  "mode": "embedding", "no_numa": True, "port": 8090},
    "embedder_1":           {"tier": "hot",  "mode": "embedding", "no_numa": True, "port": 8091},
    "embedder_2":           {"tier": "hot",  "mode": "embedding", "no_numa": True, "port": 8092},
    "embedder_3":           {"tier": "hot",  "mode": "embedding", "no_numa": True, "port": 8093},
    "embedder_4":           {"tier": "hot",  "mode": "embedding", "no_numa": True, "port": 8094},
    "embedder_5":           {"tier": "hot",  "mode": "embedding", "no_numa": True, "port": 8095},
    # ---- WARM tier (optional, --include-warm) ----
    # 2026-05-06: worker_pool deprecated in registry; warm 1.5B worker retained as inert.
    "worker_fast":          {"tier": "warm", "mode": "worker_pool", "worker_type": "fast",
                             "no_numa": True, "port": 8102},
    # architect_coding REMOVED 2026-05-06 (REAP-246B role eliminated; 139 GB freed)
    # ingest_long_context PROMOTED to HOT 2026-05-06 (Stage 1 of three_stage_summarization)
    # thinking_reasoning REMOVED 2026-05-06 (GGUF deleted from disk 2026-03-06)
}


# Embedding model: BGE-large-en-v1.5 (purpose-built for embeddings, 1024 dims)
# 6 parallel instances provide redundancy and reduce latency via fan-out
EMBEDDING_MODEL_PATH = str(_PATHS["models_dir"] / "bge-large-en-v1.5-f16.gguf")
EMBEDDER_PORTS = [8090, 8091, 8092, 8093, 8094, 8095]

# Worker pool models (FIXED paths to existing files)
# NOTE: worker_coder uses the fast 1.5B worker backend on port 8102.
WORKER_POOL_MODELS = {
    # gemma4-26B-A4B Q4_K_M MTP — swapped 2026-05-08 from Qwen3-Coder-30B-A3B Q4_K_M.
    # +18pp on tool_compliance (96% vs 78%), +6pp on full suite (90% vs 84%),
    # +36% tps on tool_compliance (60.7 vs 44.7), 3× more concise output.
    # NB: requires ik_llama.cpp PR #1744 binary (Phase 2 wires runtime_requirements
    # through the launcher; until then this path is informational only — production
    # launcher still uses default LLAMA_SERVER and will fail on gemma4 arch).
    "explore": "/mnt/raid0/llm/models/gemma-4-26B-A4B-it-Q4_K_M.gguf",
    "fast": str(_PATHS["model_base"] / "QuantFactory/Qwen2.5-Coder-1.5B-GGUF/Qwen2.5-Coder-1.5B.Q4_K_M.gguf"),
}

# Draft model for MTP speculative decoding on explore worker.
# gemma4 assistant Q8 — in-house GGUF converted from google/gemma-4-26B-A4B-it-assistant
# safetensors (no community GGUF existed). 4-layer drafter, 58% acceptance at draft_max=2.
EXPLORE_DRAFT_MODEL = "/mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-Q8_0.gguf"

# Vision models (VL) with multimodal projector
VISION_WORKER_MODEL = str(_PATHS["model_base"] / "lmstudio-community/Qwen2.5-VL-7B-Instruct-GGUF/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf")
VISION_WORKER_MMPROJ = str(_PATHS["model_base"] / "lmstudio-community/Qwen2.5-VL-7B-Instruct-GGUF/mmproj-model-f16.gguf")
VISION_ESCALATION_MODEL = str(_PATHS["model_base"] / "lmstudio-community/Qwen3-VL-30B-A3B-Instruct-GGUF/Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf")
VISION_ESCALATION_MMPROJ = str(_PATHS["model_base"] / "lmstudio-community/Qwen3-VL-30B-A3B-Instruct-GGUF/mmproj-Qwen3-VL-30B-A3B-Instruct-F16.gguf")


def _filter_by_numa_mode(servers: list[dict], mode: str) -> list[dict]:
    """Filter server list by --numa-mode {full,quarter,both}.

    For roles whose NUMA_CONFIG has both `full_instance_idx` AND multiple
    instances (i.e., a full-NUMA-node instance + per-NUMA-quarter siblings on
    overlapping CPU sets), pick one mode:
      - "full":    keep only the full instance (max single-stream tps)
      - "quarter": skip the full, keep the quarters (max aggregate under load)
      - "both":    return input unchanged (CPU oversubscription — only useful
                   when the role's per-instance -t is light enough not to
                   over-subscribe; pre-2026-05-08 Qwen3-Coder -t 24 fit this,
                   gemma4 -t 96 does NOT — see launcher-numa-mode-gating
                   handoff)

    Roles without a full+quarter mix (single-instance roles like
    architect_general / frontdoor, or single-quarter roles like
    vision_escalation) pass through untouched.
    """
    if mode == "both":
        return servers
    out: list[dict] = []
    for srv in servers:
        # The primary role is roles[0]; aliases share its NUMA_CONFIG.
        role = srv["roles"][0]
        cfg = NUMA_CONFIG.get(role)
        if not cfg or "full_instance_idx" not in cfg or len(cfg["instances"]) <= 1:
            out.append(srv)
            continue
        full_idx = cfg["full_instance_idx"]
        srv_idx = srv.get("numa_instance", 0)
        if mode == "full" and srv_idx == full_idx:
            out.append(srv)
        elif mode == "quarter" and srv_idx != full_idx:
            out.append(srv)
    return out


def _build_servers_from_classification() -> tuple[list[dict], list[dict]]:
    """Compute HOT_SERVERS + WARM_SERVERS from ROLE_LAUNCH_META + NUMA_CONFIG.

    For each role in ROLE_LAUNCH_META:
      - If "no_numa": True, emit a single server entry at meta["port"].
      - Else look up NUMA_CONFIG[role]["instances"] and emit one entry per instance.
    Mode-specific flags (vision/worker_pool/embedding) get added to each entry,
    plus mode-specific kwargs (vision_type, worker_type).
    Aliases (shared_with_first_n) are added to the first N entries only.
    """
    hot: list[dict] = []
    warm: list[dict] = []

    for role, meta in ROLE_LAUNCH_META.items():
        target = hot if meta["tier"] == "hot" else warm
        mode = meta.get("mode", "default")
        aliases = meta.get("shared_with_first_n", [])
        alias_count = meta.get("shared_with_first_n_count", 1)  # default: aliases on first instance only

        # Mode-specific flag dict applied to every entry for this role
        mode_flags: dict = {}
        if mode == "worker_pool":
            mode_flags["worker_pool"] = True
            if "worker_type" in meta:
                mode_flags["worker_type"] = meta["worker_type"]
        elif mode == "vision":
            mode_flags["vision"] = True
            if "vision_type" in meta:
                mode_flags["vision_type"] = meta["vision_type"]
        elif mode == "embedding":
            mode_flags["embedding"] = True

        if meta.get("no_numa"):
            # Single port, no NUMA pinning (embedders, worker_fast, etc.)
            entry = {"port": meta["port"], "roles": [role]}
            entry.update(mode_flags)
            target.append(entry)
            continue

        # NUMA-pinned: one entry per instance in NUMA_CONFIG[role]["instances"]
        cfg = NUMA_CONFIG.get(role)
        if not cfg:
            raise ValueError(
                f"Role '{role}' in ROLE_LAUNCH_META requires NUMA_CONFIG entry "
                f"(or set 'no_numa': True if no pinning needed)"
            )
        for idx, instance in enumerate(cfg["instances"]):
            _cpus, port, _threads = instance
            roles_list: list[str] = [role]
            if idx < alias_count:
                roles_list.extend(aliases)
            entry = {"port": port, "roles": roles_list}
            if "full_instance_idx" in cfg or len(cfg["instances"]) > 1:
                entry["numa_instance"] = idx
            entry.update(mode_flags)
            target.append(entry)

    return hot, warm


def _validate_role_classification() -> None:
    """Sanity checks on ROLE_LAUNCH_META + NUMA_CONFIG agreement.

    Catches the common bugs that hand-edited HOT_SERVERS used to allow:
      - Role in ROLE_LAUNCH_META but missing NUMA_CONFIG (and no "no_numa" flag)
      - Role in NUMA_CONFIG but missing ROLE_LAUNCH_META classification
      - Same port assigned to two different role primaries
    Raises ValueError on any inconsistency.
    """
    errors: list[str] = []

    # Check 1: every NUMA_CONFIG role has a ROLE_LAUNCH_META entry
    for role in NUMA_CONFIG:
        if role not in ROLE_LAUNCH_META:
            errors.append(
                f"NUMA_CONFIG['{role}'] has no matching ROLE_LAUNCH_META entry — "
                f"add tier/mode classification or remove from NUMA_CONFIG"
            )

    # Check 2: every NUMA-pinned ROLE_LAUNCH_META role has NUMA_CONFIG instances
    for role, meta in ROLE_LAUNCH_META.items():
        if meta.get("no_numa"):
            if "port" not in meta:
                errors.append(f"ROLE_LAUNCH_META['{role}'] no_numa=True requires 'port' field")
        else:
            if role not in NUMA_CONFIG:
                errors.append(
                    f"ROLE_LAUNCH_META['{role}'] has no_numa=False but no NUMA_CONFIG entry"
                )

    # Check 3: no port collisions between role primaries
    primary_ports: dict[int, str] = {}
    for role, meta in ROLE_LAUNCH_META.items():
        if meta.get("no_numa"):
            ports = [meta["port"]]
        else:
            cfg = NUMA_CONFIG.get(role, {})
            ports = [inst[1] for inst in cfg.get("instances", [])]
        for port in ports:
            if port in primary_ports and primary_ports[port] != role:
                errors.append(
                    f"Port {port} assigned to both '{primary_ports[port]}' and '{role}'"
                )
            primary_ports[port] = role

    if errors:
        msg = "Role classification inconsistencies detected:\n  " + "\n  ".join(errors)
        raise ValueError(msg)


# Validate at module load (fails fast on misconfiguration)
_validate_role_classification()

# Computed server lists. Single source of truth: ROLE_LAUNCH_META + NUMA_CONFIG.
HOT_SERVERS, WARM_SERVERS = _build_servers_from_classification()


def validate_against_registry(registry_yaml_path: str | None = None) -> list[str]:
    """Cross-check ROLE_LAUNCH_META against orchestration/model_registry.yaml.

    Returns list of warning strings (empty if everything is consistent).
    Called from `start` command but non-fatal: prints warnings, does not abort.
    Useful for catching drift between launcher classification and registry's
    process_layout / server_mode sections.
    """
    if registry_yaml_path is None:
        # Default: orchestration/model_registry.yaml relative to repo root
        registry_yaml_path = str(Path(__file__).parent.parent.parent / "orchestration" / "model_registry.yaml")
    warnings: list[str] = []
    try:
        import yaml
        with open(registry_yaml_path) as f:
            registry = yaml.safe_load(f)
    except Exception as e:
        return [f"could not load registry from {registry_yaml_path}: {e}"]

    pl = registry.get("process_layout", {})
    reg_hot = set(pl.get("hot_resident", []))
    reg_warm = set(pl.get("warm_mmap", []))

    # Compute launcher's HOT tier including aliases (shared_with_first_n).
    # Aliases share the same process as their primary, so for tier comparison
    # they count as HOT too.
    launcher_hot: set[str] = set()
    launcher_warm: set[str] = set()
    for role, meta in ROLE_LAUNCH_META.items():
        target = launcher_hot if meta["tier"] == "hot" else launcher_warm
        target.add(role)
        for alias in meta.get("shared_with_first_n", []):
            target.add(alias)

    # Production roles to cross-check — skip pure launcher-internal names.
    # worker_explore is a legacy alias kept for backwards-compat; embedders
    # aren't in the registry's process_layout (they're infrastructure roles);
    # worker_fast is the deprecated worker_pool warm tier.
    skip = {"worker_explore"} | {f"embedder_{i}" for i in range(6)} | {"embedder", "worker_fast"}
    launcher_hot_filtered = launcher_hot - skip

    only_in_launcher = launcher_hot_filtered - reg_hot - reg_warm
    only_in_registry_hot = reg_hot - launcher_hot - launcher_warm

    for r in sorted(only_in_launcher):
        warnings.append(f"role '{r}' is HOT in launcher but absent from registry process_layout")
    for r in sorted(only_in_registry_hot):
        # Roles that the registry says should be hot but launcher doesn't classify hot
        warnings.append(f"role '{r}' is hot_resident in registry but not in launcher's HOT tier")

    # Cross-check NUMA_CONFIG ports vs registry server_mode ports (where present)
    sm = registry.get("server_mode", {})
    for role, srv in sm.items():
        if not isinstance(srv, dict): continue
        reg_port = srv.get("port")
        if reg_port is None: continue
        cfg = NUMA_CONFIG.get(role)
        if cfg:
            launcher_ports = [inst[1] for inst in cfg["instances"]]
            if reg_port not in launcher_ports:
                warnings.append(
                    f"role '{role}': registry server_mode says port {reg_port}, "
                    f"but launcher NUMA_CONFIG ports are {launcher_ports}"
                )
        else:
            meta = ROLE_LAUNCH_META.get(role)
            if meta and meta.get("no_numa") and meta.get("port") != reg_port:
                warnings.append(
                    f"role '{role}': registry server_mode says port {reg_port}, "
                    f"but launcher ROLE_LAUNCH_META says port {meta.get('port')}"
                )
    return warnings

DEV_MODEL = "Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf"
DEV_MODEL_PATH = str(_PATHS["models_dir"] / DEV_MODEL)

# Optional orchestrator API launch profiles for repeatable debugging runs.
ORCHESTRATOR_PROFILES: dict[str, dict[str, str]] = {
    "contention-debug": {
        "ORCHESTRATOR_UVICORN_WORKERS": "6",
        "ORCHESTRATOR_FRONTDOOR_TRACE": "1",
        "ORCHESTRATOR_DELEGATION_TRACE": "1",
        "ORCHESTRATOR_DELEGATION_TOTAL_MAX_SECONDS": "55",
        "ORCHESTRATOR_DELEGATION_SPECIALIST_MAX_SECONDS": "25",
        "ORCHESTRATOR_INFERENCE_LOCK_TIMEOUT_EXCLUSIVE_S": "45",
        "ORCHESTRATOR_INFERENCE_LOCK_TIMEOUT_SHARED_S": "45",
    },
}

# =============================================================================
# Docker Services (NextPLAID multi-vector retrieval + SearXNG metasearch)
# =============================================================================

DOCKER_SERVICES = [
    {
        "name": "nextplaid-code",
        "port": 8088,
        "image": "ghcr.io/lightonai/next-plaid:cpu-1.0.4",
        "model": "lightonai/LateOn-Code",
        "description": "Multi-vector code retrieval (ColBERT)",
        # Separate index subdir to avoid cross-contamination (different models = incompatible embeddings)
        "volumes": [
            f"{_PATHS['project_root']}/cache/next-plaid/code-indices:/data/indices",
            f"{_PATHS['cache_dir']}/huggingface:/root/.cache/huggingface",
        ],
        "args": ["--host", "0.0.0.0", "--port", "8080", "--index-dir", "/data/indices",
                 "--model", "lightonai/LateOn-Code", "--int8"],
    },
    {
        "name": "nextplaid-docs",
        "port": 8089,
        "image": "ghcr.io/lightonai/next-plaid:cpu-1.0.4",
        "model": "/mnt/raid0/llm/models/gte-moderncolbert-v1-onnx",
        "description": "Multi-vector doc retrieval (ColBERT)",
        "volumes": [
            f"{_PATHS['project_root']}/cache/next-plaid/docs-indices:/data/indices",
            f"{_PATHS['cache_dir']}/huggingface:/root/.cache/huggingface",
            "/mnt/raid0/llm/models/gte-moderncolbert-v1-onnx:/models/gte-moderncolbert-v1-onnx:ro",
        ],
        "args": ["--host", "0.0.0.0", "--port", "8080", "--index-dir", "/data/indices",
                 "--model", "/models/gte-moderncolbert-v1-onnx", "--int8"],
    },
    {
        "name": "searxng",
        "port": 8090,
        "image": "docker.io/searxng/searxng:latest",
        "description": "Metasearch aggregator (JSON API for web_search)",
        "volumes": [
            f"{_PATHS['project_root']}/config/searxng:/etc/searxng:Z",
        ],
        "args": [],  # Config via mounted settings.yml, not CLI args
        "health_path": "/",  # SearXNG serves HTML on /, not /health
    },
]


# =============================================================================
# Model Path Validation
# =============================================================================


def validate_model_paths() -> list[str]:
    """Validate all model paths exist. Returns list of errors.

    This prevents hallucinations about missing models by failing fast
    with clear error messages showing exactly what's missing.
    """
    errors = []

    # HOT tier models
    if not Path(EMBEDDING_MODEL_PATH).exists():
        errors.append(f"[HOT] Embedding: {EMBEDDING_MODEL_PATH}")

    for worker_type, path in WORKER_POOL_MODELS.items():
        if not Path(path).exists():
            errors.append(f"[HOT] Worker '{worker_type}': {path}")

    # Draft model for explore worker spec decode
    if not Path(EXPLORE_DRAFT_MODEL).exists():
        errors.append(f"[HOT] Explore draft: {EXPLORE_DRAFT_MODEL}")

    # Frontdoor model (swapped to Qwen3.6-35B-A3B Q8 2026-05-04; same file shared by
    # coder_escalation + worker_summarize via mmap)
    frontdoor_model = "/mnt/raid0/llm/models/Qwen_Qwen3.6-35B-A3B-Q8_0.gguf"
    if not Path(frontdoor_model).exists():
        errors.append(f"[HOT] frontdoor: {frontdoor_model}")

    # Architect/ingest models
    # 2026-05-06: architect_coding REMOVED (REAP-246B role eliminated; 139 GB freed).
    architect_models = [
        ("architect_general", str(_PATHS["model_base"] / "unsloth/Qwen3.5-122B-A10B-GGUF/")),  # swapped 2026-03-19
        ("ingest_long_context", str(_PATHS["model_base"] / "lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF/")),
    ]
    for role, path in architect_models:
        if not Path(path).exists():
            errors.append(f"[HOT] {role}: {path}")

    # Vision models (VL with multimodal projector)
    for label, path in [
        ("worker_vision model", VISION_WORKER_MODEL),
        ("worker_vision mmproj", VISION_WORKER_MMPROJ),
        ("vision_escalation model", VISION_ESCALATION_MODEL),
        ("vision_escalation mmproj", VISION_ESCALATION_MMPROJ),
    ]:
        if not Path(path).exists():
            errors.append(f"[HOT] {label}: {path}")

    # Auxiliary services
    formalizer = _PATHS["models_dir"] / "LightOnOCR-2-1B-bbox-Q4_K_M.gguf"
    if not formalizer.exists():
        errors.append(f"[AUX] document_formalizer: {formalizer}")

    # Tool registry (required for deterministic tools)
    tool_registry = _PATHS["project_root"] / "orchestration/tool_registry.yaml"
    if not tool_registry.exists():
        errors.append(f"[TOOL] tool_registry.yaml: {tool_registry}")

    # C++ math tools (optional - warn but don't fail)
    cpp_math_tools = _PATHS["llama_cpp_bin"] / "llama-math-tools"
    if not cpp_math_tools.exists():
        # This is a warning, not an error - append with different prefix
        pass  # Will be checked separately in init_memrl_and_tools

    return errors


# =============================================================================
# State Management
# =============================================================================


@dataclass
class ProcessInfo:
    """Information about a running process."""
    role: str
    pid: int
    port: int
    started_at: str
    model_path: str
    log_file: str


def load_state() -> dict[str, ProcessInfo]:
    """Load state from file."""
    if not STATE_FILE.exists():
        return {}
    try:
        with open(STATE_FILE) as f:
            data = json.load(f)
        return {k: ProcessInfo(**v) for k, v in data.items()}
    except (json.JSONDecodeError, TypeError):
        return {}


def save_state(state: dict[str, ProcessInfo]) -> None:
    """Save state to file."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    serializable: dict[str, dict[str, Any]] = {}
    for key, value in state.items():
        if isinstance(value, ProcessInfo):
            serializable[key] = asdict(value)
            continue
        # Backward-compatible fallback: preserve minimally-typed dict records.
        if isinstance(value, dict):
            serializable[key] = dict(value)
            continue
        # Unknown record type; skip instead of crashing startup.
        continue
    with open(STATE_FILE, "w") as f:
        json.dump(serializable, f, indent=2)


# =============================================================================
# Process Management
# =============================================================================


def check_free_memory() -> int:
    """Return free memory in GB."""
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                kb = int(line.split()[1])
                return kb // (1024 * 1024)
    return 0


# =============================================================================
# Host prerequisites — applied at session start before any llama-server launch.
# Source: handoffs/active/cpu-kernel-env-flags-inventory.md §211 +
#         handoffs/active/model-registry-v5-deployment-draft.yaml host_prerequisites.
# Single source of truth for canonical inference host state.
# =============================================================================

# Required sysctl values. /etc/sysctl.d/99-epyc-inference.conf gives the boot-time
# default; these are re-verified per session per feedback_numa_balancing_self_reset.
_HOST_PREREQ_SYSCTLS = {
    "kernel.numa_balancing": "0",
    "kernel.perf_event_paranoid": "1",
}

# Required THP state — both enabled and defrag must read "always".
_HOST_PREREQ_THP = {
    "/sys/kernel/mm/transparent_hugepage/enabled": "always",
    "/sys/kernel/mm/transparent_hugepage/defrag": "always",
}

# Required CPU governor.
_HOST_PREREQ_GOVERNOR = "performance"


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
    "worker": {
        "GGML_CCD_POOLS": "1",
        "GGML_CCD_WORK_DIST": "1",
        "GGML_BARRIER_LOCAL_BETWEEN_OPS": "1",
    },
    # MoE Q8 BW-bound frontdoor — EP stack (+17% honest baseline g.1 = drone+shard, N=2).
    "frontdoor": {
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
    "architect_general": {
        "GGML_NUMA_REPACK_INTERLEAVE": "0",
    },
    # Hybrid SSM dense (Nemotron-9B-v2-class) — c3 = CPU1 stack + mbind off.
    # Activate when a hybrid_ssm_dense model is rostered.
    "hybrid_ssm_dense": {
        "GGML_CCD_POOLS": "1",
        "GGML_CCD_WORK_DIST": "1",
        "GGML_BARRIER_LOCAL_BETWEEN_OPS": "1",
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


def _role_env_overrides(role: str) -> dict[str, str]:
    """Return per-role env block for a given role. Empty dict if role not registered.
    Falls back through arch-class aliases (e.g. coder_escalation → worker)."""
    if role in _ROLE_ENV_BLOCKS:
        return dict(_ROLE_ENV_BLOCKS[role])
    # Aliases — production roles that map to v5 arch_class names.
    # 2026-05-06: coder_escalation + worker_summarize now use the SAME GGUF as frontdoor
    # (Qwen3.6-35B-A3B Q8) and should inherit frontdoor's EP-stack env block.
    # 2026-05-06: thinking_reasoning alias REMOVED (role eliminated).
    # NB: ingest_long_context (Qwen3-Next-80B-A3B hybrid SSM MoE) routes to hybrid_ssm_moe
    # (default v5 — MoE-Spec budget=40 was REAP-246B-specific, NOT validated on hybrid SSM).
    # formalizer (MathSmith-Qwen3-8B Q8 dense) routes to dense_q8 — it's not MoE at all.
    arch_aliases = {
        "coder_escalation": "frontdoor",   # Qwen3.6-35B-A3B Q8 (same model as frontdoor since 2026-05-06 swap)
        "worker_summarize": "frontdoor",   # Qwen3.6-35B-A3B Q8 (same model as frontdoor since 2026-05-06 swap)
        "worker_general": "worker",         # gemma4-26B-A4B Q4_K_M MTP — GGML_* env stripped at launch when binary_override is in effect (ik_llama.cpp PR #1744 forked at different ggml commit)
        "worker_explore": "worker",         # Legacy alias for worker_general
        "general_gemma_3_27b_it_qat": "dense_q4",
        "ingest_long_context": "hybrid_ssm_moe",  # Qwen3-Next-80B-A3B
        "formalizer": "dense_q8",                 # MathSmith-Qwen3-8B Q8 dense; NOT MoE at all
        "toolrunner": "worker",                   # gemma4-26B-A4B Q4_K_M MTP (shares with worker_general)
    }
    aliased = arch_aliases.get(role)
    if aliased and aliased in _ROLE_ENV_BLOCKS:
        return dict(_ROLE_ENV_BLOCKS[aliased])
    return {}


def build_launch_env(role: str, base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Compose the full env dict for a llama-server launch.

    Order (later overrides earlier):
        1. base_env (parent process env, typically os.environ.copy())
        2. LLVM-20 libomp prepended to LD_LIBRARY_PATH (canonical recipe)
        3. canonical OMP env stack (always applied)
        4. per-role GGML_* env block (from v5 deployment draft)

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
    env.update(_CANONICAL_OMP_ENV)
    env.update(_role_env_overrides(role))
    return env


def _runtime_requirements_for_role(
    registry: "RegistryLoader", role_name: str
) -> tuple[str | None, list[str] | None]:
    """Return (binary_dir, ld_library_paths) for a role from server_mode.<x>.runtime_requirements.

    Walks `registry._raw["server_mode"]` looking for the entry whose `model_role`
    matches `role_name`. Returns (None, None) when no entry has runtime_requirements
    or when the role isn't found — caller falls back to default LLAMA_SERVER + the
    canonical env without LD_LIBRARY_PATH overrides.

    Used by the worker_pool launch branch (currently only worker_general / gemma4
    MTP via ik_llama.cpp PR #1744). Other workers stay on the default binary.
    """
    if not registry or not hasattr(registry, "_raw"):
        return None, None
    sm = registry._raw.get("server_mode", {}) or {}
    for entry in sm.values():
        if not isinstance(entry, dict):
            continue
        if entry.get("model_role") != role_name:
            continue
        rt = entry.get("runtime_requirements") or {}
        return rt.get("binary_dir"), rt.get("ld_library_path")
    return None, None


def _read_sysctl(key: str) -> str | None:
    path = "/proc/sys/" + key.replace(".", "/")
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _read_thp_active(path: str) -> str | None:
    # /sys/kernel/mm/transparent_hugepage/enabled has format e.g. "always [madvise] never"
    # The bracketed token is the active value.
    try:
        with open(path) as f:
            content = f.read().strip()
    except OSError:
        return None
    for token in content.split():
        if token.startswith("[") and token.endswith("]"):
            return token[1:-1]
    return content


def _read_governor() -> str | None:
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") as f:
            return f.read().strip()
    except OSError:
        return None


def check_host_prerequisites() -> tuple[bool, list[str]]:
    """Audit canonical host state. Returns (all_pass, list_of_drift_messages)."""
    drift: list[str] = []

    for key, want in _HOST_PREREQ_SYSCTLS.items():
        got = _read_sysctl(key)
        if got != want:
            drift.append(f"sysctl {key}={got} (want {want})")

    for path, want in _HOST_PREREQ_THP.items():
        got = _read_thp_active(path)
        if got != want:
            drift.append(f"{path} active={got} (want {want})")

    gov = _read_governor()
    if gov != _HOST_PREREQ_GOVERNOR:
        drift.append(f"cpu0 scaling_governor={gov} (want {_HOST_PREREQ_GOVERNOR})")

    return (len(drift) == 0, drift)


def apply_host_prerequisites(auto_fix: bool = True) -> bool:
    """Verify and (optionally) apply canonical host settings.

    Returns True if host is canonical (or was successfully fixed). Returns False
    if any prereq could not be applied — caller should refuse to launch.
    """
    print("[host_prereq] Auditing canonical host state...")
    ok, drift = check_host_prerequisites()
    if ok:
        print("  [OK] All host prerequisites satisfied "
              "(numa_balancing=0, THP=always, governor=performance, perf_paranoid=1)")
        return True

    print(f"  [DRIFT] {len(drift)} setting(s) need correction:")
    for msg in drift:
        print(f"    - {msg}")

    if not auto_fix:
        print("  [SKIP] auto_fix disabled. Pass --apply-host-prereqs or fix manually.")
        return False

    print("  [FIX] Applying canonical settings (sudo -n)...")
    ok_fix = True
    for key, val in _HOST_PREREQ_SYSCTLS.items():
        if _read_sysctl(key) == val:
            continue
        try:
            subprocess.run(["sudo", "-n", "sysctl", "-w", f"{key}={val}"],
                           check=True, capture_output=True, text=True, timeout=5)
            print(f"    ✓ sysctl {key}={val}")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
            print(f"    ✗ FAILED to set sysctl {key}: {exc}")
            ok_fix = False

    for path, val in _HOST_PREREQ_THP.items():
        if _read_thp_active(path) == val:
            continue
        try:
            # tee with sudo: echo "always" | sudo tee /sys/kernel/...
            proc = subprocess.run(
                ["sudo", "-n", "tee", path],
                input=val + "\n", check=True, capture_output=True, text=True, timeout=5,
            )
            print(f"    ✓ {path} = {val}")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
            print(f"    ✗ FAILED to set {path}: {exc}")
            ok_fix = False

    if _read_governor() != _HOST_PREREQ_GOVERNOR:
        try:
            subprocess.run(
                ["sudo", "-n", "cpupower", "frequency-set", "-g", _HOST_PREREQ_GOVERNOR],
                check=True, capture_output=True, text=True, timeout=10,
            )
            print(f"    ✓ cpu governor = {_HOST_PREREQ_GOVERNOR}")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
            print(f"    ✗ FAILED to set governor: {exc}")
            ok_fix = False

    # Re-audit
    ok, drift_after = check_host_prerequisites()
    if ok:
        print("  [OK] All host prerequisites now satisfied after fix")
        return True

    print(f"  [FAIL] {len(drift_after)} setting(s) STILL drifted after fix attempt:")
    for msg in drift_after:
        print(f"    - {msg}")
    return False


def is_port_in_use(port: int) -> bool:
    """Check if a port is in use."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def _pids_on_port(port: int) -> list[int]:
    """Best-effort discovery of LISTEN pids on a TCP port."""
    try:
        result = subprocess.run(
            ["lsof", "-t", f"-i:{port}"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        pids: list[int] = []
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                pids.append(int(line))
            except ValueError:
                continue
        return pids
    except Exception:
        return []


def _pid_alive(pid: int) -> bool:
    """Return True when a pid currently exists."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _child_pids(pid: int) -> list[int]:
    """Return direct child pids for a process."""
    try:
        result = subprocess.run(
            ["ps", "-o", "pid=", "--ppid", str(pid)],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        return []

    children: list[int] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            children.append(int(line))
        except ValueError:
            continue
    return children


def _collect_descendants(root_pid: int) -> list[int]:
    """Collect all descendants of root_pid (breadth-first)."""
    descendants: list[int] = []
    queue = [root_pid]
    seen = {root_pid}
    while queue:
        parent = queue.pop(0)
        for child in _child_pids(parent):
            if child in seen:
                continue
            seen.add(child)
            descendants.append(child)
            queue.append(child)
    return descendants


def kill_process(pid: int, timeout: int = 5) -> bool:
    """Kill a process tree gracefully, then forcefully."""
    if pid <= 0:
        return True

    this_pid = os.getpid()
    targets = [p for p in (_collect_descendants(pid) + [pid]) if p > 0 and p != this_pid]
    if not targets:
        return True

    try:
        # Terminate children first, then parent.
        for target in reversed(targets):
            try:
                os.kill(target, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except PermissionError:
                print(f"  [!] Permission denied killing PID {target}")
        for _ in range(timeout):
            time.sleep(1)
            if not any(_pid_alive(target) for target in targets):
                return True
        # Force kill survivors.
        for target in reversed(targets):
            if not _pid_alive(target):
                continue
            try:
                os.kill(target, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError:
                print(f"  [!] Permission denied force-killing PID {target}")
        time.sleep(1)
        return not any(_pid_alive(target) for target in targets)
    except Exception as exc:
        print(f"  [!] Failed to kill PID {pid}: {exc}")
        return False


# =============================================================================
# Docker Container Management (NextPLAID services)
# =============================================================================


def _docker_available() -> bool:
    """Check if docker CLI is available."""
    try:
        result = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def docker_container_running(name: str) -> bool:
    """Check if a named Docker container is running."""
    try:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", name],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() == "true"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def start_docker_container(service: dict) -> ProcessInfo | None:
    """Start a Docker service. Removes any existing container with the same name first."""
    name = service["name"]
    port = service["port"]

    # Remove existing container (stopped or running) with same name
    subprocess.run(
        ["docker", "rm", "-f", name],
        capture_output=True, timeout=10,
    )

    cmd = ["docker", "run", "-d", "--name", name, "-p", f"{port}:8080"]
    for vol in service.get("volumes", []):
        cmd.extend(["-v", vol])
    cmd.append(service["image"])
    cmd.extend(service.get("args", []))

    print(f"  Starting {name} on port {port}...")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        print(f"    [FAIL] docker run failed: {result.stderr.strip()[:200]}")
        return None

    container_id = result.stdout.strip()[:12]
    print(f"    Container: {container_id}")

    # Wait for health
    health_path = service.get("health_path", "/health")
    print(f"    Waiting for health...")
    if wait_for_health(port, timeout=60, path=health_path):
        print(f"    [OK] {name} ready ({service['description']})")
        # Use container_id as PID placeholder (Docker manages the actual process)
        return ProcessInfo(
            role=name,
            pid=-1,  # Docker-managed, not a host PID
            port=port,
            started_at=datetime.now().isoformat(),
            model_path=service.get("model", service["image"]),
            log_file=f"docker logs {name}",
        )
    else:
        print(f"    [FAIL] {name} health check timed out")
        # Show last few log lines for debugging
        logs = subprocess.run(
            ["docker", "logs", "--tail", "10", name],
            capture_output=True, text=True, timeout=5,
        )
        if logs.stdout:
            print(f"    Last logs: {logs.stdout.strip()[:300]}")
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=5)
        return None


def stop_docker_container(name: str) -> bool:
    """Stop and remove a named Docker container."""
    result = subprocess.run(
        ["docker", "rm", "-f", name],
        capture_output=True, text=True, timeout=15,
    )
    return result.returncode == 0


def wait_for_health(port: int, timeout: int = _HEALTH_SERVER_STARTUP, path: str = "/health") -> bool:
    """Wait for server health endpoint."""
    import urllib.request
    import urllib.error

    url = f"http://localhost:{port}{path}"
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, ConnectionResetError, OSError):
            pass
        time.sleep(2)
    return False


# =============================================================================
# Server Launching
# =============================================================================


def build_server_command(
    role_config: Any,
    port: int,
    dev_mode: bool = False,
    embedding_mode: bool = False,
    worker_pool_mode: bool = False,
    worker_type: str = None,
    vision_mode: bool = False,
    vision_type: str = None,
    binary_override: str | None = None,
) -> list[str]:
    """Build llama-server command from role config.

    `binary_override` (Phase 2): when set, replaces `LLAMA_SERVER` for the worker_pool
    explore branch. Used by worker_general (gemma4 MTP) to launch ik_llama.cpp PR #1744
    instead of the production llama.cpp build. Other branches ignore this argument
    today — extend as needed.
    """
    # Vision server mode - VL models with multimodal projector
    if vision_mode:
        if vision_type == "escalation":
            # Qwen3-VL-30B MoE - larger model, expert reduction
            return [
                str(LLAMA_SERVER),
                "-m", VISION_ESCALATION_MODEL,
                "--mmproj", VISION_ESCALATION_MMPROJ,
                "--override-kv", "qwen3vlmoe.expert_used_count=int:4",
                "--host", "127.0.0.1",
                "--port", str(port),
                "-np", "1",
                "-c", "16384",
                "-t", "96",
                "--flash-attn", "on",
            ]
        else:
            # Qwen2.5-VL-7B - smaller worker model
            return [
                str(LLAMA_SERVER),
                "-m", VISION_WORKER_MODEL,
                "--mmproj", VISION_WORKER_MMPROJ,
                "--host", "127.0.0.1",
                "--port", str(port),
                "-np", "2",
                "-c", "8192",
                "-t", "24",
                "--flash-attn", "on",
            ]

    # Embedding server mode - BGE-large with CLS pooling
    if embedding_mode:
        return [
            str(LLAMA_SERVER),
            "-m", EMBEDDING_MODEL_PATH,
            "--host", "127.0.0.1",
            "--port", str(port),
            "-np", "4",  # 4 parallel slots for embedding requests
            "-c", "512",  # BGE works with short contexts
            "-t", "4",  # 4 threads per instance (6 instances = 24 threads total)
            "--embeddings",  # Enable embedding endpoint
            "--pooling", "cls",  # BGE uses CLS token pooling (standard BERT)
            "--flash-attn", "on",
        ]

    # Worker pool mode - heterogeneous workers with specific configs
    if worker_pool_mode and worker_type:
        model_path = WORKER_POOL_MODELS.get(worker_type)
        if not model_path:
            raise ValueError(f"Unknown worker type: {worker_type}")

        # Worker-type specific configuration
        if worker_type == "fast":
            # Fast worker: 1.5B model, 4 slots for parallel burst capacity
            return [
                str(LLAMA_SERVER),
                "-m", model_path,
                "--host", "127.0.0.1",
                "--port", str(port),
                "-np", "4",  # 4 parallel slots (consolidated from 2×2)
                "-c", "16384",  # 4K per slot
                "-t", "16",  # 16 threads for small model
                "--flash-attn", "on",
            ]
        else:
            # explore worker: gemma4-26B-A4B Q4_K_M MTP (swapped 2026-05-08 from
            # Qwen3-Coder-30B-A3B Q4_K_M). Tool_compliance 96% vs 78% prior, +36% tps.
            # Binary defaults to str(LLAMA_SERVER); start_server overrides via
            # binary_override (from server_mode.worker.runtime_requirements.binary_dir)
            # for gemma4 MTP because it needs ik_llama.cpp PR #1744 build, not the
            # production llama.cpp build.
            binary = binary_override if binary_override else str(LLAMA_SERVER)
            # Derive thread count from NUMA_CONFIG by matching this instance's port.
            # Pre-2026-05-08 was hardcoded -t 24 (1.5B-era leftover), then briefly -t 96
            # which over-subscribed the 4 quarter instances (24 cores each) and crashed
            # the load average to 420. Per-instance lookup gets full=96 + quarters=48.
            _numa_thread_count = 96  # default fallback (full canonical instance)
            for cpu_list, inst_port, threads in NUMA_CONFIG.get("worker_general", {}).get("instances", []):
                if inst_port == port:
                    _numa_thread_count = threads
                    break
            return [
                binary,
                "-m", model_path,
                "-md", EXPLORE_DRAFT_MODEL,  # MTP draft (gemma4 assistant Q8)
                "--spec-type", "mtp",        # CRITICAL: engages ik_llama.cpp PR #1744 MTP code path.
                                             # Without this, -md is treated as standard spec decode and
                                             # MTP-arch draft tensors are loaded but never assigned to a
                                             # backend buffer → "tensor buffer not set" assertion.
                "--draft-max", "2",          # MTP recipe: 58% acceptance at k=2 (research-registry tuning)
                "--draft-p-min", "0.0",      # greedy: accept top-1 drafts, verifier rejects mismatches
                "--threads-draft", "16",     # dedicate 16 threads to small 4-layer drafter
                "-ub", "512",                # MTP override of canonical -ub 8192 (per gemma4 deep-dive)
                "--no-mmap",                 # canonical recipe: bulk-read on EPYC NUMA cold-cache decode
                "--reasoning", "off",        # disable gemma4 thinking-channel (output otherwise lands in
                                             # reasoning_content not content; registry: gemma4_26b reasoning=off)
                "--jinja",                   # gemma4 ships a custom chat template embedded in the gguf;
                                             # without --jinja, llama.cpp rejects /v1/chat/completions
                                             # with "this custom template is not supported"
                "--host", "127.0.0.1",
                "--port", str(port),
                # -np 1 (single slot): MTP shares state with the target across slots in a
                # way that the ik_llama.cpp PR #1744 build asserts on with -np 2 ("tensor
                # buffer not set" at ggml-backend.cpp:236 during inference). Single slot
                # matches the working benchmark recipe. Pre-gemma4 worker_general used
                # -np 2 because external-draft spec decode (Qwen3-Coder + 0.75B draft)
                # had per-slot draft state; MTP fuses draft + target, hence -np 1.
                "-np", "1",
                "-c", "16384",  # match research-registry max_context; 8192 causes MTP buffer mismatches
                # Per-instance thread count (full=96, quarters=48). Pre-2026-05-08 was
                # hardcoded -t 24 (Qwen3-Coder tolerated it); gemma4 + MTP under
                # ik_llama.cpp PR #1744 must match the bench recipe to avoid the
                # "tensor buffer not set" MTP assertion.
                "-t", str(_numa_thread_count),
                # KV cache q8_0/q8_0 — registry-declared and required for stable MTP buffer
                # allocation. f16 default left some MTP tensor buffers uninitialized.
                "-ctk", "q8_0",
                "-ctv", "q8_0",
                "--flash-attn", "on",
            ]

    if dev_mode:
        return [
            str(LLAMA_SERVER),
            "-m", DEV_MODEL_PATH,
            "--host", "127.0.0.1",
            "--port", str(port),
            "-np", "4",
            "-c", "4096",
            "-t", "16",
            "--flash-attn", "on",  # Flash attention
        ]

    model_path = role_config.model.full_path
    accel = role_config.acceleration
    parallel_slots = "1" if role_config.name in SERIAL_ROLES else "2"

    # NUMA-aware thread count: use the configured thread count for the
    # specific instance, falling back to 96 (single NUMA node).
    numa_cfg = NUMA_CONFIG.get(role_config.name)
    if numa_cfg and numa_cfg["instances"]:
        # Default to first instance thread count (all instances same for a role)
        thread_count = str(numa_cfg["instances"][0][2])
    else:
        thread_count = "96"

    # KV cache budgets: role-aware context sizes to prevent memory pressure.
    # 2026-05-06: architect_coding REMOVED (role eliminated; entry stripped).
    _KV_CONTEXT_SIZES = {
        "architect_general": "16384",   # 122B MoE hybrid → ~16GB KV
        "ingest_long_context": "32768", # 80B SSM, needs long context (Stage 1 of three_stage_summarization)
    }
    context_size = _KV_CONTEXT_SIZES.get(role_config.name, "32768")

    # Use v2 binary for roles with v3 spec decode bug (Qwen2.5 architecture)
    _binary = LLAMA_SERVER_V2 if role_config.name in _V2_ROLES and LLAMA_SERVER_V2.exists() else LLAMA_SERVER
    cmd = [
        str(_binary),
        "-m", model_path,
        "--host", "127.0.0.1",
        "--port", str(port),
        "-np", parallel_slots,  # Parallel slots (1 for large roles, 2 otherwise)
        "-c", context_size,     # Role-aware context size
        "-t", thread_count,     # NUMA-aware thread count (48 for quarter, 96 for node)
        "--flash-attn", "on",   # Flash attention
    ]

    # --jinja: Use model's native chat template (enables thinking on Qwen3/3.5).
    # SKIP for architect_general — Qwen3.5 hybrids enter infinite <think> loops.
    # --reasoning off was insufficient: the jinja template itself primes the model
    # into think mode. Without --jinja, llama-server falls back to generic ChatML
    # which has no thinking scaffolding. architect_coding (REAP-246B, pure MoE)
    # keeps --jinja + default reasoning (no loop issue on non-hybrid architectures).
    if role_config.name != "architect_general":
        cmd.append("--jinja")

    # KV cache quantization: reduces KV memory with negligible quality impact.
    # Phase 0 benchmarks (2026-03-25): generation speed neutral, memory savings significant at 65K+.
    # CRITICAL (2026-03-28): V=q4_0 causes 71% prefill regression on pure-attention models.
    # V=f16 has ZERO prefill regression (actually 1% faster due to K bandwidth savings).
    # q4_0 K / f16 V = quality-neutral (PPL +0.017 with Hadamard), 37% KV savings, zero speed cost.
    # q4_0 / q4_0 = 71% KV savings but 71% prefill regression on pure-attn. OK for hybrid (SSM amortizes).
    # --kv-hadamard: production binary rebuilt with Hadamard support (commit b51c905ec, 2026-03-28).
    # Closes q4_0 K PPL gap from +0.055 to +0.017 vs f16. Zero throughput overhead.
    _KV_QUANT_CONFIGS = {
        # 2026-05-06: frontdoor + coder_escalation now share Qwen3.6-35B-A3B Q8 GGUF
        # (qwen35moe MoE-attention, NOT SSM hybrid). Per registry kv_quant {q8_0/q8_0}.
        "frontdoor":            ("q8_0", "q8_0"),   # Qwen3.6-35B-A3B Q8: q8_0 K/V (Qwen trained bf16)
        "coder_escalation":     ("q8_0", "q8_0"),   # same model as frontdoor
        "architect_general":    ("q4_0", "f16"),    # pure attention: q4_0 K (4x), f16 V (zero prefill cost)
        # architect_coding REMOVED 2026-05-06 (REAP-246B role eliminated)
        "ingest_long_context":  ("q4_0", "q4_0"),   # SSM-hybrid, long context, max compression
    }
    kv_quant = _KV_QUANT_CONFIGS.get(role_config.name)
    if kv_quant:
        cmd.extend(["-ctk", kv_quant[0], "-ctv", kv_quant[1]])
        # --kv-hadamard: v3 auto-enables (upstream #21038), v2 needs explicit flag
        if role_config.name in _V2_ROLES and LLAMA_SERVER_V2.exists():
            cmd.append("--kv-hadamard")

    # mlock: lock model weights in RAM to prevent page cache eviction.
    # Validated in S2: 30x latency improvement under memory pressure.
    # Requires ulimit -l unlimited in launch environment.
    if role_config.name in MLOCK_ROLES:
        cmd.append("--mlock")

    # Add acceleration based on type
    if accel.type == "moe_expert_reduction" and accel.experts:
        cmd.extend([
            "--override-kv",
            f"{accel.override_key}=int:{accel.experts}",
        ])
    elif accel.type == "speculative_decoding" and accel.draft_role:
        # Get draft model path from registry
        registry = RegistryLoader()
        draft_config = registry.get_role(accel.draft_role)
        if draft_config:
            cmd.extend([
                "-md", draft_config.model.full_path,
                "--draft-max", str(accel.k or 16),
            ])

    # MoE + spec decode combo (e.g., 480B with jukofyork draft + expert reduction)
    # draft_role is populated from speculative_decoding sub-config in registry
    if accel.type == "moe_expert_reduction" and accel.draft_role:
        registry = RegistryLoader()
        draft_config = registry.get_role(accel.draft_role)
        if draft_config:
            cmd.extend([
                "-md", draft_config.model.full_path,
                "--draft-max", str(accel.k or 16),
            ])

    # Self-speculation: same model as target and draft, draft exits early
    elif accel.type == "self_speculation" and accel.n_layer_exit_draft:
        cmd.extend([
            "-md", model_path,
            "--n-layer-exit-draft", str(accel.n_layer_exit_draft),
            "--draft-max", str(accel.k or 16),
        ])

    # Hierarchical speculation: self-spec with intermediate verification
    elif accel.type == "hierarchical_speculation":
        cmd.extend([
            "-md", model_path,
            "--n-layer-exit-draft", str(accel.n_layer_exit_draft or 0),
            "--hierarchical-spec",
            "--draft-max", str(accel.k or 16),
        ])
        if accel.n_layer_exit_intermediate:
            cmd.extend(["--n-layer-exit-intermediate", str(accel.n_layer_exit_intermediate)])

    # Tree speculation: --draft-p-split was the DySpec branching probability flag.
    # REMOVED 2026-05-04: production-consolidated-v5 kernel push stripped tree-speculation
    # support; the binary no longer accepts --draft-p-split. Spec-decode is now linear-only,
    # which matches the registry config (all 4 spec-decode roles use p_split=0 = linear).
    # Re-introduce only if a future binary restores tree speculation.
    # Historical: Coder Q4KM tree was +2.7% at 48t; hybrids tree HARMFUL (-25% to -40%).
    if False and accel.p_split is not None:  # disabled — flag stripped in v5
        cmd.extend(["--draft-p-split", str(accel.p_split)])

    # NUMA-specific spec param overrides: when NUMA thread count differs from 192t,
    # the optimal draft_max/p_split may differ. Override the registry defaults with
    # NUMA-optimal values from bench_sweep_spec_params.sh results.
    if numa_cfg and "spec_overrides" in numa_cfg:
        overrides = numa_cfg["spec_overrides"]
        if "draft_max" in overrides:
            # Replace --draft-max value in existing cmd
            for i, arg in enumerate(cmd):
                if arg == "--draft-max" and i + 1 < len(cmd):
                    cmd[i + 1] = str(overrides["draft_max"])
                    break
        if False and "p_split" in overrides:  # disabled — --draft-p-split stripped in v5 binary
            # Replace or add --draft-p-split
            replaced = False
            for i, arg in enumerate(cmd):
                if arg == "--draft-p-split" and i + 1 < len(cmd):
                    cmd[i + 1] = str(overrides["p_split"])
                    replaced = True
                    break
            if not replaced and overrides["p_split"] > 0:
                cmd.extend(["--draft-p-split", str(overrides["p_split"])])

    # Prompt n-gram lookup decode — disabled 2026-05-04: production-consolidated-v5 binary
    # stripped the bare `--lookup` boolean flag. Replaced by `--lookup-cache-static FNAME` /
    # `--lookup-cache-dynamic FNAME` (file-based) and `--spec-ngram-size-n/m N` parameters
    # for ngram-simple/ngram-map speculative-decoding modes. Per-role registry `lookup: true`
    # is now informational; emission would crash the server with `error: invalid argument: --lookup`.
    # Historical: dense/small-MoE +27% (30B), net-negative on large MoE (480B); combined with
    # spec-decode 5.4x vs 5.2x spec-only (production-consolidated commit 8e35dbc01).
    # Re-enable by writing a static cache file and emitting --lookup-cache-static if the
    # quality/speed lever is wanted back.
    if False and accel.lookup:  # disabled — bare --lookup stripped in v5 binary
        cmd.append("--lookup")

    # DS-3: KV state save/restore — enables dynamic stack slot persistence.
    # Each role gets its own subdirectory to avoid slot ID collisions.
    slot_dir = SLOT_SAVE_DIR / role_config.name
    slot_dir.mkdir(parents=True, exist_ok=True)
    cmd.extend(["--slot-save-path", str(slot_dir)])

    return cmd


def start_server(
    port: int,
    roles: list[str],
    registry: RegistryLoader,
    dev_mode: bool = False,
    embedding_mode: bool = False,
    worker_pool_mode: bool = False,
    worker_type: str = None,
    vision_mode: bool = False,
    vision_type: str = None,
    numa_instance: int = 0,
) -> ProcessInfo | None:
    """Start a llama-server for the given roles."""
    # Vision mode - VL models with multimodal projector
    if vision_mode:
        log_file = LOG_DIR / f"vision-{vision_type or 'worker'}-{port}.log"
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        if vision_type == "escalation":
            model_path = VISION_ESCALATION_MODEL
            model_name = "Qwen3-VL-30B-A3B (vision escalation)"
        else:
            model_path = VISION_WORKER_MODEL
            model_name = "Qwen2.5-VL-7B (vision worker)"

        cmd = build_server_command(
            None, port, vision_mode=True, vision_type=vision_type
        )

        print(f"  Starting vision server [{vision_type or 'worker'}] on port {port}: {model_name}")
        print(f"    Roles: {', '.join(roles)}")
        print(f"    Command: {' '.join(cmd[:6])}...")

        with open(log_file, "w") as log:
            env = build_launch_env(roles[0], os.environ.copy())
            proc = subprocess.Popen(
                _numa_prefix(roles[0]) + cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
            )

        print(f"    PID: {proc.pid}")
        print(f"    Waiting for health...")

        # VL models take longer to load (mmproj + main model)
        timeout = _HEALTH_VISION_SERVER if vision_type == "escalation" else _HEALTH_WORKER_SERVER
        if wait_for_health(port, timeout=timeout):
            print(f"    [OK] Vision server {vision_type or 'worker'} ready")
            return ProcessInfo(
                role=roles[0],
                pid=proc.pid,
                port=port,
                started_at=datetime.now().isoformat(),
                model_path=model_path,
                log_file=str(log_file),
            )
        else:
            print(f"    [FAIL] Vision server {vision_type or 'worker'} did not become healthy")
            print(f"    Check log: {log_file}")
            kill_process(proc.pid)
            return None

    # Embedding mode uses dedicated config, no registry lookup needed
    if embedding_mode:
        log_file = LOG_DIR / f"embedder-{port}.log"
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        cmd = build_server_command(None, port, dev_mode=False, embedding_mode=True)
        model_name = "BGE-large-en-v1.5 (embeddings)"
        instance_idx = port - 8090  # 0-5 for ports 8090-8095

        print(f"  Starting embedder #{instance_idx} on port {port}: {model_name}")
        print(f"    Roles: {', '.join(roles)}")
        print(f"    Command: {' '.join(cmd[:6])}...")

        with open(log_file, "w") as log:
            env = build_launch_env(roles[0], os.environ.copy())
            # NOTE: Do NOT set OMP_NUM_THREADS=1 - it disables parallel tensor repack (2.2x slower loading)
            proc = subprocess.Popen(
                _numa_prefix(roles[0]) + cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
            )

        print(f"    PID: {proc.pid}")
        print(f"    Waiting for health...")

        if wait_for_health(port, timeout=60):  # Faster timeout for small model
            print(f"    [OK] Embedder #{instance_idx} ready")
            return ProcessInfo(
                role=roles[0],  # Use actual role name (embedder, embedder_1, etc.)
                pid=proc.pid,
                port=port,
                started_at=datetime.now().isoformat(),
                model_path=EMBEDDING_MODEL_PATH,
                log_file=str(log_file),
            )
        else:
            print(f"    [FAIL] Embedder #{instance_idx} did not become healthy")
            print(f"    Check log: {log_file}")
            kill_process(proc.pid)
            return None

    # Worker pool mode - heterogeneous workers
    if worker_pool_mode and worker_type:
        log_file = LOG_DIR / f"worker-{worker_type}-{port}.log"
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        model_path = WORKER_POOL_MODELS.get(worker_type)
        if not model_path:
            print(f"  [!] Unknown worker type: {worker_type}")
            return None

        # Per-role binary + LD_LIBRARY_PATH override (Phase 2). worker_general (gemma4
        # MTP) needs ik_llama.cpp PR #1744 binary; other workers fall back to default.
        # Lookup keyed on the primary role (e.g. "worker_general"), not worker_type.
        binary_dir, ld_paths = _runtime_requirements_for_role(registry, roles[0])
        binary_override = (
            str(Path(binary_dir) / "llama-server") if binary_dir else None
        )

        cmd = build_server_command(
            None, port, worker_pool_mode=True, worker_type=worker_type,
            binary_override=binary_override,
        )
        model_name = Path(model_path).stem

        print(f"  Starting worker pool [{worker_type}] on port {port}: {model_name}")
        print(f"    Roles: {', '.join(roles)}")
        if binary_override:
            print(f"    Binary override: {binary_override}")
        print(f"    Command: {' '.join(cmd[:6])}...")

        with open(log_file, "w") as log:
            # Worker pool roles map their worker_type to the canonical "worker" role for env.
            env = build_launch_env("worker", os.environ.copy())
            # When a per-role binary override is in effect (gemma4 MTP via ik_llama.cpp
            # PR #1744), strip the production-llama.cpp-tuned GGML_* env block. Those
            # flags (GGML_CCD_POOLS / GGML_CCD_WORK_DIST / GGML_BARRIER_LOCAL_BETWEEN_OPS)
            # were validated for Qwen3-Coder-30B on the production ggml fork; the
            # ik_llama.cpp gemma-mtp branch is forked at a different ggml commit and
            # leaves MTP draft tensors with no buffer assignment when these flags are
            # set, triggering "tensor buffer not set" assertion at ggml-backend.cpp:236.
            # Bench launches confirm: gemma4 MTP works with bare OMP env, no GGML_*.
            if binary_override:
                stripped = [k for k in list(env.keys()) if k.startswith("GGML_")]
                for k in stripped:
                    del env[k]
                if stripped:
                    print(f"    [binary_override] stripped GGML_* env: {stripped}")
            # Prepend role-specific LD_LIBRARY_PATH entries (Phase 2): ik_llama.cpp
            # PR #1744 build needs its own libllama.so / libggml.so on the resolver
            # path. Prepend so the override beats system libs without touching the
            # canonical-recipe LLVM-20 libomp path that already lives in env.
            if ld_paths:
                existing = env.get("LD_LIBRARY_PATH", "")
                merged = ":".join(ld_paths) + (f":{existing}" if existing else "")
                env["LD_LIBRARY_PATH"] = merged
                print(f"    LD_LIBRARY_PATH += {ld_paths}")
            # NOTE: Do NOT set OMP_NUM_THREADS=1 - it disables parallel tensor repack (2.2x slower loading)
            proc = subprocess.Popen(
                _numa_prefix(roles[0]) + cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
            )

        print(f"    PID: {proc.pid}")
        print(f"    Waiting for health...")

        # Faster timeout for smaller models (quick_check for fast workers)
        timeout = int(_registry_timeout("health", "quick_check", 10)) * 6 if worker_type == "fast" else _HEALTH_WORKER_SERVER
        if wait_for_health(port, timeout=timeout):
            print(f"    [OK] Worker {worker_type} ready")
            return ProcessInfo(
                role=f"worker_{worker_type}",
                pid=proc.pid,
                port=port,
                started_at=datetime.now().isoformat(),
                model_path=model_path,
                log_file=str(log_file),
            )
        else:
            print(f"    [FAIL] Worker {worker_type} did not become healthy")
            print(f"    Check log: {log_file}")
            kill_process(proc.pid)
            return None

    # Use first role's config for the server
    primary_role = roles[0]
    role_config = registry.get_role(primary_role)

    if not role_config and not dev_mode:
        print(f"  [!] Role {primary_role} not found in registry")
        return None

    log_file = LOG_DIR / f"llama-server-{port}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = build_server_command(role_config, port, dev_mode)

    model_name = DEV_MODEL if dev_mode else role_config.model.name
    numa_cfg = NUMA_CONFIG.get(primary_role)
    numa_label = ""
    if numa_cfg and numa_instance < len(numa_cfg["instances"]):
        cpu_list = numa_cfg["instances"][numa_instance][0]
        numa_label = f" [NUMA {numa_instance}: cpus {cpu_list}]"

    print(f"  Starting port {port}: {model_name}{numa_label}")
    print(f"    Roles: {', '.join(roles)}")
    print(f"    Command: {' '.join(cmd[:5])}...")

    # Start process — taskset CPU-pinned per NUMA config + canonical OMP env + per-role GGML
    with open(log_file, "w") as log:
        env = build_launch_env(primary_role, os.environ.copy())
        # NOTE: Do NOT set OMP_NUM_THREADS=1 - it disables parallel tensor repack (2.2x slower loading)
        proc = subprocess.Popen(
            _numa_prefix(primary_role, numa_instance) + cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )

    print(f"    PID: {proc.pid}")

    # Wait for health
    print(f"    Waiting for health...")
    if wait_for_health(port, timeout=180):
        print(f"    [OK] Server ready")
        return ProcessInfo(
            role=primary_role,
            pid=proc.pid,
            port=port,
            started_at=datetime.now().isoformat(),
            model_path=DEV_MODEL_PATH if dev_mode else role_config.model.full_path,
            log_file=str(log_file),
        )
    else:
        print(f"    [FAIL] Server did not become healthy")
        print(f"    Check log: {log_file}")
        kill_process(proc.pid)
        return None


def _apply_orchestrator_profile(env: dict[str, str], profile: str | None) -> None:
    """Apply optional orchestrator profile env vars without overriding explicit env."""
    if not profile:
        return
    profile_vars = ORCHESTRATOR_PROFILES.get(profile)
    if not profile_vars:
        print(f"    [WARN] Unknown orchestrator profile '{profile}' (ignored)")
        return
    print(f"    Using orchestrator profile: {profile}")
    for key, value in profile_vars.items():
        env.setdefault(key, value)


def start_orchestrator(profile: str | None = None) -> ProcessInfo | None:
    """Start the orchestrator API."""
    log_file = LOG_DIR / "orchestrator.log"

    print("  Starting orchestrator API on port 8000")
    stale_pids = _pids_on_port(8000)
    if stale_pids:
        print(f"    Clearing stale listeners on :8000 ({', '.join(str(p) for p in stale_pids)})")
        for stale_pid in stale_pids:
            kill_process(stale_pid)
        time.sleep(1)

    # Set environment — enable production feature flags
    env = os.environ.copy()
    env["HF_HOME"] = str(_PATHS["cache_dir"] / "huggingface")
    env["TMPDIR"] = str(_PATHS["tmp_dir"])
    # Feature flags: enable production capabilities
    env["ORCHESTRATOR_MEMRL"] = "1"
    env["ORCHESTRATOR_ROUTING_CLASSIFIER"] = "1"
    env["ORCHESTRATOR_TOOLS"] = "1"
    env["ORCHESTRATOR_SCRIPTS"] = "1"
    # NOTE: Do NOT set ORCHESTRATOR_REPL here — it collides with
    # OrchestratorSettings.repl (REPLSettings model) in config.py.
    # The repl feature flag defaults to True in features.py already.
    env["ORCHESTRATOR_CACHING"] = "1"
    env["ORCHESTRATOR_STREAMING"] = "1"
    env["ORCHESTRATOR_MOCK_MODE"] = "0"
    env["ORCHESTRATOR_GENERATION_MONITOR"] = "1"
    env["ORCHESTRATOR_REACT_MODE"] = "1"
    env["ORCHESTRATOR_CASCADING_TOOL_POLICY"] = "1"
    env["ORCHESTRATOR_WORKER_CALL_BUDGET"] = "1"
    env["ORCHESTRATOR_TASK_TOKEN_BUDGET"] = "1"
    env.setdefault("ORCHESTRATOR_WORKER_CALL_BUDGET_CAP", "30")
    env.setdefault("ORCHESTRATOR_TASK_TOKEN_BUDGET_CAP", "200000")
    env["ORCHESTRATOR_SESSION_SCRATCHPAD"] = "1"
    env["ORCHESTRATOR_SESSION_LOG"] = "1"
    env["ORCHESTRATOR_APPROVAL_GATES"] = "1"
    env["ORCHESTRATOR_RESUME_TOKENS"] = "1"
    env["ORCHESTRATOR_SIDE_EFFECT_TRACKING"] = "1"
    env["ORCHESTRATOR_STRUCTURED_TOOL_OUTPUT"] = "1"
    # LangGraph Phase 3: per-node migration (infrastructure validated by 48 unit tests)
    env["ORCHESTRATOR_LANGGRAPH_INGEST"] = "1"
    env["ORCHESTRATOR_LANGGRAPH_WORKER"] = "1"
    env["ORCHESTRATOR_LANGGRAPH_FRONTDOOR"] = "1"
    env["ORCHESTRATOR_LANGGRAPH_CODER"] = "1"
    env["ORCHESTRATOR_LANGGRAPH_CODER_ESCALATION"] = "1"
    env["ORCHESTRATOR_LANGGRAPH_ARCHITECT"] = "1"
    env["ORCHESTRATOR_LANGGRAPH_ARCHITECT_CODING"] = "1"
    _apply_orchestrator_profile(env, profile)
    # Bound inference-lock waits by default to avoid multi-minute silent stalls
    # during iterative debugging / seeding runs.
    env.setdefault("ORCHESTRATOR_INFERENCE_LOCK_TIMEOUT_EXCLUSIVE_S", "45")
    env.setdefault("ORCHESTRATOR_INFERENCE_LOCK_TIMEOUT_SHARED_S", "45")

    with open(log_file, "w") as log:
        workers = int(env.get("ORCHESTRATOR_UVICORN_WORKERS", "6"))
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "src.api:app",
                "--host", "127.0.0.1",
                "--port", "8000",
                "--workers", str(workers),
                "--limit-concurrency", "4",  # Prevent request pile-up per worker
            ],
            cwd=str(_PATHS["project_root"]),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )

    print(f"    PID: {proc.pid}")
    print(f"    Waiting for health...")

    if wait_for_health(8000, timeout=60):
        print(f"    [OK] Orchestrator ready")
        return ProcessInfo(
            role="orchestrator",
            pid=proc.pid,
            port=8000,
            started_at=datetime.now().isoformat(),
            model_path="uvicorn",
            log_file=str(log_file),
        )
    # Health probe can fail transiently (port permissions / local sandbox),
    # while the process is actually alive. Avoid killing a healthy API due to
    # a false-negative probe; only hard-fail when process already exited.
    if proc.poll() is None:
        print("    [WARN] Health probe timed out, but API process is still running")
        print(f"    Check log: {log_file}")
        return ProcessInfo(
            role="orchestrator",
            pid=proc.pid,
            port=8000,
            started_at=datetime.now().isoformat(),
            model_path="uvicorn",
            log_file=str(log_file),
        )

    print(f"    [FAIL] Orchestrator did not start")
    print(f"    Check log: {log_file}")
    kill_process(proc.pid)
    return None


def start_document_formalizer() -> ProcessInfo | None:
    """Start the document formalizer (LightOnOCR-2) server."""
    log_file = LOG_DIR / "document_formalizer.log"
    port = 9001

    print(f"  Starting document_formalizer (LightOnOCR-2) on port {port}")

    # Set environment
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_PATHS["project_root"]) + os.pathsep + env.get("PYTHONPATH", "")
    env["LIGHTONOCR_WORKERS"] = "8"
    env["LIGHTONOCR_THREADS"] = "12"
    env["LIGHTONOCR_MAX_TOKENS"] = "2048"
    env["LIGHTONOCR_TIMEOUT"] = "120"

    with open(log_file, "w") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                str(_PATHS["project_root"] / "src/services/lightonocr_llama_server.py"),
                "--port", str(port),
            ],
            cwd=str(_PATHS["project_root"]),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )

    print(f"    PID: {proc.pid}")
    print(f"    Waiting for health...")

    if wait_for_health(port, timeout=60):
        print(f"    [OK] Document formalizer ready")
        return ProcessInfo(
            role="document_formalizer",
            pid=proc.pid,
            port=port,
            started_at=datetime.now().isoformat(),
            model_path="LightOnOCR-2-1B-bbox",
            log_file=str(log_file),
        )
    else:
        print(f"    [FAIL] Document formalizer did not start")
        print(f"    Check log: {log_file}")
        kill_process(proc.pid)
        return None


def start_sd_server() -> ProcessInfo | None:
    """Start the sd-server diffusion inference service (stable-diffusion.cpp native).

    Replaces the ComfyUI-GGUF + PyTorch path 2026-05-07 — sd.cpp's native
    ggml backend keeps Q8_0 weights packed and uses native quantized GEMM
    kernels, skipping ComfyUI-GGUF's per-layer dequant-to-BF16 step.
    Measured ~1.74× wall-clock and ~3.43× sampler s/iter speedup at 512² /
    4 steps; expected ~2× wall-clock at production 1024² / 8 steps.
    Stack-managed per feedback_stack_managed_services. Health probe uses
    /sdapi/v1/samplers (sd-server has no dedicated /health endpoint).
    """
    log_file = LOG_DIR / "sd_server.log"
    port = 8190
    launcher = _PATHS["project_root"] / "scripts/diffusion/start_sd_server.sh"

    print(f"  Starting sd_server (ERNIE-Image-Turbo, ggml native) on port {port}")

    if not launcher.exists():
        print(f"    [FAIL] Launcher not found: {launcher}")
        return None

    env = os.environ.copy()
    env["SD_SERVER_PORT"] = str(port)

    with open(log_file, "w") as log:
        proc = subprocess.Popen(
            ["bash", str(launcher)],
            cwd=str(_PATHS["project_root"]),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )

    print(f"    PID: {proc.pid}")
    print(f"    Waiting for health (path=/sdapi/v1/samplers, timeout=120s)...")

    if wait_for_health(port, timeout=120, path="/sdapi/v1/samplers"):
        print(f"    [OK] sd-server ready")
        return ProcessInfo(
            role="sd_server",
            pid=proc.pid,
            port=port,
            started_at=datetime.now().isoformat(),
            model_path="ernie-image-turbo-Q8_0.gguf + ministral-3-3b + flux2-vae (sd.cpp ggml native)",
            log_file=str(log_file),
        )
    else:
        print(f"    [FAIL] sd-server did not start")
        print(f"    Check log: {log_file}")
        kill_process(proc.pid)
        return None


def start_whisper() -> ProcessInfo | None:
    """Start the faster-whisper STT server (large-v3-turbo, int8).

    Promoted from sidecar to stack-managed 2026-05-06 per
    feedback_stack_managed_services. Reuses the existing launch script in
    epyc-inference-research; no rewrite needed.
    """
    log_file = LOG_DIR / "whisper.log"
    port = 9000
    # Whisper launcher lives in the inference-research repo (was a sidecar)
    launcher = Path("/mnt/raid0/llm/epyc-inference-research/scripts/voice/start_whisper_server.sh")

    print(f"  Starting whisper (faster-whisper large-v3-turbo) on port {port}")

    if not launcher.exists():
        print(f"    [FAIL] Launcher not found: {launcher}")
        return None

    env = os.environ.copy()
    env["WHISPER_PORT"] = str(port)

    with open(log_file, "w") as log:
        proc = subprocess.Popen(
            ["bash", str(launcher)],
            cwd=str(launcher.parent),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )

    print(f"    PID: {proc.pid}")
    print(f"    Waiting for health (path=/health, timeout=60s)...")

    if wait_for_health(port, timeout=60, path="/health"):
        print(f"    [OK] Whisper ready")
        return ProcessInfo(
            role="whisper",
            pid=proc.pid,
            port=port,
            started_at=datetime.now().isoformat(),
            model_path="faster-whisper-large-v3-turbo (int8)",
            log_file=str(log_file),
        )
    else:
        print(f"    [FAIL] Whisper did not start")
        print(f"    Check log: {log_file}")
        kill_process(proc.pid)
        return None


# =============================================================================
# Commands
# =============================================================================


def cmd_start(args: argparse.Namespace) -> int:
    """Start the orchestrator stack."""
    # DS-7 / NIB2-19: --migrate-to handler (runs before any start path)
    migrate_to = getattr(args, "migrate_to", None)
    if migrate_to:
        dry_run = getattr(args, "dry_run", False)
        try:
            from src.config.stack_migration import migrate_to_template
        except Exception as exc:  # noqa: BLE001
            print(f"[DS-7] Migration module unavailable: {exc}")
            return 1
        print(f"[DS-7] Migrating stack → template '{migrate_to}' "
              f"({'DRY-RUN' if dry_run else 'LIVE'})")
        registry_path = (
            Path(_PATHS.get("model_registry", ""))
            if _PATHS.get("model_registry") else None
        )
        result = migrate_to_template(migrate_to, dry_run=dry_run, registry_path=registry_path)
        print(result.summary())
        return 0 if result.ok else 1

    # DS-7: Stack template validation (before any other work)
    stack_profile = getattr(args, "stack_profile", None)
    validate_only = getattr(args, "validate_only", False)
    if stack_profile:
        try:
            from src.config.stack_templates import (
                load_template, validate_template, _TEMPLATES_DIR,
            )
            print(f"[DS-7] Loading stack template: {stack_profile}")
            template = load_template(stack_profile)
            print(f"  Name: {template.name}")
            print(f"  Description: {template.description}")
            print(f"  Roles: {len(template.roles)} ({', '.join(template.role_names())})")
            print(f"  Instances: {template.total_instances}")
            print(f"  RAM: {template.total_ram_gb:.0f} GB")
            print()

            registry_path = Path(_PATHS.get("model_registry", "")) if _PATHS.get("model_registry") else None
            result = validate_template(template, registry_path)
            if result.errors:
                print(f"  [FAIL] {len(result.errors)} validation errors:")
                for err in result.errors:
                    print(f"    ERROR: {err}")
            if result.warnings:
                for warn in result.warnings:
                    print(f"    WARN: {warn}")
            if result.valid:
                print("  [OK] Template valid")
            else:
                print("\n  Template validation failed. Fix errors and retry.")
                return 1

            if validate_only:
                print("\n--validate-only: exiting after validation.")
                return 0

            print(f"  (Template loaded but not yet used for server launch — "
                  f"integration pending DS-7 Phase 2)")
            print()
        except FileNotFoundError as exc:
            print(f"[DS-7] ERROR: {exc}")
            return 1
        except Exception as exc:
            print(f"[DS-7] Template load error: {exc}")
            return 1

    print("=" * 60)
    print("ORCHESTRATOR STACK STARTUP")
    print("=" * 60)
    print()

    # Host prerequisites — applied before any llama-server launch.
    # See cpu-kernel-env-flags-inventory.md §211 + model-registry-v5-deployment-draft.yaml.
    skip_host_prereqs = getattr(args, "skip_host_prereqs", False)
    if skip_host_prereqs:
        print("[host_prereq] SKIPPED (--skip-host-prereqs). Canonical state NOT enforced.")
    else:
        if not apply_host_prerequisites(auto_fix=True):
            print("[!] Host prerequisites could not be applied. Refusing to launch.")
            print("    Override with --skip-host-prereqs (NOT recommended for benchmarks).")
            return 1
    print()

    # Check memory
    free_gb = check_free_memory()
    print(f"[i] Free memory: {free_gb} GB")
    if free_gb < 100 and not args.dev:
        print("[!] WARNING: Less than 100GB free. Consider --dev mode.")
        if input("Continue? (y/N) ").lower() != "y":
            return 1
    print()

    # Load registry
    registry = RegistryLoader()
    state: dict[str, ProcessInfo] = {}

    # Validate model paths (prevents hallucinations about missing models)
    if not args.dev:
        print("[0.5] Validating model paths...")
        errors = validate_model_paths()
        if errors:
            print("[!] MODEL VALIDATION FAILED:")
            for err in errors:
                print(f"    - {err}")
            print("\nFix missing models or update paths in orchestrator_stack.py")
            print(f"Check {_PATHS['models_dir']} and {_PATHS['model_base']}")
            return 1
        print("  [OK] All model paths validated")
        print()

    # Cross-check launcher classification vs registry process_layout / server_mode.
    # Non-fatal: prints warnings but does not abort. Useful for catching drift
    # between the launcher's ROLE_LAUNCH_META and the registry's source-of-truth
    # process_layout section.
    if not args.dev:
        registry_warnings = validate_against_registry()
        if registry_warnings:
            print("[0] Registry classification warnings:")
            for w in registry_warnings:
                print(f"  ⚠ {w}")
            print()

    # Determine which servers to start
    servers_to_start = []

    if args.dev:
        print("[1] Starting in DEV mode (single 0.5B model)...")
        servers_to_start = [{"port": 8080, "roles": ["dev"]}]
    elif args.only:
        # --only: start ONLY the specified roles, nothing else
        requested = set(args.only)
        print(f"[1] Selective start: {', '.join(sorted(requested))}")
        for server in HOT_SERVERS + WARM_SERVERS:
            if requested & set(server["roles"]):
                servers_to_start.append(server)
                print(f"  Including: port {server['port']} ({', '.join(server['roles'])})")
        if not servers_to_start:
            print(f"  [!] No servers matched roles: {', '.join(sorted(requested))}")
            print(f"  Available roles: {', '.join(sorted({r for s in HOT_SERVERS + WARM_SERVERS for r in s['roles']}))}")
            return 1
    else:
        print("[1] Starting HOT servers...")
        servers_to_start = HOT_SERVERS.copy()

        # Add warm servers if requested
        if args.include_warm:
            for warm_server in WARM_SERVERS:
                for role in warm_server["roles"]:
                    if role in args.include_warm:
                        servers_to_start.append(warm_server)
                        print(f"  Including WARM server: port {warm_server['port']} ({role})")
                        break

    # Apply --numa-mode filter (default 'both' for back-compat — pre-2026-05-08 default).
    # Picks full XOR quarters for any role with full_instance_idx + multiple instances
    # (currently frontdoor + coder_escalation + worker_general); single-instance roles
    # pass through. See launcher-numa-mode-gating handoff.
    numa_mode = getattr(args, "numa_mode", "both")
    if numa_mode == "both":
        # Light advisory only — 'both' has been working for frontdoor/coder_escalation since
        # 2026-03 (Qwen3.6-35B Q8 quarters tuned to coexist with the full instance). The
        # gemma4-MTP exception is the one that needs --numa-mode full per role. We don't
        # spam at every start since most roles are fine.
        if any("worker_general" in s.get("roles", []) for s in servers_to_start):
            print(f"  [advisory] worker_general (gemma4-MTP) runs at -t 96; if its full + 4 quarters "
                  f"are all kept (default 'both'), expect 1.5× CPU oversubscription. "
                  f"Use '--numa-mode full' (single instance) or '--numa-mode quarter' (4 concurrent) "
                  f"for that role specifically. See launcher-numa-mode-gating.md.")
    pre_filter_count = len(servers_to_start)
    servers_to_start = _filter_by_numa_mode(servers_to_start, numa_mode)
    if numa_mode != "both" and len(servers_to_start) != pre_filter_count:
        dropped = pre_filter_count - len(servers_to_start)
        print(f"  [--numa-mode={numa_mode}] dropped {dropped} overlapping instance(s); "
              f"{len(servers_to_start)} server(s) to start")

    print()

    # Check target ports — skip healthy, clean up unhealthy
    target_ports = {s["port"] for s in servers_to_start}
    print("[2] Checking target ports...")
    already_healthy_ports: set[int] = set()
    for server in servers_to_start:
        port = server["port"]
        if is_port_in_use(port):
            if wait_for_health(port, timeout=3):
                print(f"  Port {port} already healthy, skipping")
                already_healthy_ports.add(port)
                continue
            print(f"  Port {port} in use but unhealthy, cleaning up...")
            # Find PID from lsof
            try:
                result = subprocess.run(
                    ["lsof", "-t", f"-i:{port}"],
                    capture_output=True,
                    text=True,
                )
                if result.stdout.strip():
                    for pid_str in result.stdout.strip().split("\n"):
                        pid = int(pid_str)
                        kill_process(pid)
            except Exception as e:
                print(f"  [!] Error cleaning port {port}: {e}")
    if already_healthy_ports:
        print(f"  Preserved {len(already_healthy_ports)} healthy server(s)")

    print()

    # Start servers sequentially (skip already-healthy ports)
    print("[3] Starting llama-servers...")
    for i, server in enumerate(servers_to_start):
        port = server["port"]
        roles = server["roles"]

        if port in already_healthy_ports:
            role_label = roles[0] if roles else str(port)
            print(f"  Skipping port {port}: {role_label} (already healthy)")
            # Record existing server in state so status reporting works
            state[f"server_{port}"] = {"port": port, "roles": roles, "status": "preserved"}
            for role in roles:
                if role not in state:
                    state[role] = {"port": port, "roles": roles, "status": "preserved"}
            continue

        embedding_mode = server.get("embedding", False)
        worker_pool_mode = server.get("worker_pool", False)
        worker_type = server.get("worker_type")
        vision_mode = server.get("vision", False)
        vision_type = server.get("vision_type")
        numa_instance = server.get("numa_instance", 0)

        info = start_server(
            port, roles, registry, args.dev,
            embedding_mode=embedding_mode,
            worker_pool_mode=worker_pool_mode,
            worker_type=worker_type,
            vision_mode=vision_mode,
            vision_type=vision_type,
            numa_instance=numa_instance,
        )
        if info:
            state[f"server_{port}"] = info
            # Also map all roles to this server
            for role in roles:
                if role not in state:
                    state[role] = info
        else:
            print(f"  [!] Failed to start server on port {port}")
            # Embedding/worker_pool/vision server failure is non-fatal (fallback available)
            is_optional = embedding_mode or worker_pool_mode or vision_mode
            if not args.dev and not is_optional:
                return 1

        # Sequential loading: wait for this server to be healthy before launching
        # the next one. Concurrent mlock on large models causes crashes even when
        # total RAM is sufficient (race condition during page fault + lock).
        is_small_model = embedding_mode or (worker_pool_mode and worker_type == "fast") or (vision_mode and vision_type != "escalation")
        if i < len(servers_to_start) - 1 and not args.dev and not is_small_model:
            if not wait_for_health(port, timeout=300):
                print(f"  [!] Server on port {port} did not become healthy within 300s")
            else:
                print(f"  Server on port {port} healthy, proceeding to next")

    print()

    # Start orchestrator (skip if already healthy, or if --only was used for model servers)
    if args.only:
        print("[4] Skipping orchestrator API (--only mode)")
        if wait_for_health(8000, timeout=2):
            print("  Orchestrator already healthy")
            state["orchestrator"] = {"port": 8000, "status": "preserved"}
        else:
            print("  [i] Orchestrator not running — start separately if needed")
    elif 8000 in already_healthy_ports:
        print("[4] Starting orchestrator API...")
        print("  Orchestrator already healthy, skipping")
        state["orchestrator"] = {"port": 8000, "status": "preserved"}
    else:
        info = start_orchestrator(getattr(args, "profile", None))
        if info:
            state["orchestrator"] = info
        else:
            print("  [!] Failed to start orchestrator")
            return 1

    print()

    # Start document formalizer (optional, non-fatal)
    if not args.dev and not args.only:
        print("[5] Starting document formalizer (LightOnOCR-2)...")
        info = start_document_formalizer()
        if info:
            state["document_formalizer"] = info
        else:
            print("  [!] Document formalizer failed (non-fatal, continuing)")

        print()

        # Start sd-server diffusion service (optional, non-fatal)
        # ERNIE-Image-Turbo Q8 GGUF + Mistral3 + flux2 VAE via stable-diffusion.cpp.
        # Replaced ComfyUI 2026-05-07 — see start_sd_server() for context.
        if 8190 in already_healthy_ports:
            print("[5a] Starting sd-server (ggml native diffusion)...")
            print("  Already healthy, skipping")
            state["sd_server"] = {"port": 8190, "status": "preserved"}
        else:
            print("[5a] Starting sd-server (ggml native diffusion)...")
            info = start_sd_server()
            if info:
                state["sd_server"] = info
            else:
                print("  [!] sd-server failed (non-fatal, image generation unavailable)")

        print()

        # Start Whisper STT service (optional, non-fatal)
        # Promoted from sidecar 2026-05-06.
        if 9000 in already_healthy_ports:
            print("[5b] Starting Whisper STT server...")
            print("  Already healthy, skipping")
            state["whisper"] = {"port": 9000, "status": "preserved"}
        else:
            print("[5b] Starting Whisper STT server...")
            info = start_whisper()
            if info:
                state["whisper"] = info
            else:
                print("  [!] Whisper failed (non-fatal, STT unavailable)")

        print()

        # Start Docker services (NextPLAID retrieval + SearXNG metasearch)
        if _docker_available():
            print("[5.5] Starting Docker services (NextPLAID retrieval + SearXNG metasearch)...")
            for service in DOCKER_SERVICES:
                info = start_docker_container(service)
                if info:
                    state[service["name"]] = info
                else:
                    svc_name = service["name"]
                    if svc_name == "searxng":
                        print(f"  [!] {svc_name} failed (non-fatal, web_search falls back to DDG HTML scraping)")
                    else:
                        print(f"  [!] {svc_name} failed (non-fatal, code_search degrades gracefully)")
            print()
        else:
            print("[5.5] Docker not available, skipping Docker containers")
            print("  code_search/doc_search will be unavailable")
            print("  web_search will use DDG HTML scraping fallback")
            print()

        # Initialize MemRL databases and tool registry
        init_memrl_and_tools()

        print()

    # Save state
    save_state(state)
    print(f"[i] State saved to {STATE_FILE}")
    print()

    # Final status
    print("=" * 60)
    print("STACK READY")
    print("=" * 60)
    cmd_status(args)

    return 0


def _find_pids_on_port(port: int) -> list[int]:
    """Find PIDs listening on a port via lsof (fallback for stale state)."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return [int(p) for p in result.stdout.strip().split("\n") if p.strip()]
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    return []


def _scan_known_ports() -> dict[int, list[int]]:
    """Scan all known orchestrator ports for running processes.

    Returns:
        {port: [pid, ...]} for ports that have listeners.
    """
    known_ports = sorted({s["port"] for s in HOT_SERVERS} | NUMA_REPLICA_PORTS | {8000})
    found: dict[int, list[int]] = {}
    for port in known_ports:
        pids = _find_pids_on_port(port)
        if pids:
            found[port] = pids
    return found


def cmd_stop(args: argparse.Namespace) -> int:
    """Stop components."""
    state = load_state()

    if not state and args.all:
        # State file empty — fall back to port scanning
        found = _scan_known_ports()
        if not found:
            print("No running components found")
            return 0

        print(f"State file empty but found processes on {len(found)} ports (port scan fallback)")
        killed = 0
        for port, pids in sorted(found.items()):
            for pid in pids:
                print(f"  Stopping PID {pid} on port {port}...")
                if kill_process(pid):
                    print(f"    [OK] Stopped")
                    killed += 1
                else:
                    print(f"    [!] Failed to stop")
        print(f"Stopped {killed} orphaned processes")
        save_state({})
        return 0

    if not state:
        print("No running components found")
        return 0

    targets = []
    if args.all:
        targets = list(state.keys())
    elif args.components:
        targets = args.components
    else:
        print("Specify --all or component names")
        return 1

    for name in targets:
        if name in state:
            info = state[name]
            if info.pid == -1:
                # Docker-managed container
                print(f"Stopping Docker container {name}...")
                if stop_docker_container(info.role):
                    del state[name]
                    print(f"  [OK] Stopped")
                else:
                    print(f"  [!] Failed to stop container {name}")
            else:
                print(f"Stopping {name} (PID {info.pid})...")
                if kill_process(info.pid):
                    del state[name]
                    print(f"  [OK] Stopped")
                else:
                    print(f"  [!] Failed to stop")
        else:
            print(f"  [?] {name} not found in state")

    save_state(state)

    # After state-based stop, scan for orphans that survived
    if args.all:
        orphans = _scan_known_ports()
        if orphans:
            print(f"\nFound {sum(len(p) for p in orphans.values())} orphaned processes on {len(orphans)} ports")
            for port, pids in sorted(orphans.items()):
                for pid in pids:
                    print(f"  Stopping orphan PID {pid} on port {port}...")
                    if kill_process(pid):
                        print(f"    [OK] Stopped")
                    else:
                        print(f"    [!] Failed to stop")

    return 0


def cmd_reload(args: argparse.Namespace) -> int:
    """Reload components."""
    state = load_state()
    registry = RegistryLoader()

    for component in args.components:
        print(f"Reloading {component}...")

        # Special case: reload all embedders at once
        if component == "embedders":
            print("  Reloading all 6 BGE embedder instances...")

            # Kill by state file entries
            for port in EMBEDDER_PORTS:
                key = f"server_{port}"
                role = "embedder" if port == 8090 else f"embedder_{port - 8090}"
                if key in state:
                    kill_process(state[key].pid)
                    del state[key]
                if role in state:
                    del state[role]

            # Also kill by port (in case state is stale)
            for port in EMBEDDER_PORTS:
                if is_port_in_use(port):
                    try:
                        result = subprocess.run(
                            ["lsof", "-t", f"-i:{port}"],
                            capture_output=True, text=True,
                        )
                        if result.stdout.strip():
                            for pid_str in result.stdout.strip().split("\n"):
                                kill_process(int(pid_str))
                                print(f"    Killed stale process on port {port}")
                    except (subprocess.TimeoutExpired, OSError, ValueError):
                        pass  # Best-effort stale process cleanup

            time.sleep(2)  # Wait for ports to free

            # Start all embedders
            success_count = 0
            for port in EMBEDDER_PORTS:
                role = "embedder" if port == 8090 else f"embedder_{port - 8090}"
                info = start_server(
                    port, [role], registry, dev_mode=False,
                    embedding_mode=True,
                )
                if info:
                    state[f"server_{port}"] = info
                    state[role] = info
                    success_count += 1

            print(f"  [OK] {success_count}/{len(EMBEDDER_PORTS)} embedders restarted")
            if success_count == 0:
                return 1
            continue

        elif component == "orchestrator":
            # Stop by authoritative listener port only.
            # State-file PIDs can go stale and be reused by unrelated processes.
            for pid in _pids_on_port(8000):
                kill_process(pid)
            time.sleep(1)

            # Start new
            info = start_orchestrator(getattr(args, "profile", None))
            if info:
                state["orchestrator"] = info
            else:
                print(f"  [!] Failed to restart orchestrator")
                return 1

        elif component in PORT_MAP:
            port = PORT_MAP[component]
            key = f"server_{port}"

            # Find roles and config for this port
            roles = [component]
            worker_pool_mode = False
            worker_type = None
            embedding_mode = False
            vision_mode = False
            vision_type = None

            for server in HOT_SERVERS + WARM_SERVERS:
                if server["port"] == port:
                    roles = server["roles"]
                    worker_pool_mode = server.get("worker_pool", False)
                    worker_type = server.get("worker_type")
                    embedding_mode = server.get("embedding", False)
                    vision_mode = server.get("vision", False)
                    vision_type = server.get("vision_type")
                    break

            # Stop existing
            # Stop by authoritative listener port only.
            # State-file PIDs can go stale and be reused by unrelated processes.
            for pid in _pids_on_port(port):
                kill_process(pid)
            time.sleep(1)

            # Start new
            info = start_server(
                port, roles, registry, dev_mode=False,
                embedding_mode=embedding_mode,
                worker_pool_mode=worker_pool_mode,
                worker_type=worker_type,
                vision_mode=vision_mode,
                vision_type=vision_type,
            )
            if info:
                state[key] = info
                for role in roles:
                    state[role] = info
            else:
                print(f"  [!] Failed to restart {component}")
                return 1

        else:
            # Check if it's a Docker service
            docker_service = None
            for svc in DOCKER_SERVICES:
                if component == svc["name"]:
                    docker_service = svc
                    break

            if docker_service:
                print(f"  Reloading Docker service {component}...")
                stop_docker_container(component)
                time.sleep(2)
                info = start_docker_container(docker_service)
                if info:
                    state[component] = info
                else:
                    print(f"  [!] Failed to restart {component}")
                    return 1
            else:
                print(f"  [?] Unknown component: {component}")

    save_state(state)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show status of all components."""
    state = load_state()

    if not state:
        print("No components running")
        return 0

    print()
    print(f"{'COMPONENT':<25} {'PORT':<8} {'PID':<10} {'STATUS':<10} {'MODEL'}")
    print("-" * 80)

    seen_pids = set()
    for name, info in sorted(state.items()):
        if info.pid != -1 and info.pid in seen_pids:
            continue  # Skip duplicates (roles sharing servers)
        seen_pids.add(info.pid)

        if info.pid == -1:
            # Docker-managed container
            alive = docker_container_running(info.role)
            # Look up health_path for this service (SearXNG uses /, others use /health)
            health_path = "/health"
            for svc in DOCKER_SERVICES:
                if svc["name"] == info.role:
                    health_path = svc.get("health_path", "/health")
                    break
            healthy = wait_for_health(info.port, timeout=3, path=health_path) if alive else False
            status = "healthy" if healthy else ("running" if alive else "stopped")
            pid_str = "docker"
        else:
            # Native process
            try:
                os.kill(info.pid, 0)
                alive = True
            except ProcessLookupError:
                alive = False
            healthy = wait_for_health(info.port, timeout=3) if alive else False
            if not alive and is_port_in_use(info.port):
                # PID drift can happen if the original launcher PID exits while
                # a listener remains healthy on the same port.
                replacement_pids = _pids_on_port(info.port)
                if replacement_pids:
                    replacement_pid = replacement_pids[0]
                    info.pid = replacement_pid
                    state[name] = info
                    alive = True
                    healthy = wait_for_health(info.port, timeout=3)
            status = "healthy" if healthy else ("running" if alive else "dead")
            pid_str = str(info.pid)

        model = Path(info.model_path).stem if info.model_path != "uvicorn" else "uvicorn"

        print(f"{name:<25} {info.port:<8} {pid_str:<10} {status:<10} {model[:30]}")

    print()
    print(f"State file: {STATE_FILE}")
    save_state(state)
    return 0


# =============================================================================
# MemRL and Tool Registry Initialization
# =============================================================================


def init_memrl_and_tools() -> bool:
    """Initialize MemRL databases and tool registry for the session.

    This ensures all deterministic tools (41 total) are ready and
    the REPL memory system is initialized with seed examples.
    """
    success = True

    # [6] REPL Memory Initialization
    print("[6] Initializing MemRL databases...")

    # Initialize REPL seed examples
    seed_loader_path = _PATHS["project_root"] / "orchestration/repl_memory/seed_loader.py"
    if seed_loader_path.exists():
        result = subprocess.run(
            [sys.executable, str(seed_loader_path), "--init"],
            capture_output=True,
            text=True,
            cwd=str(_PATHS["project_root"]),
        )
        if result.returncode == 0:
            print("  [OK] REPL seed examples loaded")
        else:
            print(f"  [WARN] Seed loader failed: {result.stderr[:100] if result.stderr else 'no output'}")

    # Warm up all embedding servers with test query
    try:
        import urllib.request
        import urllib.error

        test_payload = json.dumps({"content": "test embedding warmup"}).encode()
        healthy_count = 0
        for port in EMBEDDER_PORTS:
            try:
                req = urllib.request.Request(
                    f"http://localhost:{port}/embedding",
                    data=test_payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status == 200:
                        healthy_count += 1
            except (urllib.error.URLError, TimeoutError, OSError):
                pass  # Expected during warmup — server may still be starting
        print(f"  [OK] Embedding servers warmed up: {healthy_count}/{len(EMBEDDER_PORTS)} healthy")
    except Exception as e:
        print(f"  [WARN] Embedding warmup failed: {e}")

    # [7] Tool Registry Initialization
    print("[7] Initializing deterministic tool registry...")

    # Validate tool registry exists
    tool_registry_path = _PATHS["project_root"] / "orchestration/tool_registry.yaml"
    if not tool_registry_path.exists():
        print(f"  [!] Tool registry not found: {tool_registry_path}")
        success = False
    else:
        # Load and validate tool executor
        try:
            # Add src to path for imports
            import sys as _sys
            _sys.path.insert(0, str(_PATHS["project_root"]))
            from orchestration.tools.executor import get_executor
            executor = get_executor()
            tools = executor.list_tools()
            print(f"  [OK] Tool registry loaded: {len(tools)} tools")

            # Categorize tools
            categories: dict[str, int] = {}
            for t in tools:
                cat = t.get("category", "other")
                categories[cat] = categories.get(cat, 0) + 1
            for cat, count in sorted(categories.items()):
                print(f"      {cat}: {count}")
        except Exception as e:
            print(f"  [WARN] Tool executor init failed: {e}")

    # Verify C++ math tools binary
    cpp_binary = _PATHS["llama_cpp_bin"] / "llama-math-tools"
    if cpp_binary.exists():
        print("  [OK] C++ math tools binary found")
    else:
        print(f"  [WARN] C++ math tools not built: {cpp_binary}")
        print(f"        Run: cd {_PATHS['llm_root']}/llama.cpp && make llama-math-tools")

    return success


# =============================================================================
# Checkpoint Hooks for Self-Management Procedures
# =============================================================================

CHECKPOINT_DIR = _PATHS["project_root"] / "orchestration/checkpoints"


def checkpoint_create(name: str, include_state: bool = True) -> dict[str, Any]:
    """Create a checkpoint of the orchestrator stack state.

    Called by self-management procedures before making changes.

    Args:
        name: Descriptive checkpoint name.
        include_state: Whether to include server state.

    Returns:
        Dict with checkpoint_id and path.
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_id = f"{name}_{timestamp}"
    checkpoint_path = CHECKPOINT_DIR / f"{checkpoint_id}.json"

    checkpoint_data = {
        "id": checkpoint_id,
        "name": name,
        "created_at": datetime.now().isoformat(),
        "state": {},
        "registry_snapshot": None,
    }

    # Capture current state
    if include_state:
        state = load_state()
        checkpoint_data["state"] = {k: asdict(v) for k, v in state.items()}

    # Snapshot of registry (just metadata, not full file)
    registry_path = _PATHS["project_root"] / "orchestration/model_registry.yaml"
    if registry_path.exists():
        checkpoint_data["registry_snapshot"] = {
            "path": str(registry_path),
            "mtime": registry_path.stat().st_mtime,
            "size": registry_path.stat().st_size,
        }

    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint_data, f, indent=2)

    return {
        "checkpoint_id": checkpoint_id,
        "path": str(checkpoint_path),
        "created_at": checkpoint_data["created_at"],
    }


def checkpoint_restore(checkpoint_id: str) -> dict[str, Any]:
    """Restore orchestrator stack from a checkpoint.

    Args:
        checkpoint_id: ID from checkpoint_create.

    Returns:
        Dict with restoration status.
    """
    checkpoint_path = CHECKPOINT_DIR / f"{checkpoint_id}.json"

    if not checkpoint_path.exists():
        return {"success": False, "error": f"Checkpoint not found: {checkpoint_id}"}

    try:
        with open(checkpoint_path) as f:
            checkpoint_data = json.load(f)

        # Restore state (process info)
        if checkpoint_data.get("state"):
            saved_state = {
                k: ProcessInfo(**v)
                for k, v in checkpoint_data["state"].items()
            }
            save_state(saved_state)

        return {
            "success": True,
            "checkpoint_id": checkpoint_id,
            "restored_at": datetime.now().isoformat(),
            "original_created_at": checkpoint_data.get("created_at"),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def checkpoint_list(limit: int = 10) -> list[dict[str, Any]]:
    """List available checkpoints.

    Args:
        limit: Maximum number to return (newest first).

    Returns:
        List of checkpoint summaries.
    """
    if not CHECKPOINT_DIR.exists():
        return []

    checkpoints = []
    for cp_path in sorted(CHECKPOINT_DIR.glob("*.json"), reverse=True)[:limit]:
        try:
            with open(cp_path) as f:
                data = json.load(f)
            checkpoints.append({
                "id": data.get("id", cp_path.stem),
                "name": data.get("name"),
                "created_at": data.get("created_at"),
                "path": str(cp_path),
            })
        except (json.JSONDecodeError, OSError, KeyError):
            pass  # Skip malformed or unreadable checkpoint files

    return checkpoints


def checkpoint_delete(checkpoint_id: str) -> bool:
    """Delete a checkpoint.

    Args:
        checkpoint_id: Checkpoint to delete.

    Returns:
        True if deleted, False if not found.
    """
    checkpoint_path = CHECKPOINT_DIR / f"{checkpoint_id}.json"
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        return True
    return False


# Export hooks for use by procedure_registry
__checkpoint_hooks__ = {
    "create": checkpoint_create,
    "restore": checkpoint_restore,
    "list": checkpoint_list,
    "delete": checkpoint_delete,
}


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description="Orchestrator stack manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Start command
    start_parser = subparsers.add_parser("start", help="Start the stack")
    start_parser.add_argument("--hot-only", action="store_true", help="Start HOT models only")
    start_parser.add_argument("--include-warm", nargs="+", metavar="ROLE", help="Include WARM models")
    start_parser.add_argument("--only", nargs="+", metavar="ROLE",
                              help="Start ONLY these roles (skip everything else). "
                                   "Searches both HOT and WARM server lists.")
    start_parser.add_argument("--dev", action="store_true", help="Dev mode (single 0.5B model)")
    start_parser.add_argument(
        "--numa-mode",
        choices=["full", "quarter", "both"],
        default="both",
        help=(
            "For roles with both a full-NUMA-node instance and quarter-instance siblings "
            "(currently frontdoor + coder_escalation + worker_general — see "
            "NUMA_CONFIG[role]['full_instance_idx']), pick one mode. "
            "'full' = single full instance (max single-stream tps; recommended for single-user "
            "workloads). 'quarter' = 4 concurrent quarters (max aggregate under multi-request "
            "load). 'both' = default, preserves pre-2026-05-08 behavior with all 5 — viable "
            "when the role's -t is small enough to avoid CPU oversubscription (Qwen3-Coder -t 24 "
            "and Qwen3.6-35B Q8 quarter-tuned were OK; gemma4-MTP -t 96 will hit load 420 → "
            "9 t/s with 'both', so use --numa-mode full for that role specifically). "
            "Single-instance roles (architect_general, ingest_long_context, embedders) are "
            "unaffected by this flag."
        ),
    )
    start_parser.add_argument(
        "--skip-host-prereqs",
        action="store_true",
        help="Skip host_prereq audit/apply (numa_balancing, THP, governor). NOT recommended for benchmarks.",
    )
    start_parser.add_argument(
        "--profile",
        choices=sorted(ORCHESTRATOR_PROFILES.keys()),
        help="Optional orchestrator API env profile",
    )
    start_parser.add_argument(
        "--stack-profile",
        metavar="NAME",
        help="Load stack template from stack_templates/<NAME>.yaml (DS-7). "
             "Use --validate-only to check without launching.",
    )
    start_parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate stack template and exit (use with --stack-profile)",
    )
    start_parser.add_argument(
        "--migrate-to",
        metavar="NAME",
        help="Migrate running stack to stack_templates/<NAME>.yaml via full "
             "restart (DS-7 / NIB2-19). Use with --dry-run to plan only.",
    )
    start_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --migrate-to, plan the migration without stopping any servers.",
    )

    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop components")
    stop_parser.add_argument("--all", action="store_true", help="Stop all components")
    stop_parser.add_argument("components", nargs="*", help="Components to stop")

    # Reload command
    reload_parser = subparsers.add_parser("reload", help="Reload components")
    reload_parser.add_argument("components", nargs="+", help="Components to reload")
    reload_parser.add_argument(
        "--profile",
        choices=sorted(ORCHESTRATOR_PROFILES.keys()),
        help="Optional orchestrator API env profile (used when reloading orchestrator)",
    )

    # Status command
    subparsers.add_parser("status", help="Show status")

    args = parser.parse_args()

    if args.command == "start":
        return cmd_start(args)
    elif args.command == "stop":
        return cmd_stop(args)
    elif args.command == "reload":
        return cmd_reload(args)
    elif args.command == "status":
        return cmd_status(args)

    return 1


if __name__ == "__main__":
    sys.exit(main())
