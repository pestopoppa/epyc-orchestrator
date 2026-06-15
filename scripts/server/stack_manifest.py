"""Orchestrator stack manifest — ports, role aliases, model paths, classifications.

Extracted from orchestrator_stack.py during the 2026-05-22 Tranche-7 refactor.
Owns all the static configuration that describes WHAT the orchestrator should
launch and WHERE: PORT_MAP, HOT_ROLES, SERIAL_ROLES, ROLE_LAUNCH_META, model
paths, NUMA_REPLICA_PORTS, plus the classification helpers that compute
HOT_SERVERS / WARM_SERVERS from ROLE_LAUNCH_META + NUMA_CONFIG.

orchestrator_stack.py re-imports every name here, and the registry compiler's
`from orchestrator_stack import ROLE_LAUNCH_META` fallback path keeps working.
"""

from __future__ import annotations

from pathlib import Path

from scripts.server.stack_numa import NUMA_CONFIG
from scripts.server.stack_paths import LLAMA_MATH_TOOLS, _V2_ROLES, _PATHS


# =============================================================================
# Port assignments by role (primary ports — full-speed 1×96t instances)
# Pre-warm (2026-03-29): primary port is the full-speed instance.
# Quarter instances on offset ports (808x, 818x, 828x, 838x).
# =============================================================================

PORT_MAP = {
    "frontdoor": 8070,  # Full-speed 1×96t (quarters: 8080, 8180, 8280, 8380)
    "coder_escalation": 8070,  # Alias -> frontdoor shared Qwen3.6 server
    "worker_summarize": 8070,  # Alias -> frontdoor shared Qwen3.6 server
    "worker_general": 8072,  # Full-speed 1×96t (quarters: 8082, 8182, 8282, 8382)
    "worker_explore": 8072,  # Alias -> worker_general (legacy name; pre-2026-03-19)
    "worker_math": 8072,  # Shares with worker_general
    "toolrunner": 8072,  # Shares with worker_general
    "worker_vision": 8086,  # Dedicated VL server
    "vision_escalation": 8087,  # VL escalation (Qwen3-VL-30B MoE)
    "worker_coder": 8102,  # Fast coding worker semantic role (1.5B backend) — DEPRECATED (worker_pool)
    "worker_fast": 8102,  # Fast worker (1.5B, WARM, 4 slots) — DEPRECATED (worker_pool)
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
    "sd_server": 8190,  # ERNIE-Image-Turbo via stable-diffusion.cpp native (replaced ComfyUI 2026-05-07; ~1.7-3.4× CPU speedup)
    "whisper": 9000,  # faster-whisper STT (transcription service, not llama-server)
}

# HOT roles (always started) - NUMA-optimized
# 2026-05-06: architect_coding removed, ingest_long_context promoted (see registry).
HOT_ROLES = {
    "frontdoor",
    "coder_escalation",
    "worker_general",
    "embedder",
    "architect_general",
    "ingest_long_context",
    "worker_vision",
    "vision_escalation",
}

# All NUMA replica ports (for port scanning and cleanup) — derived from
# NUMA_CONFIG + PORT_MAP. Stays here because it depends on both the manifest's
# PORT_MAP and stack_numa.NUMA_CONFIG.
NUMA_REPLICA_PORTS = {
    port
    for cfg in NUMA_CONFIG.values()
    for _, port, _ in cfg["instances"]
    if port not in PORT_MAP.values()
}


# Roles that must never run concurrently (large/latency-sensitive paths).
# Note: frontdoor intentionally runs with 2 slots by default for better
# interactive responsiveness under concurrent traffic.
# 2026-05-06: removed architect_coding (role eliminated) + thinking_reasoning (role eliminated, no GGUF).
SERIAL_ROLES = {
    "coder_escalation",
    "worker_summarize",
    # 2026-05-11: frontdoor added to test -np 1 vs -np 2 single-instance throughput.
    # -np 2 was inherited from quarter-mode (4×48t) where two slots gave round-robin
    # admission. In full-mode 1×96t single-user serving, slot 2 is just preallocated
    # KV that we never use — wastes L3. If -np 1 measurably improves throughput,
    # keep frontdoor in SERIAL_ROLES; otherwise revert.
    "frontdoor",
    # architect_general REMOVED 2026-05-09: -np 1 + speculative tree decode
    # (draft_max=24, p_split=0) trips a llama.cpp assertion in
    # common_speculative_state_tree::draft — `init: ... starting position of Y < cache position X`
    # → `decode: failed to initialize batch` →
    # `init: invalid seq_id[1][0] = 1 >= 1` →
    # `GGML_ASSERT(logits != nullptr)` in common/sampling.cpp:152.
    # Build b8957-2ffbdbbba (production llama.cpp). Re-enable when upstream patches
    # M-RoPE rollback or we move to a binary that supports it. Throughput cost: spec
    # decode previously +25% (4.3→12.6 t/s with moe8+spec_q8); MoE-expert-reduction
    # path stays active so we keep the moe-budget gain (12.19 t/s probe-B canonical).
    "ingest_long_context",
    "vision_escalation",
    "formalizer",
    "toolrunner",
}


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
#   2. NUMA_CONFIG (stack_numa) — wiring spec: per-role NUMA instances list.
#
# Adding a new role now requires editing TWO places consistently:
#   a) Add the role's NUMA wiring in NUMA_CONFIG (or set "no_numa": True in
#      ROLE_LAUNCH_META if the role doesn't need NUMA pinning, e.g. embedders).
#   b) Add a ROLE_LAUNCH_META entry with tier/mode/aliases.
# A consistency check (`_validate_role_classification()` below) catches
# common mismatches at module load time.

# Per-role launch metadata. Source of truth for tier classification + launch mode.
ROLE_LAUNCH_META: dict[str, dict] = {
    # ---- HOT tier (always started) ----
    # 2026-05-09: coder_escalation + worker_summarize CONSOLIDATED onto frontdoor's
    # llama-server. All three share the same Qwen3.6-35B-A3B Q8 GGUF since the
    # 2026-05-06 model swap; running them as separate processes wasted 36 GB of
    # mlocked RAM and ran two competing 96-thread OMP teams (-40% to -69%
    # throughput on the cohabiting roles per 2026-05-09 measurements). The
    # orchestrator API routes by role name → registry's url field (all three
    # roles point at port 8070 in the registry).
    "frontdoor": {
        "tier": "hot",
        "mode": "default",
        "shared_with_first_n": ["coder_escalation", "worker_summarize"],
    },
    # coder_escalation entry REMOVED 2026-05-09 — consolidated into frontdoor above.
    # NUMA_CONFIG['coder_escalation'] left in place as dead key for git-history
    # blame purposes; not referenced by build_servers_from_classification anymore.
    "worker_general": {
        "tier": "hot",
        "mode": "worker_pool",
        "worker_type": "explore",
        # Aliases that share the worker_general process: worker_explore is the
        # legacy name (pre-2026-03-19 worker pool design); worker_math + toolrunner
        # share the GGUF mmap and process for routing fan-out.
        "shared_with_first_n": ["worker_explore", "worker_math", "toolrunner"],
        "shared_with_first_n_count": 2,
    },  # aliases on full + first quarter
    "architect_general": {"tier": "hot", "mode": "default"},
    "ingest_long_context": {"tier": "hot", "mode": "default"},
    "worker_vision": {"tier": "hot", "mode": "vision", "vision_type": "worker"},
    "vision_escalation": {"tier": "hot", "mode": "vision", "vision_type": "escalation"},
    # Embedders — no NUMA pinning, fixed single port each
    "embedder": {"tier": "hot", "mode": "embedding", "no_numa": True, "port": 8090},
    "embedder_1": {"tier": "hot", "mode": "embedding", "no_numa": True, "port": 8091},
    "embedder_2": {"tier": "hot", "mode": "embedding", "no_numa": True, "port": 8092},
    "embedder_3": {"tier": "hot", "mode": "embedding", "no_numa": True, "port": 8093},
    "embedder_4": {"tier": "hot", "mode": "embedding", "no_numa": True, "port": 8094},
    "embedder_5": {"tier": "hot", "mode": "embedding", "no_numa": True, "port": 8095},
    # ---- WARM tier (optional, --include-warm) ----
    # 2026-05-06: worker_pool deprecated in registry; warm 1.5B worker retained as inert.
    "worker_fast": {
        "tier": "warm",
        "mode": "worker_pool",
        "worker_type": "fast",
        "no_numa": True,
        "port": 8102,
    },
    # architect_coding REMOVED 2026-05-06 (REAP-246B role eliminated; 139 GB freed)
    # ingest_long_context PROMOTED to HOT 2026-05-06 (Stage 1 of three_stage_summarization)
    # thinking_reasoning REMOVED 2026-05-06 (GGUF deleted from disk 2026-03-06)
}


# =============================================================================
# Model paths
# =============================================================================

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
    "fast": str(
        _PATHS["model_base"] / "QuantFactory/Qwen2.5-Coder-1.5B-GGUF/Qwen2.5-Coder-1.5B.Q4_K_M.gguf"
    ),
}

# Draft model for MTP speculative decoding on explore worker.
# gemma4 assistant Q8 — in-house GGUF converted from google/gemma-4-26B-A4B-it-assistant
# safetensors (no community GGUF existed). 4-layer drafter, 58% acceptance at draft_max=2.
EXPLORE_DRAFT_MODEL = "/mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-Q8_0.gguf"

# Vision models (VL) with multimodal projector
VISION_WORKER_MODEL = str(
    _PATHS["model_base"]
    / "lmstudio-community/Qwen2.5-VL-7B-Instruct-GGUF/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
)
VISION_WORKER_MMPROJ = str(
    _PATHS["model_base"] / "lmstudio-community/Qwen2.5-VL-7B-Instruct-GGUF/mmproj-model-f16.gguf"
)
VISION_ESCALATION_MODEL = str(
    _PATHS["model_base"]
    / "lmstudio-community/Qwen3-VL-30B-A3B-Instruct-GGUF/Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf"
)
VISION_ESCALATION_MMPROJ = str(
    _PATHS["model_base"]
    / "lmstudio-community/Qwen3-VL-30B-A3B-Instruct-GGUF/mmproj-Qwen3-VL-30B-A3B-Instruct-F16.gguf"
)

DEFAULT_EFFECTIVE_CONTEXT_TOKENS = 32768
LAUNCH_CONTEXT_TOKENS = {
    "worker_general": 16384,
    "worker_fast": 16384,
    "worker_vision": 8192,
    "vision_escalation": 16384,
    "architect_general": 16384,
    "ingest_long_context": 32768,
}

DEFAULT_UBATCH_TOKENS = 8192
WORKER_MTP_UBATCH_TOKENS = 512
WORKER_MTP_SPEC_TYPE = "mtp"
WORKER_MTP_DRAFT_MAX = 2
WORKER_MTP_DRAFT_P_MIN = 0.0
WORKER_MTP_THREADS_DRAFT = 16
WORKER_MTP_KV_TYPES = ("q8_0", "q8_0")

# Effective launcher KV settings. Descriptors may preserve broader model
# capability metadata; this table witnesses the actual llama-server CLI path.
LAUNCH_KV_QUANT_CONFIGS = {
    "frontdoor": ("q8_0", "q8_0"),
    "coder_escalation": ("q8_0", "q8_0"),
    "worker_summarize": ("q8_0", "q8_0"),
    "worker_general": WORKER_MTP_KV_TYPES,
    "worker_math": WORKER_MTP_KV_TYPES,
    "toolrunner": WORKER_MTP_KV_TYPES,
    "architect_general": ("q4_0", "f16"),
    "ingest_long_context": ("q4_0", "q4_0"),
}
NO_SPEC_DECODE_ROLES = {"architect_general"}
KV_HADAMARD_ROLES = set(_V2_ROLES)

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
    "gate3-tool-telemetry": {
        "AUTOPILOT_TOOL_SENTINELS": "1",
        "ORCHESTRATOR_STRUCTURED_TOOL_OUTPUT": "1",
    },
}


# =============================================================================
# Docker Services (NextPLAID retrieval + SearXNG metasearch + Crawl4AI extraction)
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
        "args": [
            "--host",
            "0.0.0.0",
            "--port",
            "8080",
            "--index-dir",
            "/data/indices",
            "--model",
            "lightonai/LateOn-Code",
            "--int8",
        ],
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
        "args": [
            "--host",
            "0.0.0.0",
            "--port",
            "8080",
            "--index-dir",
            "/data/indices",
            "--model",
            "/models/gte-moderncolbert-v1-onnx",
            "--int8",
        ],
    },
    {
        "name": "searxng",
        # 8888 (not 8090): the embedder pool occupies 8090-8095, so leaving
        # searxng on 8090 collides with embedder_0 and the container fails
        # docker-run networking. 8888 sits outside both the llama-server
        # 80xx range and the reembed_episodic_store.py probe range.
        "port": 8888,
        "image": "docker.io/searxng/searxng:latest",
        "description": "Metasearch aggregator (JSON API for web_search)",
        "volumes": [
            f"{_PATHS['project_root']}/config/searxng:/etc/searxng:Z",
        ],
        "args": [],  # Config via mounted settings.yml, not CLI args
        "health_path": "/",  # SearXNG serves HTML on /, not /health
    },
    {
        "name": "crawl4ai",
        # Crawl4AI's maintained Docker deployment serves on 11235. Do not use
        # the old handoff hint of 8086; that is the worker_vision port.
        "port": 11235,
        "container_port": 11235,
        "image": "unclecode/crawl4ai:latest",
        "description": "Browser-backed page extraction for web_research",
        "shm_size": "1g",
        "run_timeout": 180,
        "args": [],
        "health_path": "/health",
    },
]


# =============================================================================
# Server-list classification helpers
# =============================================================================


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
        alias_count = meta.get(
            "shared_with_first_n_count", 1
        )  # default: aliases on first instance only

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
                errors.append(f"Port {port} assigned to both '{primary_ports[port]}' and '{role}'")
            primary_ports[port] = role

    if errors:
        msg = "Role classification inconsistencies detected:\n  " + "\n  ".join(errors)
        raise ValueError(msg)


# Validate at module load (fails fast on misconfiguration)
_validate_role_classification()

# Computed server lists. Single source of truth: ROLE_LAUNCH_META + NUMA_CONFIG.
HOT_SERVERS, WARM_SERVERS = _build_servers_from_classification()


def validate_against_registry(
    registry_yaml_path: str | None = None,
    *,
    roles: set[str] | None = None,
) -> list[str]:
    """Cross-check ROLE_LAUNCH_META against orchestration/model_registry.yaml.

    Returns list of warning strings (empty if everything is consistent).
    Called from `start` command but non-fatal: prints warnings, does not abort.
    Useful for catching drift between launcher classification and registry's
    process_layout / server_mode sections.
    """
    if registry_yaml_path is None:
        # Default: orchestration/model_registry.yaml relative to repo root
        registry_yaml_path = str(
            Path(__file__).parent.parent.parent / "orchestration" / "model_registry.yaml"
        )
    warnings: list[str] = []
    role_scope = set(roles) if roles is not None else None

    def in_scope(role: str) -> bool:
        return role_scope is None or role in role_scope

    try:
        import yaml

        with open(registry_yaml_path) as f:
            registry = yaml.safe_load(f)
    except Exception as e:
        return [f"could not load registry from {registry_yaml_path}: {e}"]

    pl = registry.get("process_layout", {})
    has_process_layout = isinstance(pl, dict) and any(
        key in pl for key in ("hot_resident", "warm_mmap")
    )
    reg_hot = set(pl.get("hot_resident", [])) if isinstance(pl, dict) else set()
    reg_warm = set(pl.get("warm_mmap", [])) if isinstance(pl, dict) else set()

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

    # Production roles to cross-check — skip launcher-only aliases not present
    # in the registry process_layout. This keeps the validator aligned with the
    # live hot/warm sets instead of a hand-maintained internal-name list.
    skip = {role for role in launcher_hot if role not in reg_hot and role not in reg_warm}
    launcher_hot_filtered = {role for role in launcher_hot - skip if in_scope(role)}
    reg_hot_filtered = {role for role in reg_hot if in_scope(role)}
    reg_warm_filtered = {role for role in reg_warm if in_scope(role)}

    if has_process_layout:
        only_in_launcher = launcher_hot_filtered - reg_hot_filtered - reg_warm_filtered
        only_in_registry_hot = reg_hot_filtered - launcher_hot - launcher_warm

        for r in sorted(only_in_launcher):
            warnings.append(
                f"role '{r}' is HOT in launcher but absent from registry process_layout"
            )
        for r in sorted(only_in_registry_hot):
            # Roles that the registry says should be hot but launcher doesn't classify hot
            warnings.append(
                f"role '{r}' is hot_resident in registry but not in launcher's HOT tier"
            )

    # Cross-check legacy/direct role port hints against computed launch roles.
    # PORT_MAP is still used by targeted reload/status compatibility paths; it
    # must agree with ROLE_LAUNCH_META + NUMA_CONFIG for shared aliases.
    computed_role_ports: dict[str, int] = {}
    for server in HOT_SERVERS + WARM_SERVERS:
        port = server.get("port")
        if not isinstance(port, int):
            continue
        for role in server.get("roles", []):
            computed_role_ports.setdefault(str(role), port)
    for role, port in PORT_MAP.items():
        if not in_scope(role):
            continue
        computed_port = computed_role_ports.get(role)
        if computed_port is not None and port != computed_port:
            warnings.append(
                f"role '{role}': PORT_MAP says port {port}, "
                f"but computed launch roles use port {computed_port}"
            )

    # Cross-check NUMA_CONFIG ports vs registry server_mode ports (where present)
    sm = registry.get("server_mode", {})
    for role, srv in sm.items():
        if not isinstance(srv, dict):
            continue
        covered_roles = {str(role)}
        model_role = srv.get("model_role")
        if isinstance(model_role, str):
            covered_roles.add(model_role)
        shared_with = srv.get("shared_with")
        if isinstance(shared_with, list):
            covered_roles.update(str(item) for item in shared_with if isinstance(item, str))
        if role_scope is not None and not covered_roles & role_scope:
            continue
        reg_port = srv.get("port")
        if reg_port is None:
            continue
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
            if not meta:
                for covered_role in sorted(covered_roles):
                    if not in_scope(covered_role):
                        continue
                    computed_port = computed_role_ports.get(covered_role)
                    if computed_port is not None and reg_port != computed_port:
                        warnings.append(
                            f"role '{role}': registry server_mode says port {reg_port} "
                            f"for launch role '{covered_role}', "
                            f"but computed launch roles use port {computed_port}"
                        )
    return warnings


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
        ("architect_general", str(_PATHS["model_base"] / "unsloth/Qwen3.5-122B-A10B-GGUF/")),
        (
            "ingest_long_context",
            str(_PATHS["model_base"] / "lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF/"),
        ),
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
    cpp_math_tools = LLAMA_MATH_TOOLS
    if not cpp_math_tools.exists():
        # This is a warning, not an error - append with different prefix
        pass  # Will be checked separately in init_memrl_and_tools

    return errors
