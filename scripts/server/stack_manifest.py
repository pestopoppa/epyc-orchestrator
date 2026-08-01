"""Orchestrator stack manifest — ports, role aliases, model paths, classifications.

Extracted from orchestrator_stack.py during the 2026-05-22 Tranche-7 refactor.
Owns all the static configuration that describes WHAT the orchestrator should
launch and WHERE: PORT_MAP, HOT_ROLES, SERIAL_ROLES, ROLE_LAUNCH_META, model
paths, NUMA_REPLICA_PORTS, plus the classification helpers that compute
HOT_SERVERS / WARM_SERVERS from ROLE_LAUNCH_META + NUMA_CONFIG.

orchestrator_stack.py re-imports every name here, and the registry compiler's
`from orchestrator_stack import ROLE_LAUNCH_META` fallback path keeps working.

2026-08-01 — THIS MODULE IS NOW A THIN LOADER. Every table below used to be a
Python literal; they now live in `orchestration/launch_manifest.yaml` (launch
data) and `orchestration/stack_topology.yaml` (per-role NUMA wiring, via
stack_numa). The names, types and values are unchanged — `frozenset` is still
`frozenset`, tuples are still tuples — so no consumer changed.

The reason is not tidiness. The duplication this module accumulated (and the
ring of local fallback tables around it) existed BECAUSE it was code: a Python
table invites a second Python table beside it, and every severe launcher defect
found in the 2026-07 audit had the same shape — the compiled artifact was
correct and a local literal won anyway. Data does not invite that.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from scripts.server.stack_numa import NUMA_CONFIG
from scripts.server.stack_paths import LLAMA_MATH_TOOLS, _V2_ROLES, _PATHS


# =============================================================================
# Declared-data loader
# =============================================================================

_LAUNCH_MANIFEST_PATH = Path(__file__).resolve().parents[2] / "orchestration" / "launch_manifest.yaml"

# Path placeholders the manifest may use. Substitution is literal, not
# str.format, so a value containing braces for some other reason is left alone
# instead of raising a KeyError at import.
_PATH_TOKENS: dict[str, str] = {
    "{models_dir}": str(_PATHS["models_dir"]),
    "{model_base}": str(_PATHS["model_base"]),
    "{project_root}": str(_PATHS["project_root"]),
    "{cache_dir}": str(_PATHS["cache_dir"]),
    "{llm_root}": str(_PATHS["llm_root"]),
    "{tmp_dir}": str(_PATHS["tmp_dir"]),
}

_REQUIRED_MANIFEST_SECTIONS = (
    "port_map",
    "optional_auxiliary_roles",
    "hot_roles",
    "serial_roles",
    "role_launch_meta",
    "models",
    "embedding",
    "vision",
    "launch_shape",
    "gpu_shadow_lane",
    "orchestrator_profiles",
    "docker_services",
)


def _expand_paths(value):
    """Recursively expand {models_dir}-style placeholders in loaded strings."""
    if isinstance(value, str):
        for token, replacement in _PATH_TOKENS.items():
            if token in value:
                value = value.replace(token, replacement)
        return value
    if isinstance(value, list):
        return [_expand_paths(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_paths(item) for key, item in value.items()}
    return value


def _load_launch_manifest(path: Path | None = None) -> dict:
    """Load orchestration/launch_manifest.yaml.

    No fallback table and no partial load. A launcher that cannot read its own
    manifest must fail at IMPORT rather than start something plausible — the
    failure mode this refactor exists to remove is precisely "a local default
    won over the declared value and the run looked fine".
    """
    manifest_path = _LAUNCH_MANIFEST_PATH if path is None else path
    try:
        document = yaml.safe_load(manifest_path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"stack_manifest: declared launch manifest missing at {manifest_path}. "
            f"The launcher has no fallback manifest by design."
        ) from exc
    if not isinstance(document, dict):
        raise ValueError(f"stack_manifest: {manifest_path} did not parse to a mapping")
    missing = [key for key in _REQUIRED_MANIFEST_SECTIONS if key not in document]
    if missing:
        raise ValueError(f"stack_manifest: {manifest_path} is missing section(s) {missing}")
    return _expand_paths(document)


_MANIFEST = _load_launch_manifest()


# =============================================================================
# Port assignments by role (primary ports — full-speed 1×96t instances)
# Pre-warm (2026-03-29): primary port is the full-speed instance.
# Quarter instances on offset ports (808x, 818x, 828x, 838x).
# Declared in launch_manifest.yaml `port_map:` — including the per-role comments
# that record WHY each port is what it is.
# =============================================================================

PORT_MAP = dict(_MANIFEST["port_map"])

# Stack-managed auxiliaries whose startup failure is non-fatal but must remain
# visible in the read-only status surface even when no state row was created.
OPTIONAL_AUXILIARY_ROLES = frozenset(_MANIFEST["optional_auxiliary_roles"])

# HOT roles (always started) - NUMA-optimized
HOT_ROLES = set(_MANIFEST["hot_roles"])

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
SERIAL_ROLES = set(_MANIFEST["serial_roles"])


# =============================================================================
# Single source of truth for "which roles run where" (refactored 2026-05-06)
# =============================================================================
# Pre-2026-05-06: HOT_SERVERS + WARM_SERVERS were two parallel hand-edited lists
# of dicts that duplicated wiring data already in NUMA_CONFIG. This made it easy
# to forget one when adding/removing/renaming a role and ship a broken config.
#
# Post-2026-05-06: HOT_SERVERS / WARM_SERVERS are COMPUTED from:
#   1. ROLE_LAUNCH_META — small classification dict: per-role tier + launch mode
#      + aliases + mode-specific kwargs. Declared in launch_manifest.yaml.
#   2. NUMA_CONFIG (stack_numa) — wiring spec: per-role NUMA instances list.
#      Declared in stack_topology.yaml.
#
# Adding a new role now requires editing TWO declared places consistently:
#   a) Add the role's NUMA wiring in stack_topology.yaml `numa_config:` (or set
#      `no_numa: true` in role_launch_meta if the role doesn't need NUMA pinning,
#      e.g. embedders).
#   b) Add a role_launch_meta entry with tier/mode/aliases.
# A consistency check (`_validate_role_classification()` below) catches
# common mismatches at module load time.

# Per-role launch metadata. Source of truth for tier classification + launch
# mode. Insertion order is load-bearing — HOT_SERVERS/WARM_SERVERS are emitted
# in it — and YAML mapping order is preserved on load.
ROLE_LAUNCH_META: dict[str, dict] = {
    role: dict(meta) for role, meta in _MANIFEST["role_launch_meta"].items()
}



# =============================================================================
# Model paths — declared in launch_manifest.yaml `models:` / `embedding:` /
# `vision:`. The measured provenance for each choice moved with the data.
# =============================================================================

_MODELS = _MANIFEST["models"]
_EMBEDDING = _MANIFEST["embedding"]

# Embedding model: BGE-large-en-v1.5 (purpose-built for embeddings, 1024 dims)
# 6 parallel instances provide redundancy and reduce latency via fan-out
EMBEDDING_MODEL_PATH = _MODELS["embedding_model_path"]
EMBEDDER_PORTS = list(_EMBEDDING["pool_ports"])
# Each pool port gets its OWN recipe dict (never a shared one — a caller that
# mutated a shared dict would silently retune the whole pool).
EMBEDDING_SERVER_RECIPES: dict[int, dict[str, str | int | bool]] = {
    port: {"model_path": EMBEDDING_MODEL_PATH, **_EMBEDDING["pool_recipe"]}
    for port in EMBEDDER_PORTS
}
EMBEDDING_SERVER_RECIPES.update(
    {int(port): dict(recipe) for port, recipe in _EMBEDDING["extra_recipes"].items()}
)

# Worker pool models (FIXED paths to existing files)
# NOTE: worker_coder uses the fast 1.5B worker backend on port 8102.
WORKER_POOL_MODELS = dict(_MODELS["worker_pool"])

# Draft model for MTP speculative decoding on explore worker.
EXPLORE_DRAFT_MODEL = _MODELS["explore_draft_model"]

# Vision models (VL) with multimodal projector.
_VISION_WORKER = _MANIFEST["vision"]["worker"]
_VISION_ESCALATION = _MANIFEST["vision"]["escalation"]
VISION_WORKER_MODEL = _VISION_WORKER["model"]
VISION_WORKER_MMPROJ = _VISION_WORKER["mmproj"]
VISION_WORKER_DEVICE = _VISION_WORKER["device"]
VISION_WORKER_IMAGE_MIN_TOKENS = _VISION_WORKER["image_min_tokens"]
VISION_WORKER_CACHE_RAM = _VISION_WORKER["cache_ram"]
# vision_escalation is an ALIAS on the worker's process, not a second server, so
# its model/mmproj/device are BOUND rather than declared twice. Declaring them
# separately is exactly how the retired :8087 lane drifted to a different model.
if not _VISION_ESCALATION.get("same_as_worker"):
    raise ValueError(
        "stack_manifest: vision.escalation must declare same_as_worker: true — "
        "vision_escalation is an alias on the worker_vision process, and a second "
        "independent model declaration is the drift this binding exists to prevent"
    )
VISION_ESCALATION_MODEL = VISION_WORKER_MODEL
VISION_ESCALATION_MMPROJ = VISION_WORKER_MMPROJ
VISION_ESCALATION_DEVICE = VISION_WORKER_DEVICE
VISION_ESCALATION_REASONING = _VISION_ESCALATION["reasoning"]

_LAUNCH_SHAPE = _MANIFEST["launch_shape"]
DEFAULT_EFFECTIVE_CONTEXT_TOKENS = _LAUNCH_SHAPE["default_effective_context_tokens"]
LAUNCH_CONTEXT_TOKENS = dict(_LAUNCH_SHAPE["context_tokens"])

DEFAULT_UBATCH_TOKENS = _LAUNCH_SHAPE["default_ubatch_tokens"]

# Effective launcher KV settings. Descriptors may preserve broader model
# capability metadata; this table witnesses the actual llama-server CLI path.
LAUNCH_KV_QUANT_CONFIGS = {
    role: tuple(pair) for role, pair in _LAUNCH_SHAPE["kv_quant_configs"].items()
}
NO_SPEC_DECODE_ROLES: set[str] = set(_LAUNCH_SHAPE["no_spec_decode_roles"])
KV_HADAMARD_ROLES = set(_V2_ROLES)

# =============================================================================
# GPU shadow lane launch constants (docs/gpu-shadow-lane.md; gpu-serving-tie-in
# P2-6/P0-2). Declared in launch_manifest.yaml `gpu_shadow_lane:`; INERT until a
# registry proposal adds the matching role_launch_meta + numa_config + port_map
# entries.
# =============================================================================

_GPU_SHADOW_LANE = _MANIFEST["gpu_shadow_lane"]
GPU_SHADOW_LANE_TENANT_ROLE = _GPU_SHADOW_LANE["tenant_role"]
GPU_SHADOW_LANE_DEVICE = _GPU_SHADOW_LANE["device"]
GPU_SHADOW_LANE_REASONING = _GPU_SHADOW_LANE["reasoning"]
GPU_SHADOW_LANE_FALLBACK_SLOTS = _GPU_SHADOW_LANE["fallback_slots"]
GPU_SHADOW_LANE_FALLBACK_CONTEXT_TOKENS = _GPU_SHADOW_LANE["fallback_context_tokens"]

DEV_MODEL = _MODELS["dev_model"]
DEV_MODEL_PATH = str(_PATHS["models_dir"] / DEV_MODEL)

# Optional orchestrator API launch profiles for repeatable debugging runs.
ORCHESTRATOR_PROFILES: dict[str, dict[str, str]] = {
    name: dict(env) for name, env in _MANIFEST["orchestrator_profiles"].items()
}


# =============================================================================
# Docker Services (NextPLAID retrieval + SearXNG metasearch + Crawl4AI extraction)
# Declared in launch_manifest.yaml `docker_services:`, including the port-choice
# rationale (searxng 8888 not 8090; crawl4ai 11235 not 8086).
# =============================================================================

DOCKER_SERVICES = [dict(service) for service in _MANIFEST["docker_services"]]



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
    dropped_full_aliases: dict[str, list[str]] = {}
    first_kept_by_role: dict[str, dict] = {}
    for srv in servers:
        # The primary role is roles[0]; aliases share its NUMA_CONFIG.
        roles = srv["roles"]
        role = roles[0]
        cfg = NUMA_CONFIG.get(role)
        if not cfg or "full_instance_idx" not in cfg or len(cfg["instances"]) <= 1:
            out.append(srv)
            continue
        full_idx = cfg["full_instance_idx"]
        srv_idx = srv.get("numa_instance", 0)
        if mode == "full" and srv_idx == full_idx:
            out.append(srv)
        elif mode == "quarter" and srv_idx != full_idx:
            kept = dict(srv)
            kept["roles"] = list(roles)
            out.append(kept)
            first_kept_by_role.setdefault(role, kept)
        elif mode == "quarter" and srv_idx == full_idx and len(roles) > 1:
            dropped_full_aliases.setdefault(role, []).extend(str(alias) for alias in roles[1:])
    if mode == "quarter":
        for role, aliases in dropped_full_aliases.items():
            target = first_kept_by_role.get(role)
            if not target:
                continue
            roles = target.get("roles")
            if not isinstance(roles, list):
                continue
            for alias in aliases:
                if alias not in roles:
                    roles.append(alias)
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
        elif mode == "eval_batch_frontdoor":
            mode_flags["eval_batch_frontdoor"] = True
        elif mode == "gpu_shadow_lane":
            # gpu-serving-tie-in P2-6 (P0-2): mode branch for the GPU shadow
            # lane launcher (docs/gpu-shadow-lane.md). INERT today — no
            # ROLE_LAUNCH_META entry carries this mode until the registry
            # proposal (docs/proposals/gpu-shadow-lane-registry-proposal.md)
            # is applied at the operator-gated activation. Witnessed by
            # tests/unit/test_gpu_shadow_lane.py (State-A inertness witness).
            mode_flags["gpu_shadow_lane"] = True

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
