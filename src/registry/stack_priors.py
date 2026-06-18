"""Compile derived stack priors from registry and model descriptors.

This module is intentionally additive: existing consumers can migrate to the
generated artifact one by one instead of re-parsing scattered registry comments
and hardcoded role tables.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import re
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml


REPO_ROOT = Path("/mnt/raid0/llm/epyc-orchestrator")
DEFAULT_REGISTRY = REPO_ROOT / "orchestration" / "model_registry.yaml"
DEFAULT_DESCRIPTORS = REPO_ROOT / "orchestration" / "model_descriptors.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "orchestration" / "derived" / "stack_priors.yaml"
DEFAULT_STACK_MANIFEST = REPO_ROOT / "scripts" / "server" / "stack_manifest.py"
DEFAULT_STACK_NUMA = REPO_ROOT / "scripts" / "server" / "stack_numa.py"
DEFAULT_ORCHESTRATOR_STACK = REPO_ROOT / "scripts" / "server" / "orchestrator_stack.py"
DEFAULT_STACK_PATHS = REPO_ROOT / "scripts" / "server" / "stack_paths.py"
DEFAULT_STACK_RUNTIME = REPO_ROOT / "scripts" / "server" / "stack_runtime.py"
PRECEDENCE_SPEC = REPO_ROOT / "docs" / "reference" / "stack-truth-precedence.md"

STACK_PRIORS_VERSION = 4
REQUIRED_TOP_LEVEL_FIELDS = (
    "stack_priors_version",
    "contract",
    "compiled_at",
    "status",
    "coverage_scope",
    "precedence_spec",
    "source_artifacts",
    "roles",
    "known_global_gaps",
)
REQUIRED_ROLE_FIELDS = (
    "role",
    "deployment_status",
    "status",
    "model_id",
    "display_name",
    "serving",
    "priors",
    "acceleration",
    "model",
    "evidence",
    "known_gaps",
)
REQUIRED_SERVING_FIELDS = (
    "endpoint",
    "server_role",
    "binding",
    "ports",
    "slots",
    "tier",
    "effective_context_tokens",
    "binary",
    "binary_dir",
    "numa_policy",
    "shared_mmap",
    "launch",
)
REQUIRED_LAUNCH_FIELDS = (
    "entries",
    "primary_roles",
    "modes",
    "requirements",
    "runtime",
)
REQUIRED_PRIOR_FIELDS = (
    "throughput_tps",
    "quality_overall",
    "memory_cost",
)

RESIDENCY_COST = {"hot": 1.0, "warm": 2.0, "cold": 3.0}


class StackPriorsCompileError(ValueError):
    """Stack-prior compilation found unresolved live-role gaps."""

    def __init__(self, gaps_by_role: dict[str, list[str]]) -> None:
        self.gaps_by_role = gaps_by_role
        lines = ["Stack prior compilation refused unresolved role gaps:"]
        for role, gaps in sorted(gaps_by_role.items()):
            for gap in gaps:
                lines.append(f"  - {role}: {gap}")
        super().__init__("\n".join(lines))


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return loaded


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_commit(path: Path) -> str | None:
    resolved = path.resolve()
    parents = (resolved,) if resolved.is_dir() else (resolved.parent,)
    for parent in parents[0].parents:
        if (parent / ".git").exists():
            try:
                return subprocess.check_output(
                    ["git", "-C", str(parent), "rev-parse", "--short", "HEAD"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            except (OSError, subprocess.CalledProcessError):
                return None
    return None


def _source_metadata(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "repo_commit": _repo_commit(path),
    }


def stack_priors_contract() -> dict[str, Any]:
    """Return the versioned consumer contract embedded in generated priors."""
    return {
        "schema": "epyc.stack_priors",
        "version": STACK_PRIORS_VERSION,
        "required_top_level_fields": list(REQUIRED_TOP_LEVEL_FIELDS),
        "required_role_fields": list(REQUIRED_ROLE_FIELDS),
        "required_serving_fields": list(REQUIRED_SERVING_FIELDS),
        "required_launch_fields": list(REQUIRED_LAUNCH_FIELDS),
        "required_prior_fields": list(REQUIRED_PRIOR_FIELDS),
        "fallback_policy": (
            "Consumers may use local fallback values only as explicit degraded "
            "mode when this generated contract is missing or invalid."
        ),
    }


def validate_stack_priors_contract(priors: dict[str, Any]) -> list[str]:
    """Validate the generated stack-priors consumer contract shape.

    This intentionally checks structure, not semantic freshness. Source hashes,
    retired live roles, and strict known-gap policy remain in stack_change_guard.
    """
    errors: list[str] = []
    if priors.get("stack_priors_version") != STACK_PRIORS_VERSION:
        errors.append(
            f"stack_priors_version must be {STACK_PRIORS_VERSION}, "
            f"got {priors.get('stack_priors_version')!r}"
        )

    for field in REQUIRED_TOP_LEVEL_FIELDS:
        if field not in priors:
            errors.append(f"missing top-level stack-prior field: {field}")

    contract = priors.get("contract")
    if not isinstance(contract, dict):
        errors.append("stack priors artifact has no mapping-valued contract section")
    elif contract.get("version") != STACK_PRIORS_VERSION:
        errors.append(
            f"stack-prior contract version must be {STACK_PRIORS_VERSION}, "
            f"got {contract.get('version')!r}"
        )

    roles = priors.get("roles")
    if not isinstance(roles, dict):
        errors.append("stack priors artifact has no mapping-valued roles section")
        return errors

    for role, record in sorted(roles.items()):
        if not isinstance(record, dict):
            errors.append(f"role {role!r} record is not a mapping")
            continue
        for field in REQUIRED_ROLE_FIELDS:
            if field not in record:
                errors.append(f"role {role!r} is missing contract field {field!r}")
        serving = record.get("serving")
        if not isinstance(serving, dict):
            errors.append(f"role {role!r} serving is not a mapping")
        else:
            for field in REQUIRED_SERVING_FIELDS:
                if field not in serving:
                    errors.append(f"role {role!r} serving is missing field {field!r}")
            launch = serving.get("launch")
            if not isinstance(launch, dict):
                errors.append(f"role {role!r} serving.launch is not a mapping")
            else:
                for field in REQUIRED_LAUNCH_FIELDS:
                    if field not in launch:
                        errors.append(f"role {role!r} serving.launch is missing field {field!r}")
        priors_block = record.get("priors")
        if not isinstance(priors_block, dict):
            errors.append(f"role {role!r} priors is not a mapping")
        else:
            for field in REQUIRED_PRIOR_FIELDS:
                if field not in priors_block:
                    errors.append(f"role {role!r} priors is missing field {field!r}")
        known_gaps = record.get("known_gaps")
        if not isinstance(known_gaps, list):
            errors.append(f"role {role!r} known_gaps must be a list")
    return errors


def load_stack_priors_artifact(path: Path = DEFAULT_OUTPUT) -> dict[str, Any] | None:
    """Load a generated stack-priors artifact for runtime consumers.

    Compilation and validation paths should use the stricter helpers in this
    module. Runtime consumers use this fail-closed loader so a missing or
    malformed generated artifact falls back to their explicit degraded modes.
    """
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None
    return loaded if isinstance(loaded, dict) else None


def live_stack_role_records(path: Path = DEFAULT_OUTPUT) -> dict[str, dict[str, Any]]:
    """Return live stack-prior role records keyed by role name."""
    artifact = load_stack_priors_artifact(path)
    if artifact is None:
        return {}
    roles = artifact.get("roles")
    if not isinstance(roles, dict):
        return {}

    live: dict[str, dict[str, Any]] = {}
    for role, record in roles.items():
        if not isinstance(role, str):
            continue
        if not isinstance(record, dict) or record.get("deployment_status") != "live_stack":
            continue
        live[role] = record
    return live


def canonical_stack_role_id(role_name: str) -> str | None:
    """Return the canonical role ID for a known stack role or alias."""
    from src.roles import Role

    role = Role.from_string(str(role_name))
    return role.value if role is not None else None


def live_stack_role_ids(
    path: Path = DEFAULT_OUTPUT,
    *,
    preferred_order: Sequence[str] = (),
) -> list[str]:
    """Return canonical live stack role IDs in a stable preferred order."""
    live_records = live_stack_role_records(path)
    if not live_records:
        return []

    live: list[str] = []
    for role_id, record in live_records.items():
        raw_role = record.get("role") if isinstance(record, dict) else None
        role_name = raw_role if isinstance(raw_role, str) and raw_role else role_id
        live.append(canonical_stack_role_id(role_name) or str(role_name))

    live = list(dict.fromkeys(live))
    if not live:
        return []

    live_set = set(live)
    ordered: list[str] = []
    seen: set[str] = set()
    for role in preferred_order:
        canonical = canonical_stack_role_id(str(role)) or str(role)
        if canonical in live_set and canonical not in seen:
            ordered.append(canonical)
            seen.add(canonical)
    ordered.extend(role for role in live if role not in seen)
    return ordered


def stack_prior_serving(record: dict[str, Any]) -> dict[str, Any]:
    """Return the mapping-valued ``serving`` block from a role record."""
    serving = record.get("serving")
    return serving if isinstance(serving, dict) else {}


def stack_prior_endpoint_port(serving: dict[str, Any]) -> int | None:
    """Return the endpoint port from a stack-prior serving block, if present."""
    endpoint = serving.get("endpoint")
    if not isinstance(endpoint, str):
        return None
    return urlparse(endpoint).port


def stack_prior_serving_ports(serving: dict[str, Any]) -> list[int]:
    """Return integer serving ports from a stack-prior serving block."""
    ports = serving.get("ports")
    if not isinstance(ports, list):
        return []
    return [port for port in ports if isinstance(port, int)]


def stack_prior_serving_url_value(serving: dict[str, Any]) -> str | None:
    """Return the config-compatible URL value for a serving block."""
    ports = stack_prior_serving_ports(serving)
    if ports:
        urls = [f"http://localhost:{port}" for port in ports]
        if len(urls) > 1:
            urls[0] = f"full:{urls[0]}"
        return ",".join(urls)
    endpoint = serving.get("endpoint")
    return endpoint if isinstance(endpoint, str) and endpoint.startswith("http") else None


def live_stack_serving_url_values(path: Path = DEFAULT_OUTPUT) -> dict[str, str]:
    """Return config-compatible URL values keyed by live stack role."""
    urls: dict[str, str] = {}
    for role, record in live_stack_role_records(path).items():
        url = stack_prior_serving_url_value(stack_prior_serving(record))
        if url:
            urls[role] = url
    return urls


def live_stack_serving_slot_limits(path: Path = DEFAULT_OUTPUT) -> dict[str, int]:
    """Return per-serving-URL admission slot limits from live stack priors."""
    limits: dict[str, int] = {}
    for record in live_stack_role_records(path).values():
        serving = stack_prior_serving(record)
        slots = serving.get("slots")
        if not isinstance(slots, int) or slots <= 0:
            continue
        endpoint = serving.get("endpoint")
        if isinstance(endpoint, str):
            limits[endpoint] = max(slots, limits.get(endpoint, 0))
        for port in stack_prior_serving_ports(serving):
            url = f"http://localhost:{port}"
            limits[url] = max(slots, limits.get(url, 0))
    return limits


def live_role_primary_ports(
    role_names: set[str] | frozenset[str],
    path: Path = DEFAULT_OUTPUT,
) -> dict[str, int]:
    """Return one primary serving port per requested live role."""
    ports: dict[str, int] = {}
    for role, record in live_stack_role_records(path).items():
        if role not in role_names:
            continue
        serving = stack_prior_serving(record)
        endpoint_port = stack_prior_endpoint_port(serving)
        if endpoint_port is not None:
            ports[role] = endpoint_port
            continue
        for port in stack_prior_serving_ports(serving):
            ports[role] = port
            break
    return ports


def live_warm_worker_slots(path: Path = DEFAULT_OUTPUT) -> dict[str, int]:
    """Return live warm worker roles and their stack-prior slot caps."""
    caps: dict[str, int] = {}
    for role, record in live_stack_role_records(path).items():
        if not role.startswith("worker_"):
            continue
        serving = stack_prior_serving(record)
        if serving.get("tier") != "warm":
            continue
        slots = serving.get("slots")
        caps[role] = slots if isinstance(slots, int) and slots > 0 else 1
    return caps


def _coerce_tps(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if parsed > 0 else None
    if isinstance(value, str):
        matches = [float(match) for match in re.findall(r"\d+(?:\.\d+)?", value)]
        return max(matches) if matches else None
    return None


def _residency_cost(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    return RESIDENCY_COST.get(value.strip().lower())


def _quality_prior(descriptor: dict[str, Any]) -> float | None:
    quality = descriptor.get("quality")
    if not isinstance(quality, dict):
        return None
    suite_vector = quality.get("suite_vector")
    if not isinstance(suite_vector, dict):
        return None
    value = suite_vector.get("overall")
    if isinstance(value, (int, float)) and 0 <= float(value) <= 1:
        return float(value)
    return None


def _throughput_prior(descriptor: dict[str, Any], server_cfg: dict[str, Any] | None) -> float | None:
    candidates: list[float] = []
    if isinstance(server_cfg, dict):
        tps = _coerce_tps(server_cfg.get("throughput"))
        if tps is not None:
            candidates.append(tps)

    speed = descriptor.get("speed")
    if isinstance(speed, dict):
        for key in (
            "solo_96t_tps",
            "quarter_48t_tps",
            "prefill_tps",
            "optimized_tps",
            "generation_tps_range",
        ):
            tps = _coerce_tps(speed.get(key))
            if tps is not None:
                candidates.append(tps)
    return max(candidates) if candidates else None


def _descriptor_roles(descriptor: dict[str, Any]) -> list[str]:
    role_bindings = descriptor.get("role_bindings")
    if not isinstance(role_bindings, dict):
        return []
    roles = role_bindings.get("roles")
    if not isinstance(roles, list):
        return []
    return sorted(str(role) for role in roles if isinstance(role, str))


def _descriptor_server_roles(descriptor: dict[str, Any]) -> list[str]:
    role_bindings = descriptor.get("role_bindings")
    if not isinstance(role_bindings, dict):
        return []
    server_roles = role_bindings.get("server_roles")
    if not isinstance(server_roles, list):
        return []
    return sorted(str(role) for role in server_roles if isinstance(role, str))


def _descriptor_by_role(descriptors: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_role: dict[str, dict[str, Any]] = {}
    models = descriptors.get("models")
    if not isinstance(models, list):
        return by_role
    for descriptor in models:
        if not isinstance(descriptor, dict):
            continue
        for role in _descriptor_roles(descriptor):
            by_role[role] = descriptor
    return by_role


def _server_for_role(
    role: str,
    server_mode: dict[str, Any],
    stack_aliases: dict[str, str],
    stack_roles: dict[str, dict[str, Any]],
) -> tuple[str | None, dict[str, Any] | None, str]:
    direct = server_mode.get(role)
    if isinstance(direct, dict):
        return role, direct, "server_mode.direct"

    for server_role, cfg in server_mode.items():
        if not isinstance(cfg, dict):
            continue
        if cfg.get("model_role") == role:
            return str(server_role), cfg, "server_mode.model_role"
        shared_with = cfg.get("shared_with")
        if isinstance(shared_with, list) and role in shared_with:
            return str(server_role), cfg, "server_mode.shared_with"

    primary = stack_aliases.get(role)
    if primary:
        server_role, cfg, binding = _server_for_role(primary, server_mode, {}, stack_roles)
        if cfg is not None:
            return server_role, cfg, f"stack_manifest.alias->{binding}"

    stack_cfg = stack_roles.get(role)
    if isinstance(stack_cfg, dict):
        return role, stack_cfg, "stack_manifest.role"

    return None, None, "unresolved"


def _launch_mode_for_server(server: dict[str, Any]) -> str:
    if server.get("worker_pool"):
        return "worker_pool"
    if server.get("vision"):
        return "vision"
    if server.get("embedding"):
        return "embedding"
    return "default"


def _launch_entry_for_role(server: dict[str, Any], role: str) -> dict[str, Any] | None:
    port = server.get("port")
    roles = server.get("roles")
    if not isinstance(port, int) or not isinstance(roles, list) or not roles:
        return None
    primary_role = roles[0] if isinstance(roles[0], str) else role
    entry: dict[str, Any] = {
        "port": port,
        "primary_role": primary_role,
        "mode": _launch_mode_for_server(server),
        "alias": role != primary_role,
    }
    numa_instance = server.get("numa_instance")
    if isinstance(numa_instance, int):
        entry["numa_instance"] = numa_instance
    worker_type = server.get("worker_type")
    if isinstance(worker_type, str):
        entry["worker_type"] = worker_type
    vision_type = server.get("vision_type")
    if isinstance(vision_type, str):
        entry["vision_type"] = vision_type
    return entry


def _launch_requirements_for_meta(
    meta: dict[str, Any],
    *,
    worker_pool_models: dict[str, Any],
    explore_draft_model: Any,
    vision_worker_model: Any,
    vision_worker_mmproj: Any,
    vision_escalation_model: Any,
    vision_escalation_mmproj: Any,
) -> dict[str, str]:
    requirements: dict[str, str] = {}
    mode = str(meta.get("mode") or "")
    if mode == "worker_pool":
        worker_type = str(meta.get("worker_type") or "")
        model_path = worker_pool_models.get(worker_type)
        if model_path:
            requirements["model_path"] = str(model_path)
        if worker_type == "explore" and explore_draft_model:
            requirements["draft_model_path"] = str(explore_draft_model)
    elif mode == "vision":
        vision_type = meta.get("vision_type")
        if vision_type == "worker":
            requirements["model_path"] = str(vision_worker_model)
            requirements["mmproj_path"] = str(vision_worker_mmproj)
        elif vision_type == "escalation":
            requirements["model_path"] = str(vision_escalation_model)
            requirements["mmproj_path"] = str(vision_escalation_mmproj)
    return {key: value for key, value in sorted(requirements.items()) if value}


def _positive_int(value: Any) -> int | None:
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.isdigit():
        parsed = int(value)
        return parsed if parsed > 0 else None
    return None


def _effective_context_for_meta(
    role: str,
    meta: dict[str, Any],
    *,
    kv_context_sizes: dict[str, Any],
    default_context_size: Any,
) -> int | None:
    return _positive_int(kv_context_sizes.get(role, default_context_size))


def _launch_record(
    entries: list[dict[str, Any]],
    requirements: dict[str, str] | None = None,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sorted_entries = sorted(
        entries,
        key=lambda entry: (
            entry.get("port", -1),
            str(entry.get("primary_role", "")),
            str(entry.get("mode", "")),
        ),
    )
    return {
        "entries": sorted_entries,
        "primary_roles": sorted(
            {
                str(entry["primary_role"])
                for entry in sorted_entries
                if isinstance(entry.get("primary_role"), str)
            }
        ),
        "modes": sorted(
            {str(entry["mode"]) for entry in sorted_entries if isinstance(entry.get("mode"), str)}
        ),
        "requirements": copy.deepcopy(requirements or {}),
        "runtime": copy.deepcopy(runtime or {}),
    }


def _stack_manifest_info() -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    try:
        from scripts.server.stack_manifest import (
            EXPLORE_DRAFT_MODEL,
            DEFAULT_EFFECTIVE_CONTEXT_TOKENS,
            HOT_SERVERS,
            LAUNCH_CONTEXT_TOKENS,
            PORT_MAP,
            ROLE_LAUNCH_META,
            VISION_ESCALATION_MMPROJ,
            VISION_ESCALATION_MODEL,
            VISION_WORKER_MMPROJ,
            VISION_WORKER_MODEL,
            WARM_SERVERS,
            WORKER_POOL_MODELS,
        )
    except Exception:
        return {}, {}

    launch_ports_by_role: dict[str, list[int]] = {}
    launch_entries_by_role: dict[str, list[dict[str, Any]]] = {}
    for server in HOT_SERVERS + WARM_SERVERS:
        if not isinstance(server, dict):
            continue
        port = server.get("port")
        if not isinstance(port, int):
            continue
        for role in server.get("roles") or []:
            if isinstance(role, str):
                launch_ports_by_role.setdefault(role, []).append(port)
                launch_entry = _launch_entry_for_role(server, role)
                if launch_entry is not None:
                    launch_entries_by_role.setdefault(role, []).append(launch_entry)

    aliases: dict[str, str] = {}
    roles: dict[str, dict[str, Any]] = {}
    for primary, meta in ROLE_LAUNCH_META.items():
        if not isinstance(meta, dict):
            continue
        launch_requirements = _launch_requirements_for_meta(
            meta,
            worker_pool_models=WORKER_POOL_MODELS,
            explore_draft_model=EXPLORE_DRAFT_MODEL,
            vision_worker_model=VISION_WORKER_MODEL,
            vision_worker_mmproj=VISION_WORKER_MMPROJ,
            vision_escalation_model=VISION_ESCALATION_MODEL,
            vision_escalation_mmproj=VISION_ESCALATION_MMPROJ,
        )
        ports = sorted(set(launch_ports_by_role.get(str(primary), [])))
        if ports:
            port = ports[0]
        elif meta.get("no_numa"):
            port = meta.get("port")
        else:
            port = PORT_MAP.get(primary)
        roles[str(primary)] = {
            "tier": meta.get("tier"),
            "port": port,
            "ports": ports or ([port] if isinstance(port, int) else []),
            "url": f"http://localhost:{port}" if isinstance(port, int) else None,
            "effective_context_tokens": _effective_context_for_meta(
                str(primary),
                meta,
                kv_context_sizes=LAUNCH_CONTEXT_TOKENS,
                default_context_size=DEFAULT_EFFECTIVE_CONTEXT_TOKENS,
            ),
            "launch": _launch_record(
                launch_entries_by_role.get(str(primary), []),
                launch_requirements,
            ),
        }
        shared = meta.get("shared_with_first_n") if isinstance(meta, dict) else None
        if isinstance(shared, list):
            for alias in shared:
                if isinstance(alias, str):
                    alias_ports = sorted(set(launch_ports_by_role.get(alias, [])))
                    alias_port = alias_ports[0] if alias_ports else port
                    aliases[alias] = str(primary)
                    roles[alias] = {
                        "tier": meta.get("tier"),
                        "port": alias_port,
                        "ports": alias_ports or ([alias_port] if isinstance(alias_port, int) else []),
                        "url": f"http://localhost:{alias_port}" if isinstance(alias_port, int) else None,
                        "effective_context_tokens": roles[str(primary)].get(
                            "effective_context_tokens"
                        ),
                        "launch": _launch_record(
                            launch_entries_by_role.get(alias, []),
                            launch_requirements,
                        ),
                    }
    return aliases, roles


def _first_string(values: Any) -> str | None:
    if not isinstance(values, list):
        return None
    for value in values:
        if isinstance(value, str):
            return value
    return None


def _first_launch_entry_value(launch: dict[str, Any], field: str) -> str | None:
    entries = launch.get("entries")
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get(field), str):
            return str(entry[field])
    return None


def _runtime_requirements(server_cfg: dict[str, Any] | None) -> tuple[str | None, list[str]]:
    if not isinstance(server_cfg, dict):
        return None, []
    runtime = server_cfg.get("runtime_requirements")
    if not isinstance(runtime, dict):
        return None, []
    binary_dir = runtime.get("binary_dir") if isinstance(runtime.get("binary_dir"), str) else None
    raw_ld = runtime.get("ld_library_path")
    ld_paths = [str(path) for path in raw_ld if isinstance(path, str)] if isinstance(raw_ld, list) else []
    return binary_dir, ld_paths


def _effective_acceleration(
    role_cfg: dict[str, Any] | None,
    server_cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    if isinstance(server_cfg, dict) and isinstance(server_cfg.get("acceleration"), dict):
        return copy.deepcopy(server_cfg["acceleration"])
    if isinstance(role_cfg, dict) and isinstance(role_cfg.get("acceleration"), dict):
        return copy.deepcopy(role_cfg["acceleration"])
    return {}


def _override_kv_args(acceleration: dict[str, Any]) -> list[str]:
    if acceleration.get("type") != "moe_expert_reduction":
        return []
    override_key = acceleration.get("override_key")
    experts = acceleration.get("experts")
    if not isinstance(override_key, str) or not isinstance(experts, int):
        return []
    return [f"{override_key}=int:{experts}"]


def _launch_runtime_record(
    role: str,
    descriptor: dict[str, Any],
    server_cfg: dict[str, Any] | None,
    role_cfg: dict[str, Any] | None,
    launch_cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(launch_cfg, dict):
        return {}
    launch = launch_cfg.get("launch")
    if not isinstance(launch, dict):
        return {}

    try:
        from scripts.server.stack_manifest import (
            DEFAULT_EFFECTIVE_CONTEXT_TOKENS,
            DEFAULT_UBATCH_TOKENS,
            LAUNCH_KV_QUANT_CONFIGS,
            NO_SPEC_DECODE_ROLES,
            SERIAL_ROLES,
            WORKER_MTP_DRAFT_MAX,
            WORKER_MTP_DRAFT_P_MIN,
            WORKER_MTP_SPEC_TYPE,
            WORKER_MTP_THREADS_DRAFT,
            WORKER_MTP_UBATCH_TOKENS,
        )
        from scripts.server.stack_numa import MLOCK_ROLES
        from scripts.server.stack_paths import LLAMA_SERVER, LLAMA_SERVER_V2, SLOT_SAVE_DIR, _V2_ROLES
        from src.roles import Role
    except Exception:
        return {}

    descriptor_serving = descriptor.get("serving") if isinstance(descriptor.get("serving"), dict) else {}
    primary_role = _first_string(launch.get("primary_roles")) or role
    mode = _first_string(launch.get("modes")) or "default"
    worker_type = _first_launch_entry_value(launch, "worker_type")
    vision_type = _first_launch_entry_value(launch, "vision_type")
    requirements = launch.get("requirements") if isinstance(launch.get("requirements"), dict) else {}
    acceleration = _effective_acceleration(role_cfg, server_cfg)
    binary_dir, ld_paths = _runtime_requirements(server_cfg)

    binary_path = str(Path(binary_dir) / "llama-server") if binary_dir else str(LLAMA_SERVER)
    binary_family = (
        str(descriptor_serving.get("binary"))
        if isinstance(descriptor_serving.get("binary"), str)
        else "llama.cpp-v2"
        if primary_role in _V2_ROLES and LLAMA_SERVER_V2.exists()
        else "llama.cpp"
    )
    if not binary_dir and primary_role in _V2_ROLES and LLAMA_SERVER_V2.exists():
        binary_path = str(LLAMA_SERVER_V2)

    if mode == "worker_pool":
        slots = 4 if worker_type == "fast" else 1
    elif mode == "vision":
        slots = 1 if vision_type == "escalation" else 2
    elif mode == "embedding":
        slots = 4
    else:
        slots = 1 if primary_role in SERIAL_ROLES else 2

    canonical_primary_role = str(Role.from_string(primary_role, default=None) or primary_role)
    canonical_role = str(Role.from_string(role, default=None) or role)
    kv_types = (
        LAUNCH_KV_QUANT_CONFIGS.get(canonical_primary_role)
        or LAUNCH_KV_QUANT_CONFIGS.get(canonical_role)
    )
    override_kv = ["qwen3vlmoe.expert_used_count=int:4"] if vision_type == "escalation" else []
    override_kv.extend(_override_kv_args(acceleration))
    override_kv = sorted(set(override_kv))

    spec: dict[str, Any] = {
        "enabled": False,
        "type": None,
        "disabled_by": None,
        "draft_model_path": None,
        "draft_max": None,
        "draft_p_min": None,
        "threads_draft": None,
    }
    if mode == "worker_pool" and worker_type == "explore":
        spec.update(
            {
                "enabled": True,
                "type": WORKER_MTP_SPEC_TYPE,
                "draft_model_path": str(requirements.get("draft_model_path"))
                if requirements.get("draft_model_path")
                else None,
                "draft_max": WORKER_MTP_DRAFT_MAX,
                "draft_p_min": WORKER_MTP_DRAFT_P_MIN,
                "threads_draft": WORKER_MTP_THREADS_DRAFT,
            }
        )
    elif primary_role in NO_SPEC_DECODE_ROLES and (
        acceleration.get("draft_role")
        or acceleration.get("draft_max")
        or acceleration.get("n_layer_exit_draft")
    ):
        spec["disabled_by"] = "no_spec_decode"

    context_tokens = launch_cfg.get("effective_context_tokens")
    if not isinstance(context_tokens, int):
        context_tokens = DEFAULT_EFFECTIVE_CONTEXT_TOKENS

    return {
        "binary_family": binary_family,
        "binary_path": binary_path,
        "binary_dir": binary_dir,
        "ld_library_path": ld_paths,
        "env_policy": "binary_override_strip_ggml" if binary_dir else "canonical",
        "kmp_blocktime": 10 if binary_dir else None,
        "cache": {
            "context_tokens": context_tokens,
            "slots": slots,
            "ubatch": WORKER_MTP_UBATCH_TOKENS
            if mode == "worker_pool" and worker_type == "explore"
            else DEFAULT_UBATCH_TOKENS
            if mode == "default"
            else None,
            "kv_type_k": kv_types[0] if kv_types else None,
            "kv_type_v": kv_types[1] if kv_types else None,
            "kv_hadamard": bool(primary_role in _V2_ROLES and LLAMA_SERVER_V2.exists()),
            "no_mmap": bool(mode == "worker_pool" and worker_type == "explore"),
            "mlock": bool(mode == "default" and primary_role in MLOCK_ROLES),
            "slot_save_path": str(SLOT_SAVE_DIR / primary_role) if mode == "default" else None,
        },
        "flags": {
            "flash_attn": True,
            "jinja": bool(
                (mode == "default" and primary_role != "architect_general")
                or (mode == "worker_pool" and worker_type == "explore")
            ),
            "reasoning": "off" if mode == "worker_pool" and worker_type == "explore" else None,
            "override_kv": override_kv,
            "spec": spec,
        },
    }


def _role_memory_cost(
    role: str,
    role_cfg: dict[str, Any] | None,
    server_cfg: dict[str, Any] | None,
) -> tuple[float | None, str | None, list[str]]:
    gaps: list[str] = []
    if isinstance(server_cfg, dict):
        cost = _residency_cost(server_cfg.get("tier"))
        if cost is not None:
            return cost, "server_mode.tier", gaps

    if isinstance(role_cfg, dict):
        memory = role_cfg.get("memory")
        if isinstance(memory, dict):
            cost = _residency_cost(memory.get("residency"))
            if cost is not None:
                return cost, "roles.memory.residency", gaps

    gaps.append("Missing memory residency evidence")
    return None, None, gaps


def _serving_record(
    role: str,
    descriptor: dict[str, Any],
    role_cfg: dict[str, Any] | None,
    server_role: str | None,
    server_cfg: dict[str, Any] | None,
    binding: str,
    launch_cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    serving = descriptor.get("serving")
    descriptor_serving = serving if isinstance(serving, dict) else {}
    ports: set[int] = set()
    launch_ports = (
        [port for port in launch_cfg.get("ports", []) if isinstance(port, int)]
        if isinstance(launch_cfg, dict)
        else []
    )
    if launch_ports:
        ports.update(launch_ports)
    else:
        for value in descriptor_serving.get("ports") or []:
            if isinstance(value, int):
                ports.add(value)
    if isinstance(server_cfg, dict):
        slots = server_cfg.get("slots")
        if not launch_ports:
            port = server_cfg.get("port")
            if isinstance(port, int):
                ports.add(port)
            numa_ports = server_cfg.get("numa_ports")
            if isinstance(numa_ports, list):
                ports.update(port for port in numa_ports if isinstance(port, int))
    else:
        slots = None

    launch_record = (
        copy.deepcopy(launch_cfg.get("launch") or _launch_record([]))
        if isinstance(launch_cfg, dict)
        else _launch_record([])
    )
    runtime_record = _launch_runtime_record(
        role,
        descriptor,
        server_cfg,
        role_cfg,
        launch_cfg,
    )
    launch_record["runtime"] = runtime_record

    if not isinstance(slots, int) or slots <= 0:
        cache = runtime_record.get("cache") if isinstance(runtime_record, dict) else {}
        runtime_slots = cache.get("slots") if isinstance(cache, dict) else None
        if isinstance(runtime_slots, int) and runtime_slots > 0:
            slots = runtime_slots

    return {
        "endpoint": server_cfg.get("url")
        if isinstance(server_cfg, dict)
        else launch_cfg.get("url")
        if isinstance(launch_cfg, dict)
        else None,
        "server_role": server_role,
        "binding": binding,
        "ports": sorted(ports),
        "slots": slots if isinstance(slots, int) and slots > 0 else None,
        "tier": launch_cfg.get("tier")
        if isinstance(launch_cfg, dict) and launch_cfg.get("tier") is not None
        else server_cfg.get("tier")
        if isinstance(server_cfg, dict)
        else None,
        "effective_context_tokens": launch_cfg.get("effective_context_tokens")
        if isinstance(launch_cfg, dict)
        and isinstance(launch_cfg.get("effective_context_tokens"), int)
        else None,
        "binary": descriptor_serving.get("binary"),
        "binary_dir": descriptor_serving.get("binary_dir"),
        "numa_policy": descriptor_serving.get("numa_policy"),
        "shared_mmap": bool(
            (descriptor.get("role_bindings") or {}).get("shared_mmap")
        )
        if isinstance(descriptor.get("role_bindings"), dict)
        else False,
        "launch": launch_record,
    }


def _role_record(
    role: str,
    descriptor: dict[str, Any],
    registry_roles: dict[str, Any],
    server_mode: dict[str, Any],
    stack_aliases: dict[str, str],
    stack_roles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    role_cfg = registry_roles.get(role)
    if not isinstance(role_cfg, dict):
        role_cfg = None
    server_role, server_cfg, binding = _server_for_role(
        role, server_mode, stack_aliases, stack_roles
    )
    launch_cfg = stack_roles.get(role)
    memory_cost, memory_source, memory_gaps = _role_memory_cost(role, role_cfg, server_cfg)
    known_gaps = [str(gap) for gap in descriptor.get("known_gaps") or []]
    gaps = list(dict.fromkeys(known_gaps + memory_gaps))

    throughput = _throughput_prior(descriptor, server_cfg)
    quality = _quality_prior(descriptor)
    if throughput is None:
        gaps.append("Missing throughput prior")
    if quality is None:
        gaps.append("Missing overall quality prior")
    if server_cfg is None:
        gaps.append("Missing live server binding")
    architecture = descriptor.get("architecture")
    architecture = architecture if isinstance(architecture, dict) else {}
    model_record = {
        "family": descriptor.get("family"),
        "arch": descriptor.get("arch"),
        "params_b": descriptor.get("params_b"),
        "active_b": descriptor.get("active_b"),
        "quant": descriptor.get("quant"),
        "mem_gb": descriptor.get("mem_gb"),
        "ctx_max": descriptor.get("ctx_max"),
        "modalities": copy.deepcopy(descriptor.get("modalities") or []),
    }
    for key in ("n_layers", "attention_layers"):
        if architecture.get(key) is not None:
            model_record[key] = architecture[key]

    return {
        "role": role,
        "deployment_status": "live_stack"
        if role in stack_roles
        else "benchmark_or_candidate",
        "status": "compiled_with_gaps" if gaps else "compiled",
        "model_id": descriptor.get("model_id"),
        "display_name": descriptor.get("display_name"),
        "serving": _serving_record(
            role,
            descriptor,
            role_cfg,
            server_role,
            server_cfg,
            binding,
            launch_cfg,
        ),
        "priors": {
            "throughput_tps": throughput,
            "quality_overall": quality,
            "memory_cost": memory_cost,
        },
        "acceleration": copy.deepcopy(descriptor.get("acceleration") or {}),
        "model": model_record,
        "evidence": {
            "precedence": {
                "serving": "server_mode/stack_manifest outrank roles metadata",
                "memory_cost": memory_source,
                "spec": str(PRECEDENCE_SPEC),
            },
            "descriptor_server_roles": _descriptor_server_roles(descriptor),
            "alias_overrides": copy.deepcopy(
                (descriptor.get("role_bindings") or {}).get("alias_overrides") or []
            ),
            "quality": copy.deepcopy((descriptor.get("quality") or {}).get("measured", [])),
            "speed": copy.deepcopy((descriptor.get("speed") or {}).get("measured", [])),
        },
        "known_gaps": sorted(set(gaps)),
    }


def _default_roles_from_descriptors(descriptors: dict[str, Any]) -> set[str]:
    roles: set[str] = set()
    for descriptor in (descriptors.get("models") or []):
        if isinstance(descriptor, dict):
            roles.update(_descriptor_roles(descriptor))
    return roles


def compile_stack_priors(
    *,
    registry_path: Path = DEFAULT_REGISTRY,
    descriptor_path: Path = DEFAULT_DESCRIPTORS,
    active_roles: set[str] | None = None,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    registry = _load_yaml(registry_path)
    descriptors = _load_yaml(descriptor_path)

    registry_roles = registry.get("roles") or {}
    server_mode = registry.get("server_mode") or {}
    if not isinstance(registry_roles, dict) or not isinstance(server_mode, dict):
        raise ValueError("registry must contain mapping-valued roles and server_mode sections")

    descriptor_by_role = _descriptor_by_role(descriptors)
    requested_roles = active_roles or _default_roles_from_descriptors(descriptors)
    stack_aliases, stack_roles = _stack_manifest_info()
    role_records: dict[str, Any] = {}
    gaps_by_role: dict[str, list[str]] = {}

    for role in sorted(requested_roles):
        descriptor = descriptor_by_role.get(role)
        if descriptor is None:
            gaps_by_role[role] = ["Missing model descriptor binding"]
            continue
        record = _role_record(
            role,
            descriptor,
            registry_roles,
            server_mode,
            stack_aliases,
            stack_roles,
        )
        role_records[role] = record
        if record["known_gaps"]:
            gaps_by_role[role] = record["known_gaps"]

    if gaps_by_role and not allow_incomplete:
        raise StackPriorsCompileError(gaps_by_role)

    return {
        "stack_priors_version": STACK_PRIORS_VERSION,
        "contract": stack_priors_contract(),
        "compiled_at": _timestamp(),
        "status": "compiled_with_gaps" if gaps_by_role else "compiled",
        "coverage_scope": "descriptor_role_bindings"
        if active_roles is None
        else "explicit_active_roles",
        "precedence_spec": str(PRECEDENCE_SPEC),
        "source_artifacts": {
            "registry": _source_metadata(registry_path),
            "descriptors": _source_metadata(descriptor_path),
            "stack_manifest": _source_metadata(DEFAULT_STACK_MANIFEST),
            "stack_numa": _source_metadata(DEFAULT_STACK_NUMA),
            "orchestrator_stack": _source_metadata(DEFAULT_ORCHESTRATOR_STACK),
            "stack_paths": _source_metadata(DEFAULT_STACK_PATHS),
            "stack_runtime": _source_metadata(DEFAULT_STACK_RUNTIME),
        },
        "roles": role_records,
        "known_global_gaps": {
            role: list(gaps) for role, gaps in sorted(gaps_by_role.items()) if gaps
        },
    }


def write_stack_priors(
    output_path: Path,
    *,
    registry_path: Path = DEFAULT_REGISTRY,
    descriptor_path: Path = DEFAULT_DESCRIPTORS,
    active_roles: set[str] | None = None,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    priors = compile_stack_priors(
        registry_path=registry_path,
        descriptor_path=descriptor_path,
        active_roles=active_roles,
        allow_incomplete=allow_incomplete,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(
            priors,
            fh,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
            width=200,
        )
    return priors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile derived stack priors")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--descriptors", type=Path, default=DEFAULT_DESCRIPTORS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--roles", nargs="+", help="Explicit role list")
    parser.add_argument("--dry-run", action="store_true", help="Print priors instead of writing")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Emit priors with known_gaps instead of refusing incomplete records",
    )
    args = parser.parse_args(argv)

    roles = set(args.roles) if args.roles else None
    priors = compile_stack_priors(
        registry_path=args.registry,
        descriptor_path=args.descriptors,
        active_roles=roles,
        allow_incomplete=args.allow_incomplete,
    )
    if args.dry_run:
        yaml.safe_dump(
            priors,
            sys.stdout,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
            width=200,
        )
    else:
        write_stack_priors(
            args.output,
            registry_path=args.registry,
            descriptor_path=args.descriptors,
            active_roles=roles,
            allow_incomplete=args.allow_incomplete,
        )
        print(f"OK: wrote {len(priors.get('roles', {}))} role priors to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
