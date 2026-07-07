"""Compile derived stack priors from registry and model descriptors.

This module is intentionally additive: existing consumers can migrate to the
generated artifact one by one instead of re-parsing scattered registry comments
and hardcoded role tables.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
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
DEFAULT_MODELS_DIR = Path("/mnt/raid0/llm/models")
DEFAULT_MODEL_BASE_DIR = Path("/mnt/raid0/llm/lmstudio/models")

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


def stack_prior_launch(record: dict[str, Any]) -> dict[str, Any]:
    """Return the mapping-valued ``serving.launch`` block from a role record."""
    launch = stack_prior_serving(record).get("launch")
    return launch if isinstance(launch, dict) else {}


def stack_prior_launch_entries(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Return mapping-valued launch entries from a stack-prior role record."""
    entries = stack_prior_launch(record).get("entries")
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def stack_prior_launch_modes(record: dict[str, Any]) -> set[str]:
    """Return string launch modes from a stack-prior role record."""
    modes = stack_prior_launch(record).get("modes")
    if not isinstance(modes, list):
        return set()
    return {mode for mode in modes if isinstance(mode, str)}


def stack_prior_uses_shared_worker_launch(record: dict[str, Any]) -> bool:
    """Return True when stack-prior launch metadata identifies shared worker use."""
    if "worker_pool" in stack_prior_launch_modes(record):
        return True
    for entry in stack_prior_launch_entries(record):
        if entry.get("mode") == "worker_pool":
            return True
        if entry.get("vision_type") == "worker":
            return True
    return False


def stack_prior_model_mem_gb(record: dict[str, Any]) -> float | None:
    """Return numeric model memory from a stack-prior role record, if present."""
    model = record.get("model")
    mem_gb = model.get("mem_gb") if isinstance(model, dict) else None
    if not isinstance(mem_gb, (int, float)):
        return None
    return float(mem_gb)


def live_stack_lock_role_sets(
    path: Path = DEFAULT_OUTPUT,
) -> tuple[frozenset[str], frozenset[str]] | None:
    """Derive exclusive/shared lock role sets from generated live stack priors."""
    roles = live_stack_role_records(path)
    if not roles:
        return None

    heavy: set[str] = set()
    light: set[str] = set()
    for role, record in roles.items():
        if stack_prior_uses_shared_worker_launch(record):
            light.add(role)
        else:
            heavy.add(role)

    if not heavy and not light:
        return None
    return frozenset(heavy), frozenset(light)


def live_stack_safe_non_stream_roles(
    path: Path = DEFAULT_OUTPUT,
    *,
    min_mem_gb: float,
) -> frozenset[str] | None:
    """Derive safe-mode non-stream roles from generated live stack-prior memory."""
    roles = live_stack_role_records(path)
    if not roles:
        return None

    threshold = max(0.0, float(min_mem_gb))
    derived: set[str] = set()
    saw_live_memory = False
    for role, record in roles.items():
        mem_gb = stack_prior_model_mem_gb(record)
        if mem_gb is None:
            continue
        saw_live_memory = True
        if mem_gb >= threshold:
            derived.add(role)

    if not saw_live_memory:
        return None
    return frozenset(derived)


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


def stack_prior_primary_port(serving: dict[str, Any]) -> int | None:
    """Return the endpoint port, falling back to the first declared serving port."""
    try:
        endpoint_port = stack_prior_endpoint_port(serving)
    except ValueError:
        endpoint_port = None
    if endpoint_port is not None:
        return endpoint_port
    for port in stack_prior_serving_ports(serving):
        return port
    return None


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


def _stack_prior_serves_llama_slots(serving: dict[str, Any]) -> bool:
    """Return True when a serving block identifies a llama-server slots API."""
    binary = serving.get("binary")
    if binary in {"llama.cpp", "ik-pr1744"}:
        return True
    launch = serving.get("launch")
    runtime = launch.get("runtime") if isinstance(launch, dict) else None
    binary_path = runtime.get("binary_path") if isinstance(runtime, dict) else None
    return isinstance(binary_path, str) and Path(binary_path).name == "llama-server"


def live_stack_slot_query_ports(path: Path = DEFAULT_OUTPUT) -> dict[str, list[int]]:
    """Return live llama-server slot-query ports keyed by canonical role."""
    ports_by_role: dict[str, set[int]] = {}
    for role, record in live_stack_role_records(path).items():
        serving = stack_prior_serving(record)
        if not _stack_prior_serves_llama_slots(serving):
            continue
        for entry in stack_prior_launch_entries(record):
            if entry.get("alias") is True:
                continue
            port = entry.get("port")
            if isinstance(port, int):
                ports_by_role.setdefault(role, set()).add(port)

    return {
        role: sorted(ports)
        for role, ports in sorted(ports_by_role.items())
        if ports
    }


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
        primary_port = stack_prior_primary_port(serving)
        if primary_port is not None:
            ports[role] = primary_port
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
            _filter_by_numa_mode,
        )
    except Exception:
        return {}, {}

    numa_mode = os.environ.get("ORCHESTRATOR_STACK_NUMA_MODE", "full").strip().lower()
    if numa_mode not in {"full", "quarter", "both"}:
        numa_mode = "full"
    active_servers = _filter_by_numa_mode(HOT_SERVERS + WARM_SERVERS, numa_mode)

    launch_ports_by_role: dict[str, list[int]] = {}
    launch_entries_by_role: dict[str, list[dict[str, Any]]] = {}
    for server in active_servers:
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


def _role_no_mmap_prior(
    server_cfg: dict[str, Any] | None,
    role_cfg: dict[str, Any] | None,
    *,
    default: bool,
) -> bool:
    """Resolve a role's no_mmap cache prior from its config.

    2026-06-26 v6 cutover: precedence is server_mode -> roles block -> ``default``.
    A role may set ``no_mmap: true`` directly, or under a ``cache``/``serving``
    sub-mapping. Absent any explicit setting the caller-supplied ``default`` is
    used (which preserves the legacy worker_pool+explore canonical-recipe value).
    """
    for cfg in (server_cfg, role_cfg):
        if not isinstance(cfg, dict):
            continue
        if isinstance(cfg.get("no_mmap"), bool):
            return cfg["no_mmap"]
        for nested_key in ("cache", "serving"):
            nested = cfg.get(nested_key)
            if isinstance(nested, dict) and isinstance(nested.get("no_mmap"), bool):
                return nested["no_mmap"]
    return default


def _positive_int_prior(
    *containers: dict[str, Any] | None,
    key: str,
    fallback: int,
) -> int:
    for container in containers:
        if not isinstance(container, dict):
            continue
        value = container.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return fallback


def _number_prior(
    *containers: dict[str, Any] | None,
    key: str,
    fallback: int | float,
) -> int | float:
    for container in containers:
        if not isinstance(container, dict):
            continue
        value = container.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    return fallback


def _nested_mapping(container: dict[str, Any] | None, *path: str) -> dict[str, Any] | None:
    current: Any = container
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current if isinstance(current, dict) else None


def _kv_types_prior(
    server_cfg: dict[str, Any] | None,
    role_cfg: dict[str, Any] | None,
    descriptor: dict[str, Any],
) -> tuple[str, str] | None:
    """Resolve runtime KV cache types from registry/descriptor facts.

    Registry-derived facts take precedence over launcher fallback tables. The
    accepted shapes mirror the current master registry (`server_mode.*.kv_quant`
    and `roles.*.model.kv_cache`) plus descriptor acceleration metadata for
    future generated surfaces.
    """
    candidate_maps = (
        _nested_mapping(server_cfg, "kv_quant"),
        _nested_mapping(server_cfg, "kv_cache"),
        _nested_mapping(role_cfg, "model", "kv_cache"),
        _nested_mapping(role_cfg, "kv_cache"),
        _nested_mapping(descriptor, "acceleration", "kv"),
    )
    for candidate in candidate_maps:
        if not isinstance(candidate, dict):
            continue
        key_type = candidate.get("k") or candidate.get("type_k") or candidate.get("kv_type_k")
        value_type = candidate.get("v") or candidate.get("type_v") or candidate.get("kv_type_v")
        if isinstance(key_type, str) and isinstance(value_type, str):
            return key_type, value_type
    return None


def _worker_context_prior(
    role_cfg: dict[str, Any] | None,
    *,
    fallback: int,
) -> int:
    model_cfg = _nested_mapping(role_cfg, "model")
    if not isinstance(model_cfg, dict):
        return fallback
    return _positive_int_prior(
        model_cfg,
        key="max_context",
        fallback=fallback,
    )


def _resolve_nextn_draft_path(
    requirements: dict[str, Any],
    acceleration: dict[str, Any],
    server_cfg: dict[str, Any] | None,
    *,
    models_dir: Any = None,
) -> str | None:
    """Resolve the NEXTN self-draft GGUF path for a draft-mtp role.

    2026-06-26 v6 cutover: NEXTN self-draft roles (frontdoor, architect_general)
    embed the draft head in the base GGUF. The compiled draft path intentionally
    resolves to the model path; the launcher emits draft-mtp spec flags but
    suppresses ``-md`` when both paths have the same realpath.

    Sources, in precedence order:
      1. requirements.draft_model_path (explicit full path, e.g. server_mode override)
      2. acceleration.draft_model_path (explicit full path on the accel block)
      3. acceleration.draft_model / server_cfg.draft_model (bare or relative; the
         registry's NEXTN self-draft pointer == the base file)
      4. requirements.model_path / server_cfg.model_path (full base path; self-draft)
      5. server_cfg.model (bare or relative base path)
    Bare/relative values are resolved against ``models_dir`` when available so the
    emitted path is absolute.
    """
    candidates = [
        requirements.get("draft_model_path"),
        acceleration.get("draft_model_path"),
        acceleration.get("draft_model"),
        server_cfg.get("draft_model") if isinstance(server_cfg, dict) else None,
        requirements.get("model_path"),
        server_cfg.get("model_path") if isinstance(server_cfg, dict) else None,
        server_cfg.get("model") if isinstance(server_cfg, dict) else None,
    ]
    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate:
            continue
        candidate_path = Path(candidate)
        if candidate_path.is_absolute():
            return str(candidate_path)
        if models_dir is not None:
            return str(Path(models_dir) / candidate)
        return candidate
    return None


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
        )
        from scripts.server.stack_numa import MLOCK_ROLES
        from scripts.server.stack_paths import (
            LLAMA_SERVER,
            LLAMA_SERVER_V2,
            SLOT_SAVE_DIR,
            _PATHS,
            _V2_ROLES,
        )
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
    kv_types = _kv_types_prior(server_cfg, role_cfg, descriptor) or (
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
    # 2026-06-26 v6 cutover: spec_type carries the v6 MTP token 'draft-mtp' (bare
    # 'mtp' is invalid in v6). It is preserved verbatim from the registry
    # acceleration block — no normalization/allow-list rejects it here.
    spec_type_prior = (
        str(acceleration.get("spec_type"))
        if isinstance(acceleration.get("spec_type"), str) and acceleration.get("spec_type")
        else None
    )
    if mode == "worker_pool" and worker_type == "explore":
        worker_draft_max = _positive_int_prior(
            acceleration,
            key="draft_max",
            fallback=2,
        )
        worker_threads_draft = _positive_int_prior(
            acceleration,
            key="threads_draft",
            fallback=16,
        )
        worker_draft_p_min = _number_prior(
            acceleration,
            key="draft_p_min",
            fallback=0.0,
        )
        spec.update(
            {
                "enabled": True,
                # 2026-06-26 v6 cutover: prefer the registry spec_type (draft-mtp);
                # the literal fallback is degraded mode for incomplete registries.
                "type": spec_type_prior or "draft-mtp",
                "draft_model_path": str(requirements.get("draft_model_path"))
                if requirements.get("draft_model_path")
                else None,
                "draft_max": worker_draft_max,
                "draft_p_min": worker_draft_p_min,
                "threads_draft": worker_threads_draft,
            }
        )
    elif spec_type_prior == "draft-mtp" and role == primary_role:
        # 2026-06-26 v6 cutover: emit a NON-NULL draft-mtp spec ONLY for the PRIMARY
        # role that launches the server (role == primary_role). ALIAS roles
        # (shared_with_first_n, e.g. coder_escalation / worker_summarize sharing
        # frontdoor's :8070 process) inherit the host's NEXTN draft at runtime and do
        # NOT launch their own draft — they fall through to the disabled spec so their
        # launch record matches the launch manifest (which nulls draft for aliases).
        # emit a NON-NULL draft-mtp spec for any non-worker
        # role whose registry acceleration.spec_type == 'draft-mtp' (frontdoor
        # qwen36_q8_0, architect_general). These are NEXTN self-draft models — the
        # draft head is embedded in the base GGUF, so the resolved drafter file is
        # the same file as -m. Keep the compiled path explicit for provenance; the
        # launcher suppresses -md for same-realpath drafts and preserves
        # --spec-type/--spec-draft-n-max. draft_max carries the n-max value from
        # the registry (frontdoor=4, architect=4).
        nextn_draft_path = _resolve_nextn_draft_path(
            requirements,
            acceleration,
            server_cfg,
            models_dir=_PATHS.get("models_dir"),
        )
        draft_max_prior = acceleration.get("draft_max")
        spec.update(
            {
                "enabled": True,
                "type": spec_type_prior,
                "draft_model_path": str(nextn_draft_path) if nextn_draft_path else None,
                "draft_max": draft_max_prior
                if isinstance(draft_max_prior, int) and not isinstance(draft_max_prior, bool)
                else None,
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
    if mode == "worker_pool" and worker_type == "explore":
        context_tokens = _worker_context_prior(role_cfg, fallback=context_tokens)
    worker_ubatch = _positive_int_prior(
        acceleration,
        key="ubatch",
        fallback=512,
    )

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
            "ubatch": worker_ubatch
            if mode == "worker_pool" and worker_type == "explore"
            else DEFAULT_UBATCH_TOKENS
            if mode == "default"
            else None,
            "kv_type_k": kv_types[0] if kv_types else None,
            "kv_type_v": kv_types[1] if kv_types else None,
            "kv_hadamard": bool(primary_role in _V2_ROLES and LLAMA_SERVER_V2.exists()),
            # 2026-06-26 v6 cutover: no_mmap is no longer hardcoded to worker_pool+explore.
            # It now flows through from the role's config (server_mode then roles block),
            # defaulting to False when absent — so non-worker quarter roles (N12 private
            # quarters) can request no_mmap=True without forcing it globally. The legacy
            # worker_pool+explore canonical-recipe default is preserved as a fallback.
            "no_mmap": _role_no_mmap_prior(
                server_cfg,
                role_cfg,
                default=bool(mode == "worker_pool" and worker_type == "explore"),
            ),
            "mlock": bool(mode == "default" and primary_role in MLOCK_ROLES),
            "slot_save_path": str(SLOT_SAVE_DIR / primary_role) if mode == "default" else None,
        },
        "flags": {
            "flash_attn": True,
            # 2026-06-26: architect_general no longer excluded from --jinja. The
            # 2026-04-15 exclusion (commit 0879ed56) suppressed Qwen3.5-122B hybrid
            # <think>-loops by falling back to generic ChatML, but that also made the
            # registry's enable_thinking=false inert (kwarg only applies on the
            # /v1/chat/completions+jinja path). Enrolling architect into jinja routes
            # it through chat-completions where nothink fires (frontdoor proves the
            # same-family draft-mtp+jinja+nothink path). Gated on the J12 think-loop
            # suppression probe before trusting.
            "jinja": bool(
                (mode == "default")
                or (mode == "worker_pool" and worker_type == "explore")
            ),
            "reasoning": "off" if mode == "worker_pool" and worker_type == "explore" else None,
            "override_kv": override_kv,
            "spec": spec,
        },
    }


def _models_dir() -> Path:
    try:
        from scripts.server.stack_paths import _PATHS

        models_dir = _PATHS.get("models_dir")
        if models_dir:
            return Path(models_dir)
    except Exception:
        pass
    return DEFAULT_MODELS_DIR


def _model_base_dir() -> Path:
    try:
        from src.registry.registry_loader import RegistryLoader

        return RegistryLoader(validate_paths=False).model_base_path
    except Exception:
        return DEFAULT_MODEL_BASE_DIR


def _resolved_model_path(value: Any, *, base_dir: Path | None = None) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base_dir or _models_dir()) / path)


def _role_model_path(role_cfg: dict[str, Any] | None) -> str | None:
    model_cfg = _nested_mapping(role_cfg, "model")
    if not isinstance(model_cfg, dict):
        return None
    return _resolved_model_path(model_cfg.get("path"), base_dir=_model_base_dir())


def _server_mode_launch_requirement_overrides(
    role: str,
    server_cfg: dict[str, Any] | None,
    role_cfg: dict[str, Any] | None,
) -> dict[str, str]:
    if not isinstance(server_cfg, dict):
        return {}

    overrides: dict[str, str] = {}
    explicit_model = _resolved_model_path(server_cfg.get("model_path"))
    server_model_role = server_cfg.get("model_role")
    role_model = (
        _role_model_path(role_cfg)
        if not isinstance(server_model_role, str) or server_model_role == role
        else None
    )
    server_model = _resolved_model_path(server_cfg.get("model"))
    if explicit_model or role_model or server_model:
        overrides["model_path"] = str(explicit_model or role_model or server_model)

    explicit_draft = _resolved_model_path(server_cfg.get("draft_model_path"))
    server_draft = _resolved_model_path(server_cfg.get("draft_model"))
    if explicit_draft or server_draft:
        overrides["draft_model_path"] = str(explicit_draft or server_draft)

    mmproj_path = _resolved_model_path(server_cfg.get("mmproj_path"))
    if mmproj_path:
        overrides["mmproj_path"] = mmproj_path
    return overrides


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
    requirement_overrides = (
        _server_mode_launch_requirement_overrides(role, server_cfg, role_cfg)
        if isinstance(launch_cfg, dict)
        else {}
    )
    if requirement_overrides:
        requirements = launch_record.get("requirements")
        if not isinstance(requirements, dict):
            requirements = {}
        launch_record["requirements"] = {**requirements, **requirement_overrides}
    runtime_launch_cfg = copy.deepcopy(launch_cfg) if isinstance(launch_cfg, dict) else None
    if isinstance(runtime_launch_cfg, dict):
        runtime_launch_cfg["launch"] = launch_record
    runtime_record = _launch_runtime_record(
        role,
        descriptor,
        server_cfg,
        role_cfg,
        runtime_launch_cfg,
    )
    launch_record["runtime"] = runtime_record

    if not isinstance(slots, int) or slots <= 0:
        cache = runtime_record.get("cache") if isinstance(runtime_record, dict) else {}
        runtime_slots = cache.get("slots") if isinstance(cache, dict) else None
        if isinstance(runtime_slots, int) and runtime_slots > 0:
            slots = runtime_slots

    sorted_ports = sorted(ports)
    if sorted_ports:
        endpoint = f"http://localhost:{sorted_ports[0]}"
    elif isinstance(launch_cfg, dict) and isinstance(launch_cfg.get("url"), str):
        endpoint = launch_cfg.get("url")
    elif isinstance(server_cfg, dict):
        endpoint = server_cfg.get("url")
    else:
        endpoint = None

    return {
        "endpoint": endpoint,
        "server_role": server_role,
        "binding": binding,
        "ports": sorted_ports,
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
