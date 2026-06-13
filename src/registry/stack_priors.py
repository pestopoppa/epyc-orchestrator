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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path("/mnt/raid0/llm/epyc-orchestrator")
DEFAULT_REGISTRY = REPO_ROOT / "orchestration" / "model_registry.yaml"
DEFAULT_DESCRIPTORS = REPO_ROOT / "orchestration" / "model_descriptors.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "orchestration" / "derived" / "stack_priors.yaml"
PRECEDENCE_SPEC = REPO_ROOT / "docs" / "reference" / "stack-truth-precedence.md"

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


def _stack_manifest_info() -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    try:
        from scripts.server.stack_manifest import PORT_MAP, ROLE_LAUNCH_META
    except Exception:
        return {}, {}

    aliases: dict[str, str] = {}
    roles: dict[str, dict[str, Any]] = {}
    for primary, meta in ROLE_LAUNCH_META.items():
        if not isinstance(meta, dict):
            continue
        port = meta.get("port") if meta.get("no_numa") else PORT_MAP.get(primary)
        roles[str(primary)] = {
            "tier": meta.get("tier"),
            "port": port,
            "url": f"http://localhost:{port}" if isinstance(port, int) else None,
        }
        shared = meta.get("shared_with_first_n") if isinstance(meta, dict) else None
        if isinstance(shared, list):
            for alias in shared:
                if isinstance(alias, str):
                    aliases[alias] = str(primary)
                    roles[alias] = {
                        "tier": meta.get("tier"),
                        "port": port,
                        "url": f"http://localhost:{port}" if isinstance(port, int) else None,
                    }
    return aliases, roles


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
    descriptor: dict[str, Any],
    server_role: str | None,
    server_cfg: dict[str, Any] | None,
    binding: str,
) -> dict[str, Any]:
    serving = descriptor.get("serving")
    descriptor_serving = serving if isinstance(serving, dict) else {}
    ports: set[int] = set()
    for value in descriptor_serving.get("ports") or []:
        if isinstance(value, int):
            ports.add(value)
    if isinstance(server_cfg, dict):
        port = server_cfg.get("port")
        if isinstance(port, int):
            ports.add(port)
        numa_ports = server_cfg.get("numa_ports")
        if isinstance(numa_ports, list):
            ports.update(port for port in numa_ports if isinstance(port, int))
        slots = server_cfg.get("slots")
    else:
        slots = None

    return {
        "endpoint": server_cfg.get("url") if isinstance(server_cfg, dict) else None,
        "server_role": server_role,
        "binding": binding,
        "ports": sorted(ports),
        "slots": slots if isinstance(slots, int) and slots > 0 else None,
        "tier": server_cfg.get("tier") if isinstance(server_cfg, dict) else None,
        "binary": descriptor_serving.get("binary"),
        "binary_dir": descriptor_serving.get("binary_dir"),
        "numa_policy": descriptor_serving.get("numa_policy"),
        "shared_mmap": bool(
            (descriptor.get("role_bindings") or {}).get("shared_mmap")
        )
        if isinstance(descriptor.get("role_bindings"), dict)
        else False,
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

    return {
        "role": role,
        "deployment_status": "live_stack"
        if role in stack_roles
        else "benchmark_or_candidate",
        "status": "compiled_with_gaps" if gaps else "compiled",
        "model_id": descriptor.get("model_id"),
        "display_name": descriptor.get("display_name"),
        "serving": _serving_record(descriptor, server_role, server_cfg, binding),
        "priors": {
            "throughput_tps": throughput,
            "quality_overall": quality,
            "memory_cost": memory_cost,
        },
        "acceleration": copy.deepcopy(descriptor.get("acceleration") or {}),
        "model": {
            "family": descriptor.get("family"),
            "arch": descriptor.get("arch"),
            "params_b": descriptor.get("params_b"),
            "active_b": descriptor.get("active_b"),
            "quant": descriptor.get("quant"),
            "mem_gb": descriptor.get("mem_gb"),
            "ctx_max": descriptor.get("ctx_max"),
            "modalities": copy.deepcopy(descriptor.get("modalities") or []),
        },
        "evidence": {
            "precedence": {
                "serving": "server_mode/stack_manifest outrank roles metadata",
                "memory_cost": memory_source,
                "spec": str(PRECEDENCE_SPEC),
            },
            "descriptor_server_roles": _descriptor_server_roles(descriptor),
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
        "stack_priors_version": 1,
        "compiled_at": _timestamp(),
        "status": "compiled_with_gaps" if gaps_by_role else "compiled",
        "coverage_scope": "descriptor_role_bindings"
        if active_roles is None
        else "explicit_active_roles",
        "precedence_spec": str(PRECEDENCE_SPEC),
        "source_artifacts": {
            "registry": _source_metadata(registry_path),
            "descriptors": _source_metadata(descriptor_path),
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
