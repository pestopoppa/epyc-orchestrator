"""Compile model-capability descriptors from orchestrator registries."""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Any

import yaml


DEFAULT_LEAN_REGISTRY = Path("/mnt/raid0/llm/epyc-orchestrator/orchestration/model_registry.yaml")
DEFAULT_RESEARCH_REGISTRY = Path(
    "/mnt/raid0/llm/epyc-inference-research/orchestration/model_registry.yaml"
)
DEFAULT_DESCRIPTOR_OUTPUT = Path(
    "/mnt/raid0/llm/epyc-orchestrator/orchestration/model_descriptors.yaml"
)


class DescriptorCompileError(ValueError):
    """Descriptor compilation found missing load-bearing evidence."""

    def __init__(self, missing_by_model: dict[str, list[str]]) -> None:
        self.missing_by_model = missing_by_model
        lines = ["Descriptor compilation refused incomplete model descriptors:"]
        for model_id, gaps in sorted(missing_by_model.items()):
            for gap in gaps:
                lines.append(f"  - {model_id}: {gap}")
        super().__init__("\n".join(lines))


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
    for parent in (path if path.is_dir() else path.parent).resolve().parents:
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


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _coerce_scalar(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        match = re.search(r"(\d+(?:\.\d+)?)", stripped)
        if match:
            return float(match.group(1))
    return None


def _score_fraction(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number / 100.0 if number > 1.0 else number
    if isinstance(value, str):
        frac = re.search(r"(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)", value)
        if frac and float(frac.group(2)):
            return round(float(frac.group(1)) / float(frac.group(2)), 4)
        pct = re.search(r"(\d+(?:\.\d+)?)\s*%", value)
        if pct:
            return round(float(pct.group(1)) / 100.0, 4)
    return None


def _canonical_slug(value: str) -> str:
    value = value.rsplit("/", 1)[-1]
    value = re.sub(r"\.gguf$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"-0+\d+-of-0+\d+$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"[^A-Za-z0-9.]+", "-", value).strip("-").lower()
    value = re.sub(r"-+", "-", value)
    return value


def _canonical_model_id(name: str, quant: str | None) -> str:
    slug = _canonical_slug(name)
    if quant:
        quant_slug = _canonical_slug(quant)
        if not slug.endswith(quant_slug):
            slug = f"{slug}-{quant_slug}"
    return slug


def _first_model_dict(*configs: dict[str, Any] | None) -> dict[str, Any]:
    for cfg in configs:
        if not isinstance(cfg, dict):
            continue
        model = cfg.get("model")
        if isinstance(model, dict):
            return model
    return {}


def _first_model_name(*configs: dict[str, Any] | None) -> str | None:
    model = _first_model_dict(*configs)
    if model.get("name"):
        return str(model["name"])
    for cfg in configs:
        if not isinstance(cfg, dict):
            continue
        for key in ("model", "model_path"):
            value = cfg.get(key)
            if isinstance(value, str):
                return value.rsplit("/", 1)[-1]
    return None


def _quant_from(name: str | None, model: dict[str, Any]) -> str | None:
    if model.get("quant"):
        return str(model["quant"])
    if not name:
        return None
    match = re.search(
        r"(Q\d_K_[A-Z0-9]+|Q\d_K_[A-Z]+|Q\d_K_M|Q\d_K_S|Q\d_0|Q\d(?:_[0-9])?|f16|fp16)",
        name,
        flags=re.IGNORECASE,
    )
    return match.group(1) if match else None


def _params_from(name: str | None, model: dict[str, Any]) -> float | None:
    size = model.get("params_b")
    if size is not None:
        return _as_float(size)
    if not name:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)B", name, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def _active_from(name: str | None, arch: str | None, params_b: float | None) -> float | None:
    if name:
        match = re.search(r"A(\d+(?:\.\d+)?)B", name, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    if arch and "dense" in arch.lower():
        return params_b
    return None


def _family_from(model_id: str, params_b: float | None) -> str:
    if params_b is None:
        return model_id
    marker = f"{params_b:g}b"
    parts = model_id.split("-")
    family: list[str] = []
    for part in parts:
        if part == marker:
            break
        family.append(part)
    return "-".join(family) if family else model_id


def _server_for_role(
    role: str,
    server_mode: dict[str, Any],
) -> tuple[str | None, dict[str, Any] | None, str | None]:
    direct = server_mode.get(role)
    if isinstance(direct, dict):
        return role, direct, None

    for server_role, cfg in server_mode.items():
        if not isinstance(cfg, dict):
            continue
        if cfg.get("model_role") == role:
            return str(server_role), cfg, None
        shared = cfg.get("shared_with")
        if isinstance(shared, list) and role in shared:
            return str(server_role), cfg, "shared_with"
    return None, None, None


def _expanded_active_roles(
    active_roles: set[str] | None,
    registry: dict[str, Any],
) -> set[str]:
    roles = set(active_roles or set())
    server_mode = registry.get("server_mode") or {}
    top_roles = registry.get("roles") or {}
    if not roles:
        roles.update(
            key
            for key, value in server_mode.items()
            if isinstance(value, dict) and value.get("model")
        )
    for cfg in server_mode.values():
        if not isinstance(cfg, dict):
            continue
        model_role = cfg.get("model_role")
        if model_role and model_role in top_roles:
            roles.add(str(model_role))
        shared = cfg.get("shared_with")
        if isinstance(shared, list):
            roles.update(str(item) for item in shared)
    return roles


def _quality(role_cfg: dict[str, Any], server_cfg: dict[str, Any] | None) -> dict[str, Any]:
    suite_vector: dict[str, float] = {}
    performance = role_cfg.get("performance") if isinstance(role_cfg, dict) else None
    if isinstance(performance, dict):
        for key, value in performance.items():
            score = None
            if key.endswith("_pct") or key in {"quality_score", "vl_score", "long_context_quality"}:
                score = _score_fraction(value)
            elif key.endswith("_suite") or key == "long_context":
                score = _score_fraction(value)
            if score is not None:
                suite_vector[key.replace("_pct", "")] = score

    if isinstance(server_cfg, dict):
        benchmark_score = _score_fraction(server_cfg.get("benchmark_score"))
        if benchmark_score is not None and "overall" not in suite_vector:
            suite_vector["overall"] = benchmark_score

    measured: list[dict[str, Any]] = []
    if isinstance(server_cfg, dict) and server_cfg.get("benchmark_score"):
        measured.append(
            {
                "date": _coerce_scalar(server_cfg.get("benchmark_date")),
                "protocol": "registry benchmark_score",
                "value": str(server_cfg["benchmark_score"]),
            }
        )
    elif isinstance(performance, dict) and performance:
        measured.append(
            {
                "date": _coerce_scalar(role_cfg.get("benchmark_date")),
                "protocol": "registry performance block",
                "value": {k: _coerce_scalar(v) for k, v in performance.items()},
            }
        )

    return {
        "suite_vector": suite_vector,
        "source": "orchestration/model_registry.yaml",
        "eval_protocol": "MEASUREMENT.md#canonical-quality",
        "measured": measured,
    }


def _speed(role_cfg: dict[str, Any], server_cfg: dict[str, Any] | None) -> dict[str, Any]:
    performance = role_cfg.get("performance") if isinstance(role_cfg, dict) else None
    throughput = _as_float(server_cfg.get("throughput")) if isinstance(server_cfg, dict) else None
    optimized = None
    baseline = None
    if isinstance(performance, dict):
        optimized = _as_float(performance.get("optimized_tps"))
        baseline = _as_float(performance.get("baseline_tps"))

    numa_instances = server_cfg.get("numa_instances") if isinstance(server_cfg, dict) else None
    solo_96t_tps = throughput if numa_instances in (None, 1) else baseline
    quarter_48t_tps = throughput if numa_instances == 4 else None

    measured: list[dict[str, Any]] = []
    if throughput is not None:
        measured.append(
            {
                "date": _coerce_scalar(server_cfg.get("benchmark_date")),
                "protocol": "server_mode throughput",
                "value_tps": throughput,
            }
        )
    elif optimized is not None or baseline is not None:
        measured.append(
            {
                "date": _coerce_scalar(role_cfg.get("benchmark_date")),
                "protocol": "registry performance block",
                "baseline_tps": baseline,
                "optimized_tps": optimized,
            }
        )

    return {
        "solo_96t_tps": solo_96t_tps,
        "quarter_48t_tps": quarter_48t_tps,
        "prefill_tps": None,
        "optimized_tps": optimized,
        "source": "orchestration/model_registry.yaml",
        "measured": measured,
    }


def _acceleration(role_cfg: dict[str, Any], server_cfg: dict[str, Any] | None) -> dict[str, Any]:
    accel: dict[str, Any] = {}
    for source in (role_cfg, server_cfg or {}):
        candidate = source.get("acceleration") if isinstance(source, dict) else None
        if isinstance(candidate, dict):
            accel.update(candidate)

    raw_type = accel.get("spec_type") or accel.get("type") or "none"
    spec_type = str(raw_type)
    if spec_type == "speculative_decoding":
        spec_type = "draft"

    draft_compat: list[str] = []
    for key in ("draft_role", "draft_model"):
        value = accel.get(key)
        if value:
            draft_compat.append(_canonical_slug(str(value)))

    chat_kwargs = server_cfg.get("chat_template_kwargs") if isinstance(server_cfg, dict) else None
    enable_thinking = None
    if isinstance(chat_kwargs, dict) and "enable_thinking" in chat_kwargs:
        enable_thinking = bool(chat_kwargs["enable_thinking"])
    else:
        model = role_cfg.get("model") if isinstance(role_cfg, dict) else None
        if isinstance(model, dict) and model.get("disable_thinking") is True:
            enable_thinking = False

    kv = server_cfg.get("kv_quant") if isinstance(server_cfg, dict) else None
    if not isinstance(kv, dict):
        kv = {"k": None, "v": None}

    out = {
        "spec_type": spec_type,
        "draft_compat": sorted(set(draft_compat)),
        "enable_thinking": enable_thinking,
        "kv": kv,
    }
    for key in ("draft_max", "draft_p_min", "k", "p_split", "lookup", "corpus_retrieval"):
        if key in accel:
            out[key] = accel[key]
    return out


def _serving(
    role_cfg: dict[str, Any], server_role: str | None, server_cfg: dict[str, Any] | None
) -> dict[str, Any]:
    runtime = server_cfg.get("runtime_requirements") if isinstance(server_cfg, dict) else None
    binary_dir = runtime.get("binary_dir") if isinstance(runtime, dict) else None
    if binary_dir and "ik_llama.cpp" in str(binary_dir):
        binary = "ik-pr1744"
    elif isinstance(server_cfg, dict) and server_cfg.get("backend"):
        binary = str(server_cfg["backend"])
    else:
        binary = "llama.cpp"

    ports: list[int] = []
    if isinstance(server_cfg, dict):
        port = server_cfg.get("port")
        if isinstance(port, int):
            ports.append(port)
        numa_ports = server_cfg.get("numa_ports")
        if isinstance(numa_ports, list):
            ports.extend(int(item) for item in numa_ports if isinstance(item, int))

    numa_instances = server_cfg.get("numa_instances") if isinstance(server_cfg, dict) else None
    if not server_cfg:
        numa_policy = "unresolved_no_server_binding"
    elif numa_instances == 4:
        numa_policy = "4x48t_quarter_instances"
    elif numa_instances == 1:
        numa_policy = "single_96t"
    else:
        numa_policy = "server_default"

    memory = role_cfg.get("memory") if isinstance(role_cfg, dict) else None
    return {
        "binary": binary,
        "binary_dir": str(binary_dir) if binary_dir else None,
        "numa_policy": numa_policy,
        "mlock": bool(memory.get("pinned")) if isinstance(memory, dict) else False,
        "ports": sorted(set(ports)),
        "server_role": server_role,
    }


def _merge_descriptor(target: dict[str, Any], incoming: dict[str, Any]) -> None:
    for key in ("roles", "server_roles", "runtime_aliases"):
        values = incoming["role_bindings"].get(key) or []
        existing = target["role_bindings"].setdefault(key, [])
        target["role_bindings"][key] = sorted(set(existing) | set(values))
    target["role_bindings"]["shared_mmap"] = target["role_bindings"].get("shared_mmap") or incoming[
        "role_bindings"
    ].get("shared_mmap")
    target["quality"]["suite_vector"].update(incoming["quality"]["suite_vector"])
    target["quality"]["measured"].extend(incoming["quality"]["measured"])
    if not target["speed"]["measured"]:
        target["speed"] = incoming["speed"]
    for key in ("ports",):
        target["serving"][key] = sorted(
            set(target["serving"].get(key) or []) | set(incoming["serving"].get(key) or [])
        )
    target["known_gaps"] = sorted(set(target["known_gaps"]) | set(incoming["known_gaps"]))


def _descriptor_gaps(descriptor: dict[str, Any]) -> list[str]:
    gaps: list[str] = []
    if not descriptor["quality"]["suite_vector"]:
        gaps.append("Missing quality suite_vector evidence")
    if not descriptor["speed"]["measured"]:
        gaps.append("Missing speed measurement evidence")
    if not descriptor["serving"]["ports"]:
        gaps.append("Missing serving port binding")
    if descriptor["serving"]["numa_policy"] == "unresolved_no_server_binding":
        gaps.append("Missing server_mode binding")
    if descriptor["acceleration"]["enable_thinking"] is None:
        gaps.append("Missing enable_thinking compatibility evidence")
    if descriptor["ctx_max"] is None:
        gaps.append("Missing structured ctx_max")
    return gaps


def _descriptor_for_role(
    role: str,
    role_cfg: dict[str, Any],
    server_role: str | None,
    server_cfg: dict[str, Any] | None,
    binding_kind: str | None,
) -> dict[str, Any] | None:
    model = _first_model_dict(role_cfg, server_cfg)
    model_name = _first_model_name(role_cfg, server_cfg)
    if not model_name:
        return None

    quant = _quant_from(model_name, model)
    model_id = _canonical_model_id(model_name, quant)
    arch = str(model.get("architecture") or "unknown")
    params_b = _params_from(model_name, model)
    active_b = _active_from(model_name, arch, params_b)

    mem_gb = model.get("size_gb")
    if mem_gb is None and isinstance(server_cfg, dict):
        mem_gb = server_cfg.get("memory_gb")

    server_model_name = _first_model_name(server_cfg) if server_cfg else None
    server_model_id = (
        _canonical_model_id(server_model_name, _quant_from(server_model_name, {}))
        if server_model_name
        else None
    )

    known_gaps: list[str] = []
    runtime_aliases: list[str] = []
    if binding_kind == "shared_with" and server_role:
        runtime_aliases.append(server_role)
        if server_model_id and server_model_id != model_id:
            known_gaps.append(
                "Role-server conflict: role model metadata does not match the shared runtime server model"
            )

    descriptor = {
        "model_id": model_id,
        "display_name": str(model.get("name") or model_name),
        "family": _family_from(model_id, params_b),
        "arch": arch,
        "params_b": params_b,
        "active_b": active_b,
        "quant": quant,
        "mem_gb": _as_float(mem_gb),
        "ctx_max": model.get("ctx_max") or model.get("context_length"),
        "modalities": ["text"],
        "role_bindings": {
            "roles": [role],
            "server_roles": [server_role] if server_role else [],
            "runtime_aliases": runtime_aliases,
            "shared_mmap": bool(server_cfg),
        },
        "quality": _quality(role_cfg, server_cfg),
        "speed": _speed(role_cfg, server_cfg),
        "acceleration": _acceleration(role_cfg, server_cfg),
        "serving": _serving(role_cfg, server_role, server_cfg),
        "known_gaps": known_gaps,
    }
    descriptor["known_gaps"].extend(_descriptor_gaps(descriptor))
    return descriptor


def _source_metadata(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "repo_commit": _repo_commit(path),
        "sha256": _sha256(path),
    }


def compile_model_descriptors(
    lean_registry_path: Path = DEFAULT_LEAN_REGISTRY,
    research_registry_path: Path | None = DEFAULT_RESEARCH_REGISTRY,
    *,
    active_roles: set[str] | None = None,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    """Compile one descriptor per canonical model identity.

    The current W2 compiler is intentionally conservative: it derives model identity,
    role bindings, serving, acceleration, and registry-carried quality/speed evidence
    from the lean registry. It records research registry provenance in the metadata so
    W2 follow-up work can promote richer benchmark artifacts without rewriting source
    registries in place.
    """
    registry = _load_yaml(lean_registry_path)
    if research_registry_path is not None and research_registry_path.exists():
        _load_yaml(research_registry_path)

    roles = registry.get("roles") or {}
    server_mode = registry.get("server_mode") or {}
    if not isinstance(roles, dict) or not isinstance(server_mode, dict):
        raise ValueError("registry must contain mapping-valued roles and server_mode sections")

    descriptors: dict[str, dict[str, Any]] = {}
    for role in sorted(_expanded_active_roles(active_roles, registry)):
        role_cfg = roles.get(role)
        server_role, server_cfg, binding_kind = _server_for_role(role, server_mode)
        if not isinstance(role_cfg, dict) and isinstance(server_cfg, dict):
            model_role = server_cfg.get("model_role")
            role_cfg = roles.get(model_role) if model_role else {}
        if not isinstance(role_cfg, dict):
            role_cfg = {}

        descriptor = _descriptor_for_role(role, role_cfg, server_role, server_cfg, binding_kind)
        if descriptor is None:
            continue
        existing = descriptors.get(descriptor["model_id"])
        if existing:
            _merge_descriptor(existing, descriptor)
        else:
            descriptors[descriptor["model_id"]] = descriptor

    for descriptor in descriptors.values():
        manual_gaps = [gap for gap in descriptor["known_gaps"] if not gap.startswith("Missing ")]
        descriptor["known_gaps"] = sorted(set(manual_gaps + _descriptor_gaps(descriptor)))

    missing_by_model = {
        model_id: descriptor["known_gaps"]
        for model_id, descriptor in descriptors.items()
        if descriptor["known_gaps"]
    }
    if missing_by_model and not allow_incomplete:
        raise DescriptorCompileError(missing_by_model)

    status = "compiled_with_gaps" if missing_by_model else "compiled"
    metadata: dict[str, Any] = {
        "descriptor_version": 3,
        "compiled_at": _timestamp(),
        "status": status,
        "coverage_scope": "active_roles_from_stack_manifest",
        "source_registries": {"lean": _source_metadata(lean_registry_path)},
        "model_id_policy": {
            "canonical_shape": "family-params-active-quant",
            "invariant": "model records are keyed by physical model identity, never by role name",
        },
        "known_global_gaps": [
            "Research registry and bench-artifact enrichment is provenance-recorded but not yet field-promoted.",
            "Source registry comments remain the measurement witness; do not reformat them in place.",
        ],
        "models": [descriptors[key] for key in sorted(descriptors)],
    }
    if research_registry_path is not None:
        metadata["source_registries"]["research"] = _source_metadata(research_registry_path)
    return metadata


def write_model_descriptors(
    output_path: Path,
    *,
    lean_registry_path: Path = DEFAULT_LEAN_REGISTRY,
    research_registry_path: Path | None = DEFAULT_RESEARCH_REGISTRY,
    active_roles: set[str] | None = None,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    descriptors = compile_model_descriptors(
        lean_registry_path=lean_registry_path,
        research_registry_path=research_registry_path,
        active_roles=active_roles,
        allow_incomplete=allow_incomplete,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(
            descriptors,
            fh,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
            width=200,
        )
    return descriptors


def _roles_from_stack_manifest() -> set[str]:
    sys.path.insert(0, "/mnt/raid0/llm/epyc-orchestrator/scripts/server")
    from stack_manifest import ROLE_LAUNCH_META  # type: ignore[import]

    active = set(ROLE_LAUNCH_META.keys())
    for meta in ROLE_LAUNCH_META.values():
        aliases = meta.get("shared_with_first_n") if isinstance(meta, dict) else None
        if isinstance(aliases, list):
            active.update(str(alias) for alias in aliases)
    return active


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile model-capability descriptors")
    parser.add_argument("--lean", type=Path, default=DEFAULT_LEAN_REGISTRY)
    parser.add_argument("--research", type=Path, default=DEFAULT_RESEARCH_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_DESCRIPTOR_OUTPUT)
    parser.add_argument("--roles", nargs="+", help="Explicit active role list")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print descriptors instead of writing"
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Emit descriptors with known_gaps instead of refusing incomplete records",
    )
    args = parser.parse_args(argv)

    active_roles = set(args.roles) if args.roles else _roles_from_stack_manifest()
    descriptors = compile_model_descriptors(
        lean_registry_path=args.lean,
        research_registry_path=args.research,
        active_roles=active_roles,
        allow_incomplete=args.allow_incomplete,
    )
    if args.dry_run:
        yaml.safe_dump(
            descriptors,
            sys.stdout,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
            width=200,
        )
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(
                descriptors,
                fh,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
                width=200,
            )
        print(f"OK: wrote {len(descriptors.get('models', []))} descriptors to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
