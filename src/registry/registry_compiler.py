"""Compile a lean orchestrator registry from the master research registry.

Background (2026-05-09): Today's recovery hit a bug because the orchestrator's
hand-edited `model_registry.yaml` had drifted from the master at
`epyc-inference-research/orchestration/model_registry.yaml`. The validator
catches in-file inconsistencies but the master/orchestrator coupling itself
was hand-maintained — silent drift was inevitable.

This module compiles a lean view from `(master + active_role_set)` at every
stack launch. The lean view contains exactly the roles the orchestrator will
launch plus their transitive dependencies (drafts, aliases). Sections needed
only for research/benchmarking (deprecated_models, kernel_audits, observations,
runtime_quirks) are dropped.

Cache-invalidates by SHA-256 of `(master content + sorted active role names)`.
If neither side has changed, the cached compile output is reused.

Wired into `orchestrator_stack.py:cmd_start` before the validator gate. The
auto-generated lean output replaces the orchestrator's `model_registry.yaml`.
A header banner makes the auto-generated nature visible to anyone opening the
file.

Escape hatch: setting `ORCHESTRATOR_REGISTRY_NO_COMPILE=1` skips the compile
step and falls back to the on-disk file (useful during a master-registry
schema change when you need to hand-edit before re-syncing).
"""

from __future__ import annotations

import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


# Top-level sections in the master that the orchestrator runtime DOES need.
# Anything not listed is dropped from the lean output.
_KEEP_SECTIONS_FULL = (
    "runtime_defaults",
    "worker_pool",
    "process_layout",
    "escalation_chains",
    "routing_hints",
    "command_templates",
    "dflash_drafters",
)
# Sections that are role-keyed dicts and need filtering.
_FILTER_ROLE_KEYED = ("server_mode", "roles")


def _server_mode_entry_roles(entry_name: str, entry: Any) -> set[str]:
    """Return all role names covered by a server_mode entry."""
    roles = {entry_name}
    if not isinstance(entry, dict):
        return roles

    model_role = entry.get("model_role")
    if isinstance(model_role, str):
        roles.add(model_role)

    shared_with = entry.get("shared_with")
    if isinstance(shared_with, list):
        roles.update(str(role) for role in shared_with)

    return roles


def _filter_server_mode(server_mode: dict[str, Any], needed: set[str]) -> dict[str, Any]:
    """Keep server records that directly or indirectly serve needed roles."""
    return {
        name: entry
        for name, entry in server_mode.items()
        if _server_mode_entry_roles(name, entry) & needed
    }


def cache_key(master_path: Path, active_roles: set[str]) -> str:
    """SHA-256 of master file bytes + sorted active role names.

    Recompiles when EITHER side changes. Stable across runs when neither
    changes — repeated `cmd_start` calls hit the cache.
    """
    h = hashlib.sha256()
    h.update(master_path.read_bytes())
    h.update(b"\x00")  # separator
    for r in sorted(active_roles):
        h.update(r.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _resolve_role_dependencies(master: dict, active_roles: set[str]) -> set[str]:
    """Walk role definitions to find transitive deps (draft models, aliases).

    A role's `acceleration.draft_role` references another role by name (e.g.
    `architect_general.acceleration.draft_role = "draft_qwen35_0_8b_q8_0"`).
    The draft role MUST be in the lean output too or `RegistryLoader.get_role`
    will fail when the launcher tries to look it up.
    """
    resolved = set(active_roles)
    roles_section = master.get("roles") or {}
    server_mode = master.get("server_mode") or {}

    # Iterate until fixed point — a draft role might itself reference another role.
    changed = True
    while changed:
        changed = False
        for r in list(resolved):
            for source in (roles_section.get(r), server_mode.get(r)):
                if not isinstance(source, dict):
                    continue
                accel = source.get("acceleration")
                if isinstance(accel, dict):
                    dr = accel.get("draft_role")
                    if dr and dr not in resolved:
                        resolved.add(dr)
                        changed = True
                # `model_role` field cross-refs a role definition (architect_general
                # in server_mode points at qwen35_122b_q4km — both must survive).
                mr = source.get("model_role")
                if mr and mr not in resolved:
                    resolved.add(mr)
                    changed = True
                # `shared_with` lists — defensively include
                sw = source.get("shared_with")
                if isinstance(sw, list):
                    for s in sw:
                        if s not in resolved:
                            resolved.add(s)
                            changed = True
    return resolved


def active_roles_from_launch_meta(launch_meta: dict[str, Any]) -> set[str]:
    """Return compile input roles from launcher metadata, including aliases.

    `ROLE_LAUNCH_META` keys are the primary launch targets. Some production
    roles are aliases attached to the first N instances of a primary target
    (`shared_with_first_n`), so they do not appear as top-level launch keys.
    The lean registry still needs their `roles.X` / `server_mode.X` records for
    routing, timeout, descriptor, and attestation consumers.
    """
    active = set(launch_meta.keys())
    for meta in launch_meta.values():
        if not isinstance(meta, dict):
            continue
        aliases = meta.get("shared_with_first_n")
        if isinstance(aliases, list):
            active.update(str(alias) for alias in aliases)
    return active


def compile_lean(master_path: Path, active_roles: set[str]) -> dict:
    """Project master registry through the active stack manifest.

    Returns a dict in the same shape RegistryLoader expects.
    """
    with master_path.open("r", encoding="utf-8") as f:
        master = yaml.safe_load(f)
    if not isinstance(master, dict):
        raise ValueError(f"master registry at {master_path} did not parse to a dict")

    needed = _resolve_role_dependencies(master, active_roles)

    out: dict[str, Any] = {}

    # Full-keep sections — small, no filtering needed.
    for section in _KEEP_SECTIONS_FULL:
        if section in master:
            out[section] = master[section]

    # Role-keyed sections — filter to needed roles only. `server_mode` entries
    # may be keyed by backing process name rather than route role name
    # (`worker` -> `worker_general`), so inspect entry metadata too.
    for section in _FILTER_ROLE_KEYED:
        src = master.get(section)
        if not isinstance(src, dict):
            continue
        if section == "server_mode":
            out[section] = _filter_server_mode(src, needed)
        else:
            out[section] = {k: v for k, v in src.items() if k in needed}

    return out


def _format_header_banner(
    master_path: Path, active_roles: set[str], cache_key_value: str
) -> str:
    """Top-of-file comment block for the compiled output."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    role_list = ", ".join(sorted(active_roles))
    return (
        "# =============================================================================\n"
        "# AUTO-GENERATED — DO NOT HAND-EDIT.\n"
        "#\n"
        "# This file is compiled at every `orchestrator_stack.py start` from the master\n"
        f"# registry at {master_path}\n"
        "# by src/registry/registry_compiler.py. Hand edits will be silently overwritten\n"
        "# at the next stack launch.\n"
        "#\n"
        "# To make a real change: edit the MASTER registry. The next start will detect the\n"
        "# change (cache key mismatch) and regenerate this file. To temporarily skip the\n"
        "# compile (e.g. during a master schema change), set ORCHESTRATOR_REGISTRY_NO_COMPILE=1.\n"
        "#\n"
        f"# Compiled at: {now}\n"
        f"# Cache key:   {cache_key_value[:16]}...\n"
        f"# Active roles: {role_list}\n"
        "# =============================================================================\n"
        "\n"
    )


def load_or_compile(
    master_path: Path,
    active_roles: set[str],
    output_path: Path,
    cache_key_path: Path,
) -> dict:
    """Cache-aware compile.

    On cache hit (key matches) returns the existing `output_path` parsed.
    On miss, recompiles, writes `output_path`, updates `cache_key_path`.
    """
    if os.environ.get("ORCHESTRATOR_REGISTRY_NO_COMPILE") == "1":
        # Escape hatch — read whatever's on disk, no compile.
        if output_path.exists():
            with output_path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        raise FileNotFoundError(
            f"ORCHESTRATOR_REGISTRY_NO_COMPILE=1 set but {output_path} does not exist"
        )

    if not master_path.exists():
        raise FileNotFoundError(f"master registry not found: {master_path}")

    current_key = cache_key(master_path, active_roles)
    cached_key = (
        cache_key_path.read_text().strip()
        if cache_key_path.exists()
        else None
    )

    if cached_key == current_key and output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            cached = yaml.safe_load(f)
        if isinstance(cached, dict):
            return cached
        # Cached file unparseable — fall through to recompile.

    # Compile and persist.
    compiled = compile_lean(master_path, active_roles)
    banner = _format_header_banner(master_path, active_roles, current_key)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(banner)
        yaml.safe_dump(
            compiled,
            f,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
            width=200,
        )
    cache_key_path.parent.mkdir(parents=True, exist_ok=True)
    cache_key_path.write_text(current_key)
    return compiled


# --- CLI for diagnostic / dry-run inspection -------------------------------

def _main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Compile lean orchestrator registry")
    p.add_argument(
        "--master",
        type=Path,
        default=Path(
            "/mnt/raid0/llm/epyc-inference-research/orchestration/model_registry.yaml"
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/mnt/raid0/llm/epyc-orchestrator/orchestration/model_registry.yaml"
        ),
    )
    p.add_argument(
        "--cache-key",
        type=Path,
        default=Path(
            "/mnt/raid0/llm/epyc-orchestrator/orchestration/.lean_cache_key"
        ),
    )
    p.add_argument(
        "--roles",
        nargs="+",
        help="Explicit active role list. If omitted, imports ROLE_LAUNCH_META keys.",
    )
    p.add_argument("--dry-run", action="store_true", help="Compile to stdout, do not write")
    p.add_argument("--force", action="store_true", help="Recompile even on cache hit")
    args = p.parse_args()

    if args.roles:
        active = set(args.roles)
    else:
        # Import the launcher's manifest.
        sys.path.insert(0, "/mnt/raid0/llm/epyc-orchestrator/scripts/server")
        from orchestrator_stack import ROLE_LAUNCH_META  # type: ignore[import]
        active = active_roles_from_launch_meta(ROLE_LAUNCH_META)

    if args.dry_run:
        compiled = compile_lean(args.master, active)
        yaml.safe_dump(compiled, sys.stdout, sort_keys=False, default_flow_style=False)
        return 0

    if args.force and args.cache_key.exists():
        args.cache_key.unlink()

    out = load_or_compile(args.master, active, args.output, args.cache_key)
    print(f"OK: {len(out.get('roles', {}))} roles in compiled output")
    print(f"  master:    {args.master}")
    print(f"  output:    {args.output}")
    print(f"  cache key: {args.cache_key}")
    print(f"  active:    {sorted(active)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
