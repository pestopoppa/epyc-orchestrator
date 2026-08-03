"""Registry consistency validator (added 2026-05-09).

Catches the failure modes that cost ~2 hours of debugging on 2026-05-09:

1. **Cross-section conflicts**: a role appears in both `server_mode.roles.X`
   and top-level `roles.X` with disagreeing values for fields the launcher
   reads from one section but the runtime reads from the other (e.g.
   `acceleration.type`). Edits to one path silently no-op.

2. **GGUF/port inconsistency**: two roles share the same `model.full_path`
   (= same llama-server should host them) but declare different `port:`
   values. This was the frontdoor + coder_escalation duplicate that wasted
   36 GB of mlocked weights and ran two competing OMP teams.

3. **YAML duplicate keys at the same path**: PyYAML silently keeps the LAST
   declaration. Detect and reject.

Wire-in (intended): orchestrator_stack.py:cmd_start calls `validate_all` on
the resolved registry path before any RegistryLoader instance, fails fast
on errors.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]


# Fields that, if declared in BOTH `server_mode.X` and `roles.X`,
# must agree across both (or only be declared in one).
# NOTE: `model` and `draft_model` are intentionally NOT here — server_mode
# uses short filename strings while roles uses structured dicts. They're
# different representations by design; comparing them produces noise.
# The model-path consistency check happens separately in
# `_check_gguf_port_consistency` which compares resolved GGUF paths.
_DUAL_DECLARED_CONFLICT_FIELDS = (
    "acceleration",
)


class RegistryValidationError(Exception):
    """Registry violates a consistency invariant."""


def _strict_yaml_load(path: Path) -> dict[str, Any]:
    """Load YAML and reject duplicate keys at the same mapping level.

    PyYAML's default safe_load silently keeps the last value for a duplicate
    key. We override the constructor to raise instead.
    """

    class _StrictLoader(yaml.SafeLoader):
        pass

    def _no_duplicate_keys(loader, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in mapping:
                raise RegistryValidationError(
                    f"Duplicate key {key!r} at line {key_node.start_mark.line + 1} "
                    f"in {path} (first defined earlier in same mapping)"
                )
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    _StrictLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicate_keys
    )

    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f, Loader=_StrictLoader)


def _check_cross_section_conflicts(registry: dict[str, Any]) -> list[str]:
    """Check that role-level fields don't disagree between server_mode and roles sections.

    The registry layout is:
        server_mode:
          <role_name>:    # launch config: port, slots, model, draft_model, acceleration, ...
          ...
        roles:
          <role_name>:    # model metadata: tier, description, model, acceleration, ...
          ...

    Both sections may declare overlapping fields. They MUST agree (or the field
    must only live in one section).
    """
    errors: list[str] = []
    server_mode = registry.get("server_mode") or {}
    top_roles = registry.get("roles") or {}

    # server_mode keys are role names directly (not nested under 'roles').
    # Filter to dict values only (other server_mode keys may be scalar config).
    server_roles = {k: v for k, v in server_mode.items() if isinstance(v, dict) and k in top_roles}

    for role in sorted(server_roles):
        s = server_roles[role]
        t = top_roles.get(role) or {}
        for field in _DUAL_DECLARED_CONFLICT_FIELDS:
            if field in s and field in t and s[field] != t[field]:
                # For acceleration we compare type (the most-load-bearing key);
                # avoids spurious conflicts on benchmark-date or notes fields.
                if field == "acceleration":
                    s_type = s[field].get("type") if isinstance(s[field], dict) else None
                    t_type = t[field].get("type") if isinstance(t[field], dict) else None
                    if s_type == t_type:
                        continue
                    errors.append(
                        f"role {role!r}: acceleration.type disagrees between "
                        f"server_mode.{role}.acceleration ({s_type!r}) and "
                        f"roles.{role}.acceleration ({t_type!r}). "
                        f"Pick one source of truth; remove the other section's "
                        f"acceleration block. Today's RegistryLoader reads roles.X."
                    )
                else:
                    errors.append(
                        f"role {role!r}: {field!r} disagrees between sections. "
                        f"server_mode.{role}.{field} = {s[field]!r:.80}; "
                        f"roles.{role}.{field} = {t[field]!r:.80}. "
                        f"Pick one section as authoritative."
                    )
    return errors


def _check_gguf_port_consistency(registry: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    by_model: dict[str, list[tuple[str, int]]] = defaultdict(list)

    # server_mode keys are role names directly. Filter to dict values.
    server_mode = registry.get("server_mode") or {}
    top_roles = registry.get("roles") or {}
    server_roles = {k: v for k, v in server_mode.items() if isinstance(v, dict)}

    for role, cfg in server_roles.items():
        if not isinstance(cfg, dict):
            continue
        port = cfg.get("port")
        if port is None:
            continue
        # Look up the model path: prefer server_mode.roles.X.model_path,
        # fall back to top-level roles.X.model.path or .full_path.
        model_path = cfg.get("model_path") or cfg.get("model")
        if not model_path or not isinstance(model_path, str):
            top = top_roles.get(role) or {}
            model_field = top.get("model")
            if isinstance(model_field, dict):
                model_path = model_field.get("path") or model_field.get("full_path")
        if model_path and isinstance(model_path, str):
            by_model[model_path].append((role, int(port)))

    for model_path, role_ports in by_model.items():
        ports = {p for _, p in role_ports}
        if len(ports) > 1:
            roles_list = ", ".join(f"{r}:{p}" for r, p in sorted(role_ports))
            errors.append(
                f"GGUF {model_path.split('/')[-1]!r} is declared with "
                f"multiple ports ({sorted(ports)}). Roles: {roles_list}. "
                f"Same-GGUF roles must share one llama-server (same port); "
                f"set one canonical port for the whole group."
            )
    return errors


def validate_all(registry_path: str | Path) -> list[str]:
    """Run all checks. Returns list of error strings. Empty list = OK.

    Caller should print the errors and exit non-zero if any returned.
    """
    path = Path(registry_path)
    if not path.exists():
        return [f"registry not found: {path}"]

    registry = _strict_yaml_load(path)  # may raise RegistryValidationError on dup keys

    errors: list[str] = []
    errors += _check_cross_section_conflicts(registry)
    errors += _check_gguf_port_consistency(registry)
    return errors


def validate_or_raise(registry_path: str | Path) -> None:
    """Validate; on any error, raise RegistryValidationError with combined message."""
    errors = validate_all(registry_path)
    if errors:
        joined = "\n  - ".join(errors)
        raise RegistryValidationError(
            f"Registry validation failed for {registry_path}:\n  - {joined}"
        )


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else (
        str(_REPO_ROOT / "orchestration/model_registry.yaml")
    )
    try:
        errors = validate_all(target)
    except RegistryValidationError as exc:
        print(f"FAIL: {exc}")
        sys.exit(2)
    if errors:
        print(f"FAIL: {len(errors)} validation error(s) in {target}:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print(f"OK: {target}")
