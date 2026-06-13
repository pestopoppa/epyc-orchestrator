#!/usr/bin/env python3
"""Sync procedure role validation enums from generated stack priors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path("/mnt/raid0/llm/epyc-orchestrator")
DEFAULT_PRIORS = REPO_ROOT / "orchestration" / "derived" / "stack_priors.yaml"
DEFAULT_PROCEDURE = REPO_ROOT / "orchestration" / "procedures" / "add_model_to_registry.yaml"
DEFAULT_SCHEMA = REPO_ROOT / "orchestration" / "procedure.schema.json"

sys.path.insert(0, str(REPO_ROOT))

from scripts.validate.stack_change_guard import (  # noqa: E402
    stack_prior_permission_role_choices,
    stack_prior_role_choices,
)


class ProcedureRoleSyncError(ValueError):
    """Procedure role enum sync could not be completed."""


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if not isinstance(loaded, dict):
        raise ProcedureRoleSyncError(f"{path} did not parse to a mapping")
    return loaded


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        loaded = json.load(fh)
    if not isinstance(loaded, dict):
        raise ProcedureRoleSyncError(f"{path} did not parse to a mapping")
    return loaded


def _role_input_validation(procedure: dict[str, Any]) -> dict[str, Any]:
    inputs = procedure.get("inputs")
    if not isinstance(inputs, list):
        raise ProcedureRoleSyncError("procedure has no inputs list")
    for raw_input in inputs:
        if not isinstance(raw_input, dict) or raw_input.get("name") != "role":
            continue
        validation = raw_input.setdefault("validation", {})
        if not isinstance(validation, dict):
            raise ProcedureRoleSyncError("role input validation must be a mapping")
        return validation
    raise ProcedureRoleSyncError("procedure has no input named 'role'")


def _inline_yaml_list(values: list[str]) -> str:
    escaped = [value.replace("\\", "\\\\").replace('"', '\\"') for value in values]
    return "[" + ", ".join(f'"{value}"' for value in escaped) + "]"


def _inline_json_list(values: list[str]) -> str:
    return json.dumps(values)


def _leading_spaces(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _replace_role_enum_text(text: str, roles: list[str]) -> str:
    lines = text.splitlines()
    role_idx = next(
        (idx for idx, line in enumerate(lines) if line.strip() == "- name: role"),
        None,
    )
    if role_idx is None:
        raise ProcedureRoleSyncError("procedure has no input named 'role'")

    role_indent = _leading_spaces(lines[role_idx])
    segment_end = len(lines)
    for idx in range(role_idx + 1, len(lines)):
        line = lines[idx]
        if _leading_spaces(line) <= role_indent and line.strip().startswith("- name:"):
            segment_end = idx
            break

    enum_idx = next(
        (
            idx
            for idx in range(role_idx + 1, segment_end)
            if lines[idx].strip().startswith("enum:")
        ),
        None,
    )
    if enum_idx is None:
        raise ProcedureRoleSyncError("role input validation has no enum")

    enum_indent = _leading_spaces(lines[enum_idx])
    replacement = f"{' ' * enum_indent}enum: {_inline_yaml_list(roles)}"
    block_end = enum_idx + 1
    while block_end < segment_end:
        next_indent = _leading_spaces(lines[block_end])
        next_stripped = lines[block_end].strip()
        if next_indent > enum_indent:
            block_end += 1
            continue
        if next_indent == enum_indent and next_stripped.startswith("- "):
            block_end += 1
            continue
        break

    updated = lines[:enum_idx] + [replacement] + lines[block_end:]
    return "\n".join(updated) + ("\n" if text.endswith("\n") else "")


def _schema_permission_enum(schema: dict[str, Any]) -> list[str] | None:
    try:
        enum = schema["properties"]["permissions"]["properties"]["roles"]["items"]["enum"]
    except (KeyError, TypeError):
        return None
    if not isinstance(enum, list):
        return None
    return [str(item) for item in enum if isinstance(item, str)]


def _replace_schema_permission_enum_text(text: str, roles: list[str]) -> str:
    lines = text.splitlines()
    roles_idx = next(
        (idx for idx, line in enumerate(lines) if line.strip() == '"roles": {'),
        None,
    )
    if roles_idx is None:
        raise ProcedureRoleSyncError("procedure schema has no permissions.roles block")

    roles_indent = _leading_spaces(lines[roles_idx])
    segment_end = len(lines)
    for idx in range(roles_idx + 1, len(lines)):
        if _leading_spaces(lines[idx]) <= roles_indent and lines[idx].strip().startswith('"'):
            segment_end = idx
            break

    enum_idx = next(
        (
            idx
            for idx in range(roles_idx + 1, segment_end)
            if lines[idx].strip().startswith('"enum":')
        ),
        None,
    )
    if enum_idx is None:
        raise ProcedureRoleSyncError("procedure schema permissions.roles has no enum")

    enum_indent = _leading_spaces(lines[enum_idx])
    suffix = "," if lines[enum_idx].rstrip().endswith(",") else ""
    replacement = f"{' ' * enum_indent}\"enum\": {_inline_json_list(roles)}{suffix}"
    return "\n".join(lines[:enum_idx] + [replacement] + lines[enum_idx + 1 :]) + (
        "\n" if text.endswith("\n") else ""
    )


def sync_procedure_role_enums(
    *,
    priors_path: Path = DEFAULT_PRIORS,
    procedure_path: Path = DEFAULT_PROCEDURE,
    schema_path: Path = DEFAULT_SCHEMA,
    check: bool = False,
) -> bool:
    """Update procedure role choices from stack priors.

    Returns True when both files already matched or were updated.
    Returns False in check mode when either file would need changes.
    """
    priors = _load_yaml(priors_path)
    roles = stack_prior_role_choices(priors)
    permission_roles = stack_prior_permission_role_choices(priors)
    if not roles:
        raise ProcedureRoleSyncError("stack priors produced no role choices")
    if not permission_roles:
        raise ProcedureRoleSyncError("stack priors produced no permission role choices")

    procedure = _load_yaml(procedure_path)
    validation = _role_input_validation(procedure)
    procedure_current = validation.get("enum")
    schema = _load_json(schema_path)
    schema_current = _schema_permission_enum(schema)
    changed = procedure_current != roles or schema_current != permission_roles
    if check and changed:
        return False

    if procedure_current != roles:
        updated = _replace_role_enum_text(procedure_path.read_text(encoding="utf-8"), roles)
        procedure_path.write_text(updated, encoding="utf-8")
        reloaded = _load_yaml(procedure_path)
        if _role_input_validation(reloaded).get("enum") != roles:
            raise ProcedureRoleSyncError("failed to write updated procedure role enum")
    if schema_current != permission_roles:
        updated = _replace_schema_permission_enum_text(
            schema_path.read_text(encoding="utf-8"),
            permission_roles,
        )
        schema_path.write_text(updated, encoding="utf-8")
        reloaded = _load_json(schema_path)
        if _schema_permission_enum(reloaded) != permission_roles:
            raise ProcedureRoleSyncError("failed to write updated procedure schema role enum")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sync procedure role enums from stack priors")
    parser.add_argument("--priors", type=Path, default=DEFAULT_PRIORS)
    parser.add_argument("--procedure", type=Path, default=DEFAULT_PROCEDURE)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--check", action="store_true", help="Fail if procedure role enums would change")
    args = parser.parse_args(argv)

    try:
        ok = sync_procedure_role_enums(
            priors_path=args.priors,
            procedure_path=args.procedure,
            schema_path=args.schema,
            check=args.check,
        )
    except ProcedureRoleSyncError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    if ok:
        print(f"OK: {args.procedure}")
        return 0
    print(
        "FAIL: procedure role enums are stale; rerun without --check",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
