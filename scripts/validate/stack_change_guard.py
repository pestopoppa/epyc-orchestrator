#!/usr/bin/env python3
"""Validate generated stack priors against current source artifacts."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path("/mnt/raid0/llm/epyc-orchestrator")
DEFAULT_PRIORS = REPO_ROOT / "orchestration" / "derived" / "stack_priors.yaml"
RETIRED_LIVE_ROLES = {"architect_coding"}


@dataclass(frozen=True)
class GuardResult:
    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


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


def _source_path(priors_path: Path, source: dict[str, Any]) -> Path | None:
    raw = source.get("path")
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    return (priors_path.parent / path).resolve()


def validate_stack_priors(priors_path: Path = DEFAULT_PRIORS, *, strict: bool = False) -> GuardResult:
    errors: list[str] = []
    warnings: list[str] = []
    if not priors_path.exists():
        return GuardResult(errors=[f"missing stack priors artifact: {priors_path}"], warnings=[])

    priors = _load_yaml(priors_path)
    roles = priors.get("roles")
    if not isinstance(roles, dict):
        errors.append("stack priors artifact has no mapping-valued roles section")
        roles = {}

    sources = priors.get("source_artifacts") or {}
    if not isinstance(sources, dict):
        errors.append("stack priors artifact has no source_artifacts section")
        sources = {}

    for label in ("registry", "descriptors"):
        source = sources.get(label)
        if not isinstance(source, dict):
            errors.append(f"missing source_artifacts.{label}")
            continue
        path = _source_path(priors_path, source)
        expected = source.get("sha256")
        actual = _sha256(path) if path else None
        if not path or actual is None:
            errors.append(f"source_artifacts.{label} path is missing or unreadable: {source.get('path')!r}")
        elif expected != actual:
            errors.append(
                f"source_artifacts.{label} hash mismatch: {path} expected {expected}, got {actual}"
            )

    for role in sorted(RETIRED_LIVE_ROLES & set(roles)):
        record = roles.get(role) or {}
        if record.get("deployment_status") == "live_stack":
            errors.append(f"retired role {role!r} appears as live_stack")
        else:
            warnings.append(f"retired role {role!r} appears in non-live priors")

    for role, record in sorted(roles.items()):
        if not isinstance(record, dict):
            errors.append(f"role {role!r} record is not a mapping")
            continue
        deployment_status = record.get("deployment_status")
        serving = record.get("serving") if isinstance(record.get("serving"), dict) else {}
        priors_block = record.get("priors") if isinstance(record.get("priors"), dict) else {}
        known_gaps = record.get("known_gaps") if isinstance(record.get("known_gaps"), list) else []
        if deployment_status == "live_stack":
            if not record.get("model_id"):
                errors.append(f"live role {role!r} is missing model_id")
            if not serving.get("endpoint"):
                errors.append(f"live role {role!r} is missing serving.endpoint")
            if serving.get("tier") == "hot" and priors_block.get("memory_cost") != 1.0:
                errors.append(
                    f"live HOT role {role!r} has memory_cost={priors_block.get('memory_cost')!r}"
                )
        if known_gaps:
            warnings.append(f"role {role!r} has {len(known_gaps)} known gap(s)")

    global_gaps = priors.get("known_global_gaps")
    if isinstance(global_gaps, dict):
        for role, gaps in sorted(global_gaps.items()):
            if gaps:
                warnings.append(f"known_global_gaps.{role}: {len(gaps)} gap(s)")
    elif global_gaps:
        errors.append("known_global_gaps must be a mapping when present")

    if strict and warnings:
        errors.extend(f"strict: {warning}" for warning in warnings)
        warnings = []

    return GuardResult(errors=errors, warnings=warnings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate generated stack priors")
    parser.add_argument("--priors", type=Path, default=DEFAULT_PRIORS)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on any known gaps, not only stale hashes or live-role invariants",
    )
    args = parser.parse_args(argv)

    result = validate_stack_priors(args.priors, strict=args.strict)
    if result.errors:
        print(f"FAIL: {len(result.errors)} stack-prior error(s)")
        for error in result.errors:
            print(f"  - {error}")
        return 1
    if result.warnings:
        print(f"WARN: {len(result.warnings)} stack-prior warning(s)")
        for warning in result.warnings:
            print(f"  - {warning}")
        return 0
    print(f"OK: {args.priors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
