#!/usr/bin/env python3
"""Canonical no-inference stack-change check/update pipeline."""

from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

REPO_ROOT = Path("/mnt/raid0/llm/epyc-orchestrator")
sys.path.insert(0, str(REPO_ROOT))

from scripts.registry.sync_procedure_role_enums import (  # noqa: E402
    DEFAULT_PROCEDURE,
    DEFAULT_SCHEMA,
    ProcedureRoleSyncError,
    sync_procedure_role_enums,
)
from scripts.validate.stack_change_guard import (  # noqa: E402
    DEFAULT_SURFACE_EXCEPTIONS,
    validate_stack_priors,
)
from src.registry.model_descriptors import (  # noqa: E402
    DEFAULT_DESCRIPTOR_OUTPUT,
    DEFAULT_LEAN_REGISTRY,
    DEFAULT_RESEARCH_REGISTRY,
    compile_model_descriptors,
)
from src.registry.stack_priors import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_STACK_PRIORS,
    compile_stack_priors,
    write_stack_priors,
)

Mode = Literal["check", "update"]


@dataclass(frozen=True)
class StackChangePipelineConfig:
    mode: Mode
    repo_root: Path = REPO_ROOT
    lean_registry: Path = DEFAULT_LEAN_REGISTRY
    research_registry: Path | None = DEFAULT_RESEARCH_REGISTRY
    descriptors: Path = DEFAULT_DESCRIPTOR_OUTPUT
    stack_priors: Path = DEFAULT_STACK_PRIORS
    procedure: Path = DEFAULT_PROCEDURE
    schema: Path = DEFAULT_SCHEMA
    surface_exceptions: Path = DEFAULT_SURFACE_EXCEPTIONS
    roles: set[str] | None = None
    allow_known_gaps: bool = False
    compile_incomplete: bool = True
    allow_descriptor_model_removal: bool = False


@dataclass
class PipelineStep:
    name: str
    status: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass
class PipelineReport:
    steps: list[PipelineStep] = field(default_factory=list)

    @property
    def errors(self) -> list[str]:
        return [error for step in self.steps for error in step.errors]

    @property
    def warnings(self) -> list[str]:
        return [warning for step in self.steps for warning in step.warnings]

    @property
    def ok(self) -> bool:
        return not self.errors


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return loaded


def _dump_yaml(data: dict[str, Any]) -> str:
    return yaml.safe_dump(
        data,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
        width=200,
    )


def _normalise_generated(data: dict[str, Any]) -> dict[str, Any]:
    """Remove volatile metadata before semantic generated-artifact comparison."""
    normalised = copy.deepcopy(data)
    normalised.pop("compiled_at", None)
    for section_name in ("source_artifacts", "source_registries"):
        section = normalised.get(section_name)
        if not isinstance(section, dict):
            continue
        for source in section.values():
            if isinstance(source, dict):
                source.pop("repo_commit", None)
    return normalised


def _generated_matches(path: Path, expected: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    actual = _load_yaml(path)
    return _normalise_generated(actual) == _normalise_generated(expected)


def _descriptor_model_ids(descriptors: dict[str, Any]) -> set[str]:
    models = descriptors.get("models")
    if not isinstance(models, list):
        return set()
    return {
        str(model["model_id"])
        for model in models
        if isinstance(model, dict) and isinstance(model.get("model_id"), str)
    }


def _descriptor_removal_errors(path: Path, generated: dict[str, Any]) -> list[str]:
    if not path.exists():
        return []
    current_ids = _descriptor_model_ids(_load_yaml(path))
    generated_ids = _descriptor_model_ids(generated)
    removed = sorted(current_ids - generated_ids)
    if not removed:
        return []
    return [
        "descriptor update would remove existing model_id(s): " + ", ".join(removed),
        "rerun with --allow-descriptor-model-removal only after an explicit coverage decision",
    ]


def _roles_from_stack_manifest() -> set[str]:
    from scripts.server.stack_manifest import ROLE_LAUNCH_META

    roles = set(ROLE_LAUNCH_META.keys())
    for meta in ROLE_LAUNCH_META.values():
        aliases = meta.get("shared_with_first_n") if isinstance(meta, dict) else None
        if isinstance(aliases, list):
            roles.update(str(alias) for alias in aliases)
    return roles


def _active_roles(config: StackChangePipelineConfig) -> set[str]:
    return set(config.roles) if config.roles is not None else _roles_from_stack_manifest()


def _stack_prior_roles(config: StackChangePipelineConfig) -> set[str] | None:
    return set(config.roles) if config.roles is not None else None


def _check_descriptors(config: StackChangePipelineConfig) -> PipelineStep:
    try:
        expected = compile_model_descriptors(
            lean_registry_path=config.lean_registry,
            research_registry_path=config.research_registry,
            active_roles=_active_roles(config),
            allow_incomplete=config.compile_incomplete,
        )
    except Exception as exc:  # noqa: BLE001
        return PipelineStep(
            name="descriptors",
            status="failed",
            errors=[f"descriptor compile failed: {exc}"],
        )

    if _generated_matches(config.descriptors, expected):
        return PipelineStep(
            name="descriptors",
            status="ok",
            details=[f"fresh: {config.descriptors}"],
        )
    removal_errors = _descriptor_removal_errors(config.descriptors, expected)
    if removal_errors:
        errors = [
            f"descriptor artifact is stale: {config.descriptors}",
            *removal_errors,
        ]
    else:
        errors = [
            f"descriptor artifact is stale or missing: {config.descriptors}",
            "run: uv run python scripts/registry/stack_change_pipeline.py update",
        ]
    return PipelineStep(
        name="descriptors",
        status="stale",
        errors=errors,
    )


def _update_descriptors(config: StackChangePipelineConfig) -> PipelineStep:
    try:
        descriptors = compile_model_descriptors(
            lean_registry_path=config.lean_registry,
            research_registry_path=config.research_registry,
            active_roles=_active_roles(config),
            allow_incomplete=config.compile_incomplete,
        )
    except Exception as exc:  # noqa: BLE001
        return PipelineStep(
            name="descriptors",
            status="failed",
            errors=[f"descriptor update failed: {exc}"],
        )
    if not config.allow_descriptor_model_removal:
        removal_errors = _descriptor_removal_errors(config.descriptors, descriptors)
        if removal_errors:
            return PipelineStep(
                name="descriptors",
                status="failed",
                errors=removal_errors,
            )
    config.descriptors.parent.mkdir(parents=True, exist_ok=True)
    config.descriptors.write_text(_dump_yaml(descriptors), encoding="utf-8")
    return PipelineStep(
        name="descriptors",
        status="updated",
        details=[
            f"wrote {len(descriptors.get('models', []))} descriptor(s): {config.descriptors}"
        ],
    )


def _check_stack_priors(config: StackChangePipelineConfig) -> PipelineStep:
    try:
        expected = compile_stack_priors(
            registry_path=config.lean_registry,
            descriptor_path=config.descriptors,
            active_roles=_stack_prior_roles(config),
            allow_incomplete=config.compile_incomplete,
        )
    except Exception as exc:  # noqa: BLE001
        return PipelineStep(
            name="stack_priors",
            status="failed",
            errors=[f"stack-prior compile failed: {exc}"],
        )

    if _generated_matches(config.stack_priors, expected):
        return PipelineStep(
            name="stack_priors",
            status="ok",
            details=[f"fresh: {config.stack_priors}"],
        )
    return PipelineStep(
        name="stack_priors",
        status="stale",
        errors=[
            f"stack-prior artifact is stale or missing: {config.stack_priors}",
            "run: uv run python scripts/registry/stack_change_pipeline.py update",
        ],
    )


def _update_stack_priors(config: StackChangePipelineConfig) -> PipelineStep:
    try:
        priors = write_stack_priors(
            config.stack_priors,
            registry_path=config.lean_registry,
            descriptor_path=config.descriptors,
            active_roles=_stack_prior_roles(config),
            allow_incomplete=config.compile_incomplete,
        )
    except Exception as exc:  # noqa: BLE001
        return PipelineStep(
            name="stack_priors",
            status="failed",
            errors=[f"stack-prior update failed: {exc}"],
        )
    return PipelineStep(
        name="stack_priors",
        status="updated",
        details=[f"wrote {len(priors.get('roles', {}))} role prior(s): {config.stack_priors}"],
    )


def _procedure_enums(config: StackChangePipelineConfig, *, check: bool) -> PipelineStep:
    try:
        ok = sync_procedure_role_enums(
            priors_path=config.stack_priors,
            procedure_path=config.procedure,
            schema_path=config.schema,
            check=check,
        )
    except ProcedureRoleSyncError as exc:
        return PipelineStep(
            name="procedure_enums",
            status="failed",
            errors=[f"procedure role enum sync failed: {exc}"],
        )
    if ok:
        return PipelineStep(
            name="procedure_enums",
            status="ok" if check else "updated",
            details=[f"{'checked' if check else 'synced'}: {config.procedure}, {config.schema}"],
        )
    return PipelineStep(
        name="procedure_enums",
        status="stale",
        errors=[
            "procedure role enums are stale",
            "run: uv run python scripts/registry/stack_change_pipeline.py update",
        ],
    )


def _guard_step(
    name: str,
    config: StackChangePipelineConfig,
    *,
    strict: bool,
    all_surfaces: bool,
    allow_known_gaps: bool = False,
) -> PipelineStep:
    result = validate_stack_priors(
        config.stack_priors,
        strict=strict,
        scan_surfaces=True,
        repo_root=config.repo_root,
        surface_categories=None if all_surfaces else frozenset({"production_blocker"}),
        surface_exceptions_path=config.surface_exceptions,
        procedure_path=config.procedure,
        procedure_schema_path=config.schema,
    )
    errors = list(result.errors)
    warnings = list(result.warnings)
    status = "ok"
    if errors and strict and allow_known_gaps:
        warnings.extend(f"known-gap: {error}" for error in errors)
        errors = []
        status = "known_gaps"
    elif errors:
        status = "failed"
    elif warnings:
        status = "warnings"
    return PipelineStep(name=name, status=status, errors=errors, warnings=warnings)


def run_stack_change_pipeline(config: StackChangePipelineConfig) -> PipelineReport:
    report = PipelineReport()
    if config.mode == "check":
        report.steps.append(_check_descriptors(config))
        report.steps.append(_check_stack_priors(config))
        report.steps.append(_procedure_enums(config, check=True))
    else:
        descriptor_step = _update_descriptors(config)
        report.steps.append(descriptor_step)
        if descriptor_step.ok:
            report.steps.append(_update_stack_priors(config))
            report.steps.append(_procedure_enums(config, check=False))
        else:
            report.steps.append(
                PipelineStep(
                    name="stack_priors",
                    status="skipped",
                    warnings=["skipped because descriptor update failed"],
                )
            )
            report.steps.append(
                PipelineStep(
                    name="procedure_enums",
                    status="skipped",
                    warnings=["skipped because descriptor update failed"],
                )
            )

    report.steps.append(
        _guard_step("guard", config, strict=False, all_surfaces=False)
    )
    report.steps.append(
        _guard_step("guard_all_surfaces", config, strict=False, all_surfaces=True)
    )
    report.steps.append(
        _guard_step(
            "guard_strict",
            config,
            strict=True,
            all_surfaces=False,
            allow_known_gaps=config.allow_known_gaps,
        )
    )
    return report


def _print_report(report: PipelineReport) -> None:
    for step in report.steps:
        print(f"{step.name}: {step.status}")
        for detail in step.details:
            print(f"  detail: {detail}")
        for warning in step.warnings:
            print(f"  warn: {warning}")
        for error in step.errors:
            print(f"  error: {error}")
    print(f"summary: {'ok' if report.ok else 'failed'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the no-inference stack-change check/update pipeline"
    )
    parser.add_argument("mode", choices=("check", "update"))
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--lean-registry", type=Path, default=DEFAULT_LEAN_REGISTRY)
    parser.add_argument("--research-registry", type=Path, default=DEFAULT_RESEARCH_REGISTRY)
    parser.add_argument("--descriptors", type=Path, default=DEFAULT_DESCRIPTOR_OUTPUT)
    parser.add_argument("--stack-priors", type=Path, default=DEFAULT_STACK_PRIORS)
    parser.add_argument("--procedure", type=Path, default=DEFAULT_PROCEDURE)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--surface-exceptions", type=Path, default=DEFAULT_SURFACE_EXCEPTIONS)
    parser.add_argument("--roles", nargs="+", help="Explicit active roles")
    parser.add_argument(
        "--strict-descriptor-compile",
        action="store_true",
        help="Refuse incomplete descriptor/stack-prior records instead of reporting known gaps",
    )
    parser.add_argument(
        "--allow-known-gaps",
        action="store_true",
        help="Report strict known-gap failures as warnings while the current gap-closure work is active",
    )
    parser.add_argument(
        "--allow-descriptor-model-removal",
        action="store_true",
        help="Permit update mode to remove model IDs from the descriptor artifact",
    )
    args = parser.parse_args(argv)

    config = StackChangePipelineConfig(
        mode=args.mode,
        repo_root=args.repo_root,
        lean_registry=args.lean_registry,
        research_registry=args.research_registry,
        descriptors=args.descriptors,
        stack_priors=args.stack_priors,
        procedure=args.procedure,
        schema=args.schema,
        surface_exceptions=args.surface_exceptions,
        roles=set(args.roles) if args.roles else None,
        allow_known_gaps=args.allow_known_gaps,
        compile_incomplete=not args.strict_descriptor_compile,
        allow_descriptor_model_removal=args.allow_descriptor_model_removal,
    )
    report = run_stack_change_pipeline(config)
    _print_report(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
