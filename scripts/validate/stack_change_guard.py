#!/usr/bin/env python3
"""Validate generated stack priors against current source artifacts."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from src.registry.stack_priors import validate_stack_priors_contract


REPO_ROOT = Path("/mnt/raid0/llm/epyc-orchestrator")
DEFAULT_PRIORS = REPO_ROOT / "orchestration" / "derived" / "stack_priors.yaml"
DEFAULT_SURFACE_EXCEPTIONS = REPO_ROOT / "orchestration" / "stack_change_guard_exceptions.yaml"
DEFAULT_ADD_MODEL_PROCEDURE = REPO_ROOT / "orchestration" / "procedures" / "add_model_to_registry.yaml"
DEFAULT_PROCEDURE_SCHEMA = REPO_ROOT / "orchestration" / "procedure.schema.json"
RETIRED_LIVE_ROLES = {"architect_coding"}
REQUIRED_SOURCE_ARTIFACTS = ("registry", "descriptors", "stack_manifest", "stack_numa")
SURFACE_SCAN_ALLOW_MARKER = "stack-change-guard: allow"
SURFACE_SCAN_MAX_FILE_BYTES = 512 * 1024
SURFACE_EXCEPTION_CLASSIFICATIONS = frozenset(
    {
        "degraded_fallback",
        "legacy_test",
        "historical_doc",
        "intentional_live_exception",
    }
)


@dataclass(frozen=True)
class GuardResult:
    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class HardcodedSurfaceRule:
    rule_id: str
    category: str
    pattern: str
    path_globs: tuple[str, ...]
    remediation: str
    exclude_globs: tuple[str, ...] = ()
    ignore_comment_lines: bool = False


@dataclass(frozen=True)
class SurfaceFinding:
    rule_id: str
    category: str
    path: Path
    line: int
    snippet: str
    remediation: str

    def to_warning(self) -> str:
        return (
            f"hardcoded_surface.{self.category}.{self.rule_id}: "
            f"{self.path}:{self.line}: {self.snippet} "
            f"[remediation: {self.remediation}]"
        )


@dataclass(frozen=True)
class SurfaceException:
    rule_id: str
    category: str
    path_glob: str
    classification: str
    owner: str
    rationale: str
    expires: str
    line: int | None = None

    def matches(self, finding: SurfaceFinding) -> bool:
        if self.rule_id != finding.rule_id or self.category != finding.category:
            return False
        if self.line is not None and self.line != finding.line:
            return False
        return fnmatch.fnmatch(finding.path.as_posix(), self.path_glob)

    def warning_suffix(self) -> str:
        return (
            f"classification={self.classification}; owner={self.owner}; "
            f"expires={self.expires}; rationale={self.rationale}"
        )


HARDCODED_SURFACE_RULES: tuple[HardcodedSurfaceRule, ...] = (
    HardcodedSurfaceRule(
        rule_id="retired_role_in_active_code",
        category="production_blocker",
        pattern=r"\barchitect_coding\b",
        path_globs=(
            "src/**/*.py",
            "scripts/benchmark/*.py",
            "scripts/server/*.py",
        ),
        exclude_globs=(
            "scripts/validate/stack_change_guard.py",
            "scripts/benchmark/deprecated/**",
        ),
        remediation="remove from live behavior or mark explicit legacy/test-only",
        ignore_comment_lines=True,
    ),
    HardcodedSurfaceRule(
        rule_id="retired_role_env_flag",
        category="production_blocker",
        pattern=r"\b(?:ORCHESTRATOR(?:_FEATURE)?_)?LANGGRAPH_ARCHITECT_CODING\b",
        path_globs=("scripts/server/*.py",),
        remediation="do not enable retired LangGraph architect_coding launch flags",
        ignore_comment_lines=True,
    ),
    HardcodedSurfaceRule(
        rule_id="stale_procedure_role_enum",
        category="production_blocker",
        pattern=r"\barchitect_coding\b",
        path_globs=("orchestration/procedures/*.yaml",),
        remediation="compile procedure role choices from stack priors",
    ),
    HardcodedSurfaceRule(
        rule_id="retired_role_in_tests",
        category="legacy_test",
        pattern=r"\barchitect_coding\b",
        path_globs=("tests/**/*.py",),
        exclude_globs=("tests/unit/test_stack_change_guard.py",),
        remediation="label as retired-role coverage or migrate fixture to stack priors",
    ),
    HardcodedSurfaceRule(
        rule_id="retired_role_in_operator_docs",
        category="historical_doc",
        pattern=r"\barchitect_coding\b",
        path_globs=("docs/**/*.md",),
        remediation="generate current stack tables or label snapshot as historical",
    ),
    HardcodedSurfaceRule(
        rule_id="bilinear_model_specs_table",
        category="production_blocker",
        pattern=r"\bmodel_specs\b|\barchitect_coding\b",
        path_globs=("orchestration/repl_memory/bilinear_scorer.py",),
        remediation="derive model features from stack priors/descriptors",
    ),
    HardcodedSurfaceRule(
        rule_id="seeding_baseline_tps_table",
        category="production_blocker",
        pattern=r"\bDEFAULT_BASELINE_TPS\b|\bbaseline_tps\b|\barchitect_coding\b",
        path_globs=("scripts/benchmark/seeding_rewards.py",),
        remediation="derive seeding reward costs from stack priors",
    ),
    HardcodedSurfaceRule(
        rule_id="legacy_cli_port_probe_map",
        category="production_blocker",
        pattern=r"\b8084\b|\barchitect_coding\b|ports\s*=\s*\[8080",
        path_globs=("src/cli_orch.py",),
        remediation="derive status probes from stack priors or stack manifest API",
    ),
    HardcodedSurfaceRule(
        rule_id="stale_autopilot_program_stack_guidance",
        category="production_blocker",
        pattern=r"\b(?:8071|8084|architect_coding|512GB)\b|\bTarget ports\b|\bWARM tier demotion\b",
        path_globs=("scripts/autopilot/program.md",),
        remediation="derive AutoPilot operator endpoints and tier guidance from stack priors/system card",
    ),
)


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


def _port_from_endpoint(endpoint: Any) -> int | None:
    if not isinstance(endpoint, str):
        return None
    match = re.search(r":(\d+)(?:/|$)", endpoint)
    if not match:
        return None
    return int(match.group(1))


def _launch_manifest_targets() -> dict[str, dict[str, Any]]:
    """Return live launch ports/tier per role from the computed manifest."""
    try:
        from scripts.server.stack_manifest import HOT_SERVERS, WARM_SERVERS
    except Exception:
        return {}

    targets: dict[str, dict[str, Any]] = {}
    for tier, servers in (("hot", HOT_SERVERS), ("warm", WARM_SERVERS)):
        for server in servers:
            if not isinstance(server, dict):
                continue
            port = server.get("port")
            if not isinstance(port, int):
                continue
            for role in server.get("roles") or []:
                if isinstance(role, str):
                    target = targets.setdefault(role, {"port": port, "ports": [], "tier": tier})
                    target["ports"].append(port)
    return targets


def validate_launch_manifest_serving_alignment(
    priors: dict[str, Any],
    *,
    launch_manifest_targets: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    """Validate generated live serving records against current launch roles."""
    targets = _launch_manifest_targets() if launch_manifest_targets is None else launch_manifest_targets
    if not targets:
        return []

    roles = priors.get("roles")
    if not isinstance(roles, dict):
        return []

    errors: list[str] = []
    for role, record in sorted(roles.items()):
        if not isinstance(role, str) or not isinstance(record, dict):
            continue
        if record.get("deployment_status") != "live_stack":
            continue
        target = targets.get(role)
        if target is None:
            errors.append(f"live role {role!r} is absent from current launch manifest")
            continue
        serving = record.get("serving")
        if not isinstance(serving, dict):
            continue
        target_port = target.get("port")
        target_tier = target.get("tier")
        raw_target_ports = target.get("ports")
        target_ports = (
            {port for port in raw_target_ports if isinstance(port, int)}
            if isinstance(raw_target_ports, list)
            else set()
        )
        endpoint_port = _port_from_endpoint(serving.get("endpoint"))
        ports = serving.get("ports")
        port_set = {port for port in ports if isinstance(port, int)} if isinstance(ports, list) else set()
        if isinstance(target_port, int):
            if endpoint_port != target_port:
                errors.append(
                    f"role {role!r} serving.endpoint port {endpoint_port!r} "
                    f"does not match launch manifest port {target_port}"
                )
            if target_port not in port_set:
                errors.append(
                    f"role {role!r} serving.ports {sorted(port_set)} "
                    f"does not include launch manifest port {target_port}"
                )
        if target_ports:
            missing_ports = sorted(target_ports - port_set)
            extra_ports = sorted(port_set - target_ports)
            if missing_ports:
                errors.append(
                    f"role {role!r} serving.ports {sorted(port_set)} "
                    f"missing launch manifest port(s) {missing_ports}"
                )
            if extra_ports:
                errors.append(
                    f"role {role!r} serving.ports {sorted(port_set)} "
                    f"include non-launch port(s) {extra_ports}"
                )
        if isinstance(target_tier, str) and serving.get("tier") != target_tier:
            errors.append(
                f"role {role!r} serving.tier {serving.get('tier')!r} "
                f"does not match launch manifest tier {target_tier!r}"
            )
    return errors


def _matches_any(path: Path, patterns: tuple[str, ...]) -> bool:
    rel = path.as_posix()
    return any(fnmatch.fnmatch(rel, pattern) for pattern in patterns)


def _display_path(path: Path, repo_root: Path) -> Path:
    try:
        return path.relative_to(repo_root)
    except ValueError:
        return path


def _candidate_paths(repo_root: Path, rule: HardcodedSurfaceRule) -> list[Path]:
    paths: dict[str, Path] = {}
    for pattern in rule.path_globs:
        for path in repo_root.glob(pattern):
            if not path.is_file():
                continue
            rel_path = path.relative_to(repo_root)
            if _matches_any(rel_path, rule.exclude_globs):
                continue
            paths[rel_path.as_posix()] = path
    return [paths[key] for key in sorted(paths)]


def scan_hardcoded_surfaces(
    repo_root: Path = REPO_ROOT,
    *,
    rules: tuple[HardcodedSurfaceRule, ...] = HARDCODED_SURFACE_RULES,
    categories: frozenset[str] | None = None,
) -> list[SurfaceFinding]:
    """Find curated hardcoded model/stack surfaces that can drift.

    This is intentionally narrower than a repository-wide grep. The goal is to
    turn known risky model-specific surfaces into a validator signal without
    treating historical artifacts, benchmark outputs, or generated backups as
    live stack truth.
    """
    findings: list[SurfaceFinding] = []
    for rule in rules:
        if categories is not None and rule.category not in categories:
            continue
        compiled = re.compile(rule.pattern)
        for path in _candidate_paths(repo_root, rule):
            try:
                if path.stat().st_size > SURFACE_SCAN_MAX_FILE_BYTES:
                    continue
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError):
                continue
            rel_path = path.relative_to(repo_root)
            for line_no, line in enumerate(lines, start=1):
                stripped = line.strip()
                if SURFACE_SCAN_ALLOW_MARKER in line:
                    continue
                if rule.ignore_comment_lines and stripped.startswith("#"):
                    continue
                if not compiled.search(line):
                    continue
                findings.append(
                    SurfaceFinding(
                        rule_id=rule.rule_id,
                        category=rule.category,
                        path=rel_path,
                        line=line_no,
                        snippet=line.strip()[:160],
                        remediation=rule.remediation,
                    )
                )
    return findings


def _surface_exception_from_raw(index: int, raw: Any) -> tuple[SurfaceException | None, list[str]]:
    errors: list[str] = []
    prefix = f"surface exception #{index}"
    if not isinstance(raw, dict):
        return None, [f"{prefix} is not a mapping"]

    def required_str(field: str) -> str:
        value = raw.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{prefix} missing non-empty {field!r}")
            return ""
        return value.strip()

    path_glob = raw.get("path_glob", raw.get("path"))
    if not isinstance(path_glob, str) or not path_glob.strip():
        errors.append(f"{prefix} missing non-empty 'path_glob' or 'path'")
        path_glob = ""
    else:
        path_glob = path_glob.strip()

    line_raw = raw.get("line")
    line: int | None = None
    if line_raw is not None:
        if not isinstance(line_raw, int) or line_raw <= 0:
            errors.append(f"{prefix} line must be a positive integer when present")
        else:
            line = line_raw

    classification = required_str("classification")
    if classification and classification not in SURFACE_EXCEPTION_CLASSIFICATIONS:
        allowed = ", ".join(sorted(SURFACE_EXCEPTION_CLASSIFICATIONS))
        errors.append(f"{prefix} classification {classification!r} is not one of: {allowed}")

    expires = required_str("expires")
    if expires:
        try:
            expires_date = date.fromisoformat(expires)
        except ValueError:
            errors.append(f"{prefix} expires must be an ISO date YYYY-MM-DD")
        else:
            if expires_date < date.today():
                errors.append(f"{prefix} expired on {expires}")

    exception = SurfaceException(
        rule_id=required_str("rule_id"),
        category=required_str("category"),
        path_glob=path_glob,
        classification=classification,
        owner=required_str("owner"),
        rationale=required_str("rationale"),
        expires=expires,
        line=line,
    )
    return (None, errors) if errors else (exception, [])


def load_surface_exceptions(path: Path = DEFAULT_SURFACE_EXCEPTIONS) -> tuple[list[SurfaceException], list[str]]:
    """Load documented hardcoded-surface exceptions.

    Exceptions are not silent suppressions. Matching findings remain warnings,
    but strict mode does not promote them to errors while the metadata is valid.
    """
    if not path.exists():
        return [], []
    try:
        loaded = _load_yaml(path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return [], [f"failed to load surface exception file {path}: {exc}"]
    raw_exceptions = loaded.get("exceptions", [])
    if raw_exceptions is None:
        return [], []
    if not isinstance(raw_exceptions, list):
        return [], [f"surface exception file {path} has non-list 'exceptions'"]

    exceptions: list[SurfaceException] = []
    errors: list[str] = []
    for index, raw in enumerate(raw_exceptions, start=1):
        exception, entry_errors = _surface_exception_from_raw(index, raw)
        errors.extend(entry_errors)
        if exception is not None:
            exceptions.append(exception)
    return exceptions, errors


def _matching_surface_exception(
    finding: SurfaceFinding,
    exceptions: list[SurfaceException],
) -> SurfaceException | None:
    for exception in exceptions:
        if exception.matches(finding):
            return exception
    return None


def _surface_warning_for_finding(
    finding: SurfaceFinding,
    exceptions: list[SurfaceException],
) -> str:
    exception = _matching_surface_exception(finding, exceptions)
    if exception is None:
        return finding.to_warning()
    return (
        f"hardcoded_surface.waived.{finding.category}.{finding.rule_id}: "
        f"{finding.path}:{finding.line}: {finding.snippet} "
        f"[exception: {exception.warning_suffix()}]"
    )


def stack_prior_role_choices(priors: dict[str, Any]) -> list[str]:
    """Return model role choices that procedure inputs should accept."""
    roles = priors.get("roles")
    if not isinstance(roles, dict):
        return []

    choices: list[str] = []
    for role, record in roles.items():
        if not isinstance(role, str) or not isinstance(record, dict):
            continue
        if role in RETIRED_LIVE_ROLES:
            continue
        if record.get("deployment_status") == "retired":
            continue
        choices.append(role)
    return sorted(choices)


def stack_prior_permission_role_choices(priors: dict[str, Any]) -> list[str]:
    """Return live executor roles accepted by the procedure schema."""
    roles = priors.get("roles")
    if not isinstance(roles, dict):
        return []

    choices: list[str] = []
    for role, record in roles.items():
        if not isinstance(role, str) or not isinstance(record, dict):
            continue
        if role in RETIRED_LIVE_ROLES:
            continue
        if record.get("deployment_status") == "live_stack":
            choices.append(role)
    return sorted(choices) + ["admin"]


def _procedure_input_enum(procedure_path: Path, input_name: str) -> list[str] | None:
    procedure = _load_yaml(procedure_path)
    inputs = procedure.get("inputs")
    if not isinstance(inputs, list):
        return None
    for raw_input in inputs:
        if not isinstance(raw_input, dict) or raw_input.get("name") != input_name:
            continue
        validation = raw_input.get("validation")
        if not isinstance(validation, dict):
            return None
        enum = validation.get("enum")
        if not isinstance(enum, list):
            return None
        return [str(item) for item in enum if isinstance(item, str)]
    return None


def _procedure_schema_permission_enum(schema_path: Path) -> list[str] | None:
    with schema_path.open("r", encoding="utf-8") as fh:
        schema = json.load(fh)
    try:
        enum = schema["properties"]["permissions"]["properties"]["roles"]["items"]["enum"]
    except (KeyError, TypeError):
        return None
    if not isinstance(enum, list):
        return None
    return [str(item) for item in enum if isinstance(item, str)]


def validate_procedure_role_enums(
    priors: dict[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    procedure_path: Path | None = None,
    schema_path: Path | None = None,
) -> list[str]:
    """Validate generated procedure role enums against stack priors."""
    errors: list[str] = []
    raw_procedure_path = procedure_path
    if raw_procedure_path is None:
        resolved_procedure_path = repo_root / DEFAULT_ADD_MODEL_PROCEDURE.relative_to(REPO_ROOT)
    else:
        resolved_procedure_path = (
            raw_procedure_path
            if raw_procedure_path.is_absolute()
            else repo_root / raw_procedure_path
        )
    if resolved_procedure_path.exists():
        expected = stack_prior_role_choices(priors)
        actual = _procedure_input_enum(resolved_procedure_path, "role")
        if actual is None:
            rel_path = _display_path(resolved_procedure_path, repo_root)
            errors.append(f"procedure role enum missing: {rel_path} input 'role'")
        elif actual != expected:
            rel_path = _display_path(resolved_procedure_path, repo_root)
            errors.append(
                f"procedure role enum drift: {rel_path} input 'role' expected {expected} "
                f"from stack priors, got {actual} "
                "[run: scripts/registry/sync_procedure_role_enums.py]"
            )

    raw_schema_path = schema_path
    if raw_schema_path is None:
        resolved_schema_path = repo_root / DEFAULT_PROCEDURE_SCHEMA.relative_to(REPO_ROOT)
    else:
        resolved_schema_path = (
            raw_schema_path if raw_schema_path.is_absolute() else repo_root / raw_schema_path
        )
    if resolved_schema_path.exists():
        expected_permissions = stack_prior_permission_role_choices(priors)
        actual_permissions = _procedure_schema_permission_enum(resolved_schema_path)
        if actual_permissions is None:
            rel_path = _display_path(resolved_schema_path, repo_root)
            errors.append(f"procedure schema permission enum missing: {rel_path}")
        elif actual_permissions != expected_permissions:
            rel_path = _display_path(resolved_schema_path, repo_root)
            errors.append(
                f"procedure schema permission enum drift: {rel_path} expected "
                f"{expected_permissions} from live stack priors plus admin, "
                f"got {actual_permissions} "
                "[run: scripts/registry/sync_procedure_role_enums.py]"
            )
    return errors


def validate_stack_priors(
    priors_path: Path = DEFAULT_PRIORS,
    *,
    strict: bool = False,
    scan_surfaces: bool = False,
    repo_root: Path = REPO_ROOT,
    surface_categories: frozenset[str] | None = frozenset({"production_blocker"}),
    surface_exceptions_path: Path | None = DEFAULT_SURFACE_EXCEPTIONS,
    procedure_path: Path | None = None,
    procedure_schema_path: Path | None = None,
    launch_manifest_targets: dict[str, dict[str, Any]] | None = None,
) -> GuardResult:
    errors: list[str] = []
    warnings: list[str] = []
    if not priors_path.exists():
        return GuardResult(errors=[f"missing stack priors artifact: {priors_path}"], warnings=[])

    priors = _load_yaml(priors_path)
    errors.extend(validate_stack_priors_contract(priors))
    errors.extend(
        validate_launch_manifest_serving_alignment(
            priors,
            launch_manifest_targets=launch_manifest_targets,
        )
    )
    roles = priors.get("roles")
    if not isinstance(roles, dict):
        errors.append("stack priors artifact has no mapping-valued roles section")
        roles = {}

    sources = priors.get("source_artifacts") or {}
    if not isinstance(sources, dict):
        errors.append("stack priors artifact has no source_artifacts section")
        sources = {}

    for label in REQUIRED_SOURCE_ARTIFACTS:
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

    if scan_surfaces:
        errors.extend(
            validate_procedure_role_enums(
                priors,
                repo_root=repo_root,
                procedure_path=procedure_path,
                schema_path=procedure_schema_path,
            )
        )
        surface_exceptions: list[SurfaceException] = []
        if surface_exceptions_path is not None:
            surface_exceptions, exception_errors = load_surface_exceptions(surface_exceptions_path)
            errors.extend(exception_errors)
        for finding in scan_hardcoded_surfaces(repo_root, categories=surface_categories):
            warnings.append(_surface_warning_for_finding(finding, surface_exceptions))

    if strict and warnings:
        retained_warnings: list[str] = []
        for warning in warnings:
            if warning.startswith("hardcoded_surface.waived."):
                retained_warnings.append(warning)
            else:
                errors.append(f"strict: {warning}")
        warnings = retained_warnings

    return GuardResult(errors=errors, warnings=warnings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate generated stack priors")
    parser.add_argument("--priors", type=Path, default=DEFAULT_PRIORS)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on any known gaps, not only stale hashes or live-role invariants",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root used by the hardcoded-surface scanner",
    )
    parser.add_argument(
        "--skip-hardcoded-surface-scan",
        action="store_true",
        help="Skip curated hardcoded model/stack surface warnings",
    )
    parser.add_argument(
        "--all-hardcoded-surfaces",
        action="store_true",
        help="Report legacy-test and historical-doc surfaces in addition to production blockers",
    )
    parser.add_argument(
        "--hardcoded-surface-category",
        action="append",
        choices=sorted({rule.category for rule in HARDCODED_SURFACE_RULES}),
        help="Surface category to report; defaults to production_blocker",
    )
    parser.add_argument(
        "--surface-exceptions",
        type=Path,
        default=DEFAULT_SURFACE_EXCEPTIONS,
        help="YAML file documenting hardcoded-surface exceptions",
    )
    args = parser.parse_args(argv)
    if args.all_hardcoded_surfaces:
        surface_categories = None
    elif args.hardcoded_surface_category:
        surface_categories = frozenset(args.hardcoded_surface_category)
    else:
        surface_categories = frozenset({"production_blocker"})

    result = validate_stack_priors(
        args.priors,
        strict=args.strict,
        scan_surfaces=not args.skip_hardcoded_surface_scan,
        repo_root=args.repo_root,
        surface_categories=surface_categories,
        surface_exceptions_path=args.surface_exceptions,
    )
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
