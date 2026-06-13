#!/usr/bin/env python3
"""Validate generated stack priors against current source artifacts."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path("/mnt/raid0/llm/epyc-orchestrator")
DEFAULT_PRIORS = REPO_ROOT / "orchestration" / "derived" / "stack_priors.yaml"
RETIRED_LIVE_ROLES = {"architect_coding"}
SURFACE_SCAN_ALLOW_MARKER = "stack-change-guard: allow"
SURFACE_SCAN_MAX_FILE_BYTES = 512 * 1024


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


def _matches_any(path: Path, patterns: tuple[str, ...]) -> bool:
    rel = path.as_posix()
    return any(fnmatch.fnmatch(rel, pattern) for pattern in patterns)


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


def validate_stack_priors(
    priors_path: Path = DEFAULT_PRIORS,
    *,
    strict: bool = False,
    scan_surfaces: bool = False,
    repo_root: Path = REPO_ROOT,
    surface_categories: frozenset[str] | None = frozenset({"production_blocker"}),
) -> GuardResult:
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

    if scan_surfaces:
        for finding in scan_hardcoded_surfaces(repo_root, categories=surface_categories):
            warnings.append(finding.to_warning())

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
