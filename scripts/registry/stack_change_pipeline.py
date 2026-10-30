#!/usr/bin/env python3
"""Canonical no-inference stack-change check/update pipeline."""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.registry.sync_procedure_role_enums import (  # noqa: E402
    DEFAULT_PROCEDURE,
    DEFAULT_SCHEMA,
    ProcedureRoleSyncError,
    sync_procedure_role_enums,
)
from scripts.registry.render_stack_summary import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_OPERATOR_SUMMARY,
    render_current_stack_summary,
    write_current_stack_summary,
)
from scripts.validate.check_shared_with_derivations import (  # noqa: E402
    SourceError as SharedWithSourceError,
    check_all as check_shared_with_derivations,
    load_sources as load_shared_with_sources,
)
from scripts.validate.stack_change_guard import (  # noqa: E402
    DEFAULT_ACCEPTED_GAPS,
    DEFAULT_SURFACE_EXCEPTIONS,
    DEFAULT_SURFACE_MANIFEST,
    validate_stack_priors,
)
from scripts.validate.reasoning_effort_certifications import (  # noqa: E402
    DEFAULT_CERTIFICATIONS as DEFAULT_EFFORT_CERTIFICATIONS,
    validate_reasoning_effort_certifications,
)
from orchestration.repl_memory.q_scorer import (  # noqa: E402
    validate_live_q_scorer_prior_sources,
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
DESCRIPTOR_DIFF_LIMIT = 12
MODEL_FIELD_DIFF_LIMIT = 8
SIMULATED_FIXTURE_TARGET = "tests/unit/test_stack_change_pipeline_simulated_fixtures.py"
LAUNCH_PARITY_TARGET = "tests/unit/test_build_server_command_helpers.py"
BENCHMARK_PREFLIGHT_TARGETS = (
    "tests/unit/test_seeding_infra.py",
    "tests/unit/test_seeding_infra_additional.py",
    "tests/unit/test_seeding_infra_branching.py",
    "tests/unit/test_seed_specialist_routing_main_and_retry.py",
)
PROMOTION_GATE_TARGETS = (
    SIMULATED_FIXTURE_TARGET,
    LAUNCH_PARITY_TARGET,
    *BENCHMARK_PREFLIGHT_TARGETS,
)
SURFACE_INVENTORY_COMMAND = (
    "uv run python scripts/validate/stack_change_guard.py "
    "--list-hardcoded-surface-rules"
)
SURFACE_WARNING_ORDER = (
    "production_blocker",
    "waived_production_blocker",
    "legacy_test",
    "historical_doc",
)
ACCEPTED_GAP_WARNING_PREFIX = "accepted_gap."


DEFAULT_STACK_TOPOLOGY = REPO_ROOT / "orchestration" / "stack_topology.yaml"
# Mirrors scripts/server/stack_numa_mode.VALID_STACK_NUMA_MODES. Duplicated as a
# literal rather than imported because that module sits behind the stack_paths /
# stack_manifest circular-import chain that already fails at runtime here.
VALID_NUMA_MODES = frozenset({"full", "quarter", "both"})


AMBIENT_NUMA_MODE_ENV = "ORCHESTRATOR_STACK_NUMA_MODE"


def resolve_declared_numa_mode(
    topology_path: Path = DEFAULT_STACK_TOPOLOGY,
) -> tuple[str | None, str]:
    """Resolve the DECLARED NUMA mode and REPORT WHERE IT CAME FROM.

    Compilation must be a pure function of its declared inputs. Probing the live
    fleet to decide the lineup made the output depend on machine state, so a
    clean-shell compile and a fleet-up compile produced different artifacts from
    identical sources.

    Reads ONLY the topology declaration. The ambient
    ``ORCHESTRATOR_STACK_NUMA_MODE`` is deliberately NOT consulted here any more
    (NIB2-69, 2026-09-15): a value inherited from whatever shell ran the check made
    the verdict a function of that shell. A deliberate non-production lineup is
    requested with ``--numa-mode`` (``StackChangePipelineConfig.numa_mode``) — see
    ``resolve_pipeline_numa_mode``.

    Returns ``(mode, source)``. ``mode`` is None when nothing is declared.
    ``source`` is always a human-readable label — a value without a provenance
    label is how a wrong lineup goes unattributed.
    """
    try:
        declared = yaml.safe_load(topology_path.read_text()) or {}
        mode = declared.get("numa_mode")
        if isinstance(mode, str) and mode.strip() in VALID_NUMA_MODES:
            return mode.strip(), f"declared:{topology_path.name}"
        if mode is not None:
            return None, f"declared:{topology_path.name} INVALID value {mode!r}"
    except FileNotFoundError:
        return None, "no declared topology file"
    except Exception as exc:  # noqa: BLE001
        return None, f"declared topology unreadable: {exc}"
    return None, "no numa_mode declared"


@dataclass(frozen=True)
class NumaModeResolution:
    """The ONE NUMA mode a pipeline run evaluates, with provenance.

    Resolved once per run and threaded into BOTH the stack-prior compile and the
    guard's launch-manifest view, so the artifact under test and the view it is
    checked against always describe the same lineup.
    """

    mode: str | None
    source: str
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def resolve_pipeline_numa_mode(
    config: "StackChangePipelineConfig",
    *,
    environ: Mapping[str, str] | None = None,
) -> NumaModeResolution:
    """Explicit ``--numa-mode`` > declared topology. Ambient env is never used.

    A disagreeing ambient ``ORCHESTRATOR_STACK_NUMA_MODE`` is reported as a
    warning (so nobody believes their export took effect) but does not change
    the evaluated lineup. With nothing explicit and nothing declared, ``check``
    fails: it cannot say which lineup it evaluated, so it cannot pass. ``update``
    keeps the ESC-8 realized-fleet backstop (which refuses rather than defaulting
    to ``full``) and says so.
    """
    env = os.environ if environ is None else environ
    errors: list[str] = []
    warnings: list[str] = []
    if config.numa_mode is not None:
        explicit = str(config.numa_mode).strip().lower()
        if explicit not in VALID_NUMA_MODES:
            return NumaModeResolution(
                mode=None,
                source="explicit:--numa-mode",
                errors=(
                    f"invalid --numa-mode {config.numa_mode!r}; expected one of "
                    f"{sorted(VALID_NUMA_MODES)}",
                ),
            )
        mode: str | None = explicit
        source = "explicit:--numa-mode"
        declared, declared_source = resolve_declared_numa_mode(_topology_path(config))
        if declared is not None and declared != explicit:
            warnings.append(
                f"--numa-mode {explicit} overrides the declared production mode "
                f"{declared} ({declared_source}); this run does NOT evaluate production"
            )
    else:
        mode, source = resolve_declared_numa_mode(_topology_path(config))
        if mode is None:
            if config.mode == "check":
                errors.append(
                    f"no NUMA mode to evaluate ({source}): declare numa_mode in "
                    "orchestration/stack_topology.yaml or pass --numa-mode; check "
                    "refuses to inherit a lineup from the shell or the live fleet"
                )
            else:
                warnings.append(
                    f"no declared NUMA mode ({source}); update falls back to the "
                    "realized-fleet probe (refuses when nothing is listening)"
                )
    ambient = env.get(AMBIENT_NUMA_MODE_ENV)
    if ambient is not None and ambient.strip().lower() != (mode or ""):
        warnings.append(
            f"ambient {AMBIENT_NUMA_MODE_ENV}={ambient!r} IGNORED; this run evaluates "
            f"{mode or '<unresolved>'} ({source}). Pass --numa-mode to evaluate another lineup"
        )
    return NumaModeResolution(
        mode=mode, source=source, errors=tuple(errors), warnings=tuple(warnings)
    )


def _topology_path(config: "StackChangePipelineConfig") -> Path:
    """Topology declaration belonging to THIS config's repo_root."""
    if config.stack_topology is not None:
        return config.stack_topology
    return config.repo_root / "orchestration" / "stack_topology.yaml"


def _numa_mode_kwargs(resolution: NumaModeResolution) -> tuple[dict, str]:
    """Build the compile kwargs for the NUMA mode, plus the provenance label."""
    if resolution.mode is None:
        # Backstop only. Refusing beats defaulting to "full": ESC-8 kill chain A4
        # rewrote stack_priors.yaml to the dead full lineup that way.
        return {"require_realized_mode": True}, f"{resolution.source} -> realized-fleet probe"
    return {"numa_mode": resolution.mode}, resolution.source


def _numa_mode_step(resolution: NumaModeResolution) -> PipelineStep:
    return PipelineStep(
        name="numa_mode",
        status="failed" if resolution.errors else "ok",
        errors=list(resolution.errors),
        warnings=list(resolution.warnings),
        details=[
            f"evaluated numa_mode: {resolution.mode or '<unresolved>'} "
            f"(source: {resolution.source})"
        ],
    )


@dataclass(frozen=True)
class StackChangePipelineConfig:
    mode: Mode
    repo_root: Path = REPO_ROOT
    lean_registry: Path = DEFAULT_LEAN_REGISTRY
    # None => resolve under this config's repo_root, so a tmp_path fixture does not
    # silently inherit the production topology declaration.
    stack_topology: Path | None = None
    research_registry: Path | None = DEFAULT_RESEARCH_REGISTRY
    descriptors: Path = DEFAULT_DESCRIPTOR_OUTPUT
    stack_priors: Path = DEFAULT_STACK_PRIORS
    operator_summary: Path = DEFAULT_OPERATOR_SUMMARY
    procedure: Path = DEFAULT_PROCEDURE
    schema: Path = DEFAULT_SCHEMA
    surface_exceptions: Path = DEFAULT_SURFACE_EXCEPTIONS
    surface_manifest: Path = DEFAULT_SURFACE_MANIFEST
    accepted_gaps: Path = DEFAULT_ACCEPTED_GAPS
    effort_certifications: Path = DEFAULT_EFFORT_CERTIFICATIONS
    roles: set[str] | None = None
    allow_known_gaps: bool = False
    allow_production_blocker_waivers: bool = False
    compile_incomplete: bool = True
    allow_descriptor_model_removal: bool = False
    run_promotion_gate: bool = False
    # Explicit NUMA lineup to evaluate. None => the declared topology mode
    # (production). Never inherited from the ambient environment.
    numa_mode: str | None = None
    # Deliberate offline use: skip the research-registry-vs-origin/main check.
    allow_stale_research_registry: bool = False


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

    @property
    def unique_warnings(self) -> list[str]:
        return sorted(set(self.warnings))

    def hardcoded_surface_warning_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for warning in self.unique_warnings:
            bucket = _hardcoded_surface_warning_bucket(warning)
            if bucket is not None:
                counts[bucket] += 1
        return dict(counts)

    def accepted_gap_lines(self) -> list[str]:
        """Report every tolerated gap WITH its expiry.

        An acceptance nobody reads is a silenced check. These lines print on both
        the pass and the blocked path so an operator always sees what the gate is
        tolerating and until when.
        """
        declarations: dict[tuple[str, str], str] = {}
        for warning in self.unique_warnings:
            if not warning.startswith(ACCEPTED_GAP_WARNING_PREFIX):
                continue
            head, _, detail = warning.partition(": ")
            role = head.rsplit(".", 1)[-1]
            gap, _, suffix = detail.partition(" [declaration: ")
            expires = "unknown"
            for part in suffix.rstrip("]").split("; "):
                if part.startswith("expires="):
                    expires = part[len("expires="):]
                    break
            declarations[(role, gap.strip())] = expires
        if not declarations:
            return []
        lines = [f"accepted_gaps: {len(declarations)} declared and unexpired"]
        for (role, gap), expires in sorted(declarations.items()):
            lines.append(f"  accepted_gap: {role}: {gap} (expires {expires})")
        return lines

    def acceptance_lines(self) -> list[str]:
        if self.ok:
            lines = ["acceptance: no-inference checks passed"]
            if self.warnings:
                lines.append(
                    f"warnings: {len(self.unique_warnings)} unique "
                    f"({len(self.warnings)} total)"
                )
                surface_counts = self.hardcoded_surface_warning_counts()
                if surface_counts:
                    ordered_keys = [
                        key
                        for key in SURFACE_WARNING_ORDER
                        if key in surface_counts
                    ]
                    ordered_keys.extend(
                        sorted(set(surface_counts) - set(ordered_keys))
                    )
                    lines.append(
                        "surface_warnings: "
                        + ", ".join(
                            f"{key}={surface_counts[key]}" for key in ordered_keys
                        )
                    )
            lines.extend(self.accepted_gap_lines())
            lines.append(
                "promotion_gate: run uv run pytest -q "
                + " ".join(PROMOTION_GATE_TARGETS)
            )
            lines.append(f"surface_inventory: run {SURFACE_INVENTORY_COMMAND}")
            return lines
        return [
            "acceptance: blocked",
            *self.accepted_gap_lines(),
            f"promotion_gate: fix {len(self.errors)} error(s) before promotion",
        ]


def _hardcoded_surface_warning_bucket(warning: str) -> str | None:
    prefix = "hardcoded_surface."
    if not warning.startswith(prefix):
        return None
    suffix = warning[len(prefix):]
    parts = suffix.split(".", 3)
    if not parts:
        return None
    if parts[0] == "waived" and len(parts) >= 2:
        return f"waived_{parts[1]}"
    return parts[0]


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


def _descriptor_models_by_id(descriptors: dict[str, Any]) -> dict[str, dict[str, Any]]:
    models = descriptors.get("models")
    if not isinstance(models, list):
        return {}
    by_id: dict[str, dict[str, Any]] = {}
    for model in models:
        if not isinstance(model, dict) or not isinstance(model.get("model_id"), str):
            continue
        by_id[str(model["model_id"])] = model
    return by_id


def _format_limited(values: list[str], *, limit: int = DESCRIPTOR_DIFF_LIMIT) -> str:
    shown = values[:limit]
    suffix = f" (+{len(values) - limit} more)" if len(values) > limit else ""
    return ", ".join(shown) + suffix


def _diff_paths(current: Any, expected: Any, *, prefix: str = "") -> list[str]:
    if current == expected:
        return []
    if isinstance(current, dict) and isinstance(expected, dict):
        paths: list[str] = []
        for key in sorted(set(current) | set(expected), key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in current:
                paths.append(f"{path} added")
            elif key not in expected:
                paths.append(f"{path} removed")
            else:
                paths.extend(_diff_paths(current[key], expected[key], prefix=path))
        return paths
    if isinstance(current, list) and isinstance(expected, list):
        if len(current) != len(expected):
            return [f"{prefix} length {len(current)} -> {len(expected)}"]
        return [prefix or "<root>"]
    return [prefix or "<root>"]


def _descriptor_drift_details(path: Path, generated: dict[str, Any]) -> list[str]:
    if not path.exists():
        return []

    current = _normalise_generated(_load_yaml(path))
    expected = _normalise_generated(generated)
    current_models = _descriptor_models_by_id(current)
    expected_models = _descriptor_models_by_id(expected)
    current_ids = set(current_models)
    expected_ids = set(expected_models)

    details: list[str] = []
    added = sorted(expected_ids - current_ids)
    removed = sorted(current_ids - expected_ids)
    if added:
        details.append(f"descriptor generated adds model_id(s): {_format_limited(added)}")
    if removed:
        details.append(f"descriptor generated removes model_id(s): {_format_limited(removed)}")

    for model_id in sorted(current_ids & expected_ids):
        field_paths = _diff_paths(current_models[model_id], expected_models[model_id])
        if field_paths:
            details.append(
                f"descriptor changed {model_id}: "
                f"{_format_limited(field_paths, limit=MODEL_FIELD_DIFF_LIMIT)}"
            )
            if len(details) >= DESCRIPTOR_DIFF_LIMIT:
                break

    current_top = {key: value for key, value in current.items() if key != "models"}
    expected_top = {key: value for key, value in expected.items() if key != "models"}
    top_paths = _diff_paths(current_top, expected_top)
    if top_paths and len(details) < DESCRIPTOR_DIFF_LIMIT:
        details.append(
            "descriptor top-level drift: "
            f"{_format_limited(top_paths, limit=MODEL_FIELD_DIFF_LIMIT)}"
        )

    return details


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


def _descriptor_operator_decision_details(
    *, removal_blocked: bool, policy_blocked: bool
) -> list[str]:
    if removal_blocked:
        return [
            "operator decision required: descriptor generation removes model_id(s); "
            "approve coverage/removal before rerunning with --allow-descriptor-model-removal"
        ]
    if policy_blocked:
        return [
            "operator decision required: fix descriptor compiler/source registry before updating"
        ]
    return [
        "operator decision required: review descriptor drift details, then run "
        "uv run python scripts/registry/stack_change_pipeline.py update"
    ]


def _descriptor_policy_errors(generated: dict[str, Any]) -> list[str]:
    models = generated.get("models")
    if not isinstance(models, list):
        return []

    errors: list[str] = []
    for model in models:
        if not isinstance(model, dict):
            continue
        gaps = model.get("known_gaps")
        if not isinstance(gaps, list):
            continue
        conflicts = [
            str(gap)
            for gap in gaps
            if isinstance(gap, str) and gap.startswith("Role-server conflict:")
        ]
        if not conflicts:
            continue
        bindings = model.get("role_bindings")
        roles = (
            sorted(str(role) for role in bindings.get("roles", []) if isinstance(role, str))
            if isinstance(bindings, dict)
            else []
        )
        role_text = f" roles={','.join(roles)}" if roles else ""
        errors.append(
            f"descriptor generated role/server conflict for {model.get('model_id')}{role_text}: "
            + "; ".join(conflicts)
        )
    if errors:
        errors.append(
            "fix descriptor compiler/source registry before updating descriptor artifacts"
        )
    return errors


def _roles_from_stack_manifest() -> set[str]:
    from scripts.server.stack_manifest import ROLE_LAUNCH_META
    from src.registry.registry_compiler import launcher_tenant_roles

    roles = set(ROLE_LAUNCH_META.keys())
    for meta in ROLE_LAUNCH_META.values():
        aliases = meta.get("shared_with_first_n") if isinstance(meta, dict) else None
        if isinstance(aliases, list):
            roles.update(str(alias) for alias in aliases)
    # gpu-serving-tie-in P2-6 (P0-1): registry TENANT roles named by
    # launcher-only entries (tenant_role meta key) must flow through the
    # descriptor/priors pipeline too (empty today — no entry carries the key).
    roles.update(launcher_tenant_roles(ROLE_LAUNCH_META))
    return roles


def _active_roles(config: StackChangePipelineConfig) -> set[str]:
    return set(config.roles) if config.roles is not None else _roles_from_stack_manifest()


def _stack_prior_roles(config: StackChangePipelineConfig) -> set[str] | None:
    return set(config.roles) if config.roles is not None else None


def _lean_registry_step(
    config: StackChangePipelineConfig, *, check: bool
) -> PipelineStep:
    """Recompile the LEAN registry from the master registry.

    THE MISSING HOP. Until 2026-07-31 nothing in this pipeline recompiled lean;
    it imported only ``launcher_tenant_roles`` and treated the lean registry as a
    read-only input. So `update` regenerated descriptors and stack priors FROM A
    STALE LEAN REGISTRY and reported `descriptors: updated / stack_priors:
    updated / guard: ok` while faithfully re-emitting values master had already
    corrected. A green result computed over the wrong input.

    Concretely: master reversed the composed speculative recipe at 17:38 and
    corrected four throughput priors, and production kept launching the reversed
    recipe because every layer below lean was regenerated from lean.

    The chain is master -> lean -> descriptors -> stack_priors -> argv. This step
    makes the pipeline own the whole chain instead of its last three links.
    """
    if config.research_registry is None:
        return PipelineStep(
            name="lean_registry",
            status="ok",
            details=["skipped: no research (master) registry configured"],
        )
    bootstrap_warnings: list[str] = []
    try:
        from src.registry.registry_compiler import (
            active_roles_from_launch_meta,
            compile_lean,
            cache_key,
            load_or_compile,
        )

        # Use the COMPILER's own notion of the active set, not the pipeline's.
        # _active_roles() additionally carries launcher tenants (eval_batch_frontdoor),
        # which descriptors and priors legitimately need but which the lean registry
        # does not project. Mixing the two makes the cache key never match, so the
        # step would report "stale" on every run — a guard that always fires is as
        # useless as one that never does.
        if config.roles is not None:
            active = set(config.roles)
        else:
            try:
                from scripts.server.stack_manifest import ROLE_LAUNCH_META
            except ValueError as import_exc:
                if check:
                    raise
                # BOOTSTRAP (STACKCHG-DFLASH2-20261003). stack_manifest validates the
                # launcher against the LEAN at import — the very file this step is about
                # to regenerate. A change that flips a field both sides declare (here
                # worker_vision tier hot -> warm) is refused against the STALE lean, so
                # the lean could never be recompiled and the pipeline could never apply
                # it. During `update` only, derive the same active set from the same
                # inputs _build_role_launch_meta uses — launch_manifest role_launch_meta
                # plus MASTER server_mode.<role>.shared_with — compile, and let the later
                # steps re-import stack_manifest against the fresh lean (a failed import
                # is not cached), where every import-time gate runs for real.
                ROLE_LAUNCH_META = _launch_meta_from_sources(config)
                bootstrap_warnings.append(
                    "BOOTSTRAP: stack_manifest refused to import against the stale lean "
                    f"({str(import_exc).splitlines()[-1].strip()[:240]}); active set derived "
                    "from launch_manifest.yaml + master shared_with. Later steps re-import "
                    "against the regenerated lean."
                )
            active = active_roles_from_launch_meta(ROLE_LAUNCH_META)
        cache_path = config.lean_registry.parent / ".lean_cache_key"
        topology = _topology_path(config)
        expected_key = cache_key(config.research_registry, active, topology)
        current_key = cache_path.read_text().strip() if cache_path.exists() else ""

        if check:
            # Compare CONTENT without writing. The cache key lives in a
            # GITIGNORED per-clone file, so it cannot speak for the COMMITTED
            # lean registry in either direction: every fresh worktree has no key
            # (false "stale"), the shared clone's key ages whenever master bytes
            # change outside the active projection (false "stale" — origin/main
            # 34e27fdf failed here while the committed lean equalled the current
            # projection exactly), and a matching key over a hand-edited lean
            # would be a false "fresh". NIB2-69: the projection is the fact; the
            # key is reported only as cache state.
            compiled = compile_lean(config.research_registry, active, topology)
            committed = (
                _load_yaml(config.lean_registry) if config.lean_registry.exists() else None
            )
            key_state = (
                "matches"
                if current_key == expected_key
                else f"{current_key[:12] or '<none>'} != {expected_key[:12]}"
            )
            if committed == compiled:
                return PipelineStep(
                    name="lean_registry",
                    status="ok",
                    details=[
                        f"content fresh vs master: {config.research_registry}",
                        f"local cache key (gitignored): {key_state}",
                    ],
                )
            return PipelineStep(
                name="lean_registry",
                status="stale",
                errors=[
                    "lean registry content is stale against the master registry "
                    f"projection (local cache key: {key_state})",
                    "everything below is compiled FROM lean, so a stale lean makes "
                    "descriptors/stack_priors green over the wrong input",
                    "run: uv run python scripts/registry/stack_change_pipeline.py update",
                ],
            )

        if cache_path.exists():
            cache_path.unlink()  # force, as the CLI's --force does
        before = config.lean_registry.read_bytes() if config.lean_registry.exists() else None
        out = load_or_compile(
            config.research_registry, active, config.lean_registry, cache_path, topology
        )
        changed = before != (
            config.lean_registry.read_bytes() if config.lean_registry.exists() else None
        )
        evicted = _evict_lean_derived_modules(config) if changed else []
    except Exception as exc:  # noqa: BLE001
        return PipelineStep(
            name="lean_registry",
            status="failed",
            errors=[
                f"lean registry compile from master failed: {exc}",
                f"master: {config.research_registry}",
                f"drafter selection: {_topology_path(config)} (drafter_selection)",
            ],
        )
    details = [
        f"recompiled from master: {config.research_registry}"
        f" -> {config.lean_registry}",
        f"{len(out.get('roles', {}))} roles projected through the active manifest",
    ]
    if evicted:
        details.append(
            "lean changed: evicted " + ", ".join(evicted) + " so the later steps of this "
            "run import it against the regenerated lean, not the one it was loaded from"
        )
    return PipelineStep(
        name="lean_registry",
        status="ok",
        warnings=bootstrap_warnings,
        details=details,
    )


# Modules whose MODULE-LEVEL constants are computed from the lean registry at import.
_LEAN_DERIVED_MODULES = ("scripts.server.stack_manifest", "stack_manifest")


def _evict_lean_derived_modules(config: StackChangePipelineConfig) -> list[str]:
    """Forget stack_manifest after `update` rewrote the lean it reads.

    STACKCHG-KVPOOL-20261003. stack_manifest derives LAUNCH_CONTEXT_TOKENS,
    ROLE_LAUNCH_META, LAUNCH_KV_QUANT_CONFIGS, HOT/WARM_SERVERS, ... from the lean
    AT IMPORT, and this run imported it (the step above, `_active_roles`) before
    the lean was rewritten. Every later step of the same `update` then compiled
    stack priors from the OLD lean: raising :8083 n_ctx 196608 -> 393216 wrote
    `effective_context_tokens: 196608` into the priors, `update` reported
    `guard: ok`, and the next `check` (a fresh process) failed with "stack-prior
    artifact is stale" and three "serving.effective_context_tokens 196608 does not
    match launch context 393216" guard errors. The consumers on this path import
    it inside functions, so dropping it from sys.modules is enough for them to
    re-execute it against the fresh lean (and re-run its import-time gates).
    Only the lean stack_manifest actually reads is eligible.
    """
    default_lean = config.repo_root / "orchestration" / "model_registry.yaml"
    if config.lean_registry.resolve() != default_lean.resolve():
        return []
    evicted = [name for name in _LEAN_DERIVED_MODULES if sys.modules.pop(name, None) is not None]
    parent = sys.modules.get("scripts.server")
    if parent is not None and "stack_manifest" in vars(parent):
        delattr(parent, "stack_manifest")
    return evicted


def _launch_meta_from_sources(config: StackChangePipelineConfig) -> dict[str, dict]:
    """ROLE_LAUNCH_META's role/alias skeleton, read from its SOURCES without importing
    stack_manifest (whose import-time gates read the lean being regenerated).

    Mirrors stack_manifest._build_role_launch_meta/_derived_aliases: role keys and their
    meta from launch_manifest.yaml `role_launch_meta`; aliases = `launcher_only_aliases`
    then MASTER `server_mode.<role>.shared_with`. Only the keys
    active_roles_from_launch_meta reads are produced.
    """
    manifest = _load_yaml(config.repo_root / "orchestration" / "launch_manifest.yaml") or {}
    master = _load_yaml(config.research_registry) or {}
    server_mode = master.get("server_mode") or {}
    out: dict[str, dict] = {}
    for role, declared in (manifest.get("role_launch_meta") or {}).items():
        meta = dict(declared) if isinstance(declared, dict) else {}
        extras = [str(a) for a in meta.pop("launcher_only_aliases", None) or []]
        shared = (server_mode.get(role) or {}).get("shared_with") or []
        aliases = extras + [str(a) for a in shared if isinstance(a, str) and a not in extras]
        if aliases:
            meta["shared_with_first_n"] = aliases
        out[str(role)] = meta
    return out


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
        policy_errors = _descriptor_policy_errors(expected)
        if policy_errors:
            return PipelineStep(
                name="descriptors",
                status="failed",
                errors=policy_errors,
            )
        return PipelineStep(
            name="descriptors",
            status="ok",
            details=[f"fresh: {config.descriptors}"],
        )
    removal_errors = _descriptor_removal_errors(config.descriptors, expected)
    policy_errors = _descriptor_policy_errors(expected)
    drift_details = _descriptor_drift_details(config.descriptors, expected)
    drift_details.extend(
        _descriptor_operator_decision_details(
            removal_blocked=bool(removal_errors),
            policy_blocked=bool(policy_errors),
        )
    )
    if removal_errors or policy_errors:
        errors = [
            f"descriptor artifact is stale: {config.descriptors}",
            *removal_errors,
            *policy_errors,
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
        details=drift_details,
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
    policy_errors = _descriptor_policy_errors(descriptors)
    if policy_errors:
        return PipelineStep(
            name="descriptors",
            status="failed",
            errors=policy_errors,
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


def _check_stack_priors(
    config: StackChangePipelineConfig, resolution: NumaModeResolution
) -> PipelineStep:
    try:
        # The mode comes from the DECLARED topology so that check and update
        # compute the same lineup from the same inputs, whatever the fleet is
        # doing. The realized-fleet probe remains only as a backstop when nothing
        # is declared (ESC-8 Fix 6: refuse rather than default to "full").
        mode_kwargs, _mode_source = _numa_mode_kwargs(resolution)
        expected = compile_stack_priors(
            registry_path=config.lean_registry,
            descriptor_path=config.descriptors,
            active_roles=_stack_prior_roles(config),
            allow_incomplete=config.compile_incomplete,
            **mode_kwargs,
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


def _update_stack_priors(
    config: StackChangePipelineConfig, resolution: NumaModeResolution
) -> PipelineStep:
    try:
        # Same declared-topology resolution as _check_stack_priors, so an update
        # can never write a lineup a check would call stale. ESC-8 Fix 6 survives
        # as the backstop: with nothing declared this still refuses rather than
        # defaulting to "full" and rewriting priors to the dead full lineup.
        mode_kwargs, mode_source = _numa_mode_kwargs(resolution)
        priors = write_stack_priors(
            config.stack_priors,
            registry_path=config.lean_registry,
            descriptor_path=config.descriptors,
            active_roles=_stack_prior_roles(config),
            allow_incomplete=config.compile_incomplete,
            **mode_kwargs,
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
        details=[
            f"wrote {len(priors.get('roles', {}))} role prior(s): {config.stack_priors}",
            # Provenance, not decoration: a lineup written from the wrong source is
            # otherwise indistinguishable from one written from the right source.
            f"numa mode source: {mode_source}",
        ],
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


def _check_operator_summary(config: StackChangePipelineConfig) -> PipelineStep:
    expected = render_current_stack_summary(
        stack_priors_path=config.stack_priors,
        registry_path=config.lean_registry,
        descriptor_path=config.descriptors,
    )
    try:
        current = config.operator_summary.read_text(encoding="utf-8")
    except OSError:
        current = ""
    if current == expected:
        return PipelineStep(
            name="operator_summary",
            status="ok",
            details=[f"fresh: {config.operator_summary}"],
        )
    return PipelineStep(
        name="operator_summary",
        status="stale",
        errors=[
            f"operator stack summary is stale or missing: {config.operator_summary}",
            "run: uv run python scripts/registry/stack_change_pipeline.py update",
        ],
    )


def _update_operator_summary(config: StackChangePipelineConfig) -> PipelineStep:
    try:
        write_current_stack_summary(
            config.operator_summary,
            stack_priors_path=config.stack_priors,
            registry_path=config.lean_registry,
            descriptor_path=config.descriptors,
        )
    except OSError as exc:
        return PipelineStep(
            name="operator_summary",
            status="failed",
            errors=[f"operator stack summary update failed: {exc}"],
        )
    return PipelineStep(
        name="operator_summary",
        status="updated",
        details=[f"wrote generated operator summary: {config.operator_summary}"],
    )


def _shared_with_derivations_step(config: StackChangePipelineConfig) -> PipelineStep:
    """Recompute the five surfaces that restate `shared_with` and diff them.

    RUNS FIRST, before the lean compile, because every one of these disagreements
    is visible from the hand-edited sources alone. On 2026-09-22 they surfaced one
    class per pipeline run -- nine runs, one error each, with the stack down --
    only because nothing recomputed them until a later stage happened to trip over
    the copy. `sync_procedure_role_enums` already does this correctly for the fifth
    surface (the procedure role enums); this is the same move for the other five.

    The fifth of those five is `src/config/models.py::_LEGACY_SERVER_URL_FALLBACKS`
    (SSU-F8): the last-resort URL table `_server_url_default` ends in a bare subscript
    of. It is passed explicitly because `load_shared_with_sources` defaults it OFF --
    a caller handing the checker a synthetic stack has no models.py to recompute.
    """
    if config.research_registry is None:
        return PipelineStep(
            name="shared_with_derivations",
            status="skipped",
            details=["no master registry configured for this pipeline config"],
        )
    manifest_path = config.repo_root / "orchestration" / "launch_manifest.yaml"
    topology_path = _topology_path(config)
    # models.py is OPTIONAL to this step, unlike the three declared surfaces: a config
    # whose repo_root holds only orchestration/ (the check-mode fixtures do) declares no
    # sixth surface, and an absent surface is another step's problem, not a failure here.
    models_path = config.repo_root / "src" / "config" / "models.py"
    if not models_path.exists():
        models_path = None
    missing = [
        str(path)
        for path in (config.research_registry, manifest_path, topology_path)
        if not path.exists()
    ]
    if missing:
        # A surface this config does not declare is another step's problem. Only a
        # PRESENT-but-unparseable source is this step's failure.
        return PipelineStep(
            name="shared_with_derivations",
            status="skipped",
            details=[f"source not present in this config: {path}" for path in missing],
        )
    try:
        sources = load_shared_with_sources(
            config.research_registry, manifest_path, topology_path
        ).attach_models_py(models_path)
        findings = check_shared_with_derivations(sources)
    except SharedWithSourceError as exc:
        return PipelineStep(
            name="shared_with_derivations", status="failed", errors=[str(exc)]
        )
    if not findings:
        return PipelineStep(
            name="shared_with_derivations",
            status="ok",
            details=[
                "port_map, role_launch_meta, numa_config, roles.<alias>.model and "
                "_LEGACY_SERVER_URL_FALLBACKS all recompute from "
                "server_mode.*.shared_with"
            ],
        )
    errors = [
        f"{finding.surface} [{finding.role}] {finding.message} "
        f"({finding.file}:{finding.line}; found {finding.found}, expected {finding.expected}"
        + ("; fixable: check_shared_with_derivations.py --fix)" if finding.fixable else ")")
        for finding in findings
    ]
    return PipelineStep(name="shared_with_derivations", status="failed", errors=errors)


def _guard_step(
    name: str,
    config: StackChangePipelineConfig,
    *,
    strict: bool,
    all_surfaces: bool,
    allow_known_gaps: bool = False,
    launch_numa_mode: str | None = None,
) -> PipelineStep:
    result = validate_stack_priors(
        config.stack_priors,
        strict=strict,
        scan_surfaces=True,
        repo_root=config.repo_root,
        surface_categories=None if all_surfaces else frozenset({"production_blocker"}),
        surface_exceptions_path=config.surface_exceptions,
        surface_manifest_path=config.surface_manifest,
        procedure_path=config.procedure,
        procedure_schema_path=config.schema,
        registry_path=config.lean_registry,
        descriptor_path=config.descriptors,
        allow_production_blocker_waivers=config.allow_production_blocker_waivers,
        accepted_gaps_path=config.accepted_gaps,
        launch_numa_mode=launch_numa_mode,
    )
    errors = list(result.errors)
    warnings = list(result.warnings)
    status = "ok"
    if errors and strict and allow_known_gaps:
        # --allow-known-gaps downgrades ONLY strict-promoted warnings. Everything
        # the non-strict guard also reports as an error (launch-manifest/port
        # alignment, source-artifact hash pins, contract violations) stays an
        # error here too, so the flag can never take a structural defect green
        # (NIB2-69). The downgraded ones are listed one by one, never counted.
        known_gap_errors = [error for error in errors if error.startswith("strict: ")]
        hard_errors = [error for error in errors if not error.startswith("strict: ")]
        warnings.extend(f"known-gap: {error}" for error in known_gap_errors)
        errors = hard_errors
        status = "failed" if hard_errors else "known_gaps"
    elif errors:
        status = "failed"
    elif warnings:
        status = "warnings"
    return PipelineStep(name=name, status=status, errors=errors, warnings=warnings)


def _reasoning_effort_certifications_step(config: StackChangePipelineConfig) -> PipelineStep:
    result = validate_reasoning_effort_certifications(
        registry_path=config.lean_registry,
        certifications_path=config.effort_certifications,
    )
    if result.errors:
        return PipelineStep(
            name="reasoning_effort_certifications",
            status="failed",
            errors=result.errors,
        )
    return PipelineStep(
        name="reasoning_effort_certifications",
        status="ok",
        details=["declared prompt-effort levels match their certified model/quant/kernel era"],
    )


def _artifact_integrity_step(config: StackChangePipelineConfig) -> PipelineStep:
    """SEC-ART-1/3/4: artifact pins, served-template pins and the community-uploader review gate.

    Malformed pins and a community tool-bearing role without a complete review record are ALWAYS errors.
    Missing artifact pins (and a tool-bearing role with no uploader_class) are errors by default since
    SEC-ART-1.1 (``REGISTRY_ARTIFACT_PINS_STRICT=0`` opts out), the same rule as the registry validator.
    Missing template pins stay warnings until ``REGISTRY_TEMPLATE_PINS_STRICT`` /
    ``runtime_defaults.artifact_integrity.template_pins_strict`` flips them.
    """
    from src.registry import artifact_integrity as ai
    from src.registry.artifact_review import check_uploader_review
    from src.registry.registry_validator import (
        check_artifact_pins,
        check_template_pins,
        validator_artifact_pins_strict,
    )

    try:
        registry = yaml.safe_load(config.lean_registry.read_text()) or {}
    except OSError as exc:  # the lean_registry step owns the missing-file finding
        return PipelineStep(name="artifact_integrity", status="skipped", warnings=[f"lean registry unreadable: {exc}"])
    except yaml.YAMLError as exc:
        return PipelineStep(name="artifact_integrity", status="failed", errors=[f"lean registry is not valid YAML: {exc}"])
    pins_strict = validator_artifact_pins_strict()
    ledger_strict = ai.strict_enabled(ai.PINS_STRICT, registry)
    tmpl_strict = ai.strict_enabled(ai.TEMPLATE_PINS_STRICT, registry)
    malformed, missing = check_artifact_pins(registry)
    t_malformed, t_missing = check_template_pins(registry)
    r_errors, r_warnings = check_uploader_review(registry, strict=pins_strict, repo_root=config.repo_root)
    errors = [f"artifact pin: {m}" for m in malformed] + [f"template pin: {m}" for m in t_malformed]
    errors += [f"artifact review: {m}" for m in r_errors]
    warnings = list(r_warnings)
    (errors if pins_strict else warnings).extend(f"artifact pin missing: {m}" for m in missing)
    (errors if tmpl_strict else warnings).extend(f"template pin missing: {m}" for m in t_missing)
    return PipelineStep(
        name="artifact_integrity",
        status="failed" if errors else "ok",
        errors=errors,
        warnings=warnings,
        details=[f"artifact_pins_strict={pins_strict} launcher_ledger_strict={ledger_strict} "
                 f"template_pins_strict={tmpl_strict}"],
    )


def _promotion_gate_command() -> list[str]:
    # The pipeline can validate a fully isolated temporary stack world, where
    # ``config.repo_root`` is not a Python project. Run the unchanged gate suite
    # from this source checkout explicitly while keeping the caller's temp world
    # as the subprocess cwd.
    # Reuse the already-running project interpreter. Calling ``uv run`` here
    # can synchronize the full application dependency graph as a side effect;
    # a missing test dependency should fail visibly instead of triggering an
    # implicit install.
    return [sys.executable, "-m", "pytest", "-q",
            "-c", str(REPO_ROOT / "pyproject.toml"), f"--rootdir={REPO_ROOT}",
            "-o", "addopts=", "-p", "no:cacheprovider",
            *(str(REPO_ROOT / target) for target in PROMOTION_GATE_TARGETS)]


def _runtime_attestation_warnings() -> list[str]:
    from scripts.server.stack_commands import runtime_attestation_warnings

    return runtime_attestation_warnings()


def _stack_manifest_registry_warnings(config: StackChangePipelineConfig) -> list[str]:
    from scripts.server.stack_manifest import validate_against_registry

    return validate_against_registry(str(config.lean_registry), roles=config.roles)


def _fresh_capacity_import(repo_root: Path) -> subprocess.CompletedProcess:
    """Validate the newly compiled lean in a fresh interpreter.

    Import alone is deliberately safe for oversized lineups; repair/status tools
    must remain available. The explicit call preserves the fail-closed check.
    """
    return subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, sys.argv[1]); "
         "from scripts.server.stack_manifest import validate_serving_shape_capacity; "
         "validate_serving_shape_capacity()",
         str(repo_root)],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def _serving_shape_capacity_step(config: StackChangePipelineConfig) -> PipelineStep:
    """Re-run the explicit VRAM/host capacity gate on this run's compiled lean."""
    default_lean = config.repo_root / "orchestration" / "model_registry.yaml"
    manifest = config.repo_root / "scripts" / "server" / "stack_manifest.py"
    if config.lean_registry.resolve() != default_lean.resolve() or not manifest.exists():
        return PipelineStep(
            name="serving_shape_capacity",
            status="skipped",
            # A detail, not a warning: only fixture/lane configs point the lean
            # elsewhere; every production run uses the default lean and runs the gate.
            details=["COULD-NOT-CHECK: stack_manifest reads <repo_root>/orchestration/"
                     "model_registry.yaml only; this config's lean is elsewhere"],
        )
    try:
        result = _fresh_capacity_import(config.repo_root)
    except OSError as exc:
        return PipelineStep(name="serving_shape_capacity", status="failed",
                            errors=[f"capacity import failed to launch: {exc}"])
    if result.returncode == 0:
        return PipelineStep(
            name="serving_shape_capacity",
            status="ok",
            details=["fresh-process serving capacity validation passed on "
                     f"{default_lean}"],
        )
    tail = (result.stderr or result.stdout or "").strip().splitlines()
    reason = next((line for line in reversed(tail) if line.strip()), "no output")
    return PipelineStep(
        name="serving_shape_capacity",
        status="failed",
        errors=[f"serving capacity validation fails on the compiled lean: {_clip_output(reason)}"],
    )


def _q_scorer_prior_sources_step(
    config: StackChangePipelineConfig,
    *,
    prior_ok: bool,
) -> PipelineStep:
    if not prior_ok:
        return PipelineStep(
            name="q_scorer_priors",
            status="skipped",
            warnings=["skipped because earlier stack-change checks failed"],
        )
    errors = validate_live_q_scorer_prior_sources(config.stack_priors)
    if errors:
        return PipelineStep(name="q_scorer_priors", status="failed", errors=errors)
    return PipelineStep(
        name="q_scorer_priors",
        status="ok",
        details=["live q_scorer prior sources are stack-prior derived"],
    )


def _runtime_attestation_step(*, prior_ok: bool) -> PipelineStep:
    if not prior_ok:
        return PipelineStep(
            name="runtime_attestation",
            status="skipped",
            warnings=["skipped because earlier stack-change checks failed"],
        )
    try:
        warnings = _runtime_attestation_warnings()
    except Exception as exc:  # noqa: BLE001
        return PipelineStep(
            name="runtime_attestation",
            status="failed",
            errors=[f"runtime attestation failed to run: {exc}"],
        )
    if warnings:
        return PipelineStep(
            name="runtime_attestation",
            status="failed",
            errors=[f"live process drift: {warning}" for warning in warnings],
        )
    return PipelineStep(
        name="runtime_attestation",
        status="ok",
        details=["no concrete live process drift detected"],
    )


def _declared_env_attestation_result():
    """Seam for tests (like _runtime_attestation_warnings): reads the LIVE processes."""
    from scripts.server.env_attestation import attest
    from scripts.server.orchestrator_stack import _pids_on_port, load_state

    return attest(load_state(), pids_on_port=_pids_on_port)


def _declared_env_attestation_step() -> PipelineStep:
    """Declared-vs-live launch env (scripts/server/env_attestation.py). Read-only /proc.

    Deliberately NOT gated on prior_ok: it reads the live processes and the declared env
    blocks, never a compiled artifact, so a stale descriptor must not hide it. It exists
    because runtime_attestation never compared the declared env blocks, and every declared
    GGML knob was dropped at launch for eight weeks without a signal (b060dd56 .. 2026-09-26).
    """
    try:
        result = _declared_env_attestation_result()
    except Exception as exc:  # noqa: BLE001
        return PipelineStep(
            name="declared_env_attestation",
            status="failed",
            errors=[f"declared env attestation failed to run: {exc}"],
        )
    details = [f"compared: {line}" for line in result.compared]
    details += [f"not attested (no declared env contract): {line}" for line in result.not_attested]
    # UFH-12 A3 / UFH-14: deviations covered by a live, unexpired env override record (the
    # embedder OpenMP record, or a per-component diagnostic record in logs/env_overrides/) are
    # recorded as warnings naming the experiment -- visible, never an error, never silent.
    expected = [f"EXPECTED env deviation: {line}" for line in getattr(result, "expected", [])]
    if result.verdict == "failed":
        return PipelineStep(
            name="declared_env_attestation",
            status="failed",
            errors=[f"declared env drift: {line}" for line in result.errors],
            warnings=expected,
            details=details,
        )
    if result.verdict == "could-not-check":
        return PipelineStep(
            name="declared_env_attestation",
            status="skipped",
            warnings=["COULD-NOT-CHECK: no managed live process in this tree's state file "
                      "(a lane worktree has none); this is not a pass"],
            details=details,
        )
    return PipelineStep(name="declared_env_attestation", status="ok", warnings=expected, details=details)


def _stack_manifest_registry_step(
    config: StackChangePipelineConfig,
    *,
    prior_ok: bool,
) -> PipelineStep:
    if not prior_ok:
        return PipelineStep(
            name="stack_manifest_registry",
            status="skipped",
            warnings=["skipped because earlier stack-change checks failed"],
        )
    try:
        warnings = _stack_manifest_registry_warnings(config)
    except Exception as exc:  # noqa: BLE001
        return PipelineStep(
            name="stack_manifest_registry",
            status="failed",
            errors=[f"stack manifest registry check failed to run: {exc}"],
        )
    if warnings:
        return PipelineStep(
            name="stack_manifest_registry",
            status="failed",
            errors=[f"stack manifest registry drift: {warning}" for warning in warnings],
        )
    return PipelineStep(
        name="stack_manifest_registry",
        status="ok",
        details=["stack manifest launch roles match registry process_layout/server_mode"],
    )


def _clip_output(text: str, *, max_chars: int = 2000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _promotion_gate_step(config: StackChangePipelineConfig, *, prior_ok: bool) -> PipelineStep:
    command = _promotion_gate_command()
    command_text = " ".join(command)
    if not config.run_promotion_gate:
        return PipelineStep(
            name="promotion_gate",
            status="reference",
            details=[f"no-inference promotion target: {command_text}"],
        )
    if not prior_ok:
        return PipelineStep(
            name="promotion_gate",
            status="skipped",
            warnings=["skipped because earlier stack-change checks failed"],
        )
    try:
        test_env = os.environ.copy()
        test_env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        test_env["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            command,
            cwd=config.repo_root,
            env=test_env,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        return PipelineStep(
            name="promotion_gate",
            status="failed",
            errors=[f"promotion gate failed to launch: {exc}"],
            details=[f"command: {command_text}"],
        )
    details = [f"command: {command_text}"]
    if result.stdout:
        details.append("stdout:\n" + _clip_output(result.stdout.rstrip()))
    if result.stderr:
        details.append("stderr:\n" + _clip_output(result.stderr.rstrip()))
    if result.returncode == 0:
        return PipelineStep(name="promotion_gate", status="ok", details=details)
    return PipelineStep(
        name="promotion_gate",
        status="failed",
        errors=[f"promotion gate exited {result.returncode}"],
        details=details,
    )


def _git_out(repo: Path, *args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _research_registry_staleness_step(config: StackChangePipelineConfig) -> PipelineStep:
    """Refuse to compile from a research (master) registry that is not origin/main's.

    The default research registry is a SHARED clone that can sit many commits behind
    origin/main; compiling from it silently drops operator-ratified declarations
    (2026-10-24: ten ``ctx_checkpoints: 32`` lines). Compares
    ``git hash-object <file>`` with ``git rev-parse origin/main:<relpath>``.
    ``git fetch`` only moves the remote-tracking ref (never the worktree); if it
    fails, the existing origin/main ref is used and the step says so.
    """
    name = "research_registry_fresh"
    path = config.research_registry
    if path is None:
        return PipelineStep(name=name, status="skipped", details=["no research registry configured"])
    if config.allow_stale_research_registry:
        return PipelineStep(
            name=name,
            status="skipped",
            warnings=[
                "LOUD: --allow-stale-research-registry: research registry NOT checked against "
                "origin/main; output may drop landed declarations -- do not commit it"
            ],
        )
    from src.registry.research_origin import provenance_for

    origin = provenance_for(Path(path))
    if origin is not None:
        return PipelineStep(
            name=name,
            status="ok",
            details=[f"materialized from research {origin[0]} at commit {origin[1]} (not a working copy)"],
        )
    path = Path(path)
    if not path.exists():
        return PipelineStep(name=name, status="skipped", details=[f"{path} does not exist (reported by the compile step)"])
    resolved = path.resolve()
    top = _git_out(resolved.parent, "rev-parse", "--show-toplevel")
    if top.returncode != 0:
        return PipelineStep(
            name=name,
            status="skipped",
            details=[f"{resolved} is not inside a git checkout; origin/main comparison not possible"],
        )
    repo = Path(top.stdout.strip())
    relpath = resolved.relative_to(repo.resolve()).as_posix()
    details: list[str] = []
    try:
        fetch = _git_out(repo, "fetch", "--quiet", "origin", "main", timeout=30)
        fetched = fetch.returncode == 0
    except subprocess.TimeoutExpired:
        fetched = False
    if not fetched:
        details.append("git fetch failed; compared against the existing origin/main ref (may itself be stale)")
    remote = _git_out(repo, "rev-parse", f"origin/main:{relpath}")
    if remote.returncode != 0:
        return PipelineStep(
            name=name,
            status="skipped",
            details=details + [f"origin/main:{relpath} not resolvable in {repo}; comparison not possible"],
        )
    local = _git_out(repo, "hash-object", str(resolved))
    remote_hash, local_hash = remote.stdout.strip(), local.stdout.strip()
    if remote_hash != local_hash:
        return PipelineStep(
            name=name,
            status="failed",
            details=details,
            errors=[
                f"research registry {resolved} is stale or modified vs origin/main of {repo}: "
                f"on-disk blob {local_hash} != origin/main:{relpath} blob {remote_hash}. "
                "Compiling from it would drop landed declarations. Pass "
                "--research-registry <path to a fresh origin/main checkout>/orchestration/model_registry.yaml, "
                "or --allow-stale-research-registry for deliberate offline use."
            ],
        )
    return PipelineStep(
        name=name, status="ok", details=details + [f"matches origin/main:{relpath} ({remote_hash})"]
    )


def run_stack_change_pipeline(config: StackChangePipelineConfig) -> PipelineReport:
    report = PipelineReport()
    # Resolved ONCE and recorded as the first step, so the output always says
    # which lineup was evaluated and the compile and the guard cannot disagree.
    numa_resolution = resolve_pipeline_numa_mode(config)
    report.steps.append(_numa_mode_step(numa_resolution))
    # Before anything compiles from the research registry. A quiet skip (not a git
    # checkout) adds no step; a pass, a bypass warning or a refusal is always shown.
    fresh_step = _research_registry_staleness_step(config)
    if fresh_step.status != "skipped" or fresh_step.warnings or not fresh_step.ok:
        report.steps.append(fresh_step)
    if not fresh_step.ok:
        return report
    report.steps.append(_shared_with_derivations_step(config))
    if config.mode == "check":
        report.steps.append(_lean_registry_step(config, check=True))
        report.steps.append(_check_descriptors(config))
        report.steps.append(_check_stack_priors(config, numa_resolution))
        report.steps.append(_procedure_enums(config, check=True))
        report.steps.append(_check_operator_summary(config))
    else:
        # master -> lean FIRST. Everything below compiles from lean, so this must
        # lead or the rest is regenerated over a stale input and reports green.
        lean_step = _lean_registry_step(config, check=False)
        report.steps.append(lean_step)
        if not lean_step.ok:
            report.steps.append(
                PipelineStep(
                    name="descriptors",
                    status="failed",
                    errors=["skipped: lean registry did not compile from master"],
                )
            )
            return report
        descriptor_step = _update_descriptors(config)
        report.steps.append(descriptor_step)
        if descriptor_step.ok:
            report.steps.append(_update_stack_priors(config, numa_resolution))
            report.steps.append(_procedure_enums(config, check=False))
            report.steps.append(_update_operator_summary(config))
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
                PipelineStep(
                    name="operator_summary",
                    status="skipped",
                    warnings=["skipped because descriptor update failed"],
                )
            )

    # Every guard step builds its launch-manifest view for the SAME resolved mode
    # the priors were compiled for. Any launch/serving alignment error is a hard
    # error in all three steps (strict or not, known gaps allowed or not), so a
    # single manifest/port mismatch fails `check`.
    launch_mode = numa_resolution.mode
    report.steps.append(
        _guard_step(
            "guard", config, strict=False, all_surfaces=False, launch_numa_mode=launch_mode
        )
    )
    report.steps.append(
        _guard_step(
            "guard_all_surfaces",
            config,
            strict=False,
            all_surfaces=True,
            launch_numa_mode=launch_mode,
        )
    )
    report.steps.append(
        _guard_step(
            "guard_strict",
            config,
            strict=True,
            all_surfaces=False,
            allow_known_gaps=config.allow_known_gaps,
            launch_numa_mode=launch_mode,
        )
    )
    report.steps.append(_reasoning_effort_certifications_step(config))
    report.steps.append(_artifact_integrity_step(config))
    report.steps.append(_serving_shape_capacity_step(config))
    report.steps.append(_stack_manifest_registry_step(config, prior_ok=report.ok))
    report.steps.append(_q_scorer_prior_sources_step(config, prior_ok=report.ok))
    report.steps.append(_runtime_attestation_step(prior_ok=report.ok))
    report.steps.append(_declared_env_attestation_step())
    report.steps.append(
        PipelineStep(
            name="simulated_fixtures",
            status="reference",
            details=[
                "data-only stack-change fixture target: "
                f"uv run pytest -q {SIMULATED_FIXTURE_TARGET}"
            ],
        )
    )
    report.steps.append(_promotion_gate_step(config, prior_ok=report.ok))
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
    for line in report.acceptance_lines():
        print(line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the no-inference stack-change check/update pipeline. The master registry "
            "is read from research origin/main by default; never sync or edit the shared "
            "research clone to make a rebuild work; pass --research-registry only for a "
            "deliberate alternative checkout."
        ),
    )
    parser.add_argument("mode", choices=("check", "update"))
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--lean-registry", type=Path, default=DEFAULT_LEAN_REGISTRY)
    parser.add_argument(
        "--research-registry",
        type=Path,
        default=None,
        help=(
            "Master registry to compile from. DEFAULT: research origin/main, fetched "
            "(ref-only) and materialized from git into a private temp dir -- the master "
            "registry is read from research origin/main by default; never sync or edit "
            "the shared research clone to make a rebuild work. Pass this only for a "
            "deliberate alternative checkout (it is still checked against that "
            "checkout's origin/main)"
        ),
    )
    parser.add_argument("--descriptors", type=Path, default=DEFAULT_DESCRIPTOR_OUTPUT)
    parser.add_argument("--stack-priors", type=Path, default=DEFAULT_STACK_PRIORS)
    parser.add_argument("--operator-summary", type=Path, default=DEFAULT_OPERATOR_SUMMARY)
    parser.add_argument("--procedure", type=Path, default=DEFAULT_PROCEDURE)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--surface-exceptions", type=Path, default=DEFAULT_SURFACE_EXCEPTIONS)
    parser.add_argument("--surface-manifest", type=Path, default=DEFAULT_SURFACE_MANIFEST)
    parser.add_argument(
        "--accepted-gaps",
        type=Path,
        default=DEFAULT_ACCEPTED_GAPS,
        help=(
            "YAML file declaring owned, expiring acceptances of known stack-prior "
            "gaps; declared gaps are reported with their expiry instead of blocking"
        ),
    )
    parser.add_argument(
        "--effort-certifications", type=Path, default=DEFAULT_EFFORT_CERTIFICATIONS
    )
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
        "--allow-production-blocker-waivers",
        action="store_true",
        help=(
            "Permit documented production-blocker hardcoded-surface waivers as "
            "visible warnings; default rejects them fail-closed"
        ),
    )
    parser.add_argument(
        "--allow-stale-research-registry",
        action="store_true",
        help=(
            "Skip the check that the research registry file equals origin/main's "
            "version and, with no --research-registry, compile from the shared research "
            "clone's working copy. Deliberate offline use only: a stale master drops "
            "landed declarations"
        ),
    )
    parser.add_argument(
        "--allow-descriptor-model-removal",
        action="store_true",
        help="Permit update mode to remove model IDs from the descriptor artifact",
    )
    parser.add_argument(
        "--run-promotion-gate",
        action="store_true",
        help="Run the no-inference pytest promotion gate after checks pass",
    )
    parser.add_argument(
        "--numa-mode",
        choices=sorted(VALID_NUMA_MODES),
        default=None,
        help=(
            "NUMA lineup to compile and guard. Default: the declared production "
            "mode in orchestration/stack_topology.yaml. The ambient "
            "ORCHESTRATOR_STACK_NUMA_MODE is never consulted; the evaluated mode "
            "and its source are printed as the numa_mode step"
        ),
    )
    args = parser.parse_args(argv)

    materialized = None
    research_registry = args.research_registry
    if research_registry is None:
        if args.allow_stale_research_registry:
            research_registry = DEFAULT_RESEARCH_REGISTRY
            print(
                "WARNING: --allow-stale-research-registry with no --research-registry: "
                f"compiling from the shared research clone working copy {research_registry}, "
                "which can be many commits behind origin/main. Output may DROP landed "
                "declarations; do not commit it.",
                file=sys.stderr,
            )
        else:
            from src.registry.research_origin import (
                ResearchOriginError,
                materialize_research_registry,
            )

            try:
                materialized = materialize_research_registry()
            except ResearchOriginError as exc:
                print(
                    f"ERROR: cannot materialize research origin/main: {exc}\n"
                    "Pass --research-registry <fresh checkout> or, for deliberate offline "
                    "use only, --allow-stale-research-registry.",
                    file=sys.stderr,
                )
                return 1
            research_registry = materialized.path
            for warning in materialized.warnings:
                print(f"WARNING: {warning}", file=sys.stderr)
            print(f"master registry: {materialized.label} @ {materialized.sha}", file=sys.stderr)

    config = StackChangePipelineConfig(
        mode=args.mode,
        repo_root=args.repo_root,
        lean_registry=args.lean_registry,
        research_registry=research_registry,
        descriptors=args.descriptors,
        stack_priors=args.stack_priors,
        operator_summary=args.operator_summary,
        procedure=args.procedure,
        schema=args.schema,
        surface_exceptions=args.surface_exceptions,
        surface_manifest=args.surface_manifest,
        accepted_gaps=args.accepted_gaps,
        effort_certifications=args.effort_certifications,
        roles=set(args.roles) if args.roles else None,
        allow_known_gaps=args.allow_known_gaps,
        allow_production_blocker_waivers=args.allow_production_blocker_waivers,
        compile_incomplete=not args.strict_descriptor_compile,
        allow_descriptor_model_removal=args.allow_descriptor_model_removal,
        run_promotion_gate=args.run_promotion_gate,
        numa_mode=args.numa_mode,
        allow_stale_research_registry=args.allow_stale_research_registry,
    )
    # NIB2-102: the guard's retired-role derivation reads the SAME master as every
    # other step of this run (origin/main by default), not its own clone default.
    from scripts.validate import stack_change_guard as _guard

    _guard.use_master_registry(research_registry)
    try:
        report = run_stack_change_pipeline(config)
    finally:
        _guard.use_master_registry(None)
        if materialized is not None:
            materialized.cleanup()
    _print_report(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
