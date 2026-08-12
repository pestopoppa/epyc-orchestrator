#!/usr/bin/env python3
"""Parity gate: `stack_templates/*.yaml` topology vs the single source.

WHY THIS EXISTS (P1-4, `handoffs/active/numa-topology-cutover-resume-20260730.md`)
---------------------------------------------------------------------------------
The declared-vs-deployed NUMA topology drifted for months because ONE fact —
"which ports, on which cpuset, with how many threads" — is written down in four
places. The single source is `orchestration/stack_topology.yaml` `numa_config:`,
loaded as ``scripts.server.stack_numa.NUMA_CONFIG``. The other three restate it.
`stack_templates/default.yaml` is the fourth copy.

Re-synchronising the copies is what people kept doing, and it is why the drift
recurs. The only thing that makes a restatement safe is a gate that fails when
it stops matching. Before this module, the topology fields of default.yaml were
covered like this:

  * `model`/`quant`/`tier`      -> `scripts/validate/stack_change_guard.py`
                                   `stale_role_fact_table` (vs compiled priors).
  * `port`                      -> `src/config/stack_templates.
                                   _validate_stack_prior_parity` (vs compiled
                                   priors) -- but it resolves a per-role
                                   `numa_mode` until SOME mode matches, so a
                                   template full port the priors do not list
                                   (e.g. frontdoor :8070) is not a mismatch.
  * `numa` and `threads`        -> NOTHING, except `frontdoor` and
                                   `architect_general`, pinned ad hoc in
                                   `tests/unit/test_dynamic_stack.py`.

So `worker_general`, `architect_critic`, `ingest_long_context` and
`worker_vision` could each declare any cpuset shape and any thread count and no
gate would notice. That is exactly the hole defect c3 went through (an SMT
oversubscribed `threads: 96` frozen into a template that no gate re-derives).

WHAT THIS GATE ASSERTS
----------------------
For every deployable role, the template's declared instances are EQUAL to the
instances derived from the single source: same set of ports, and for each port
the same `numa:` shape name and the same `threads:`. Plus the structural claim
that the template's `full:` entry is the source's full instance.

It is deliberately a comparison against the SOURCE (`NUMA_CONFIG`), not against
the compiled priors: the priors are themselves a derivation, and comparing two
derivations of the same fact leaves the source unpinned.

NON-VACUITY
-----------
A parity check that iterates an empty set of roles passes trivially. This one
refuses to pass unless it actually compared a plausibly sized fleet, and unless
the role sets on the two sides account for each other exactly:

  * at least ``MIN_COMPARED_ROLES`` roles and ``MIN_COMPARED_INSTANCES``
    instances were compared;
  * every source role is either in the template or in
    ``SOURCE_ROLES_NOT_IN_TEMPLATE`` (a DECLARED exception with a reason, so a
    newly added source role cannot be silently missed);
  * every template role absent from the source is an alias or an
    embedding-mode role (which legitimately declare no NUMA placement).

WHY THIS IS A GATE AND NOT YET THE DERIVATION (standing proposal)
-----------------------------------------------------------------
A gate still leaves four copies; it only makes the fourth one honest. Removing
the copy means the template STOPS DECLARING the topology and has it filled in
at load time. That edit lands in `src/config/stack_templates.py`, which this
change deliberately does not touch. The proposal, in full:

  1. In ``load_template``, after parsing a role's yaml, resolve its instances
     from ``NUMA_CONFIG``/``NUMA_INSTANCE_SHAPES`` instead of from the document
     (``derive_expected`` here is the whole computation, ~25 lines).
  2. Make `full:` / `quarters:` OPTIONAL in the schema, and REJECT them for any
     role the source declares -- a template that still spells the topology out
     is the defect, so it must error, not be tolerated.
  3. Delete the blocks from `stack_templates/*.yaml`, leaving the template
     owning only what a template legitimately overrides: `model`, `quant`,
     `tier`, `ram_gb`, `mode`, `alias_to`, `slot_save_path`, `spec_overrides`
     and `resource_budget`.
  4. Keep this module: after step 3 its role-set accounting still answers "does
     the template name the roles the source declares", and `--render` becomes
     the migration's diffing tool.

Blocking consideration for step 1: three consumers read a loaded template --
`scripts/server/stack_commands.py`, `scripts/server/orchestrator_stack.py`
(`--validate-only`) and `src/config/stack_migration.py` (`--migrate-to`). None
of them launches from it (`stack_commands.py` prints "Template loaded but not
yet used for server launch — integration pending DS-7 Phase 2" and
`stack_migration._phase_start_target` is a stub that starts nothing), so step 1
cannot change what the fleet does. It CAN change what
``_validate_stack_prior_parity`` compares, which is the only real review
surface.

Usage:
    python3 scripts/validate/default_template_topology_parity.py            # check
    python3 scripts/validate/default_template_topology_parity.py --json
    python3 scripts/validate/default_template_topology_parity.py --render   # derived blocks
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE_PATH = REPO_ROOT / "stack_templates" / "default.yaml"

# --- non-vacuity floors ----------------------------------------------------
# Documented FLOORS, not the answer. The live fleet is 6 roles / 10 instances
# (1x96t FULL + 2x48t HALF for each of frontdoor, worker_general and
# ingest_long_context; a single instance for architect_critic,
# architect_general and worker_vision). These floors exist so that a bug that
# empties either side -- a renamed YAML key, a failed load, a filter that
# matches nothing -- fails LOUD instead of reporting "0 mismatches".
MIN_COMPARED_ROLES = 6
MIN_COMPARED_INSTANCES = 10

# Roles the single source declares that the steady-state template deliberately
# does NOT carry. Each needs a reason; this is an exception list, not a skip
# list -- a source role that is neither in the template nor named here is a
# FAILURE, so adding a role to stack_topology.yaml cannot silently bypass the
# template.
SOURCE_ROLES_NOT_IN_TEMPLATE: Mapping[str, str] = {
    "eval_batch_frontdoor": (
        "eval fan-out batch lane (:18070, HALF_A) -- launched by the eval tower, "
        "not part of the steady-state static-prewarm stack the template describes"
    ),
}

# Template `numa:` values that name no CPU placement at all.
NON_PLACED_NUMA_VALUES = frozenset({"NONE", ""})


# --------------------------------------------------------------------------
# derivation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Instance:
    """One launchable instance, as the template spells it."""

    port: int
    numa: str
    threads: int

    def as_dict(self) -> dict[str, Any]:
        return {"port": self.port, "numa": self.numa, "threads": self.threads}


def shape_to_template_numa(shape_name: str) -> str:
    """Spell a `_CPU_SHAPES` key the way a template `numa:` field spells it.

    Inverse of the resolution `tests/unit/test_dynamic_stack.py` performs:
    ``shape if shape in _CPU_SHAPES else f"NUMA_{shape}"``. `NUMA_HALF_A` is
    written `HALF_A`; `GPU_HOST_LANE` has no prefix and is written verbatim.
    """
    if shape_name.startswith("NUMA_"):
        return shape_name[len("NUMA_") :]
    return shape_name


def template_numa_to_shape(numa_value: str, known_shapes: Iterable[str]) -> str:
    """Resolve a template `numa:` value back to a `_CPU_SHAPES` key."""
    known = set(known_shapes)
    return numa_value if numa_value in known else f"NUMA_{numa_value}"


def derive_expected(
    numa_config: Mapping[str, Mapping[str, Any]],
    instance_shapes: Mapping[str, tuple[str, ...]],
) -> dict[str, dict[str, Any]]:
    """Derive the topology every template must restate, from the single source.

    Returns ``{role: {"instances": [Instance, ...], "full_port": int | None}}``.
    ``full_port`` is the source's ``full_instance_idx`` instance, or the sole
    instance when the role declares exactly one.
    """
    expected: dict[str, dict[str, Any]] = {}
    for role, cfg in numa_config.items():
        instances = list(cfg.get("instances") or [])
        shapes = tuple(instance_shapes.get(role, ()))
        if len(shapes) != len(instances):
            raise ValueError(
                f"source role {role!r} declares {len(instances)} instances but "
                f"{len(shapes)} shape names -- stack_topology.yaml is internally "
                "inconsistent; refusing to derive"
            )
        derived: list[Instance] = []
        for (_cpuset, port, threads), shape in zip(instances, shapes):
            derived.append(
                Instance(
                    port=int(port),
                    numa=shape_to_template_numa(shape),
                    threads=int(threads),
                )
            )
        full_idx = cfg.get("full_instance_idx")
        if full_idx is None and len(derived) == 1:
            full_idx = 0
        full_port = derived[full_idx].port if full_idx is not None else None
        expected[role] = {"instances": derived, "full_port": full_port}
    return expected


# --------------------------------------------------------------------------
# template reading
# --------------------------------------------------------------------------


def _instance_from_block(block: Mapping[str, Any]) -> Instance:
    return Instance(
        port=int(block["port"]),
        numa=str(block.get("numa", "")),
        threads=int(block.get("threads", 0)),
    )


def read_template_topology(document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Extract declared instances per role from a loaded template document.

    Reads the RAW yaml rather than ``src.config.stack_templates.load_template``
    on purpose: ``load_template`` substitutes defaults for missing ``numa`` and
    ``threads`` keys (``"NODE0"``/96 for a full, 48 for a quarter), which would
    let a template that OMITS the fields pass a parity check against values it
    never declared.
    """
    roles = document.get("roles") or {}
    out: dict[str, dict[str, Any]] = {}
    for role, spec in roles.items():
        spec = spec or {}
        instances: list[Instance] = []
        full_port: int | None = None
        full_block = spec.get("full")
        if isinstance(full_block, Mapping):
            inst = _instance_from_block(full_block)
            instances.append(inst)
            full_port = inst.port
        for key in ("quarters", "replicas"):
            for block in spec.get(key) or []:
                instances.append(_instance_from_block(block))
        out[role] = {
            "instances": instances,
            "full_port": full_port,
            "is_alias": bool(spec.get("alias_to"))
            or str(spec.get("tier", "")).upper() == "ALIAS",
            "mode": spec.get("mode"),
        }
    return out


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------


@dataclass
class ParityReport:
    problems: list[str]
    compared_roles: list[str]
    compared_instances: int

    @property
    def ok(self) -> bool:
        return not self.problems

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "problems": list(self.problems),
            "compared_roles": list(self.compared_roles),
            "compared_instances": self.compared_instances,
        }


def check_parity(
    numa_config: Mapping[str, Mapping[str, Any]],
    instance_shapes: Mapping[str, tuple[str, ...]],
    template_document: Mapping[str, Any],
    known_shapes: Mapping[str, tuple[str, int]],
    *,
    min_roles: int = MIN_COMPARED_ROLES,
    min_instances: int = MIN_COMPARED_INSTANCES,
    source_roles_not_in_template: Mapping[str, str] | None = None,
) -> ParityReport:
    """Compare a template's declared topology against the single source.

    Pure: every input is injected, so a test can perturb EITHER side.
    """
    exceptions = (
        SOURCE_ROLES_NOT_IN_TEMPLATE
        if source_roles_not_in_template is None
        else source_roles_not_in_template
    )
    problems: list[str] = []

    try:
        expected = derive_expected(numa_config, instance_shapes)
    except ValueError as exc:
        return ParityReport([str(exc)], [], 0)

    declared = read_template_topology(template_document)

    # -- role-set accounting: neither side may quietly omit the other ------
    for role in sorted(expected):
        if role in declared:
            continue
        if role in exceptions:
            continue
        problems.append(
            f"source role {role!r} is declared in stack_topology.yaml numa_config "
            f"but absent from the template, and is not a declared exception "
            f"(add it to the template, or to SOURCE_ROLES_NOT_IN_TEMPLATE with a reason)"
        )
    for role in sorted(declared):
        if role in expected:
            continue
        info = declared[role]
        if info["is_alias"]:
            if info["instances"]:
                problems.append(
                    f"template role {role!r} is an alias but declares "
                    f"{len(info['instances'])} launch instance(s)"
                )
            continue
        if info["mode"] == "embedding":
            placed = [
                inst
                for inst in info["instances"]
                if inst.numa.upper() not in NON_PLACED_NUMA_VALUES
            ]
            if placed:
                problems.append(
                    f"template role {role!r} is embedding-mode but declares NUMA "
                    f"placement {[i.numa for i in placed]!r}; it has no numa_config "
                    "entry to be checked against"
                )
            continue
        problems.append(
            f"template role {role!r} declares launch instances "
            f"{[i.as_dict() for i in info['instances']]!r} but has NO numa_config "
            "entry in stack_topology.yaml -- it is an unsourced topology copy"
        )

    # -- per-role field comparison -----------------------------------------
    compared_roles: list[str] = []
    compared_instances = 0
    for role in sorted(set(expected) & set(declared)):
        exp = expected[role]
        got = declared[role]
        exp_by_port = {inst.port: inst for inst in exp["instances"]}
        got_by_port = {inst.port: inst for inst in got["instances"]}

        if len(got_by_port) != len(got["instances"]):
            problems.append(f"{role}: template declares the same port twice")

        missing = sorted(set(exp_by_port) - set(got_by_port))
        extra = sorted(set(got_by_port) - set(exp_by_port))
        if missing:
            problems.append(
                f"{role}: template is missing source port(s) {missing} "
                f"(source declares {sorted(exp_by_port)}, template declares {sorted(got_by_port)})"
            )
        if extra:
            problems.append(
                f"{role}: template declares port(s) {extra} that the source does not "
                f"(source declares {sorted(exp_by_port)})"
            )

        for port in sorted(set(exp_by_port) & set(got_by_port)):
            e, g = exp_by_port[port], got_by_port[port]
            if e.numa != g.numa:
                problems.append(
                    f"{role}:{port}: numa shape is {g.numa!r} in the template but "
                    f"{e.numa!r} in the source"
                )
            else:
                shape_key = template_numa_to_shape(g.numa, known_shapes)
                if shape_key not in known_shapes:
                    problems.append(
                        f"{role}:{port}: numa {g.numa!r} resolves to unknown CPU shape "
                        f"{shape_key!r}"
                    )
            if e.threads != g.threads:
                problems.append(
                    f"{role}:{port}: threads is {g.threads} in the template but "
                    f"{e.threads} in the source"
                )
            compared_instances += 1

        if exp["full_port"] != got["full_port"]:
            problems.append(
                f"{role}: template `full:` port is {got['full_port']} but the source's "
                f"full instance is {exp['full_port']}"
            )
        compared_roles.append(role)

    # -- non-vacuity: refuse to report a clean pass on nothing --------------
    if len(compared_roles) < min_roles:
        problems.append(
            f"VACUOUS: compared only {len(compared_roles)} role(s) "
            f"({compared_roles!r}); at least {min_roles} expected. The comparison "
            "found nothing to compare, so a clean result would be meaningless."
        )
    if compared_instances < min_instances:
        problems.append(
            f"VACUOUS: compared only {compared_instances} instance(s); at least "
            f"{min_instances} expected."
        )

    return ParityReport(problems, compared_roles, compared_instances)


# --------------------------------------------------------------------------
# live wiring
# --------------------------------------------------------------------------


def load_source() -> tuple[dict, dict, dict]:
    """Load ``(NUMA_CONFIG, NUMA_INSTANCE_SHAPES, _CPU_SHAPES)`` from the source."""
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.server.stack_numa import (  # noqa: PLC0415
        _CPU_SHAPES,
        NUMA_CONFIG,
        NUMA_INSTANCE_SHAPES,
    )

    return NUMA_CONFIG, NUMA_INSTANCE_SHAPES, _CPU_SHAPES


def load_template_document(path: Path | None = None) -> dict[str, Any]:
    # Resolved at CALL time, not def time: a default argument would capture
    # DEFAULT_TEMPLATE_PATH at import and silently ignore any later override,
    # which makes the gate impossible to point at a candidate file -- and makes
    # a test that "redirects" it pass while still reading the original.
    path = DEFAULT_TEMPLATE_PATH if path is None else path
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def run_check(template_path: Path | None = None) -> ParityReport:
    numa_config, instance_shapes, known_shapes = load_source()
    document = load_template_document(template_path)
    return check_parity(numa_config, instance_shapes, document, known_shapes)


def render_derived_blocks() -> str:
    """Emit the topology blocks as the source would have them written.

    This is the derivation made visible: paste-able YAML for the `full:` /
    `quarters:` keys of every sourced role. It is the mechanism the migration
    proposal would move into ``src/config/stack_templates.load_template``.
    """
    numa_config, instance_shapes, _shapes = load_source()
    expected = derive_expected(numa_config, instance_shapes)
    lines: list[str] = []
    for role in sorted(expected):
        if role in SOURCE_ROLES_NOT_IN_TEMPLATE:
            continue
        exp = expected[role]
        lines.append(f"  {role}:")
        siblings = []
        for inst in exp["instances"]:
            if inst.port == exp["full_port"]:
                lines.append("    full:")
                lines.append(f"      port: {inst.port}")
                lines.append(f"      numa: {inst.numa}")
                lines.append(f"      threads: {inst.threads}")
            else:
                siblings.append(inst)
        if siblings:
            lines.append("    quarters:")
            for inst in siblings:
                lines.append(
                    f"      - {{ port: {inst.port}, numa: {inst.numa}, threads: {inst.threads} }}"
                )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--template",
        type=Path,
        default=None,
        help="template yaml to check (default: stack_templates/default.yaml)",
    )
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument(
        "--render",
        action="store_true",
        help="print the topology blocks derived from the single source and exit",
    )
    args = parser.parse_args(argv)

    if args.render:
        sys.stdout.write(render_derived_blocks())
        return 0

    template_path = DEFAULT_TEMPLATE_PATH if args.template is None else args.template
    report = run_check(template_path)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        print(
            f"topology parity: {template_path.name} vs stack_topology.yaml numa_config "
            f"-- compared {len(report.compared_roles)} roles / "
            f"{report.compared_instances} instances"
        )
        if report.ok:
            print("  [OK] template topology matches the single source")
        else:
            print(f"  [FAIL] {len(report.problems)} problem(s):")
            for problem in report.problems:
                print(f"    - {problem}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
