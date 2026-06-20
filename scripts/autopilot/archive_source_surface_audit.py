#!/usr/bin/env python3
"""Static audit for AutoPilot archive-source authority surfaces."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]


@dataclass(frozen=True)
class SurfaceRequirement:
    name: str
    path: str
    must_contain: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class SurfaceAuditResult:
    name: str
    path: str
    ok: bool
    missing: tuple[str, ...]
    reason: str


REQUIREMENTS: tuple[SurfaceRequirement, ...] = (
    SurfaceRequirement(
        name="operator_read_commands_default_to_journal",
        path="scripts/autopilot/autopilot.py",
        must_contain=(
            "ARCHIVE_SOURCE_JOURNAL_ALL",
            "ARCHIVE_SOURCE_STATE",
            "state is a legacy fallback",
            "f\"{source}->state-empty-fallback\"",
            "_archive_for_read_command",
        ),
        reason=(
            "status/report/plot/digest must default to journal reconstruction and "
            "make the state cache an explicit legacy fallback."
        ),
    ),
    SurfaceRequirement(
        name="dashboard_state_fallback_is_diagnostic",
        path="src/api/routes/dashboard.py",
        must_contain=(
            "legacy_state_archive_warning",
            "using_legacy_state_archive",
            "dashboard fell back to autopilot_state.json:pareto_archive",
            "archive_authority",
            "journal_rows_available",
        ),
        reason=(
            "operator dashboard may fall back to state cache only with visible "
            "diagnostics and a legacy warning."
        ),
    ),
    SurfaceRequirement(
        name="safety_gate_uses_journal_archive_context",
        path="scripts/autopilot/safety_gate.py",
        must_contain=(
            "_pareto_archive_for_safety_guard",
            "ExperimentJournal",
            "pareto_archive_from_journal_rows",
            "current_run_only=False",
            "Archive-max guard",
        ),
        reason=(
            "baseline safety checks must derive Pareto context from journal rows, "
            "not from a stale state archive cache."
        ),
    ),
    SurfaceRequirement(
        name="preflight_has_strict_archive_authority_gate",
        path="scripts/autopilot/preflight_audit.py",
        must_contain=(
            "archive_authority_diagnostic",
            "audit_archive_authority",
            "Archive Authority",
            "snapshot_readiness",
            "state_archive_present",
        ),
        reason=(
            "restart preflight must retain a strict state-vs-journal archive "
            "authority gate."
        ),
    ),
    SurfaceRequirement(
        name="repair_removes_legacy_state_cache",
        path="scripts/autopilot/archive_authority_repair.py",
        must_contain=(
            "build_repaired_state",
            "repaired.pop(\"pareto_archive\", None)",
            "build_archive_authority_report",
            "postcheck_failed",
            "--expect-trial-counter",
        ),
        reason=(
            "archive repair must remove the legacy cache only after a report-backed "
            "postcheck and stale-counter guard."
        ),
    ),
    SurfaceRequirement(
        name="restart_readiness_includes_archive_authority",
        path="scripts/autopilot/restart_readiness_report.py",
        must_contain=(
            "build_archive_authority_report",
            "archive authority is not aligned",
            "snapshot_replay",
            "baseline_authority",
            "sequential_cutover",
        ),
        reason=(
            "aggregate restart readiness must block on archive authority before "
            "sequential or baseline cutover decisions."
        ),
    ),
)


def _check_requirement(root: Path, requirement: SurfaceRequirement) -> SurfaceAuditResult:
    path = root / requirement.path
    if not path.exists():
        return SurfaceAuditResult(
            name=requirement.name,
            path=requirement.path,
            ok=False,
            missing=("file does not exist",),
            reason=requirement.reason,
        )
    text = path.read_text(encoding="utf-8")
    missing = tuple(fragment for fragment in requirement.must_contain if fragment not in text)
    return SurfaceAuditResult(
        name=requirement.name,
        path=requirement.path,
        ok=not missing,
        missing=missing,
        reason=requirement.reason,
    )


def build_archive_source_surface_audit(root: Path = ORCH_ROOT) -> dict[str, object]:
    """Audit known archive-source authority surfaces without importing them."""
    resolved = root.expanduser().resolve()
    results = [_check_requirement(resolved, requirement) for requirement in REQUIREMENTS]
    failed = [result for result in results if not result.ok]
    return {
        "ok": not failed,
        "root": str(resolved),
        "surface_count": len(results),
        "failed_count": len(failed),
        "results": [asdict(result) for result in results],
    }


def render_markdown(report: dict[str, object]) -> str:
    lines = [
        "# AutoPilot Archive Source Surface Audit",
        "",
        f"- OK: {str(report['ok']).lower()}",
        f"- Surfaces: {report['surface_count']}",
        f"- Failed: {report['failed_count']}",
    ]
    for result in report["results"]:  # type: ignore[index]
        status = "ok" if result["ok"] else "failed"
        lines.extend([
            "",
            f"## {result['name']}",
            "",
            f"- Status: {status}",
            f"- Path: `{result['path']}`",
            f"- Reason: {result['reason']}",
        ])
        missing = result.get("missing") or []
        if missing:
            lines.append(f"- Missing: {', '.join(f'`{item}`' for item in missing)}")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only static audit that known archive-source surfaces default "
            "to journal authority or expose explicit legacy-state diagnostics."
        )
    )
    parser.add_argument("--root", type=Path, default=ORCH_ROOT)
    parser.add_argument("--json", action="store_true", help="Emit structured JSON.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if any archive-source surface invariant fails.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_archive_source_surface_audit(args.root)
    if args.json:
        print(json.dumps(report, sort_keys=True, default=str))
    else:
        print(render_markdown(report))
    if args.strict and not report["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
