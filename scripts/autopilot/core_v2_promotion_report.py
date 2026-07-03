#!/usr/bin/env python3
"""Read-only core_v2 promotion readiness report."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(ORCH_ROOT))

from src.autopilot_core.instrument_era_guard import designed_core_activation_guard  # noqa: E402

DEFAULT_CORE_DIR = ORCH_ROOT / "benchmarks" / "prompts"
DEFAULT_REPORT_DIR = ORCH_ROOT / "orchestration" / "reports"
CORE_METADATA_KEY = "__core_metadata__"


def _utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return obj if isinstance(obj, dict) else {}


def _default_core_id() -> str:
    return os.environ.get("AUTOPILOT_T1_CORE_ID", "").strip() or "core_v2"


def _core_path(core_id: str, path: Path | None = None) -> Path:
    if path is not None:
        return path.expanduser().resolve()
    override = os.environ.get("AUTOPILOT_T1_CORE_PATH", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_CORE_DIR / f"{core_id}.jsonl"


def _inspect_core(path: Path, core_id: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    question_rows = 0
    first_error = ""
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    first_error = f"{path}:{line_no}: invalid JSONL row: {exc}"
                    break
                if not isinstance(row, dict):
                    first_error = f"{path}:{line_no}: core row must be an object"
                    break
                if row.get(CORE_METADATA_KEY):
                    metadata = row
                else:
                    question_rows += 1
    except OSError as exc:
        return {
            "ok": False,
            "path": str(path),
            "exists": False,
            "question_rows": 0,
            "metadata": {},
            "error": str(exc),
        }

    metadata_core_id = str(metadata.get("core_id", "")).strip()
    blockers: list[str] = []
    if first_error:
        blockers.append(first_error)
    if not metadata:
        blockers.append("core metadata row is missing")
    elif metadata_core_id != core_id:
        blockers.append(
            f"metadata core_id={metadata_core_id!r} does not match requested {core_id!r}"
        )
    if question_rows <= 0:
        blockers.append("core file contains no question rows")
    selected_count = metadata.get("selected_count")
    if isinstance(selected_count, int) and selected_count != question_rows:
        blockers.append(
            f"metadata selected_count={selected_count} does not match question_rows={question_rows}"
        )
    return {
        "ok": not blockers,
        "path": str(path),
        "exists": True,
        "question_rows": question_rows,
        "metadata": _metadata_summary(metadata),
        "embedded_selection_report": _selection_summary(
            metadata.get("selection_report") if isinstance(metadata, dict) else None
        ),
        "blockers": blockers,
    }


def _metadata_summary(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "core_id": metadata.get("core_id"),
        "generated_at": metadata.get("generated_at"),
        "generator": metadata.get("generator"),
        "selected_count": metadata.get("selected_count"),
        "target_size": metadata.get("target_size"),
    }


def _selection_summary(report: Any) -> dict[str, Any]:
    if not isinstance(report, Mapping):
        return {
            "ok": False,
            "present": False,
            "blockers": ["selection report is missing"],
        }
    provenance = report.get("source_provenance")
    if not isinstance(provenance, Mapping):
        provenance = {}
    parameters = report.get("parameters")
    if not isinstance(parameters, Mapping):
        parameters = {}
    selected_count = _as_int(report.get("selected_count"))
    target_size = _as_int(parameters.get("target_size"))
    shortfall = _as_int(report.get("shortfall")) or 0
    unresolved = _as_int(report.get("unresolved_selected_count")) or 0
    blockers: list[str] = []
    if selected_count is None or selected_count <= 0:
        blockers.append("selection report has no selected items")
    if target_size is not None and selected_count is not None and selected_count < target_size:
        blockers.append(f"selected_count={selected_count} is below target_size={target_size}")
    if shortfall:
        blockers.append(f"selection shortfall={shortfall}")
    if unresolved:
        blockers.append(f"unresolved_selected_count={unresolved}")
    return {
        "ok": not blockers,
        "present": True,
        "core_id": report.get("core_id"),
        "generated_at": report.get("generated_at"),
        "selected_count": selected_count,
        "eligible_items": _as_int(report.get("eligible_items")),
        "observed_items": _as_int(report.get("observed_items")),
        "source_rows": _as_int(report.get("source_rows")),
        "shortfall": shortfall,
        "unresolved_selected_count": unresolved,
        "parameters": {
            "source": parameters.get("source"),
            "min_attempts": parameters.get("min_attempts"),
            "target_size": target_size,
            "include_partitions": parameters.get("include_partitions"),
            "p_min": parameters.get("p_min"),
            "p_max": parameters.get("p_max"),
        },
        "source_provenance": {
            "trusted_rows": _as_int(provenance.get("trusted_rows")),
            "untrusted_rows": _as_int(provenance.get("untrusted_rows")),
            "era_excluded_rows": _as_int(provenance.get("era_excluded_rows")),
            "exclude_before_ts": provenance.get("exclude_before_ts"),
            "journal_batches": provenance.get("journal_batches"),
        },
        "blockers": blockers,
    }


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_core_v2_promotion_report(
    core_id: str,
    *,
    core_path: Path | None = None,
    selection_report_path: Path | None = None,
    eras_path: Path | None = None,
) -> dict[str, Any]:
    """Build a no-write readiness report for activating a designed T1 core."""
    resolved_core_path = _core_path(core_id, core_path)
    core = _inspect_core(resolved_core_path, core_id)
    explicit_selection = (
        _selection_summary(_load_json(selection_report_path))
        if selection_report_path is not None
        else core.get("embedded_selection_report", {})
    )
    guard = designed_core_activation_guard(
        core_id,
        path=eras_path,
    )
    blockers: list[str] = []
    if not core.get("ok"):
        blockers.extend(f"core artifact: {item}" for item in core.get("blockers", []))
    selection_core_id = explicit_selection.get("core_id")
    if explicit_selection.get("present") and selection_core_id and selection_core_id != core_id:
        blockers.append(
            f"selection report core_id={selection_core_id!r} does not match {core_id!r}"
        )
    if not explicit_selection.get("ok"):
        blockers.extend(
            f"selection evidence: {item}" for item in explicit_selection.get("blockers", [])
        )
    if not guard.get("ok"):
        blockers.append(f"instrument era: {guard.get('reason', 'not authorized')}")

    return {
        "ok": not blockers,
        "promotion_ready": not blockers,
        "core_id": core_id,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "core": core,
        "selection": explicit_selection,
        "instrument_era_guard": guard,
        "activation_env": {
            "AUTOPILOT_T1_CORE_ID": core_id,
            "AUTOPILOT_T1_CORE_PATH": str(resolved_core_path),
        },
        "blockers": blockers,
        "recommendation": _recommendation(blockers, guard),
    }


def _recommendation(blockers: list[str], guard: Mapping[str, Any]) -> str:
    if not blockers:
        return "core_v2 candidate is ready for operator-authorized activation"
    if guard.get("status") in {"missing_core_era", "core_mismatch", "missing_registry"}:
        return (
            "candidate artifact is inspected, but activation remains blocked until "
            "the operator appends a matching autopilot_quality instrument-era row"
        )
    return "resolve listed blockers before enabling AUTOPILOT_T1_CORE_ID"


def render_markdown(report: Mapping[str, Any]) -> str:
    core = report.get("core") if isinstance(report.get("core"), Mapping) else {}
    selection = report.get("selection") if isinstance(report.get("selection"), Mapping) else {}
    guard = (
        report.get("instrument_era_guard")
        if isinstance(report.get("instrument_era_guard"), Mapping)
        else {}
    )
    provenance = (
        selection.get("source_provenance")
        if isinstance(selection.get("source_provenance"), Mapping)
        else {}
    )
    lines = [
        "# core_v2 Promotion Readiness Report",
        "",
        f"- Status: {'ready' if report.get('promotion_ready') else 'blocked'}",
        f"- Recommendation: {report.get('recommendation')}",
        f"- Core ID: `{report.get('core_id')}`",
        f"- Core artifact: `{core.get('path')}`",
        f"- Core rows: {core.get('question_rows')} question row(s)",
        (
            "- Selection evidence: "
            f"selected={selection.get('selected_count')}, "
            f"eligible={selection.get('eligible_items')}, "
            f"observed={selection.get('observed_items')}, "
            f"source_rows={selection.get('source_rows')}, "
            f"unresolved={selection.get('unresolved_selected_count')}"
        ),
        (
            "- Ledger provenance: "
            f"trusted_rows={provenance.get('trusted_rows')}, "
            f"untrusted_rows={provenance.get('untrusted_rows')}, "
            f"era_excluded_rows={provenance.get('era_excluded_rows')}, "
            f"exclude_before_ts={provenance.get('exclude_before_ts')}"
        ),
        (
            "- Instrument-era guard: "
            f"status={guard.get('status')}, ok={guard.get('ok')}, "
            f"path=`{guard.get('path')}`"
        ),
        "",
        "## Blockers",
        "",
    ]
    blockers = report.get("blockers") or []
    if blockers:
        lines.extend(f"- {blocker}" for blocker in blockers)
    else:
        lines.append("- none")
    lines.extend(["", "## Activation Env", ""])
    env = report.get("activation_env") if isinstance(report.get("activation_env"), Mapping) else {}
    for key in ("AUTOPILOT_T1_CORE_ID", "AUTOPILOT_T1_CORE_PATH"):
        lines.append(f"- `{key}={env.get(key)}`")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-id", default=_default_core_id())
    parser.add_argument("--core-path", type=Path)
    parser.add_argument("--selection-report", type=Path)
    parser.add_argument("--eras-path", type=Path)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-md", type=Path)
    parser.add_argument("--json", action="store_true", help="Print JSON report to stdout")
    parser.add_argument("--markdown", action="store_true", help="Print Markdown report to stdout")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_core_v2_promotion_report(
        args.core_id,
        core_path=args.core_path,
        selection_report_path=args.selection_report,
        eras_path=args.eras_path,
    )
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(render_markdown(report))
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    if args.markdown:
        print(render_markdown(report))
    if not any([args.out_json, args.out_md, args.json, args.markdown]):
        stamp = _utc_compact()
        out_json = DEFAULT_REPORT_DIR / f"core_v2_promotion_readiness_{stamp}.json"
        out_md = DEFAULT_REPORT_DIR / f"core_v2_promotion_readiness_{stamp}.md"
        out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        out_md.write_text(render_markdown(report))
        print(out_json)
        print(out_md)
    return 0 if report.get("promotion_ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
