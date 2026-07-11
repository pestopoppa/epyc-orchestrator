#!/usr/bin/env python3
"""Plan and optionally apply the P0.3 era-fenced AutoPilot blacklist purge.

Default mode is report-only. Rewriting the blacklist requires both ``--apply``
and an explicit approval token so agents do not silently reopen live search
surfaces while the operator decision is still pending.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BLACKLIST = SCRIPT_DIR / "failure_blacklist.yaml"
APPROVAL_TOKEN = "ERA_FENCED_BLACKLIST_PURGE_2026_07_11"


@dataclass(frozen=True)
class PurgeTarget:
    key: str
    pattern: dict[str, Any]
    source_trial: int | None
    rationale: str


DEFAULT_TARGETS: tuple[PurgeTarget, ...] = (
    PurgeTarget(
        key="architect_delegation_t655_tool_use_axis_bug",
        pattern={"type": "structural_experiment", "flags": {"architect_delegation": True}},
        source_trial=655,
        rationale="Auto-blacklist came from the pre-repair tool_use/delegation instrument era.",
    ),
    PurgeTarget(
        key="specialist_routing_t664_tool_use_axis_bug",
        pattern={"type": "structural_experiment", "flags": {"specialist_routing": True}},
        source_trial=664,
        rationale="Auto-blacklist came from the pre-repair tool_use/delegation instrument era.",
    ),
    PurgeTarget(
        key="specialist_routing_t864_tool_use_axis_bug",
        pattern={"type": "structural_experiment", "flags": {"specialist_routing": False}},
        source_trial=864,
        rationale="Auto-blacklist came from the pre-repair tool_use/delegation instrument era.",
    ),
    PurgeTarget(
        key="frontdoor_prompt_mutation_restart_freeze",
        pattern={"type": "prompt_mutation", "file": "frontdoor.md"},
        source_trial=-1,
        rationale="Manual restart-recovery freeze; extractor fix landed and the freeze is P0.3 purge-scoped.",
    ),
    PurgeTarget(
        key="frontdoor_gepa_restart_freeze",
        pattern={"type": "gepa_optimize", "file": "frontdoor.md"},
        source_trial=-1,
        rationale="Manual restart-recovery freeze companion for GEPA/frontdoor; P0.3 purge-scoped.",
    ),
)


def retryable_reexploration_target(
    entry: dict[str, Any],
    *,
    targets: tuple[PurgeTarget, ...] = DEFAULT_TARGETS,
) -> dict[str, Any] | None:
    """Return P0.3 retry metadata for automated instrument-era targets.

    Manual freeze entries still require the explicit purge approval token. This
    helper only marks source-trial-backed automated entries as eligible for live
    re-exploration while the destructive YAML rewrite remains operator-gated.
    """
    if not isinstance(entry, dict):
        return None
    for target in targets:
        if target.source_trial is None or target.source_trial < 0:
            continue
        if not _entry_matches_target(entry, target):
            continue
        return {
            "target_key": target.key,
            "pattern": target.pattern,
            "source_trial": target.source_trial,
            "rationale": target.rationale,
            "retry_scope": "p0_3_auto_instrument_era",
        }
    return None


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_blacklist_document(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    entries = payload.get("blacklist")
    if entries is None:
        payload["blacklist"] = []
    elif not isinstance(entries, list):
        raise ValueError(f"{path} field 'blacklist' must be a list")
    return payload


def _entry_matches_target(entry: dict[str, Any], target: PurgeTarget) -> bool:
    if entry.get("pattern") != target.pattern:
        return False
    if target.source_trial is None:
        return True
    return entry.get("source_trial") == target.source_trial


def build_purge_report(
    entries: list[dict[str, Any]],
    *,
    targets: tuple[PurgeTarget, ...] = DEFAULT_TARGETS,
    applied: bool = False,
) -> dict[str, Any]:
    removable: list[dict[str, Any]] = []
    retryable: list[dict[str, Any]] = []
    matched_keys: set[str] = set()
    removable_indices: set[int] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        for target in targets:
            if _entry_matches_target(entry, target):
                matched_keys.add(target.key)
                removable_indices.add(index)
                removable.append(
                    {
                        "index": index,
                        "target_key": target.key,
                        "pattern": entry.get("pattern"),
                        "source_trial": entry.get("source_trial"),
                        "reason": entry.get("reason", ""),
                        "rationale": target.rationale,
                    }
                )
                break
        retry_target = retryable_reexploration_target(entry, targets=targets)
        if retry_target is not None:
            retryable.append(
                {
                    "index": index,
                    "reason": entry.get("reason", ""),
                    **retry_target,
                }
            )

    unmatched = [
        {
            "target_key": target.key,
            "pattern": target.pattern,
            "source_trial": target.source_trial,
            "rationale": target.rationale,
        }
        for target in targets
        if target.key not in matched_keys
    ]
    return {
        "schema_version": "autopilot_blacklist_purge_plan.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "approval_token_required": APPROVAL_TOKEN,
        "applied": applied,
        "entry_count_before": len(entries),
        "entry_count_after": len(entries) - len(removable_indices),
        "removable_count": len(removable_indices),
        "preserved_count": len(entries) - len(removable_indices),
        "removable_entries": removable,
        "retryable_count": len(retryable),
        "retryable_entries": retryable,
        "unmatched_targets": unmatched,
    }


def apply_purge(
    document: dict[str, Any],
    *,
    targets: tuple[PurgeTarget, ...] = DEFAULT_TARGETS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    entries = document.get("blacklist") or []
    if not isinstance(entries, list):
        raise ValueError("field 'blacklist' must be a list")
    report = build_purge_report(entries, targets=targets, applied=True)
    remove_indices = {
        int(item["index"])
        for item in report["removable_entries"]
        if isinstance(item.get("index"), int)
    }
    next_document = dict(document)
    next_document["blacklist"] = [
        entry for index, entry in enumerate(entries) if index not in remove_indices
    ]
    report["entry_count_after"] = len(next_document["blacklist"])
    report["preserved_count"] = len(next_document["blacklist"])
    return next_document, report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# AutoPilot Blacklist Purge Plan",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Applied: `{str(report.get('applied', False)).lower()}`",
        f"Approval token required for apply: `{report['approval_token_required']}`",
        "",
        "| entry_count_before | removable_count | entry_count_after |",
        "|---:|---:|---:|",
        "| {before} | {removable} | {after} |".format(
            before=report["entry_count_before"],
            removable=report["removable_count"],
            after=report["entry_count_after"],
        ),
        "",
    ]
    if report["removable_entries"]:
        lines.extend(
            [
                "## Removable Entries",
                "",
                "| index | target | source_trial | pattern | rationale |",
                "|---:|---|---:|---|---|",
            ]
        )
        for item in report["removable_entries"]:
            lines.append(
                "| {index} | {target} | {trial} | `{pattern}` | {rationale} |".format(
                    index=item["index"],
                    target=item["target_key"],
                    trial=item["source_trial"],
                    pattern=json.dumps(item["pattern"], sort_keys=True),
                    rationale=item["rationale"],
                )
            )
        lines.append("")
    else:
        lines.append("No matching purge-scoped blacklist entries were found.")
        lines.append("")

    if report.get("retryable_entries"):
        lines.extend(
            [
                "## Retryable Re-Exploration Entries",
                "",
                (
                    "These automated instrument-era entries may be retried without "
                    "rewriting the blacklist file. Manual freeze entries still require "
                    "the approval token above."
                ),
                "",
                "| index | target | source_trial | pattern |",
                "|---:|---|---:|---|",
            ]
        )
        for item in report["retryable_entries"]:
            lines.append(
                "| {index} | {target} | {trial} | `{pattern}` |".format(
                    index=item["index"],
                    target=item["target_key"],
                    trial=item["source_trial"],
                    pattern=json.dumps(item["pattern"], sort_keys=True),
                )
            )
        lines.append("")

    if report["unmatched_targets"]:
        lines.extend(
            ["## Unmatched Targets", "", "| target | source_trial | pattern |", "|---|---:|---|"]
        )
        for item in report["unmatched_targets"]:
            lines.append(
                "| {target} | {trial} | `{pattern}` |".format(
                    target=item["target_key"],
                    trial=item["source_trial"],
                    pattern=json.dumps(item["pattern"], sort_keys=True),
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_yaml_atomic(path: Path, document: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(
        yaml.dump(document, default_flow_style=False, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def write_report(report: dict[str, Any], json_path: Path | None, md_path: Path | None) -> None:
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    if md_path is not None:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_markdown(report), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan/apply the P0.3 blacklist purge")
    parser.add_argument("--blacklist", type=Path, default=DEFAULT_BLACKLIST)
    parser.add_argument("--apply", action="store_true", help="Rewrite the blacklist file")
    parser.add_argument(
        "--approval-token",
        default="",
        help=f"Required with --apply; must equal {APPROVAL_TOKEN}",
    )
    parser.add_argument("--backup-dir", type=Path, default=None)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--report-md", type=Path, default=None)
    parser.add_argument("--print-md", action="store_true")
    args = parser.parse_args(argv)

    document = load_blacklist_document(args.blacklist)
    entries = document.get("blacklist") or []
    if not isinstance(entries, list):
        raise ValueError("field 'blacklist' must be a list")

    if args.apply:
        if args.approval_token != APPROVAL_TOKEN:
            print(
                f"--apply requires --approval-token {APPROVAL_TOKEN}",
                file=sys.stderr,
            )
            return 2
        backup_dir = args.backup_dir or args.blacklist.parent
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / f"{args.blacklist.name}.bak-p0_3-{_utc_stamp()}"
        backup_path.write_text(args.blacklist.read_text(encoding="utf-8"), encoding="utf-8")
        next_document, report = apply_purge(document)
        report["backup_path"] = str(backup_path)
        _write_yaml_atomic(args.blacklist, next_document)
    else:
        report = build_purge_report(entries)

    write_report(report, args.report_json, args.report_md)
    if args.print_md:
        print(render_markdown(report), end="")
    else:
        print(
            "removable={removable} preserved={preserved} applied={applied}".format(
                removable=report["removable_count"],
                preserved=report["preserved_count"],
                applied=str(report.get("applied", False)).lower(),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
