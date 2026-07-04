#!/usr/bin/env python3
"""Read-only status report for A9 pairwise expanded-gap collection windows."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "orchestration"
    / "reports"
    / "offline_reward_oracle_token_coverage_final_labels_20260621"
    / "offline_reward_pairwise_expanded_gap_collection_manifest.json"
)
EXPECTED_SCHEMA = "offline_reward_pairwise_collection_window.v1"


def _active_processes(pattern: str) -> list[str]:
    try:
        proc = subprocess.run(
            ["pgrep", "-af", pattern],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return [f"process probe failed: {exc}"]
    if proc.returncode not in (0, 1):
        detail = (proc.stderr or proc.stdout or "").strip()
        return [f"process probe failed with exit {proc.returncode}: {detail}"]
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _validate_manifest(manifest: dict[str, Any], manifest_path: Path) -> tuple[list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []

    schema = manifest.get("schema_version")
    if schema != EXPECTED_SCHEMA:
        blockers.append(f"unexpected manifest schema: {schema!r}")

    batches = manifest.get("batches")
    if not isinstance(batches, list) or not batches:
        warnings.append("manifest has no runnable collection batches")
        batches = []

    batch_count = manifest.get("batch_count")
    if batch_count != len(batches):
        blockers.append(
            f"batch_count mismatch: manifest={batch_count!r}, actual={len(batches)}"
        )

    if manifest.get("requires_active_autopilot_absent") is not True:
        warnings.append("manifest does not explicitly require AutoPilot absence")

    for idx, batch in enumerate(batches, start=1):
        if not isinstance(batch, dict):
            blockers.append(f"batch {idx} is not an object")
            continue
        target = batch.get("target") or f"batch {idx}"
        command = str(batch.get("command") or "")
        durable_path = str(batch.get("durable_source_path") or "")
        template = str(batch.get("durable_source_path_template") or durable_path)
        if "seed_specialist_routing.py" not in command:
            blockers.append(f"{target}: command is not a seed_specialist_routing invocation")
        if "--dry-run" not in command.split():
            blockers.append(f"{target}: command is missing --dry-run")
        if not durable_path:
            blockers.append(f"{target}: durable_source_path is missing")
        if "<YYYYMMDDTHHMMSSZ>" not in template and not Path(durable_path).suffix:
            warnings.append(f"{target}: durable output path is not timestamp-templated")
        parent = Path(durable_path.replace("<YYYYMMDDTHHMMSSZ>", "20990101T000000Z")).parent
        if not parent.exists():
            warnings.append(f"{target}: output parent does not exist yet: {parent}")

    pipeline = manifest.get("post_collection_pipeline")
    if not isinstance(pipeline, list) or not pipeline:
        blockers.append("post_collection_pipeline is missing")

    if not manifest_path.exists():
        blockers.append(f"manifest path disappeared while validating: {manifest_path}")

    return blockers, warnings


def build_status(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    blockers: list[str] = []
    warnings: list[str] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {
            "schema_version": "offline_reward_pairwise_collection_status.v1",
            "ready": False,
            "status": "invalid",
            "manifest_path": str(manifest_path),
            "blockers": [f"manifest not found: {manifest_path}"],
            "warnings": [],
        }
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "schema_version": "offline_reward_pairwise_collection_status.v1",
            "ready": False,
            "status": "invalid",
            "manifest_path": str(manifest_path),
            "blockers": [f"cannot read manifest: {exc}"],
            "warnings": [],
        }

    manifest_blockers, manifest_warnings = _validate_manifest(manifest, manifest_path)
    blockers.extend(manifest_blockers)
    warnings.extend(manifest_warnings)

    guard = manifest.get("autopilot_guard") or {}
    pattern = str(guard.get("process_pattern") or "scripts/autopilot/autopilot.py start")
    active = _active_processes(pattern)
    autopilot_blocked = bool(active)

    batches = manifest.get("batches") if isinstance(manifest.get("batches"), list) else []
    status = "ready"
    if manifest_blockers:
        status = "invalid"
    elif not batches:
        status = "no_runnable_batches"
    elif autopilot_blocked:
        status = "blocked"

    if autopilot_blocked and status == "blocked":
        blockers.append(f"active AutoPilot process(es): {'; '.join(active)}")

    return {
        "schema_version": "offline_reward_pairwise_collection_status.v1",
        "ready": status == "ready",
        "status": status,
        "manifest_path": str(manifest_path),
        "manifest_schema_version": manifest.get("schema_version"),
        "source_plan_decision": manifest.get("source_plan_decision"),
        "batch_count": len(batches),
        "post_collection_step_count": len(manifest.get("post_collection_pipeline") or []),
        "autopilot_guard": {
            "process_pattern": pattern,
            "active_processes": active,
            "refusal_exit_code": guard.get("refusal_exit_code", 75),
        },
        "blockers": blockers,
        "warnings": warnings,
    }


def render_markdown(status: dict[str, Any]) -> str:
    lines = [
        "# A9 Pairwise Collection Status",
        "",
        f"- Status: `{status['status']}`",
        f"- Ready: `{str(status['ready']).lower()}`",
        f"- Manifest: `{status['manifest_path']}`",
        f"- Batches: `{status.get('batch_count', 0)}`",
        f"- Post-collection steps: `{status.get('post_collection_step_count', 0)}`",
    ]
    blockers = status.get("blockers") or []
    warnings = status.get("warnings") or []
    if blockers:
        lines.extend(["", "## Blockers"])
        lines.extend(f"- {item}" for item in blockers)
    if warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {item}" for item in warnings)
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--markdown", action="store_true", help="Emit Markdown instead of JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    status = build_status(args.manifest)
    if args.markdown:
        print(render_markdown(status), end="")
    else:
        print(json.dumps(status, indent=2, sort_keys=True))
    if status["status"] == "ready":
        return 0
    if status["status"] == "blocked":
        return int(status.get("autopilot_guard", {}).get("refusal_exit_code") or 75)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
