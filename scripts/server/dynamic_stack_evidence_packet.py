#!/usr/bin/env python3
"""Build a read-only Dynamic Stack Phase-E evidence packet.

DS-E1 is a decision gate, not a scheduler implementation gate. This report
packages the currently available no-inference evidence and makes missing inputs
explicit so DS-7/DS-6 work does not start from stale assumptions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from glob import glob
import json
from pathlib import Path
import re
import sys
from typing import Any

import yaml

SCRIPT_PATH = Path(__file__).resolve()
ORCH_ROOT = SCRIPT_PATH.parents[2]
WORKSPACE_ROOT = ORCH_ROOT.parent
RESEARCH_ROOT = WORKSPACE_ROOT / "epyc-inference-research"
DEFAULT_STACK_PRIORS = ORCH_ROOT / "orchestration" / "derived" / "stack_priors.yaml"
DEFAULT_CLASSIFIER_CONFIG = ORCH_ROOT / "orchestration" / "classifier_config.yaml"
DEFAULT_CONTENTION_MATRIX = ORCH_ROOT / "orchestration" / "contention_matrix.yaml"
DEFAULT_RESEARCH_MANIFEST = RESEARCH_ROOT / "docs" / "MODEL_MANIFEST.md"
DEFAULT_RI10_REPORT_GLOB = "orchestration/reports/ri10_canary_sample_report*.json"
DEFAULT_KV_GLOBS = (
    "orchestration/reports/ds_e1*kv*",
    "orchestration/reports/dynamic_stack*kv*",
    "../epyc-inference-research/data/dynamic_stack/**/kv*",
    "../epyc-inference-research/data/kv_measurements/**",
)

sys.path.insert(0, str(ORCH_ROOT))
sys.path.insert(0, str(ORCH_ROOT / "scripts" / "server"))


@dataclass(frozen=True)
class EvidenceSection:
    """One DS-E1 input status."""

    key: str
    status: str
    summary: str
    details: dict[str, Any]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not contain a mapping")
    return loaded


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _live_role_rows(stack_priors: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    roles = stack_priors.get("roles") or {}
    if not isinstance(roles, dict):
        return rows
    for role, record in sorted(roles.items()):
        if not isinstance(record, dict):
            continue
        if record.get("deployment_status") != "live_stack":
            continue
        serving = record.get("serving") or {}
        priors = record.get("priors") or {}
        model = record.get("model") or {}
        rows.append(
            {
                "role": role,
                "model_id": record.get("model_id"),
                "endpoint": serving.get("endpoint"),
                "ports": serving.get("ports") or [],
                "tier": serving.get("tier"),
                "effective_context_tokens": serving.get("effective_context_tokens"),
                "throughput_tps": priors.get("throughput_tps"),
                "model_mem_gb": model.get("mem_gb"),
            }
        )
    return rows


def stack_roster_section(stack_priors_path: Path = DEFAULT_STACK_PRIORS) -> EvidenceSection:
    if not stack_priors_path.exists():
        return EvidenceSection(
            "stack_roster",
            "missing",
            "Generated stack priors are missing.",
            {"path": str(stack_priors_path)},
        )
    try:
        stack_priors = _load_yaml(stack_priors_path)
    except Exception as exc:
        return EvidenceSection(
            "stack_roster",
            "invalid",
            "Generated stack priors could not be parsed.",
            {"path": str(stack_priors_path), "error": str(exc)},
        )
    rows = _live_role_rows(stack_priors)
    status = "ready" if rows and stack_priors.get("status") == "compiled" else "incomplete"
    return EvidenceSection(
        "stack_roster",
        status,
        f"{len(rows)} live stack-prior roles packaged from generated truth.",
        {
            "path": str(stack_priors_path),
            "compiled_at": stack_priors.get("compiled_at"),
            "source_commit": ((stack_priors.get("source_artifacts") or {}).get("registry") or {}).get(
                "repo_commit"
            ),
            "roles": rows,
        },
    )


def _manifest_compiled_at(text: str) -> str | None:
    match = re.search(r"compiled at `([^`]+)`", text)
    return match.group(1) if match else None


def ds5_manifest_section(
    manifest_path: Path = DEFAULT_RESEARCH_MANIFEST,
    stack_priors_path: Path = DEFAULT_STACK_PRIORS,
) -> EvidenceSection:
    if not manifest_path.exists():
        return EvidenceSection(
            "ds5_roster_manifest",
            "missing",
            "Research MODEL_MANIFEST.md is missing.",
            {"path": str(manifest_path)},
        )
    text = manifest_path.read_text(encoding="utf-8")
    manifest_compiled_at = _manifest_compiled_at(text)
    stack_compiled_at = None
    if stack_priors_path.exists():
        try:
            stack_compiled_at = _load_yaml(stack_priors_path).get("compiled_at")
        except Exception:
            stack_compiled_at = None
    status = "ready"
    summary = "Research model manifest exists for DS-5 roster context."
    if stack_compiled_at and manifest_compiled_at and manifest_compiled_at != stack_compiled_at:
        status = "stale"
        summary = "Research model manifest exists but references an older stack-prior compile."
    return EvidenceSection(
        "ds5_roster_manifest",
        status,
        summary,
        {
            "path": str(manifest_path),
            "manifest_compiled_at": manifest_compiled_at,
            "stack_priors_compiled_at": stack_compiled_at,
        },
    )


def contention_section(matrix_path: Path = DEFAULT_CONTENTION_MATRIX) -> EvidenceSection:
    try:
        from stack_numa import NUMA_CONFIG
        from src.scheduling.contention import MatrixStatus, matrix_status, topology_fingerprint
    except Exception as exc:
        return EvidenceSection(
            "contention_matrix",
            "invalid",
            "Contention matrix freshness imports failed.",
            {"error": str(exc), "path": str(matrix_path)},
        )
    current_hash = topology_fingerprint(NUMA_CONFIG)
    status = matrix_status(
        matrix_path,
        current_topology_hash=current_hash,
        max_age_days=30,
    )
    mapped = {
        MatrixStatus.OK: "ready",
        MatrixStatus.MISSING: "missing",
        MatrixStatus.STALE: "stale",
        MatrixStatus.INVALID: "invalid",
    }.get(status, "invalid")
    return EvidenceSection(
        "contention_matrix",
        mapped,
        (
            f"Contention matrix status is {mapped} "
            f"for topology {current_hash[:8]}."
        ),
        {
            "path": str(matrix_path),
            "topology_hash": current_hash,
            "matrix_status": getattr(status, "value", str(status)),
        },
    )


def ri10_canary_section(config_path: Path = DEFAULT_CLASSIFIER_CONFIG) -> EvidenceSection:
    if not config_path.exists():
        return EvidenceSection(
            "ri10_canary",
            "missing",
            "Classifier config is missing; RI-10 canary state cannot be read.",
            {"path": str(config_path)},
        )
    try:
        config = _load_yaml(config_path)
    except Exception as exc:
        return EvidenceSection(
            "ri10_canary",
            "invalid",
            "Classifier config could not be parsed.",
            {"path": str(config_path), "error": str(exc)},
        )
    factual = config.get("factual_risk") or {}
    mode = factual.get("mode")
    canary_ratio = factual.get("canary_ratio")
    canary_roles = factual.get("canary_roles") or []
    report_paths = _resolve_glob(ORCH_ROOT, DEFAULT_RI10_REPORT_GLOB)
    latest_report: dict[str, Any] = {}
    latest_report_path: Path | None = report_paths[-1] if report_paths else None
    if latest_report_path:
        try:
            loaded = json.loads(latest_report_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                latest_report = loaded
        except Exception as exc:
            latest_report = {"error": str(exc)}

    status = "missing_data"
    summary = "RI-10 config is present, but no current canary sample-count artifact was found."
    if latest_report:
        if latest_report.get("canary_decision_ready") is True:
            status = "ready"
            summary = "RI-10 canary sample-count and enforce/shadow arm telemetry are present."
        elif latest_report.get("sample_count_ready") is True:
            status = "insufficient_data"
            summary = (
                "RI-10 high-risk sample-count coverage exists, but enforce/shadow canary "
                "decision telemetry is not sufficient."
            )
        else:
            status = "insufficient_data"
            summary = "RI-10 canary sample-count artifact exists but is below the high-risk sample gate."
    report_summary = {
        key: latest_report.get(key)
        for key in (
            "generated_at",
            "canary_start",
            "decision_gate_high_risk_samples",
            "high_risk_rows_since_canary_start",
            "frontdoor_high_risk_rows_since_canary_start",
            "sample_count_ready",
            "canary_decision_ready",
            "decision_reason",
            "canary_arm_counts_since_canary_start",
            "high_risk_gate_actions_since_canary_start",
        )
        if key in latest_report
    }
    return EvidenceSection(
        "ri10_canary",
        status,
        summary,
        {
            "path": str(config_path),
            "mode": mode,
            "canary_ratio": canary_ratio,
            "canary_roles": canary_roles,
            "decision_gate": ">=50 high-risk samples",
            "report_path": str(latest_report_path) if latest_report_path else None,
            "report_summary": report_summary,
        },
    )


def _resolve_glob(root: Path, pattern: str) -> list[Path]:
    return sorted(
        Path(match).resolve()
        for match in glob(str(root / pattern), recursive=True)
        if Path(match).is_file()
    )


def kv_measurement_section(
    root: Path = ORCH_ROOT,
    patterns: tuple[str, ...] = DEFAULT_KV_GLOBS,
) -> EvidenceSection:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(_resolve_glob(root, pattern))
    unique = sorted({path for path in matches})
    if not unique:
        return EvidenceSection(
            "kv_size_measurements",
            "missing",
            "No direct DS-E1 production KV-size measurement series was found.",
            {
                "searched_globs": list(patterns),
                "required_contexts": ["2K", "8K", "32K"],
            },
        )
    return EvidenceSection(
        "kv_size_measurements",
        "candidate",
        f"Found {len(unique)} candidate KV measurement artifact(s); operator review required.",
        {
            "paths": [str(path) for path in unique],
            "required_contexts": ["2K", "8K", "32K"],
        },
    )


def build_packet() -> dict[str, Any]:
    sections = [
        stack_roster_section(),
        ds5_manifest_section(),
        contention_section(),
        ri10_canary_section(),
        kv_measurement_section(),
    ]
    blocking_statuses = {
        "missing",
        "missing_data",
        "insufficient_data",
        "stale",
        "invalid",
        "incomplete",
    }
    blockers = [
        f"{section.key}: {section.summary}"
        for section in sections
        if section.status in blocking_statuses
    ]
    return {
        "generated_at": _iso_now(),
        "scope": "dynamic-stack-concurrency DS-E1 evidence packet",
        "ready_for_profile_decision": not blockers,
        "blockers": blockers,
        "sections": [asdict(section) for section in sections],
    }


def render_markdown(packet: dict[str, Any]) -> str:
    lines = [
        "# Dynamic Stack DS-E1 Evidence Packet",
        "",
        f"Generated: {packet['generated_at']}",
        f"Ready for DS-7/DS-6 profile decision: {str(packet['ready_for_profile_decision']).lower()}",
        "",
    ]
    blockers = packet.get("blockers") or []
    if blockers:
        lines.extend(["## Blockers", ""])
        lines.extend(f"- {blocker}" for blocker in blockers)
        lines.append("")
    lines.extend(["## Evidence Sections", ""])
    for section in packet["sections"]:
        lines.extend(
            [
                f"### {section['key']}",
                "",
                f"- Status: `{section['status']}`",
                f"- Summary: {section['summary']}",
            ]
        )
        details = section.get("details") or {}
        if details:
            lines.append("- Details:")
            for key, value in details.items():
                rendered = json.dumps(value, sort_keys=True, default=str)
                lines.append(f"  - `{key}`: {rendered}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of Markdown.")
    parser.add_argument("--output", type=Path, help="Write the rendered packet to this path.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when the DS-E1 evidence packet is not decision-ready.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    packet = build_packet()
    rendered = (
        json.dumps(packet, indent=2, sort_keys=True, default=str) + "\n"
        if args.json
        else render_markdown(packet)
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    if args.strict and not packet["ready_for_profile_decision"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
