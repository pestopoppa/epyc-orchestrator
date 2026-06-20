#!/usr/bin/env python3
"""Aggregate read-only Fable5 gate status.

This report is intentionally a composition layer over existing authoritative
reports. It does not collect evidence, mutate state, or reinterpret pass/fail
thresholds owned by the underlying gates.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]
SERVER_DIR = ORCH_ROOT / "scripts" / "server"
VALIDATE_DIR = ORCH_ROOT / "scripts" / "validate"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SERVER_DIR))
sys.path.insert(0, str(VALIDATE_DIR))
sys.path.insert(0, str(ORCH_ROOT))

from dynamic_stack_evidence_packet import build_packet as build_ds_e1_packet  # noqa: E402
from phase_status import PHASE_PATH, build_phase_health_report  # noqa: E402
from preflight_audit import JOURNAL_PATH, STATE_PATH, _load_jsonl  # noqa: E402
from restart_readiness_report import build_restart_readiness_report  # noqa: E402
from validate_xmas_winner_table import (  # noqa: E402
    DEFAULT_CLASSIFIER_CONFIG,
    validate_config as validate_xmas_config,
    validate_table as validate_xmas_table,
)

DEFAULT_XMAS_TABLE = ORCH_ROOT / "orchestration" / "xmas_winner_table.yaml"
DEFAULT_XMAS_AB_ROOT = ORCH_ROOT / "benchmarks" / "results" / "runs" / "xmas_live_ab"
DEFAULT_DS_E1_KV_PORT = 8194
DEFAULT_XMAS_HELDOUT_PROMPTS_ARG = (
    "benchmarks/results/runs/xmas_live_ab/20260618-heldout-resilient/prompts.jsonl"
)
DEFAULT_XMAS_CONSTRAINED_OUTPUT_ARG = (
    "benchmarks/results/runs/xmas_live_ab/"
    "$(date -u +%Y%m%dT%H%M%SZ)-constrained-policy"
)
XMAS_QUIET_WINDOW_PROCESS_PATTERNS: tuple[tuple[str, str], ...] = (
    ("AutoPilot", "scripts/autopilot/autopilot.py start"),
    ("AutoPilot", "autopilot.py start"),
    ("X-MAS cheap-kill", "xmas_cheap_kill.py"),
    ("X-MAS function-axis sweep", "xmas_function_axis_sweep.py"),
    ("BEP A/B", "bep_ab.py"),
    ("DCP J7 A/B", "dcp_j7_ab.py"),
    ("DS-E1 KV measurement", "ds_e1_kv_measurements.sh"),
    ("benchmark runner", "run_benchmark.py"),
    ("seeding runner", "seed_specialist_routing.py"),
    ("seeding runner v2", "seed_specialist_routing_v2.py"),
    ("migration probe", "migration_probe.py"),
    ("placement fanout probe", "placement_fanout_probe.py"),
)
STRICT_RESTART_READINESS_COMMAND = (
    "cd /mnt/raid0/llm/epyc-orchestrator && "
    "python3 scripts/autopilot/restart_readiness_report.py "
    "--json --strict --require-seq-cutover --require-w6-audit"
)
STRICT_FABLE5_GATE_COMMAND = "python3 scripts/autopilot/fable5_gate_report.py --json --strict"
REQUIRED_XMAS_AB_POLICY = "incumbent_constrained_v1"


@dataclass(frozen=True)
class GateSection:
    """One Fable5 gate summary."""

    key: str
    status: str
    summary: str
    blockers: list[str]
    details: dict[str, Any]


def _load_json_object(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return loaded


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return loaded


def phase_section(phase_report: dict[str, Any]) -> GateSection:
    blockers = list(phase_report.get("blockers") or [])
    eval_progress = ""
    if phase_report.get("eval_total_questions") is not None:
        eval_progress = (
            f" ({phase_report.get('eval_label') or 'eval'} "
            f"{phase_report.get('eval_completed_questions')}/"
            f"{phase_report.get('eval_total_questions')})"
        )
    return GateSection(
        key="phase_health",
        status="ready" if phase_report.get("ok") else "blocked",
        summary=(
            f"AutoPilot phase heartbeat is {phase_report.get('status')} "
            f"at trial {phase_report.get('trial_id')} / {phase_report.get('phase')}"
            f"{eval_progress}."
        ),
        blockers=blockers,
        details={
            "ok": phase_report.get("ok"),
            "status": phase_report.get("status"),
            "trial_id": phase_report.get("trial_id"),
            "phase": phase_report.get("phase"),
            "action_type": phase_report.get("action_type"),
            "heartbeat_age_s": phase_report.get("heartbeat_age_s"),
            "pid": phase_report.get("pid"),
            "pid_alive": phase_report.get("pid_alive"),
            "eval_label": phase_report.get("eval_label"),
            "eval_completed_questions": phase_report.get("eval_completed_questions"),
            "eval_total_questions": phase_report.get("eval_total_questions"),
            "eval_correct_questions": phase_report.get("eval_correct_questions"),
            "eval_correct_pct": phase_report.get("eval_correct_pct"),
            "eval_concurrency": phase_report.get("eval_concurrency"),
        },
    )


def restart_section(restart_report: dict[str, Any]) -> GateSection:
    summary = restart_report.get("summary") or {}
    blockers = list(restart_report.get("blockers") or [])
    archive_authority = restart_report.get("archive_authority") or {}
    snapshot_replay = restart_report.get("snapshot_replay") or {}
    return GateSection(
        key="w4_w6_restart_cutover",
        status="ready" if restart_report.get("restart_ready") else "blocked",
        summary=(
            "W4/W6 strict restart cutover "
            f"{'is ready' if restart_report.get('restart_ready') else 'remains blocked'}."
        ),
        blockers=blockers,
        details={
            "restart_ready": restart_report.get("restart_ready"),
            "seq_cutover_ready": summary.get("seq_cutover_ready"),
            "seq_trusted_vector_trials": summary.get("seq_trusted_vector_trials"),
            "seq_min_trusted_vector_trials": summary.get(
                "seq_min_trusted_vector_trials"
            ),
            "seq_trusted_vector_trials_remaining": summary.get(
                "seq_trusted_vector_trials_remaining"
            ),
            "seq_shadow_rows": summary.get("seq_shadow_rows"),
            "seq_min_shadow_rows": summary.get("seq_min_shadow_rows"),
            "seq_shadow_rows_remaining": summary.get("seq_shadow_rows_remaining"),
            "w6_audit_cutover_ready": summary.get("w6_audit_cutover_ready"),
            "w6_audited_trial_count": summary.get("w6_audited_trial_count"),
            "w6_min_audited_trials": summary.get("w6_min_audited_trials"),
            "w6_audited_trial_count_remaining": summary.get(
                "w6_audited_trial_count_remaining"
            ),
            "w6_raw_audited_trial_count": summary.get("w6_raw_audited_trial_count"),
            "w6_trusted_audited_trial_count": summary.get(
                "w6_trusted_audited_trial_count"
            ),
            "w6_untrusted_audited_trial_count": summary.get(
                "w6_untrusted_audited_trial_count"
            ),
            "w6_untrusted_audited_trial_ids": summary.get(
                "w6_untrusted_audited_trial_ids"
            ),
            "w6_gaming_alarm": summary.get("w6_gaming_alarm"),
            "w6_potential_overfit_divergences": summary.get(
                "w6_potential_overfit_divergences"
            ),
            "baseline_seed_append_ready": summary.get("baseline_seed_append_ready"),
            "baseline_seed_append_expect_trial_counter": summary.get(
                "baseline_seed_append_expect_trial_counter"
            ),
            "baseline_seed_append_expect_journal_max_trial_id": summary.get(
                "baseline_seed_append_expect_journal_max_trial_id"
            ),
            "durable_journal_max_trial_id": archive_authority.get(
                "journal_max_trial_id"
            ),
            "state_trial_counter": archive_authority.get("state_trial_counter"),
            "snapshot_restart_readiness": summary.get("snapshot_restart_readiness"),
            "snapshot_payload_journal_max_trial_id": snapshot_replay.get(
                "payload_journal_max_trial_id"
            ),
        },
    )


def ds_e1_section(
    packet: dict[str, Any],
    *,
    clean_window: dict[str, Any] | None = None,
) -> GateSection:
    blockers = list(packet.get("blockers") or [])
    clean_window_report = clean_window or ds_e1_clean_window_report()
    clean_window_blockers = list(clean_window_report.get("blockers") or [])
    return GateSection(
        key="ds_e1_dynamic_stack",
        status="ready" if packet.get("ready_for_profile_decision") else "blocked",
        summary=(
            "DS-E1 dynamic-stack packet "
            f"{'is decision-ready' if packet.get('ready_for_profile_decision') else 'is not decision-ready'}."
        ),
        blockers=blockers,
        details={
            "ready_for_profile_decision": packet.get("ready_for_profile_decision"),
            "generated_at": packet.get("generated_at"),
            "section_statuses": {
                section.get("key"): section.get("status")
                for section in packet.get("sections") or []
                if isinstance(section, dict)
            },
            "clean_window_ready": clean_window_report.get("ready"),
            "clean_window_blockers": clean_window_blockers,
        },
    )


def ds_e1_clean_window_report() -> dict[str, Any]:
    """Report whether the DS-E1 KV harness can run without contaminating evidence."""
    blockers: list[str] = []
    autopilot = _pgrep("scripts/autopilot/autopilot.py start")
    llama = _pgrep_exact("llama-server")
    measurement_port_in_use = _tcp_port_accepting(DEFAULT_DS_E1_KV_PORT)
    if autopilot:
        blockers.append(f"active AutoPilot process(es): {_compact_processes(autopilot)}")
    if llama:
        blockers.append(f"live llama-server process(es): {_compact_processes(llama)}")
    if measurement_port_in_use:
        blockers.append(
            f"measurement port {DEFAULT_DS_E1_KV_PORT} is already accepting connections"
        )
    return {
        "ready": not blockers,
        "blockers": blockers,
        "active_autopilot": autopilot,
        "live_llama_server": llama,
        "measurement_port": DEFAULT_DS_E1_KV_PORT,
        "measurement_port_in_use": measurement_port_in_use,
    }


def _pgrep(pattern: str) -> list[str]:
    result = subprocess.run(
        ["pgrep", "-af", pattern],
        capture_output=True,
        text=True,
        check=False,
    )
    current_pid = str(os.getpid())
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and not line.startswith(f"{current_pid} ")
    ]


def _pgrep_exact(name: str) -> list[str]:
    result = subprocess.run(
        ["pgrep", "-a", "-x", name],
        capture_output=True,
        text=True,
        check=False,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _tcp_port_accepting(port: int, *, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


def _compact_processes(lines: list[str], *, limit: int = 4) -> str:
    if len(lines) <= limit:
        return "; ".join(lines)
    return "; ".join(lines[:limit]) + f"; ... +{len(lines) - limit} more"


def xmas_section(
    *,
    config_path: Path = DEFAULT_CLASSIFIER_CONFIG,
    candidate_table_path: Path = DEFAULT_XMAS_TABLE,
    ab_root: Path = DEFAULT_XMAS_AB_ROOT,
    quiet_window: dict[str, Any] | None = None,
) -> GateSection:
    blockers: list[str] = []
    quiet_window_report = quiet_window or xmas_quiet_window_report()
    details: dict[str, Any] = {
        "config_path": str(config_path),
        "candidate_table_path": str(candidate_table_path),
        "quiet_window_ready": quiet_window_report.get("ready"),
        "quiet_window_blockers": list(quiet_window_report.get("blockers") or []),
    }
    try:
        config = _load_yaml_mapping(config_path)
        raw = config.get("xmas_routing") or {}
        if not isinstance(raw, dict):
            raw = {}
            blockers.append("xmas_routing config is not a mapping")
    except Exception as exc:
        raw = {}
        blockers.append(f"failed to read X-MAS config: {exc}")

    mode = str(raw.get("mode", "off")).strip().lower()
    configured_table = str(raw.get("winner_table_path") or "").strip()
    details.update(
        {
            "mode": mode,
            "winner_table_path": configured_table,
            "require_complete_table": raw.get("require_complete_table"),
        }
    )

    config_errors = validate_xmas_config(config_path)
    details["config_validation_errors"] = config_errors
    if mode == "enforce":
        blockers.extend(config_errors)
    else:
        blockers.append(f"xmas_routing.mode is {mode}; enforce remains default-off")
        if not configured_table:
            blockers.append("xmas_routing.winner_table_path is not configured")

    if candidate_table_path.exists():
        table_errors = validate_xmas_table(
            candidate_table_path,
            require_evidence=True,
            require_function_axis=True,
        )
        details["candidate_table_errors"] = table_errors
        details["candidate_table_ready"] = not table_errors
    else:
        details["candidate_table_errors"] = ["candidate winner table is missing"]
        details["candidate_table_ready"] = False
        blockers.append(f"candidate X-MAS winner table is missing: {candidate_table_path}")

    ab_summary = _latest_xmas_ab_summary(ab_root)
    details["latest_ab_summary_path"] = (
        str(ab_summary["path"]) if ab_summary else None
    )
    details["latest_ab_decision_status"] = (
        ab_summary.get("decision_status") if ab_summary else None
    )
    details["latest_ab_score_delta"] = (
        ab_summary.get("score_delta_xmas_minus_baseline") if ab_summary else None
    )
    details["latest_ab_latency_ratio"] = (
        ab_summary.get("latency_ratio_xmas_over_baseline") if ab_summary else None
    )
    details["latest_ab_blockers"] = (
        ab_summary.get("decision_blockers") if ab_summary else []
    )
    details["latest_ab_policy"] = ab_summary.get("xmas_policy") if ab_summary else None
    details["required_ab_policy"] = REQUIRED_XMAS_AB_POLICY
    if not ab_summary:
        blockers.append("no X-MAS held-out A/B summary artifact was found")
    else:
        if ab_summary.get("xmas_policy") != REQUIRED_XMAS_AB_POLICY:
            blockers.append(
                "latest X-MAS held-out A/B policy is "
                f"{ab_summary.get('xmas_policy') or '<missing>'}; "
                f"required {REQUIRED_XMAS_AB_POLICY}"
            )
        if ab_summary.get("decision_status") != "promote_candidate":
            blockers.append(
                "latest X-MAS held-out A/B decision is "
                f"{ab_summary.get('decision_status') or '<missing>'}"
            )

    return GateSection(
        key="xmas_production_path",
        status="ready" if not blockers else "blocked",
        summary=(
            "X-MAS production routing "
            f"{'is enforce-ready' if not blockers else 'remains gated'}."
        ),
        blockers=blockers,
        details=details,
    )


def xmas_quiet_window_report() -> dict[str, Any]:
    """Report whether the X-MAS held-out A/B runner preflight can pass."""
    blockers: list[str] = []
    matches_by_label: dict[str, list[str]] = {}
    for label, pattern in XMAS_QUIET_WINDOW_PROCESS_PATTERNS:
        matches = _pgrep(pattern)
        if not matches:
            continue
        labeled_matches = matches_by_label.setdefault(label, [])
        for match in matches:
            if match not in labeled_matches:
                labeled_matches.append(match)

    for label, matches in matches_by_label.items():
        blockers.append(f"active {label} process(es): {_compact_processes(matches)}")

    return {
        "ready": not blockers,
        "blockers": blockers,
        "active_processes": matches_by_label,
    }


def _latest_xmas_ab_summary(root: Path) -> dict[str, Any] | None:
    """Return the newest parseable X-MAS live A/B summary."""
    candidates = sorted(root.glob("*/summary.json"), key=lambda path: path.stat().st_mtime)
    for path in reversed(candidates):
        try:
            loaded = _load_json_object(path)
        except Exception:
            continue
        decision = loaded.get("decision") or {}
        if not isinstance(decision, dict):
            decision = {}
        xmas_policy = loaded.get("xmas_policy")
        if not isinstance(xmas_policy, str) or not xmas_policy.strip():
            xmas_policy = "unknown_legacy"
        return {
            "path": path,
            "decision_status": decision.get("status"),
            "decision_blockers": list(decision.get("blockers") or []),
            "xmas_policy": xmas_policy.strip(),
            "score_delta_xmas_minus_baseline": loaded.get(
                "score_delta_xmas_minus_baseline"
            ),
            "latency_ratio_xmas_over_baseline": loaded.get(
                "latency_ratio_xmas_over_baseline"
            ),
        }
    return None


def build_fable5_gate_report(
    *,
    state: dict[str, Any],
    journal_rows: list[dict[str, Any]],
    phase_report: dict[str, Any],
    ds_e1_packet: dict[str, Any],
    config_path: Path = DEFAULT_CLASSIFIER_CONFIG,
    xmas_table_path: Path = DEFAULT_XMAS_TABLE,
    xmas_ab_root: Path = DEFAULT_XMAS_AB_ROOT,
) -> dict[str, Any]:
    sections = [
        phase_section(phase_report),
        restart_section(
            build_restart_readiness_report(
                state,
                journal_rows,
                require_seq_cutover=True,
                require_w6_audit=True,
            )
        ),
        ds_e1_section(ds_e1_packet),
        xmas_section(
            config_path=config_path,
            candidate_table_path=xmas_table_path,
            ab_root=xmas_ab_root,
        ),
    ]
    blockers = [
        f"{section.key}: {blocker}"
        for section in sections
        for blocker in section.blockers
    ]
    return {
        "ready": not blockers,
        "blockers": blockers,
        "sections": [asdict(section) for section in sections],
        "next_actions": build_next_actions(sections),
    }


def build_next_actions(sections: list[GateSection]) -> list[dict[str, Any]]:
    """Return deterministic operator next steps without changing gate semantics."""
    by_key = {section.key: section for section in sections}
    actions: list[dict[str, Any]] = []

    phase = by_key.get("phase_health")
    if phase and phase.status != "ready":
        actions.append(
            {
                "key": "recover_autopilot_phase",
                "priority": "P0",
                "status": "blocked",
                "reason": "AutoPilot phase health is not ready; do not trust evidence accrual until recovered.",
                "blocked_by": phase.blockers,
                "command": "python3 scripts/autopilot/phase_health_report.py --json",
            }
        )
        return actions

    restart = by_key.get("w4_w6_restart_cutover")
    if restart and restart.status != "ready":
        details = restart.details
        actions.append(
            {
                "key": "continue_w4_w6_accrual",
                "priority": "P0",
                "status": "active" if phase and phase.details.get("status") == "active" else "blocked",
                "reason": "Sequential authority and W6 audit cutover need more trusted rows before any flip.",
                "evidence": {
                    "trusted_vectors": details.get("seq_trusted_vector_trials"),
                    "trusted_vectors_required": details.get(
                        "seq_min_trusted_vector_trials"
                    ),
                    "trusted_vectors_remaining": details.get(
                        "seq_trusted_vector_trials_remaining"
                    ),
                    "seq_shadow_rows": details.get("seq_shadow_rows"),
                    "seq_shadow_rows_required": details.get("seq_min_shadow_rows"),
                    "seq_shadow_rows_remaining": details.get(
                        "seq_shadow_rows_remaining"
                    ),
                    "w6_audited_rows": details.get("w6_audited_trial_count"),
                    "w6_audited_rows_required": details.get("w6_min_audited_trials"),
                    "w6_audited_rows_remaining": details.get(
                        "w6_audited_trial_count_remaining"
                    ),
                    "w6_gaming_alarm": details.get("w6_gaming_alarm"),
                },
                "command": STRICT_RESTART_READINESS_COMMAND,
                "follow_up": STRICT_FABLE5_GATE_COMMAND,
            }
        )

    ds_e1 = by_key.get("ds_e1_dynamic_stack")
    if ds_e1 and ds_e1.status != "ready":
        details = ds_e1.details
        section_statuses = details.get("section_statuses") or {}
        if section_statuses.get("kv_size_measurements") != "ready":
            actions.append(
                {
                    "key": "run_ds_e1_kv_measurements",
                    "priority": "P0",
                    "status": "ready" if details.get("clean_window_ready") else "blocked",
                    "reason": "DS-E1 cannot decide DS-7/DS-6 profiles until direct production KV-size rows exist.",
                    "requires": "attested clean window with AutoPilot and live llama-server processes stopped/coordinated",
                    "blocked_by": details.get("clean_window_blockers") or [],
                    "command": (
                        "cd /mnt/raid0/llm/epyc-inference-research && "
                        "scripts/benchmark/ds_e1_kv_measurements.sh --execute"
                    ),
                    "follow_up": (
                        "cd /mnt/raid0/llm/epyc-orchestrator && "
                        "python3 scripts/server/dynamic_stack_evidence_packet.py "
                        "--output orchestration/reports/ds_e1_evidence_packet_20260620.md --strict"
                    ),
                }
            )
        if section_statuses.get("ri10_canary") != "ready":
            actions.append(
                {
                    "key": "collect_ri10_canary_arm_telemetry",
                    "priority": "P0",
                    "status": "active",
                    "reason": "RI-10 has enough high-risk samples, but fresh enforce/shadow canary-arm telemetry is still missing.",
                    "command": "python3 scripts/analysis/ri10_canary_sample_report.py",
                }
            )

    xmas = by_key.get("xmas_production_path")
    if xmas and xmas.status != "ready":
        quiet_window_blockers = list(xmas.details.get("quiet_window_blockers") or [])
        actions.append(
            {
                "key": "run_xmas_constrained_policy_ab",
                "priority": "P0",
                "status": "ready" if xmas.details.get("quiet_window_ready") else "blocked",
                "reason": "X-MAS enforce needs a fresh held-out A/B carrying incumbent_constrained_v1 and a promote_candidate verdict.",
                "requires": "attested quiet window; runner preflight refuses AutoPilot and competing benchmark coordinators",
                "blocked_by": quiet_window_blockers,
                "evidence_blockers": xmas.blockers,
                "prompt_manifest": DEFAULT_XMAS_HELDOUT_PROMPTS_ARG,
                "required_policy": REQUIRED_XMAS_AB_POLICY,
                "command": (
                    "cd /mnt/raid0/llm/epyc-orchestrator && "
                    "uv run python scripts/benchmark/xmas_live_ab.py "
                    f"--prompts {DEFAULT_XMAS_HELDOUT_PROMPTS_ARG} "
                    "--reps 2 --host-quiet-confirmed "
                    f"--output {DEFAULT_XMAS_CONSTRAINED_OUTPUT_ARG}"
                ),
            }
        )

    return actions


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Fable5 Gate Report",
        "",
        f"Ready: {str(report['ready']).lower()}",
        "",
    ]
    blockers = list(report.get("blockers") or [])
    if blockers:
        lines.extend(["## Blockers", ""])
        lines.extend(f"- {blocker}" for blocker in blockers)
        lines.append("")
    next_actions = list(report.get("next_actions") or [])
    if next_actions:
        lines.extend(["## Next Actions", ""])
        for action in next_actions:
            lines.extend(
                [
                    f"### {action.get('key')}",
                    "",
                    f"- Priority: `{action.get('priority')}`",
                    f"- Status: `{action.get('status')}`",
                    f"- Reason: {action.get('reason')}",
                ]
            )
            if action.get("requires"):
                lines.append(f"- Requires: {action['requires']}")
            if action.get("blocked_by"):
                lines.append("- Blocked by:")
                lines.extend(f"  - {blocker}" for blocker in action["blocked_by"])
            if action.get("evidence"):
                lines.append("- Evidence:")
                for key, value in action["evidence"].items():
                    lines.append(f"  - `{key}`: {json.dumps(value, sort_keys=True, default=str)}")
            if action.get("command"):
                lines.append(f"- Command: `{action['command']}`")
            if action.get("follow_up"):
                lines.append(f"- Follow-up: `{action['follow_up']}`")
            lines.append("")
    lines.extend(["## Sections", ""])
    for section in report.get("sections") or []:
        lines.extend(
            [
                f"### {section['key']}",
                "",
                f"- Status: `{section['status']}`",
                f"- Summary: {section['summary']}",
            ]
        )
        if section.get("blockers"):
            lines.append("- Blockers:")
            lines.extend(f"  - {blocker}" for blocker in section["blockers"])
        details = section.get("details") or {}
        if details:
            lines.append("- Details:")
            for key, value in details.items():
                lines.append(f"  - `{key}`: {json.dumps(value, sort_keys=True, default=str)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=STATE_PATH)
    parser.add_argument("--journal", type=Path, default=JOURNAL_PATH)
    parser.add_argument("--phase", type=Path, default=PHASE_PATH)
    parser.add_argument("--classifier-config", type=Path, default=DEFAULT_CLASSIFIER_CONFIG)
    parser.add_argument("--xmas-table", type=Path, default=DEFAULT_XMAS_TABLE)
    parser.add_argument("--xmas-ab-root", type=Path, default=DEFAULT_XMAS_AB_ROOT)
    parser.add_argument("--json", action="store_true", help="Emit structured JSON.")
    parser.add_argument("--out-json", type=Path, help="Write structured JSON to this path.")
    parser.add_argument("--out-md", type=Path, help="Write Markdown report to this path.")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when any gate blocks.")
    return parser.parse_args(argv)


def _write_text(path: Path, text: str) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        state = _load_json_object(args.state.expanduser().resolve())
        journal_rows = _load_jsonl(args.journal.expanduser().resolve())
        phase_report = build_phase_health_report(path=args.phase.expanduser().resolve())
        ds_e1_packet = build_ds_e1_packet()
        report = build_fable5_gate_report(
            state=state,
            journal_rows=journal_rows,
            phase_report=phase_report,
            ds_e1_packet=ds_e1_packet,
            config_path=args.classifier_config.expanduser().resolve(),
            xmas_table_path=args.xmas_table.expanduser().resolve(),
            xmas_ab_root=args.xmas_ab_root.expanduser().resolve(),
        )
    except Exception as exc:
        print(f"failed to build Fable5 gate report: {exc}", file=sys.stderr)
        return 2
    if args.out_json:
        _write_text(
            args.out_json,
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        )
    if args.out_md:
        _write_text(args.out_md, render_markdown(report))
    if args.json:
        print(json.dumps(report, sort_keys=True, default=str))
    else:
        print(render_markdown(report), end="")
    if args.strict and not report["ready"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
