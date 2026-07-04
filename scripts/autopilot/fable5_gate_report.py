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
import re
import socket
import subprocess
import sys
from typing import Any
import urllib.error
import urllib.request

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
from scripts.graph_router.offline_reward_pairwise_collection_status import (  # noqa: E402
    DEFAULT_MANIFEST as DEFAULT_A9_COLLECTION_MANIFEST,
    build_status as build_a9_collection_status,
)
from phase_status import PHASE_PATH, build_phase_health_report  # noqa: E402
from preflight_audit import JOURNAL_PATH, STATE_PATH, _load_jsonl  # noqa: E402
from restart_readiness_report import build_restart_readiness_report  # noqa: E402
from validate_xmas_winner_table import (  # noqa: E402
    DEFAULT_CLASSIFIER_CONFIG,
    validate_config as validate_xmas_config,
    validate_table as validate_xmas_table,
)
from w8_promotion_trajectory_report import build_w8_trajectory_report  # noqa: E402

DEFAULT_XMAS_TABLE = ORCH_ROOT / "orchestration" / "xmas_winner_table.yaml"
DEFAULT_XMAS_AB_ROOT = ORCH_ROOT / "benchmarks" / "results" / "runs" / "xmas_live_ab"
DEFAULT_A9_COLLECTION_SCRIPT = (
    ORCH_ROOT
    / "orchestration"
    / "reports"
    / "offline_reward_oracle_token_coverage_final_labels_20260621"
    / "collect_offline_reward_pairwise_expanded_gap.sh"
)
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
    "uv run python scripts/autopilot/restart_readiness_report.py "
    "--json --strict --require-seq-cutover --require-w6-audit --require-current-code"
)
STRICT_FABLE5_GATE_COMMAND = (
    "uv run python scripts/autopilot/fable5_gate_report.py --json --strict"
)
REQUIRED_XMAS_AB_POLICY = "incumbent_constrained_cheapfirst_v2"
TOOL_SENTINEL_ENV = "AUTOPILOT_TOOL_SENTINELS"
DEFAULT_CONFIG_ATTEST_URL = "http://localhost:8000/config/attest"


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


def _process_env(pid: Any) -> dict[str, str]:
    """Read a live process environment from /proc without raising on races."""
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return {}
    if pid_int <= 0:
        return {}
    try:
        raw = Path("/proc") / str(pid_int) / "environ"
        parts = raw.read_bytes().split(b"\0")
    except OSError:
        return {}
    env: dict[str, str] = {}
    for part in parts:
        if not part or b"=" not in part:
            continue
        key, value = part.split(b"=", 1)
        env[key.decode(errors="replace")] = value.decode(errors="replace")
    return env


def _config_attest(url: str = DEFAULT_CONFIG_ATTEST_URL) -> dict[str, Any]:
    """Fetch live API feature/env attestation; failure is advisory only."""
    try:
        with urllib.request.urlopen(url, timeout=1.0) as response:
            loaded = json.loads(response.read().decode("utf-8"))
    except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return {"error": str(exc)}
    return loaded if isinstance(loaded, dict) else {"error": "attest response is not an object"}


_TOOL_METRIC_KEYS = {
    "total_tool_calls",
    "tool_name_counts",
    "mean_tools_used",
    "tool_use_rate",
    "tool_helpfulness",
    "per_suite_tool_helpfulness",
}


def _tool_metrics_from_row(row: dict[str, Any]) -> dict[str, Any]:
    details = row.get("eval_details")
    if not isinstance(details, dict):
        return {}
    if not any(key in details for key in _TOOL_METRIC_KEYS):
        return {}
    return {
        "trial_id": row.get("trial_id"),
        "total_tool_calls": details.get("total_tool_calls"),
        "tool_name_counts": details.get("tool_name_counts"),
        "mean_tools_used": details.get("mean_tools_used"),
        "tool_use_rate": details.get("tool_use_rate"),
        "tool_helpfulness": details.get("tool_helpfulness"),
        "per_suite_tool_helpfulness": details.get("per_suite_tool_helpfulness"),
    }


def _tool_call_count(metrics: dict[str, Any]) -> int:
    try:
        return int(metrics.get("total_tool_calls") or 0)
    except (TypeError, ValueError):
        return 0


def _latest_tool_metrics(journal_rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in reversed(journal_rows):
        if not isinstance(row, dict):
            continue
        metrics = _tool_metrics_from_row(row)
        if metrics:
            return metrics
    return {}


def _recent_tool_metrics_summary(
    journal_rows: list[dict[str, Any]],
    *,
    window: int = 10,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for row in reversed(journal_rows):
        if not isinstance(row, dict):
            continue
        metrics = _tool_metrics_from_row(row)
        if not metrics:
            continue
        rows.append(metrics)
        if len(rows) >= window:
            break

    newest_first = rows
    chronological = list(reversed(newest_first))
    latest_nonzero = next(
        (metrics for metrics in newest_first if _tool_call_count(metrics) > 0),
        {},
    )
    return {
        "window": window,
        "evaluated_rows": len(chronological),
        "nonzero_rows": sum(
            1 for metrics in chronological if _tool_call_count(metrics) > 0
        ),
        "total_tool_calls": sum(
            _tool_call_count(metrics) for metrics in chronological
        ),
        "trial_ids": [
            metrics.get("trial_id")
            for metrics in chronological
            if metrics.get("trial_id") is not None
        ],
        "latest_nonzero_tool_metrics": latest_nonzero,
    }


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
            "process_started_at_s": phase_report.get("process_started_at_s"),
            "require_current_code": phase_report.get("require_current_code"),
            "code_stale": phase_report.get("code_stale"),
            "code_stale_paths": phase_report.get("code_stale_paths"),
            "eval_label": phase_report.get("eval_label"),
            "eval_completed_questions": phase_report.get("eval_completed_questions"),
            "eval_total_questions": phase_report.get("eval_total_questions"),
            "eval_correct_questions": phase_report.get("eval_correct_questions"),
            "eval_correct_pct": phase_report.get("eval_correct_pct"),
            "eval_concurrency": phase_report.get("eval_concurrency"),
            "planner_hints_enabled": phase_report.get("planner_hints_enabled"),
            "seq_verdict_enabled": phase_report.get("seq_verdict_enabled"),
            "w6_audit_accrual_enabled": phase_report.get("w6_audit_accrual_enabled"),
            "w6_audit_shadow_only": phase_report.get("w6_audit_shadow_only"),
            "w6_audit_n": phase_report.get("w6_audit_n"),
            "w6_audit_every_n_trials": phase_report.get("w6_audit_every_n_trials"),
            "autopilot_planner_timeout": phase_report.get("autopilot_planner_timeout"),
        },
    )


def tool_use_activation_section(
    *,
    phase_report: dict[str, Any],
    journal_rows: list[dict[str, Any]],
    api_attest: dict[str, Any] | None = None,
    autopilot_env: dict[str, str] | None = None,
    api_env: dict[str, str] | None = None,
) -> GateSection:
    """Report whether the planner-visible tool-use lane is actually active.

    This is advisory, not a hard Fable5 blocker: it exposes when StrategyStore can
    steer toward tool use but the measurement lane that proves tool execution is
    not enabled in the live API/AutoPilot pair.
    """
    if autopilot_env is None:
        autopilot_env = _process_env(phase_report.get("pid"))
    if api_attest is None:
        api_attest = _config_attest()
    flags = api_attest.get("flags") if isinstance(api_attest, dict) else {}
    if not isinstance(flags, dict):
        flags = {}
    api_pid = api_attest.get("pid") if isinstance(api_attest, dict) else None
    if api_env is None:
        api_env = _process_env(api_pid)

    latest_metrics = _latest_tool_metrics(journal_rows)
    recent_metrics = _recent_tool_metrics_summary(journal_rows)
    autopilot_sentinel = autopilot_env.get(TOOL_SENTINEL_ENV) == "1"
    api_sentinel = api_env.get(TOOL_SENTINEL_ENV) == "1"
    api_tools_ready = bool(flags.get("tools")) and bool(flags.get("repl"))
    structured_output_ready = bool(flags.get("structured_tool_output"))
    latest_total_calls = latest_metrics.get("total_tool_calls")
    recent_nonzero_tool_rows = int(recent_metrics.get("nonzero_rows") or 0)

    gaps: list[str] = []
    if not autopilot_sentinel:
        gaps.append(f"autopilot_env_missing_{TOOL_SENTINEL_ENV}")
    if not api_sentinel:
        gaps.append(f"api_env_missing_{TOOL_SENTINEL_ENV}")
    if flags and not api_tools_ready:
        gaps.append("api_tools_or_repl_not_enabled")
    if flags and not structured_output_ready:
        gaps.append("api_structured_tool_output_not_enabled")
    if latest_metrics and latest_total_calls == 0 and recent_nonzero_tool_rows == 0:
        gaps.append("latest_eval_total_tool_calls_zero")
    elif not latest_metrics:
        gaps.append("latest_eval_tool_metrics_missing")
    if isinstance(api_attest, dict) and api_attest.get("error"):
        gaps.append("api_config_attest_unavailable")

    status = "ready" if not gaps else "attention"
    summary = (
        "Tool-use planner hints are backed by active API/AutoPilot sentinel "
        "telemetry."
        if status == "ready"
        else "Tool-use planner hints are visible, but the live sentinel/telemetry lane is not fully active."
    )
    return GateSection(
        key="tool_use_activation",
        status=status,
        summary=summary,
        blockers=[],
        details={
            "autopilot_pid": phase_report.get("pid"),
            "api_pid": api_pid,
            "autopilot_tool_sentinels_enabled": autopilot_sentinel,
            "api_tool_sentinels_enabled": api_sentinel,
            "api_tools_enabled": flags.get("tools"),
            "api_repl_enabled": flags.get("repl"),
            "api_structured_tool_output_enabled": flags.get("structured_tool_output"),
            "activation_gaps": gaps,
            "latest_tool_metrics": latest_metrics,
            "recent_tool_metrics": recent_metrics,
            "config_attest_error": (
                api_attest.get("error") if isinstance(api_attest, dict) else None
            ),
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
            "archive_source_surface_ok": summary.get("archive_source_surface_ok"),
            "archive_source_surface_count": summary.get("archive_source_surface_count"),
            "archive_source_surface_failed_count": summary.get(
                "archive_source_surface_failed_count"
            ),
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
            "w8_promotion_status": summary.get("w8_promotion_status"),
            "w8_open_requirements": summary.get("w8_open_requirements"),
            "w8_pending_candidate": summary.get("w8_pending_candidate"),
            "w8_pending_source_trial_id": summary.get("w8_pending_source_trial_id"),
            "w8_pending_attempts": summary.get("w8_pending_attempts"),
            "w8_last_finalized_trial_id": summary.get("w8_last_finalized_trial_id"),
            "w8_last_finalized_candidate": summary.get("w8_last_finalized_candidate"),
            "w8_last_finalized_combined_E": summary.get(
                "w8_last_finalized_combined_E"
            ),
            "w8_last_finalized_delta_excludes_regression": summary.get(
                "w8_last_finalized_delta_excludes_regression"
            ),
            "w8_last_blocked_trial_id": summary.get("w8_last_blocked_trial_id"),
            "w8_last_blocked_candidate": summary.get("w8_last_blocked_candidate"),
            "w8_last_blocked_reason": summary.get("w8_last_blocked_reason"),
            "w8_latest_seq_trial_id": summary.get("w8_latest_seq_trial_id"),
            "w8_latest_candidate": summary.get("w8_latest_candidate"),
            "w8_latest_combined_E": summary.get("w8_latest_combined_E"),
            "w8_latest_required_E": summary.get("w8_latest_required_E"),
            "w8_latest_confirmed": summary.get("w8_latest_confirmed"),
            "w8_latest_seq_state": summary.get("w8_latest_seq_state"),
            "w8_latest_baseline_reference_state": summary.get(
                "w8_latest_baseline_reference_state"
            ),
            "w8_latest_fresh_eval": summary.get("w8_latest_fresh_eval"),
            "w8_baseline_reference_last_forced_trial_id": summary.get(
                "w8_baseline_reference_last_forced_trial_id"
            ),
            "w8_baseline_reference_last_forced_reason": summary.get(
                "w8_baseline_reference_last_forced_reason"
            ),
            "w8_baseline_reference_last_forced_stale": summary.get(
                "w8_baseline_reference_last_forced_stale"
            ),
            "w8_baseline_reference_blocked_trial_id": summary.get(
                "w8_baseline_reference_blocked_trial_id"
            ),
            "w8_baseline_reference_blocked_reason": summary.get(
                "w8_baseline_reference_blocked_reason"
            ),
            "w6_audit_cutover_ready": summary.get("w6_audit_cutover_ready"),
            "w6_audited_trial_count": summary.get("w6_audited_trial_count"),
            "w6_min_audited_trials": summary.get("w6_min_audited_trials"),
            "w6_audited_trial_count_remaining": summary.get(
                "w6_audited_trial_count_remaining"
            ),
            "w6_alarm_clearance_clean_trials_required": summary.get(
                "w6_alarm_clearance_clean_trials_required"
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
            "cutover_horizon_clean_trials_remaining": summary.get(
                "cutover_horizon_clean_trials_remaining"
            ),
            "cutover_horizon_blocker": summary.get("cutover_horizon_blocker"),
            "cutover_horizon_components": summary.get(
                "cutover_horizon_components"
            ),
            "baseline_seed_append_ready": summary.get("baseline_seed_append_ready"),
            "baseline_seed_append_required": summary.get(
                "baseline_seed_append_required"
            ),
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


def w8_trajectory_section(trajectory_report: dict[str, Any]) -> GateSection:
    """Surface W8 replay concentration without changing authority semantics."""
    concentration = trajectory_report.get("replay_concentration") or {}
    status_counts = trajectory_report.get("status_counts") or {}
    recent_active_candidates = trajectory_report.get("recent_active_candidates") or []
    stale_accumulating_candidates = (
        trajectory_report.get("stale_accumulating_candidates") or []
    )
    candidate_generation_required = _w8_candidate_generation_required(
        status_counts=status_counts,
        recent_active_candidates=recent_active_candidates,
        stale_accumulating_candidates=stale_accumulating_candidates,
    )
    blockers: list[str] = []
    warning_reason = concentration.get("warning_reason")
    if concentration.get("warning"):
        blockers.append(
            "replay_concentration_warning"
            + (f": {warning_reason}" if warning_reason else "")
        )
    status = "blocked" if blockers else "ready"
    return GateSection(
        key="w8_promotion_trajectory",
        status=status,
        summary=(
            "W8 replay trajectory "
            f"{'has concentration warnings' if blockers else 'has no concentration warning'}."
        ),
        blockers=blockers,
        details={
            "status": trajectory_report.get("status"),
            "ok": trajectory_report.get("ok"),
            "latest_trial_id": trajectory_report.get("latest_trial_id"),
            "snapshot_count": trajectory_report.get("snapshot_count"),
            "candidate_count": trajectory_report.get("candidate_count"),
            "status_counts": status_counts,
            "open_requirements": trajectory_report.get("open_requirements"),
            "candidate_generation_required": candidate_generation_required,
            "recent_active_candidates": recent_active_candidates,
            "stale_accumulating_candidate_count": len(stale_accumulating_candidates),
            "replay_concentration": concentration,
        },
    )


def _w8_candidate_generation_required(
    *,
    status_counts: dict[str, Any],
    recent_active_candidates: list[Any],
    stale_accumulating_candidates: list[Any],
) -> bool:
    """Return True when W8 has no replayable accumulating candidate surface."""
    if recent_active_candidates or stale_accumulating_candidates:
        return False
    replayable_statuses = {
        "active_recent_replay",
        "single_observation",
        "stale_accumulating",
        "confirmed_waiting_fresh_eval",
    }
    if any(int(status_counts.get(status) or 0) > 0 for status in replayable_statuses):
        return False
    terminal_statuses = {"reverted", "excluded", "refuted", "finalized"}
    return any(int(status_counts.get(status) or 0) > 0 for status in terminal_statuses)


def ds_e1_section(
    packet: dict[str, Any],
    *,
    clean_window: dict[str, Any] | None = None,
) -> GateSection:
    blockers = list(packet.get("blockers") or [])
    clean_window_report = clean_window or ds_e1_clean_window_report()
    clean_window_blockers = list(clean_window_report.get("blockers") or [])
    sections = [
        section for section in packet.get("sections") or [] if isinstance(section, dict)
    ]
    section_statuses = {
        section.get("key"): section.get("status")
        for section in sections
    }
    kv_section = next(
        (section for section in sections if section.get("key") == "kv_size_measurements"),
        {},
    )
    kv_details = kv_section.get("details") if isinstance(kv_section, dict) else {}
    if not isinstance(kv_details, dict):
        kv_details = {}
    ri10_section = next(
        (section for section in sections if section.get("key") == "ri10_canary"),
        {},
    )
    ri10_details = ri10_section.get("details") if isinstance(ri10_section, dict) else {}
    if not isinstance(ri10_details, dict):
        ri10_details = {}
    ri10_summary = ri10_details.get("report_summary")
    if not isinstance(ri10_summary, dict):
        ri10_summary = {}
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
            "section_statuses": section_statuses,
            "kv_required_measurements": kv_details.get("required_measurements"),
            "kv_observed_measurements": kv_details.get("observed_measurements"),
            "kv_missing_measurements": kv_details.get("missing_measurements"),
            "kv_expected_csv_columns": kv_details.get("expected_csv_columns"),
            "kv_candidate_paths": kv_details.get("paths"),
            "kv_searched_globs": kv_details.get("searched_globs"),
            "clean_window_ready": clean_window_report.get("ready"),
            "clean_window_blockers": clean_window_blockers,
            "ri10_telemetry_collection_blocker": ri10_details.get(
                "telemetry_collection_blocker"
            ),
            "ri10_telemetry_collection_reason": ri10_details.get(
                "telemetry_collection_reason"
            ),
            "ri10_canary_role_sample_deficit": ri10_details.get(
                "canary_role_sample_deficit_since_telemetry_health_start"
            ),
            "ri10_canary_arm_volume_deficit": ri10_details.get(
                "canary_arm_volume_deficit_since_telemetry_health_start"
            ),
            "ri10_canary_arm_balance_deficits": ri10_details.get(
                "canary_arm_balance_deficits_since_telemetry_health_start"
            ),
            "ri10_high_risk_by_role_current": ri10_summary.get(
                "high_risk_by_role_since_telemetry_health_start"
            ),
            "ri10_canary_role_high_risk_by_role_current": ri10_summary.get(
                "canary_role_high_risk_by_role_since_telemetry_health_start"
            ),
            "ri10_canary_arm_counts_current": ri10_summary.get(
                "canary_arm_counts_since_telemetry_health_start"
            ),
            "ri10_canary_arm_counts_by_role_current": ri10_summary.get(
                "canary_arm_counts_by_role_since_telemetry_health_start"
            ),
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
        if line.strip()
        and not line.startswith(f"{current_pid} ")
        and _process_line_matches_pattern(line, pattern)
    ]


def _process_line_matches_pattern(line: str, pattern: str) -> bool:
    """Return True when a pgrep row is an actual script invocation.

    Controller prompts can quote benchmark commands, so `pgrep -af script.py`
    may otherwise match the planner process instead of the script process.
    """
    parts = line.split()
    if len(parts) < 2:
        return False
    argv = parts[1:]
    pattern_parts = pattern.split()
    script_idx = next(
        (
            idx
            for idx, token in enumerate(pattern_parts)
            if re.search(r"\.(py|sh)$", Path(token).name)
        ),
        None,
    )
    if script_idx is None:
        return pattern in " ".join(argv)

    script_name = Path(pattern_parts[script_idx]).name
    trailing = pattern_parts[script_idx + 1 :]
    for idx, token in enumerate(argv):
        if Path(token).name != script_name:
            continue
        if not _script_token_has_runner_context(argv, idx):
            continue
        if trailing and any(part not in argv[idx + 1 :] for part in trailing):
            continue
        return True
    return False


def _script_token_has_runner_context(argv: list[str], idx: int) -> bool:
    token = argv[idx]
    if idx <= 1 or token.endswith(".sh"):
        return True
    prior = {Path(item).name for item in argv[max(0, idx - 3) : idx]}
    return bool(prior & {"python", "python3", "uv", "bash", "sh"})


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


def a9_collection_section(
    manifest_path: Path = DEFAULT_A9_COLLECTION_MANIFEST,
) -> GateSection:
    """Surface the guarded A9 pairwise source-acquisition window."""
    status = build_a9_collection_status(manifest_path)
    blockers = list(status.get("blockers") or [])
    ready = bool(status.get("ready"))
    status_label = str(status.get("status") or "unknown")
    section_status = "ready" if ready else "blocked"
    if status_label == "no_runnable_batches":
        section_status = "attention"
    return GateSection(
        key="a9_pairwise_collection",
        status=section_status,
        summary=(
            "A9 pairwise source-acquisition window "
            f"is {status_label} with {status.get('batch_count', 0)} batch(es)."
        ),
        blockers=blockers,
        details={
            "ready": ready,
            "status": status_label,
            "manifest_path": status.get("manifest_path"),
            "manifest_schema_version": status.get("manifest_schema_version"),
            "source_plan_decision": status.get("source_plan_decision"),
            "batch_count": status.get("batch_count"),
            "post_collection_step_count": status.get("post_collection_step_count"),
            "autopilot_guard": status.get("autopilot_guard"),
            "blockers": blockers,
            "warnings": list(status.get("warnings") or []),
        },
    )


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
    details["latest_ab_results_path"] = (
        str(ab_summary["path"].with_name("results.jsonl")) if ab_summary else None
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
    latest_ab_ready = False
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
        latest_ab_ready = (
            ab_summary.get("xmas_policy") == REQUIRED_XMAS_AB_POLICY
            and ab_summary.get("decision_status") == "promote_candidate"
        )
    details["latest_ab_ready"] = latest_ab_ready

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
    a9_collection_manifest: Path = DEFAULT_A9_COLLECTION_MANIFEST,
    include_tool_use_activation: bool = True,
) -> dict[str, Any]:
    sections = [
        phase_section(phase_report),
        restart_section(
            build_restart_readiness_report(
                state,
                journal_rows,
                require_seq_cutover=True,
                require_w6_audit=True,
                require_current_code=bool(phase_report.get("require_current_code")),
            )
        ),
        w8_trajectory_section(build_w8_trajectory_report(journal_rows)),
        ds_e1_section(ds_e1_packet),
        a9_collection_section(a9_collection_manifest),
        xmas_section(
            config_path=config_path,
            candidate_table_path=xmas_table_path,
            ab_root=xmas_ab_root,
        ),
    ]
    if include_tool_use_activation:
        sections.append(
            tool_use_activation_section(
                phase_report=phase_report,
                journal_rows=journal_rows,
            )
        )
    blockers = [
        f"{section.key}: {blocker}"
        for section in sections
        for blocker in section.blockers
    ]
    next_actions = build_next_actions(sections)
    return {
        "ready": not blockers,
        "summary": build_report_summary(sections, blockers, next_actions),
        "blockers": blockers,
        "sections": [asdict(section) for section in sections],
        "next_actions": next_actions,
    }


def build_report_summary(
    sections: list[GateSection],
    blockers: list[str],
    next_actions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return compact read-only status for dashboards and handoff triage."""
    by_key = {section.key: section for section in sections}
    restart = by_key.get("w4_w6_restart_cutover")
    w8_trajectory = by_key.get("w8_promotion_trajectory")
    ds_e1 = by_key.get("ds_e1_dynamic_stack")
    a9 = by_key.get("a9_pairwise_collection")
    xmas = by_key.get("xmas_production_path")
    phase = by_key.get("phase_health")
    tool_use = by_key.get("tool_use_activation")
    return {
        "ready": not blockers,
        "blocker_count": len(blockers),
        "blocked_sections": [
            section.key
            for section in sections
            if section.blockers or section.status == "blocked"
        ],
        "section_statuses": {section.key: section.status for section in sections},
        "next_action_keys": [str(action.get("key")) for action in next_actions],
        "next_action_statuses": {
            str(action.get("key")): action.get("status") for action in next_actions
        },
        "active_next_action_keys": [
            str(action.get("key"))
            for action in next_actions
            if action.get("status") == "active"
        ],
        "blocked_next_action_keys": [
            str(action.get("key"))
            for action in next_actions
            if action.get("status") == "blocked"
        ],
        "ready_next_action_keys": [
            str(action.get("key"))
            for action in next_actions
            if action.get("status") == "ready"
        ],
        "phase_status": phase.details.get("status") if phase else None,
        "phase_trial_id": phase.details.get("trial_id") if phase else None,
        "phase_action_type": phase.details.get("action_type") if phase else None,
        "restart_ready": (
            restart.details.get("restart_ready") if restart is not None else None
        ),
        "w8_promotion_status": (
            restart.details.get("w8_promotion_status") if restart is not None else None
        ),
        "w8_open_requirements": (
            restart.details.get("w8_open_requirements") if restart is not None else None
        ),
        "w8_latest_seq_trial_id": (
            restart.details.get("w8_latest_seq_trial_id") if restart is not None else None
        ),
        "w8_latest_combined_E": (
            restart.details.get("w8_latest_combined_E") if restart is not None else None
        ),
        "w8_latest_required_E": (
            restart.details.get("w8_latest_required_E") if restart is not None else None
        ),
        "w8_latest_seq_state": (
            restart.details.get("w8_latest_seq_state") if restart is not None else None
        ),
        "w8_latest_fresh_eval": (
            restart.details.get("w8_latest_fresh_eval") if restart is not None else None
        ),
        "w8_replay_concentration_warning": (
            (w8_trajectory.details.get("replay_concentration") or {}).get("warning")
            if w8_trajectory is not None
            else None
        ),
        "w8_replay_top_active_candidate": (
            (w8_trajectory.details.get("replay_concentration") or {}).get(
                "top_active_candidate"
            )
            if w8_trajectory is not None
            else None
        ),
        "w8_replay_stale_accumulating_count": (
            w8_trajectory.details.get("stale_accumulating_candidate_count")
            if w8_trajectory is not None
            else None
        ),
        "ds_e1_ready_for_profile_decision": (
            ds_e1.details.get("ready_for_profile_decision") if ds_e1 is not None else None
        ),
        "ds_e1_clean_window_ready": (
            ds_e1.details.get("clean_window_ready") if ds_e1 is not None else None
        ),
        "ds_e1_clean_window_blockers": (
            ds_e1.details.get("clean_window_blockers") if ds_e1 is not None else None
        ),
        "a9_collection_status": (
            a9.details.get("status") if a9 is not None else None
        ),
        "a9_collection_ready": (
            a9.details.get("ready") if a9 is not None else None
        ),
        "a9_collection_batch_count": (
            a9.details.get("batch_count") if a9 is not None else None
        ),
        "a9_collection_blockers": (
            a9.details.get("blockers") if a9 is not None else None
        ),
        "xmas_mode": xmas.details.get("mode") if xmas is not None else None,
        "xmas_quiet_window_ready": (
            xmas.details.get("quiet_window_ready") if xmas is not None else None
        ),
        "xmas_latest_ab_policy": (
            xmas.details.get("latest_ab_policy") if xmas is not None else None
        ),
        "xmas_latest_ab_decision_status": (
            xmas.details.get("latest_ab_decision_status") if xmas is not None else None
        ),
        "tool_use_activation_status": (
            tool_use.status if tool_use is not None else None
        ),
        "tool_use_activation_gaps": (
            tool_use.details.get("activation_gaps") if tool_use is not None else None
        ),
        "tool_use_latest_total_tool_calls": (
            (tool_use.details.get("latest_tool_metrics") or {}).get(
                "total_tool_calls"
            )
            if tool_use is not None
            else None
        ),
        "tool_use_recent_nonzero_rows": (
            (tool_use.details.get("recent_tool_metrics") or {}).get("nonzero_rows")
            if tool_use is not None
            else None
        ),
        "tool_use_recent_total_tool_calls": (
            (tool_use.details.get("recent_tool_metrics") or {}).get(
                "total_tool_calls"
            )
            if tool_use is not None
            else None
        ),
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
                "command": "uv run python scripts/autopilot/phase_health_report.py --json",
            }
        )
        return actions

    restart = by_key.get("w4_w6_restart_cutover")
    if restart and restart.status != "ready":
        details = restart.details
        if details.get("baseline_seed_append_required"):
            command_parts = [
                "cd /mnt/raid0/llm/epyc-orchestrator &&",
                "uv run python scripts/autopilot/baseline_authority_seed.py",
                "--append",
            ]
            expected_trial_counter = details.get(
                "baseline_seed_append_expect_trial_counter"
            )
            expected_journal_max = details.get(
                "baseline_seed_append_expect_journal_max_trial_id"
            )
            if expected_trial_counter is not None:
                command_parts.extend(
                    ["--expect-trial-counter", str(expected_trial_counter)]
                )
            if expected_journal_max is not None:
                command_parts.extend(
                    ["--expect-journal-max-trial-id", str(expected_journal_max)]
                )
            command_parts.append("--json")
            autopilot_active = bool(phase and phase.details.get("status") == "active")
            actions.append(
                {
                    "key": "append_baseline_seed_event",
                    "priority": "P0",
                    "status": "blocked" if autopilot_active else "ready",
                    "reason": (
                        "Baseline-as-fold authority has a ready seed preflight; "
                        "append the guarded seed event before restart cutover."
                    ),
                    "blocked_by": (
                        ["active AutoPilot process; seed tool refuses live append"]
                        if autopilot_active
                        else []
                    ),
                    "evidence": {
                        "baseline_seed_append_ready": details.get(
                            "baseline_seed_append_ready"
                        ),
                        "baseline_seed_append_required": details.get(
                            "baseline_seed_append_required"
                        ),
                        "expect_trial_counter": expected_trial_counter,
                        "expect_journal_max_trial_id": expected_journal_max,
                    },
                    "command": " ".join(command_parts),
                    "follow_up": STRICT_RESTART_READINESS_COMMAND,
                }
            )
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
                    "w6_alarm_clearance_clean_trials_required": details.get(
                        "w6_alarm_clearance_clean_trials_required"
                    ),
                    "cutover_horizon_clean_trials_remaining": details.get(
                        "cutover_horizon_clean_trials_remaining"
                    ),
                    "cutover_horizon_blocker": details.get(
                        "cutover_horizon_blocker"
                    ),
                    "cutover_horizon_components": details.get(
                        "cutover_horizon_components"
                    ),
                },
                "command": STRICT_RESTART_READINESS_COMMAND,
                "follow_up": STRICT_FABLE5_GATE_COMMAND,
            }
        )

    if restart and restart.status == "ready":
        details = restart.details
        if details.get("w8_promotion_status") != "finalized":
            w8_trajectory = by_key.get("w8_promotion_trajectory")
            concentration = (
                (w8_trajectory.details.get("replay_concentration") or {})
                if w8_trajectory is not None
                else {}
            )
            candidate_generation_required = (
                bool(w8_trajectory.details.get("candidate_generation_required"))
                if w8_trajectory is not None
                else False
            )
            actions.append(
                {
                    "key": "collect_w8_promotion_eval_evidence",
                    "priority": "P0",
                    "status": (
                        "active"
                        if phase and phase.details.get("status") == "active"
                        else "blocked"
                    ),
                    "reason": (
                        "W4/W6 authority is restart-ready; W8 needs a new "
                        "keepable candidate before replay/promotion evidence can accrue."
                        if candidate_generation_required
                        else (
                            "W4/W6 authority is restart-ready; W8 still needs live "
                            "promotion-eval finalization evidence before closing the tail."
                        )
                    ),
                    "evidence": {
                        "w8_promotion_status": details.get("w8_promotion_status"),
                        "open_requirements": details.get("w8_open_requirements"),
                        "pending_candidate": details.get("w8_pending_candidate"),
                        "pending_source_trial_id": details.get(
                            "w8_pending_source_trial_id"
                        ),
                        "pending_attempts": details.get("w8_pending_attempts"),
                        "last_blocked_trial_id": details.get(
                            "w8_last_blocked_trial_id"
                        ),
                        "last_blocked_candidate": details.get(
                            "w8_last_blocked_candidate"
                        ),
                        "last_blocked_reason": details.get("w8_last_blocked_reason"),
                        "latest_seq_trial_id": details.get("w8_latest_seq_trial_id"),
                        "latest_candidate": details.get("w8_latest_candidate"),
                        "latest_combined_E": details.get("w8_latest_combined_E"),
                        "latest_required_E": details.get("w8_latest_required_E"),
                        "latest_confirmed": details.get("w8_latest_confirmed"),
                        "latest_fresh_eval": details.get("w8_latest_fresh_eval"),
                        "latest_seq_state": details.get("w8_latest_seq_state"),
                        "latest_baseline_reference_state": details.get(
                            "w8_latest_baseline_reference_state"
                        ),
                        "baseline_reference_last_forced_trial_id": details.get(
                            "w8_baseline_reference_last_forced_trial_id"
                        ),
                        "baseline_reference_last_forced_reason": details.get(
                            "w8_baseline_reference_last_forced_reason"
                        ),
                        "baseline_reference_blocked_reason": details.get(
                            "w8_baseline_reference_blocked_reason"
                        ),
                        "candidate_generation_required": (
                            candidate_generation_required
                        ),
                        "candidate_status_counts": (
                            w8_trajectory.details.get("status_counts")
                            if w8_trajectory is not None
                            else None
                        ),
                        "recent_active_candidates": (
                            w8_trajectory.details.get("recent_active_candidates")
                            if w8_trajectory is not None
                            else None
                        ),
                        "replay_concentration_warning": concentration.get("warning"),
                        "replay_concentration_reason": concentration.get(
                            "warning_reason"
                        ),
                        "replay_top_active_candidate": concentration.get(
                            "top_active_candidate"
                        ),
                        "replay_top_active_attempt_share": concentration.get(
                            "top_active_attempt_share"
                        ),
                        "replay_stale_accumulating_count": (
                            w8_trajectory.details.get(
                                "stale_accumulating_candidate_count"
                            )
                            if w8_trajectory is not None
                            else None
                        ),
                    },
                    "command": (
                        "cd /mnt/raid0/llm/epyc-orchestrator && "
                        "uv run python scripts/autopilot/w8_promotion_trajectory_report.py "
                        "--journal orchestration"
                    ),
                    "follow_up": STRICT_FABLE5_GATE_COMMAND,
                }
            )

    tool_use = by_key.get("tool_use_activation")
    if tool_use and tool_use.status != "ready":
        phase_active = bool(phase and phase.details.get("status") == "active")
        activation_gaps = list(tool_use.details.get("activation_gaps") or [])
        sentinels_active = bool(
            tool_use.details.get("autopilot_tool_sentinels_enabled")
            and tool_use.details.get("api_tool_sentinels_enabled")
            and tool_use.details.get("api_tools_enabled")
            and tool_use.details.get("api_repl_enabled")
        )
        evidence = {
            "activation_gaps": activation_gaps,
            "autopilot_tool_sentinels_enabled": tool_use.details.get(
                "autopilot_tool_sentinels_enabled"
            ),
            "api_tool_sentinels_enabled": tool_use.details.get(
                "api_tool_sentinels_enabled"
            ),
            "api_tools_enabled": tool_use.details.get("api_tools_enabled"),
            "api_repl_enabled": tool_use.details.get("api_repl_enabled"),
            "latest_tool_metrics": tool_use.details.get("latest_tool_metrics"),
            "recent_tool_metrics": tool_use.details.get("recent_tool_metrics"),
        }
        if sentinels_active and activation_gaps and all(
            gap
            in {
                "latest_eval_total_tool_calls_zero",
                "latest_eval_tool_metrics_missing",
            }
            for gap in activation_gaps
        ):
            actions.append(
                {
                    "key": "collect_tool_use_sentinel_journal_evidence",
                    "priority": "P0",
                    "status": "active" if phase_active else "ready",
                    "reason": (
                        "The API and AutoPilot tool-sentinel lane is active; "
                        "the remaining gap is waiting for a sentinel-enabled "
                        "eval to journal nonzero tool telemetry."
                    ),
                    "requires": (
                        "one completed AutoPilot eval with "
                        "AUTOPILOT_TOOL_SENTINELS=1 and tool-use sentinels loaded"
                    ),
                    "blocked_by": [],
                    "evidence": evidence,
                    "command": (
                        "Let the current sentinel-enabled AutoPilot eval finish, "
                        "then rerun "
                        "uv run python scripts/autopilot/fable5_gate_report.py "
                        "--json --strict"
                    ),
                    "follow_up": STRICT_FABLE5_GATE_COMMAND,
                }
            )
        else:
            actions.append(
                {
                    "key": "activate_tool_use_sentinel_lane",
                    "priority": "P0",
                    "status": "blocked" if phase_active else "ready",
                    "reason": (
                        "StrategyStore already exposes tool-use hints to the planner; "
                        "the remaining gap is activating the API and AutoPilot "
                        "tool-sentinel telemetry lane so tool use is measured."
                    ),
                    "requires": (
                        "coordinated API reload plus AutoPilot restart at a trial "
                        "boundary; this changes the active eval mix"
                    ),
                    "blocked_by": (
                        [
                            "active AutoPilot process; "
                            "wait for a controlled trial boundary"
                        ]
                        if phase_active
                        else []
                    ),
                    "evidence": evidence,
                    "command": (
                        "At a controlled trial boundary, reload the orchestrator API "
                        "with AUTOPILOT_TOOL_SENTINELS=1, restart AutoPilot with "
                        "AUTOPILOT_TOOL_SENTINELS=1 plus the existing W4/W6/planner "
                        "env, then run AUTOPILOT_TOOL_SENTINELS=1 uv run python "
                        "scripts/autopilot/gate3_tool_telemetry.py"
                    ),
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
                        "uv run python scripts/server/dynamic_stack_evidence_packet.py "
                        "--output orchestration/reports/"
                        "ds_e1_evidence_packet_$(date -u +%Y%m%dT%H%M%SZ).md "
                        "--strict"
                    ),
                }
            )
        if section_statuses.get("ri10_canary") != "ready":
            ri10_evidence = {
                key.removeprefix("ri10_"): value
                for key, value in details.items()
                if key.startswith("ri10_") and value is not None
            }
            actions.append(
                {
                    "key": "collect_ri10_canary_arm_telemetry",
                    "priority": "P0",
                    "status": "active",
                    "reason": "RI-10 has raw high-risk samples, but arm-attributed canary telemetry is not yet decision-grade.",
                    "evidence": ri10_evidence,
                    "command": "uv run python scripts/analysis/ri10_canary_sample_report.py",
                }
            )

    xmas = by_key.get("xmas_production_path")
    if xmas and xmas.status != "ready":
        latest_policy = xmas.details.get("latest_ab_policy")
        latest_decision = xmas.details.get("latest_ab_decision_status")
        latest_results_path = xmas.details.get("latest_ab_results_path")
        latest_summary_path = xmas.details.get("latest_ab_summary_path")
        if xmas.details.get("latest_ab_ready"):
            deployment_blockers = [
                blocker
                for blocker in xmas.blockers
                if not blocker.startswith("xmas_routing.mode is ")
            ]
            actions.append(
                {
                    "key": "decide_xmas_enforce_enablement",
                    "priority": "P0",
                    "status": "blocked" if deployment_blockers else "ready",
                    "reason": (
                        "The repaired X-MAS held-out A/B passed; the remaining "
                        "gate is an explicit production enablement, reload, and "
                        "attestation decision."
                    ),
                    "blocked_by": deployment_blockers,
                    "evidence_blockers": xmas.blockers,
                    "latest_ab_summary_path": latest_summary_path,
                    "latest_ab_results_path": latest_results_path,
                    "latest_ab_decision_status": latest_decision,
                    "latest_ab_score_delta": xmas.details.get(
                        "latest_ab_score_delta"
                    ),
                    "latest_ab_latency_ratio": xmas.details.get(
                        "latest_ab_latency_ratio"
                    ),
                    "required_policy": REQUIRED_XMAS_AB_POLICY,
                    "follow_up": (
                        "If accepted, set xmas_routing.mode=enforce through the "
                        "normal config/reload path and rerun runtime attestation; "
                        "do not rerun the held-out A/B unless the policy/table "
                        "changes."
                    ),
                    "command": (
                        "cd /mnt/raid0/llm/epyc-orchestrator && "
                        "uv run python scripts/benchmark/xmas_live_ab.py "
                        f"--summarize-results {latest_results_path or '<results.jsonl>'} "
                        "--output /tmp/xmas-constrained-policy-diagnostics"
                    ),
                }
            )
        elif latest_policy == REQUIRED_XMAS_AB_POLICY and latest_decision not in (
            None,
            "promote_candidate",
        ):
            actions.append(
                {
                    "key": "diagnose_xmas_policy_regressions",
                    "priority": "P0",
                    "status": "ready",
                    "reason": (
                        "The latest X-MAS held-out A/B used the required "
                        "incumbent-constrained policy and did not promote; "
                        "repair the policy/table regressions before rerunning."
                    ),
                    "evidence_blockers": xmas.blockers,
                    "latest_ab_summary_path": xmas.details.get(
                        "latest_ab_summary_path"
                    ),
                    "latest_ab_results_path": latest_results_path,
                    "latest_ab_decision_status": latest_decision,
                    "latest_ab_score_delta": xmas.details.get(
                        "latest_ab_score_delta"
                    ),
                    "latest_ab_latency_ratio": xmas.details.get(
                        "latest_ab_latency_ratio"
                    ),
                    "command": (
                        "cd /mnt/raid0/llm/epyc-orchestrator && "
                        "uv run python scripts/benchmark/xmas_live_ab.py "
                        f"--summarize-results {latest_results_path or '<results.jsonl>'} "
                        "--output /tmp/xmas-constrained-policy-diagnostics"
                    ),
                }
            )
            return actions
        else:
            quiet_window_blockers = list(
                xmas.details.get("quiet_window_blockers") or []
            )
            actions.append(
                {
                    "key": "run_xmas_constrained_policy_ab",
                    "priority": "P0",
                    "status": (
                        "ready" if xmas.details.get("quiet_window_ready") else "blocked"
                    ),
                    "reason": (
                        "X-MAS enforce needs a fresh held-out A/B carrying "
                        f"{REQUIRED_XMAS_AB_POLICY} and a promote_candidate verdict."
                    ),
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

    a9 = by_key.get("a9_pairwise_collection")
    if a9 is not None:
        if a9.details.get("status") == "no_runnable_batches":
            actions.append(
                {
                    "key": "revise_a9_reward_oracle_or_reference_source",
                    "priority": "P1",
                    "status": "active",
                    "reason": (
                        "A9 clean-window acquisition is exhausted for the "
                        "current reference_token_coverage oracle; remaining "
                        "instruction-precision targets have no reference text "
                        "for that scorer."
                    ),
                    "requires": (
                        "materially different scorer/feature design or a "
                        "reference-bearing instruction-following source"
                    ),
                    "blocked_by": [],
                    "manifest": a9.details.get("manifest_path"),
                    "batch_count": a9.details.get("batch_count"),
                    "post_collection_step_count": a9.details.get(
                        "post_collection_step_count"
                    ),
                    "source_plan_decision": a9.details.get("source_plan_decision"),
                    "follow_up": (
                        "Do not rerun the current collection script; regenerate "
                        "A9 only after the oracle/source contract changes."
                    ),
                }
            )
        else:
            actions.append(
                {
                    "key": "run_a9_pairwise_collection_window",
                    "priority": "P1",
                    "status": "ready" if a9.details.get("ready") else "blocked",
                    "reason": (
                        "A9 offline reward-oracle pairwise holdouts need the "
                        "guarded priority-0/1 collection window before another "
                        "pairwise contract rebuild."
                    ),
                    "requires": "coordinated clean window; collection script refuses active AutoPilot",
                    "blocked_by": a9.blockers,
                    "manifest": a9.details.get("manifest_path"),
                    "batch_count": a9.details.get("batch_count"),
                    "post_collection_step_count": a9.details.get(
                        "post_collection_step_count"
                    ),
                    "source_plan_decision": a9.details.get("source_plan_decision"),
                    "command": (
                        "cd /mnt/raid0/llm/epyc-orchestrator && "
                        f"{DEFAULT_A9_COLLECTION_SCRIPT.relative_to(ORCH_ROOT)}"
                    ),
                    "follow_up": (
                        "cd /mnt/raid0/llm/epyc-orchestrator && "
                        "uv run python scripts/graph_router/offline_reward_pairwise_collection_status.py"
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
    parser.add_argument(
        "--require-current-code",
        action="store_true",
        help=(
            "Block when the live AutoPilot process predates runtime source changes. "
            "Enabled automatically by --strict."
        ),
    )
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
        phase_report = build_phase_health_report(
            path=args.phase.expanduser().resolve(),
            require_current_code=args.require_current_code or args.strict,
        )
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
