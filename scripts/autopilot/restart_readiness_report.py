#!/usr/bin/env python3
"""Read-only AutoPilot restart readiness report."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ORCH_ROOT))

from archive_authority_report import build_archive_authority_report  # noqa: E402
from audit_block_report import build_report as build_audit_block_report  # noqa: E402
from baseline_authority_report import build_baseline_authority_report  # noqa: E402
from baseline_authority_seed import build_baseline_seed_event  # noqa: E402
from preflight_audit import JOURNAL_PATH, STATE_PATH, _load_jsonl  # noqa: E402
from seq_readiness_report import build_seq_readiness_report  # noqa: E402
from src.autopilot_core.journal_snapshot_replay import (  # noqa: E402
    archive_payload_from_verified_snapshot,
    build_snapshot_replay_diagnostic,
)


def _ledger_events(journal_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in journal_rows
        if isinstance(row, dict) and row.get("type") and "trial_id" not in row
    ]


def _diagnostic_dict(diagnostic: Any) -> dict[str, Any]:
    if is_dataclass(diagnostic):
        return asdict(diagnostic)
    return dict(vars(diagnostic))


def _snapshot_restart_report(journal_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ledger_events = _ledger_events(journal_rows)
    diagnostic = build_snapshot_replay_diagnostic(journal_rows, ledger_events)
    payload = archive_payload_from_verified_snapshot(journal_rows, ledger_events)
    if diagnostic.bounded_replay_readiness == "current":
        readiness = "current"
    elif payload is not None:
        readiness = "tail_fold_ready"
    else:
        readiness = diagnostic.bounded_replay_readiness
    return {
        "ok": payload is not None,
        "restart_readiness": readiness,
        "payload_available": payload is not None,
        "payload_journal_max_trial_id": (
            payload.get("journal_max_trial_id") if isinstance(payload, dict) else None
        ),
        "diagnostic": _diagnostic_dict(diagnostic),
    }


def _baseline_restart_report(
    state: dict[str, Any],
    journal_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    report = build_baseline_authority_report(state, journal_rows)
    state_cache_present = isinstance(state.get("baseline_state"), dict)
    startup_safe = bool(report.get("ok")) or state_cache_present
    authority_source = (
        "ledger_fold"
        if report.get("ok")
        else "state_baseline"
        if state_cache_present
        else "missing"
    )
    seed_preflight = _baseline_seed_preflight(
        state,
        journal_rows,
        ledger_fold_ready=bool(report.get("ok")),
    )
    return {
        **report,
        "startup_safe": startup_safe,
        "authority_source": authority_source,
        "state_baseline_present": state_cache_present,
        "seed_preflight": seed_preflight,
    }


def _baseline_seed_preflight(
    state: dict[str, Any],
    journal_rows: list[dict[str, Any]],
    *,
    ledger_fold_ready: bool,
) -> dict[str, Any]:
    if ledger_fold_ready:
        return {
            "status": "ledger_fold_ready",
            "append_required": False,
            "append_ready": False,
            "warning": "",
        }
    result = build_baseline_seed_event(state, journal_rows)
    event = result.event or {}
    proof = event.get("proof") if isinstance(event.get("proof"), dict) else {}
    after = result.after or {}
    return {
        "status": result.status,
        "append_required": result.status == "ready",
        "append_ready": result.status == "ready",
        "warning": result.warning,
        "before": result.before,
        "after": result.after,
        "post_append_cutover_ready": after.get("cutover_ready"),
        "event_source_trial_id": event.get("source_trial_id"),
        "event_tier": event.get("tier"),
        "event_new_quality": event.get("new_quality"),
        "append_expect_trial_counter": proof.get("state_trial_counter"),
        "append_expect_journal_max_trial_id": proof.get("journal_max_trial_id"),
    }


def _w6_audit_restart_report(
    journal_rows: list[dict[str, Any]],
    *,
    min_audited_trials: int,
) -> dict[str, Any]:
    report = build_audit_block_report(journal_rows, alarm_window=min_audited_trials)
    audited_trial_count = int(report.get("audited_trial_count") or 0)
    gaming_alarm = bool(report.get("gaming_alarm"))
    blockers: list[str] = []
    if audited_trial_count < min_audited_trials:
        blockers.append(
            f"audited trial history too small: {audited_trial_count} < {min_audited_trials}"
        )
    if gaming_alarm:
        blockers.append("W6 audit gaming alarm is triggered")
    return {
        "cutover_ready": not blockers,
        "min_audited_trials": min_audited_trials,
        "audited_trial_count": audited_trial_count,
        "raw_audited_trial_count": report.get("raw_audited_trial_count"),
        "trusted_audited_trial_count": report.get("trusted_audited_trial_count"),
        "untrusted_audited_trial_count": report.get("untrusted_audited_trial_count"),
        "untrusted_audited_trial_ids": report.get("untrusted_audited_trial_ids"),
        "alarm_window": report.get("gaming_alarm_window"),
        "alarm_window_trial_count": report.get("gaming_alarm_window_trial_count"),
        "gaming_alarm": gaming_alarm,
        "potential_overfit_divergences": (
            report.get("transfer_diagnostic") or {}
        ).get("potential_overfit_divergences"),
        "cumulative_gaming_alarm": report.get("cumulative_gaming_alarm"),
        "cumulative_potential_overfit_divergences": (
            report.get("transfer_diagnostic") or {}
        ).get("cumulative_potential_overfit_divergences"),
        "blockers": blockers,
        "report": report,
    }


def _summary_report(report: dict[str, Any]) -> dict[str, Any]:
    seq = report["sequential_cutover"]
    w6 = report["w6_audit_cutover"]
    snapshot = report["snapshot_replay"]
    archive = report["archive_authority"]
    baseline = report["baseline_authority"]
    baseline_seed = baseline.get("seed_preflight") or {}
    return {
        "restart_ready": report["restart_ready"],
        "blockers": report["blockers"],
        "archive_status": archive["diagnostic"].get("status"),
        "snapshot_restart_readiness": snapshot.get("restart_readiness"),
        "snapshot_payload_available": snapshot.get("payload_available"),
        "baseline_authority_source": baseline.get("authority_source"),
        "baseline_startup_safe": baseline.get("startup_safe"),
        "baseline_seed_status": baseline_seed.get("status"),
        "baseline_seed_append_ready": baseline_seed.get("append_ready"),
        "baseline_seed_append_required": baseline_seed.get("append_required"),
        "baseline_seed_append_expect_trial_counter": baseline_seed.get(
            "append_expect_trial_counter"
        ),
        "baseline_seed_append_expect_journal_max_trial_id": baseline_seed.get(
            "append_expect_journal_max_trial_id"
        ),
        "seq_cutover_ready": seq.get("cutover_ready"),
        "seq_trusted_vector_trials": seq.get("trusted_vector_trials"),
        "seq_shadow_rows": (seq.get("seq_shadow") or {}).get("seq_shadow_rows"),
        "w6_audit_cutover_ready": w6.get("cutover_ready"),
        "w6_audited_trial_count": w6.get("audited_trial_count"),
        "w6_raw_audited_trial_count": w6.get("raw_audited_trial_count"),
        "w6_trusted_audited_trial_count": w6.get("trusted_audited_trial_count"),
        "w6_untrusted_audited_trial_count": w6.get("untrusted_audited_trial_count"),
        "w6_untrusted_audited_trial_ids": w6.get("untrusted_audited_trial_ids"),
        "w6_min_audited_trials": w6.get("min_audited_trials"),
        "w6_alarm_window": w6.get("alarm_window"),
        "w6_alarm_window_trial_count": w6.get("alarm_window_trial_count"),
        "w6_gaming_alarm": w6.get("gaming_alarm"),
        "w6_potential_overfit_divergences": w6.get("potential_overfit_divergences"),
        "w6_cumulative_gaming_alarm": w6.get("cumulative_gaming_alarm"),
        "w6_cumulative_potential_overfit_divergences": w6.get(
            "cumulative_potential_overfit_divergences"
        ),
    }


def build_restart_readiness_report(
    state: dict[str, Any],
    journal_rows: list[dict[str, Any]],
    *,
    require_seq_cutover: bool = False,
    require_w6_audit: bool = False,
    min_w6_audited_trials: int = 30,
) -> dict[str, Any]:
    """Build a no-write report for safe AutoPilot restart/cutover decisions."""
    archive_report = build_archive_authority_report(state, journal_rows)
    snapshot_report = _snapshot_restart_report(journal_rows)
    baseline_report = _baseline_restart_report(state, journal_rows)
    seq_report = build_seq_readiness_report(journal_rows)
    w6_report = _w6_audit_restart_report(
        journal_rows,
        min_audited_trials=min_w6_audited_trials,
    )

    blockers: list[str] = []
    if not archive_report.get("ok"):
        status = (archive_report.get("diagnostic") or {}).get("status", "unknown")
        blockers.append(f"archive authority is not aligned: {status}")
    if not snapshot_report.get("ok"):
        readiness = snapshot_report.get("restart_readiness", "unknown")
        blockers.append(f"journal snapshot is not current or foldable: {readiness}")
    if not baseline_report.get("startup_safe"):
        blockers.append("no safe baseline startup source")
    if require_seq_cutover and not seq_report.get("cutover_ready"):
        blockers.append("sequential verdict cutover readiness is blocked")
    if require_w6_audit and not w6_report.get("cutover_ready"):
        blockers.append(
            "W6 audit cutover readiness is blocked: "
            + "; ".join(w6_report.get("blockers") or ["unknown"])
        )

    report = {
        "ok": not blockers,
        "restart_ready": not blockers,
        "require_seq_cutover": require_seq_cutover,
        "require_w6_audit": require_w6_audit,
        "blockers": blockers,
        "archive_authority": archive_report,
        "snapshot_replay": snapshot_report,
        "baseline_authority": baseline_report,
        "sequential_cutover": seq_report,
        "w6_audit_cutover": w6_report,
    }
    report["summary"] = _summary_report(report)
    return report


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# AutoPilot Restart Readiness Report",
        "",
        f"- Restart ready: {str(report['restart_ready']).lower()}",
        f"- Sequential cutover required: {str(report['require_seq_cutover']).lower()}",
        f"- W6 audit cutover required: {str(report['require_w6_audit']).lower()}",
        f"- Archive authority: {summary['archive_status']}",
        (
            "- Snapshot replay: "
            f"{summary['snapshot_restart_readiness']} "
            f"(payload_available={summary['snapshot_payload_available']})"
        ),
        (
            "- Baseline startup source: "
            f"{summary['baseline_authority_source']} "
            f"(startup_safe={summary['baseline_startup_safe']})"
        ),
        (
            "- Baseline seed preflight: "
            f"{summary['baseline_seed_status']} "
            f"(append_ready={summary['baseline_seed_append_ready']}, "
            f"append_required={summary['baseline_seed_append_required']})"
        ),
        (
            "- Sequential cutover: "
            f"ready={summary['seq_cutover_ready']}, "
            f"trusted_vectors={summary['seq_trusted_vector_trials']}, "
            f"seq_shadow_rows={summary['seq_shadow_rows']}"
        ),
        (
            "- W6 audit cutover: "
            f"ready={summary['w6_audit_cutover_ready']}, "
            f"audited_trials={summary['w6_audited_trial_count']}/"
            f"{summary['w6_min_audited_trials']}, "
            f"untrusted_audited_trials={summary['w6_untrusted_audited_trial_count']}, "
            f"alarm_window={summary['w6_alarm_window_trial_count']}/"
            f"{summary['w6_alarm_window']}, "
            f"gaming_alarm={summary['w6_gaming_alarm']}, "
            "potential_overfit_divergences="
            f"{summary['w6_potential_overfit_divergences']}, "
            "cumulative_divergences="
            f"{summary['w6_cumulative_potential_overfit_divergences']}"
        ),
    ]
    if report["blockers"]:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {blocker}" for blocker in report["blockers"])
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report no-inference AutoPilot restart readiness across archive, "
            "snapshot, baseline, and sequential-verdict gates."
        )
    )
    parser.add_argument("--state", type=Path, default=STATE_PATH)
    parser.add_argument("--journal", type=Path, default=JOURNAL_PATH)
    parser.add_argument("--json", action="store_true", help="Emit structured JSON.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when restart readiness blockers exist.",
    )
    parser.add_argument(
        "--require-seq-cutover",
        action="store_true",
        help="Treat blocked sequential-verdict cutover readiness as a restart blocker.",
    )
    parser.add_argument(
        "--require-w6-audit",
        action="store_true",
        help="Treat blocked W6 rotating-audit cutover readiness as a restart blocker.",
    )
    parser.add_argument(
        "--min-w6-audited-trials",
        type=int,
        default=30,
        help="Minimum audited trial rows required when checking W6 audit cutover readiness.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.min_w6_audited_trials < 1:
        print("--min-w6-audited-trials must be >= 1", file=sys.stderr)
        return 2
    state_path = args.state.expanduser().resolve()
    journal_path = args.journal.expanduser().resolve()
    if not state_path.exists():
        print(f"state file does not exist: {state_path}", file=sys.stderr)
        return 2
    if not journal_path.exists():
        print(f"journal file does not exist: {journal_path}", file=sys.stderr)
        return 2

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"state file is not valid JSON: {state_path}: {exc}", file=sys.stderr)
        return 2
    if not isinstance(state, dict):
        print(f"state file is not a JSON object: {state_path}", file=sys.stderr)
        return 2

    journal_rows = _load_jsonl(journal_path)
    report = build_restart_readiness_report(
        state,
        journal_rows,
        require_seq_cutover=args.require_seq_cutover,
        require_w6_audit=args.require_w6_audit,
        min_w6_audited_trials=args.min_w6_audited_trials,
    )
    if args.json:
        print(json.dumps(report, sort_keys=True, default=str))
    else:
        print(render_markdown(report))
    if args.strict and not report["restart_ready"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
