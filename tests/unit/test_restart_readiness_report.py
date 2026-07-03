"""Tests for the read-only AutoPilot restart readiness report."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "autopilot"))

import restart_readiness_report as report_mod  # noqa: E402


def _state() -> dict[str, Any]:
    return {
        "trial_counter": 2,
        "baseline_state": {"baselines_by_tier": {"1": 1.8}},
    }


def _journal_row(trial_id: int, timestamp: str = "2026-06-15T00:00:01Z") -> dict[str, Any]:
    return {
        "trial_id": trial_id,
        "timestamp": timestamp,
        "species": "unit",
        "action_type": "seed_batch",
        "tier": 1,
        "quality": 1.2,
        "speed": 40.0,
        "cost": 0.2,
        "reliability": 0.9,
        "pareto_status": "frontier",
    }


def _seq_ready(*, ready: bool = False) -> dict[str, Any]:
    return {
        "cutover_ready": ready,
        "trusted_vector_trials": 61,
        "seq_shadow": {"seq_shadow_rows": 9},
        "w8_promotion_evidence": {"status": "none"},
        "cutover_blockers": [] if ready else ["trusted vector history too small"],
    }


def _audit_report(
    *,
    audited_trial_count: int = 0,
    gaming_alarm: bool = False,
    divergences: int = 0,
) -> dict[str, Any]:
    return {
        "trial_count": audited_trial_count,
        "raw_audited_trial_count": audited_trial_count,
        "trusted_audited_trial_count": audited_trial_count,
        "untrusted_audited_trial_count": 0,
        "untrusted_audited_trial_ids": [],
        "audited_trial_count": audited_trial_count,
        "totals": {
            "core_correct": 0,
            "core_total": 0,
            "audit_correct": 0,
            "audit_total": 0,
            "core_quality_0_3": 0.0,
            "audit_quality_0_3": 0.0,
            "delta_audit_minus_core": 0.0,
        },
        "trials": [],
        "gaming_alarm": gaming_alarm,
        "gaming_events": [{"trial_id": 2}] if gaming_alarm else [],
        "gaming_alarm_window": 30,
        "gaming_alarm_window_trial_count": min(audited_trial_count, 30),
        "gaming_alarm_clearance_clean_trials_required": 4 if gaming_alarm else 0,
        "cumulative_gaming_alarm": gaming_alarm,
        "cumulative_gaming_events": [{"trial_id": 2}] if gaming_alarm else [],
        "transfer_diagnostic": {
            "audited_trial_count": audited_trial_count,
            "potential_overfit_divergences": divergences,
            "events": [{"trial_id": 2}] if divergences else [],
            "clearance_clean_trials_required": 4 if gaming_alarm else 0,
            "cumulative_potential_overfit_divergences": divergences,
            "cumulative_events": [{"trial_id": 2}] if divergences else [],
            "alarm_window": 30,
            "alarm_window_trial_count": min(audited_trial_count, 30),
        },
    }


def _patch_ready_dependencies(monkeypatch, *, audit_report: dict[str, Any] | None = None) -> None:
    monkeypatch.setattr(
        report_mod,
        "build_archive_authority_report",
        lambda state, rows: {"ok": True, "diagnostic": {"status": "match"}},
    )
    monkeypatch.setattr(
        report_mod,
        "build_archive_source_surface_audit",
        lambda root=report_mod.ORCH_ROOT: {
            "ok": True,
            "surface_count": 4,
            "failed_count": 0,
            "results": [],
        },
    )
    monkeypatch.setattr(
        report_mod,
        "build_baseline_authority_report",
        lambda state, rows: {"ok": False, "status": "no_events"},
    )
    monkeypatch.setattr(
        report_mod,
        "build_seq_readiness_report",
        lambda rows, **kwargs: _seq_ready(ready=True),
    )
    monkeypatch.setattr(
        report_mod,
        "build_snapshot_replay_diagnostic",
        lambda rows, events: SimpleNamespace(
            bounded_replay_readiness="current",
            event_count=1,
            status="archive_prefix_match",
            hash_status="match",
            latest_event=None,
            through_trial_id=10,
            policy_version="journal-archive-snapshot-v1",
            snapshot_hash="abc",
            parent_snapshot_hash="",
            tail_trial_count=0,
            tail_max_trial_id=None,
            journal_max_trial_id=10,
            post_snapshot_prefix_event_count=0,
            warnings=[],
        ),
    )
    monkeypatch.setattr(
        report_mod,
        "archive_payload_from_verified_snapshot",
        lambda rows, events: {"journal_max_trial_id": 10},
    )
    if audit_report is not None:
        monkeypatch.setattr(
            report_mod,
            "build_audit_block_report",
            lambda rows, **kwargs: audit_report,
        )


def test_restart_ready_accepts_tail_fold_snapshot_and_state_baseline(monkeypatch) -> None:
    monkeypatch.setattr(
        report_mod,
        "build_archive_authority_report",
        lambda state, rows: {"ok": True, "diagnostic": {"status": "match"}},
    )
    monkeypatch.setattr(
        report_mod,
        "build_baseline_authority_report",
        lambda state, rows: {"ok": False, "status": "no_events"},
    )
    monkeypatch.setattr(
        report_mod,
        "build_seq_readiness_report",
        lambda rows, **kwargs: _seq_ready(ready=False),
    )
    monkeypatch.setattr(
        report_mod,
        "build_snapshot_replay_diagnostic",
        lambda rows, events: SimpleNamespace(
            bounded_replay_readiness="tail_unverified",
            event_count=1,
            status="archive_prefix_match",
            hash_status="match",
            latest_event=None,
            through_trial_id=10,
            policy_version="journal-archive-snapshot-v1",
            snapshot_hash="abc",
            parent_snapshot_hash="",
            tail_trial_count=1,
            tail_max_trial_id=11,
            journal_max_trial_id=11,
            post_snapshot_prefix_event_count=0,
            warnings=[],
        ),
    )
    monkeypatch.setattr(
        report_mod,
        "archive_payload_from_verified_snapshot",
        lambda rows, events: {"journal_max_trial_id": 11},
    )

    report = report_mod.build_restart_readiness_report(_state(), [])

    assert report["restart_ready"] is True
    assert report["blockers"] == []
    assert report["summary"]["snapshot_restart_readiness"] == "tail_fold_ready"
    assert report["summary"]["baseline_authority_source"] == "state_baseline"
    assert report["summary"]["baseline_authority_enabled"] is False
    assert report["summary"]["baseline_seed_status"] == "ready"
    assert report["summary"]["baseline_seed_append_ready"] is True
    assert report["summary"]["baseline_seed_append_required"] is True
    assert report["summary"]["baseline_seed_append_expect_trial_counter"] == 2
    assert report["summary"]["baseline_seed_append_expect_journal_max_trial_id"] is None
    assert report["baseline_authority"]["seed_preflight"]["event_tier"] == 1
    assert (
        report["baseline_authority"]["seed_preflight"][
            "append_expect_trial_counter"
        ]
        == 2
    )
    assert report["summary"]["seq_cutover_ready"] is False
    assert report["summary"]["seq_min_trusted_vector_trials"] is None
    assert report["summary"]["seq_trusted_vector_trials_remaining"] is None
    assert report["summary"]["seq_min_shadow_rows"] is None
    assert report["summary"]["seq_shadow_rows_remaining"] is None
    assert report["summary"]["w8_promotion_status"] == "none"
    assert report["summary"]["w6_audited_trial_count"] == 0
    assert report["summary"]["w6_audited_trial_count_remaining"] == 30
    assert report["summary"]["cutover_horizon_clean_trials_remaining"] == 30
    assert report["summary"]["cutover_horizon_blocker"] == "w6_audited_trials"
    assert report["summary"]["cutover_horizon_components"] == {
        "w6_audited_trials": 30,
        "w6_alarm_clearance": 0,
    }


def test_restart_summary_projects_w8_promotion_finalization(monkeypatch) -> None:
    _patch_ready_dependencies(monkeypatch, audit_report=_audit_report(audited_trial_count=30))
    monkeypatch.setattr(
        report_mod,
        "build_seq_readiness_report",
        lambda rows, state=None: {
            "cutover_ready": True,
            "trusted_vector_trials": 120,
            "seq_shadow": {"seq_shadow_rows": 30},
            "thresholds": {
                "min_trusted_vector_trials": 120,
                "min_seq_shadow_rows": 30,
            },
            "w8_promotion_evidence": {
                "status": "finalized",
                "open_requirements": [],
                "last_finalized_trial_id": 44,
                "last_finalized_candidate": "candidate-a",
                "last_finalized_combined_E": 110.0,
                "last_finalized_delta_excludes_regression": True,
                "latest_seq_trial_id": 44,
                "latest_combined_E": 110.0,
                "latest_required_E": 100.0,
                "latest_baseline_reference_state": "fresh",
            },
        },
    )
    state = {
        **_state(),
        "seq_last_promotion_finalized": {
            "trial_id": 44,
            "candidate": "candidate-a",
            "combined_E": 110.0,
            "delta_ci": {"excludes_regression": True, "lower_bound": 0.01},
        },
    }

    report = report_mod.build_restart_readiness_report(state, [], require_w6_audit=True)

    assert report["summary"]["w8_promotion_status"] == "finalized"
    assert report["summary"]["w8_open_requirements"] == []
    assert report["summary"]["w8_last_finalized_trial_id"] == 44
    assert report["summary"]["w8_last_finalized_candidate"] == "candidate-a"
    assert report["summary"]["w8_last_finalized_combined_E"] == 110.0
    assert report["summary"]["w8_last_finalized_delta_excludes_regression"] is True
    assert report["summary"]["w8_latest_seq_trial_id"] == 44
    assert report["summary"]["w8_latest_combined_E"] == 110.0
    assert report["summary"]["w8_latest_required_E"] == 100.0
    assert report["summary"]["w8_latest_baseline_reference_state"] == "fresh"
    rendered = report_mod.render_markdown(report)
    assert "W8 promotion evidence: status=finalized" in rendered
    assert "latest_combined_E=110.0/100.0" in rendered
    assert "open_requirements=[]" in rendered


def test_restart_report_blocks_on_archive_source_surface_audit_failure(monkeypatch) -> None:
    _patch_ready_dependencies(monkeypatch, audit_report=_audit_report(audited_trial_count=30))
    monkeypatch.setattr(
        report_mod,
        "build_archive_source_surface_audit",
        lambda root=report_mod.ORCH_ROOT: {
            "ok": False,
            "surface_count": 4,
            "failed_count": 1,
            "results": [
                {
                    "name": "legacy fallback",
                    "path": "scripts/autopilot/example.py",
                    "ok": False,
                    "missing": ["archive-source guard"],
                    "reason": "unit fixture",
                }
            ],
        },
    )

    report = report_mod.build_restart_readiness_report(_state(), [])

    assert report["restart_ready"] is False
    assert "archive source surface audit failed: 1 surface(s)" in report["blockers"]
    assert report["summary"]["archive_source_surface_ok"] is False
    assert report["summary"]["archive_source_surface_count"] == 4
    assert report["summary"]["archive_source_surface_failed_count"] == 1
    assert (
        "Archive source surfaces: ok=False, failed=1/4"
        in report_mod.render_markdown(report)
    )


def test_restart_ready_accepts_full_replay_when_snapshot_invalidated(monkeypatch) -> None:
    monkeypatch.setattr(
        report_mod,
        "build_baseline_authority_report",
        lambda state, rows: {"ok": False, "status": "no_events"},
    )
    monkeypatch.setattr(
        report_mod,
        "build_seq_readiness_report",
        lambda rows, **kwargs: _seq_ready(ready=False),
    )
    monkeypatch.setattr(
        report_mod,
        "build_snapshot_replay_diagnostic",
        lambda rows, events: SimpleNamespace(
            bounded_replay_readiness="prefix_invalidated",
            event_count=1,
            status="archive_prefix_drift",
            hash_status="mismatch",
            latest_event=None,
            through_trial_id=10,
            policy_version="journal-archive-snapshot-v1",
            snapshot_hash="abc",
            parent_snapshot_hash="",
            tail_trial_count=0,
            tail_max_trial_id=None,
            journal_max_trial_id=1,
            post_snapshot_prefix_event_count=0,
            warnings=["latest journal snapshot prefix is invalidated"],
        ),
    )
    monkeypatch.setattr(
        report_mod,
        "archive_payload_from_verified_snapshot",
        lambda rows, events: None,
    )

    report = report_mod.build_restart_readiness_report(
        _state(),
        [_journal_row(1)],
    )

    assert report["restart_ready"] is True
    assert report["blockers"] == []
    assert report["summary"]["archive_status"] == "match"
    assert report["summary"]["snapshot_restart_readiness"] == "full_replay_ready"
    assert report["summary"]["snapshot_payload_available"] is False
    assert report["snapshot_replay"]["full_replay_payload_available"] is True


def test_baseline_seed_preflight_skips_when_ledger_fold_ready(monkeypatch) -> None:
    _patch_ready_dependencies(monkeypatch)
    monkeypatch.setattr(
        report_mod,
        "build_baseline_authority_report",
        lambda state, rows: {"ok": True, "status": "match"},
    )

    report = report_mod.build_restart_readiness_report(_state(), [])

    assert report["summary"]["baseline_authority_source"] == "state_baseline"
    assert report["summary"]["baseline_authority_enabled"] is False
    assert report["summary"]["baseline_seed_status"] == "ledger_fold_ready"
    assert report["summary"]["baseline_seed_append_ready"] is False
    assert report["summary"]["baseline_seed_append_required"] is False
    assert report["baseline_authority"]["seed_preflight"] == {
        "status": "ledger_fold_ready",
        "append_required": False,
        "append_ready": False,
        "warning": "",
    }


def test_require_seq_cutover_blocks_when_seq_report_not_ready(monkeypatch) -> None:
    monkeypatch.setattr(
        report_mod,
        "build_archive_authority_report",
        lambda state, rows: {"ok": True, "diagnostic": {"status": "match"}},
    )
    monkeypatch.setattr(
        report_mod,
        "build_baseline_authority_report",
        lambda state, rows: {"ok": False, "status": "no_events"},
    )
    monkeypatch.setattr(
        report_mod,
        "build_seq_readiness_report",
        lambda rows, **kwargs: _seq_ready(),
    )
    monkeypatch.setattr(
        report_mod,
        "build_snapshot_replay_diagnostic",
        lambda rows, events: SimpleNamespace(
            bounded_replay_readiness="current",
            event_count=1,
            status="archive_prefix_match",
            hash_status="match",
            latest_event=None,
            through_trial_id=10,
            policy_version="journal-archive-snapshot-v1",
            snapshot_hash="abc",
            parent_snapshot_hash="",
            tail_trial_count=0,
            tail_max_trial_id=None,
            journal_max_trial_id=10,
            post_snapshot_prefix_event_count=0,
            warnings=[],
        ),
    )
    monkeypatch.setattr(
        report_mod,
        "archive_payload_from_verified_snapshot",
        lambda rows, events: {"journal_max_trial_id": 10},
    )

    report = report_mod.build_restart_readiness_report(
        _state(),
        [],
        require_seq_cutover=True,
    )

    assert report["restart_ready"] is False
    assert report["blockers"] == [
        "sequential verdict cutover readiness is blocked"
    ]


def test_w6_audit_summary_is_visible_without_blocking_restart(monkeypatch) -> None:
    _patch_ready_dependencies(
        monkeypatch,
        audit_report=_audit_report(
            audited_trial_count=29,
            gaming_alarm=True,
            divergences=5,
        ),
    )

    report = report_mod.build_restart_readiness_report(_state(), [])

    assert report["restart_ready"] is True
    assert report["blockers"] == []
    assert report["summary"]["w6_audit_cutover_ready"] is False
    assert report["summary"]["w6_audited_trial_count"] == 29
    assert report["summary"]["w6_audited_trial_count_remaining"] == 1
    assert report["summary"]["w6_untrusted_audited_trial_count"] == 0
    assert report["summary"]["w6_untrusted_audited_trial_ids"] == []
    assert report["summary"]["w6_min_audited_trials"] == 30
    assert report["summary"]["w6_alarm_window"] == 30
    assert report["summary"]["w6_alarm_window_trial_count"] == 29
    assert report["summary"]["w6_alarm_clearance_clean_trials_required"] == 4
    assert report["summary"]["w6_gaming_alarm"] is True
    assert report["summary"]["w6_potential_overfit_divergences"] == 5
    assert report["summary"]["w6_cumulative_potential_overfit_divergences"] == 5
    assert report["w6_audit_cutover"]["blockers"] == [
        "audited trial history too small: 29 < 30",
        "W6 audit gaming alarm is triggered",
    ]


def test_cutover_horizon_reports_largest_remaining_clean_trial_blocker(
    monkeypatch,
) -> None:
    _patch_ready_dependencies(
        monkeypatch,
        audit_report=_audit_report(
            audited_trial_count=34,
            gaming_alarm=True,
            divergences=4,
        ),
    )
    monkeypatch.setattr(
        report_mod,
        "build_seq_readiness_report",
        lambda rows, **kwargs: {
            "cutover_ready": False,
            "trusted_vector_trials": 80,
            "seq_shadow": {"seq_shadow_rows": 28},
            "thresholds": {
                "min_trusted_vector_trials": 120,
                "min_seq_shadow_rows": 30,
            },
            "cutover_blockers": ["trusted vector history too small"],
        },
    )

    report = report_mod.build_restart_readiness_report(_state(), [])

    assert report["summary"]["seq_trusted_vector_trials_remaining"] == 40
    assert report["summary"]["seq_shadow_rows_remaining"] == 2
    assert report["summary"]["w6_audited_trial_count_remaining"] == 0
    assert report["summary"]["w6_alarm_clearance_clean_trials_required"] == 4
    assert report["summary"]["cutover_horizon_clean_trials_remaining"] == 40
    assert report["summary"]["cutover_horizon_blocker"] == "seq_trusted_vectors"
    assert report["summary"]["cutover_horizon_components"] == {
        "seq_trusted_vectors": 40,
        "seq_shadow_rows": 2,
        "w6_audited_trials": 0,
        "w6_alarm_clearance": 4,
    }


def test_require_w6_audit_blocks_on_sample_size_and_alarm(monkeypatch) -> None:
    _patch_ready_dependencies(
        monkeypatch,
        audit_report=_audit_report(
            audited_trial_count=29,
            gaming_alarm=True,
            divergences=5,
        ),
    )

    report = report_mod.build_restart_readiness_report(
        _state(),
        [],
        require_w6_audit=True,
    )

    assert report["restart_ready"] is False
    assert report["blockers"] == [
        "W6 audit cutover readiness is blocked: "
        "audited trial history too small: 29 < 30; "
        "W6 audit gaming alarm is triggered"
    ]


def test_require_w6_audit_accepts_clean_minimum(monkeypatch) -> None:
    _patch_ready_dependencies(
        monkeypatch,
        audit_report=_audit_report(audited_trial_count=30),
    )

    report = report_mod.build_restart_readiness_report(
        _state(),
        [],
        require_w6_audit=True,
    )

    assert report["restart_ready"] is True
    assert report["blockers"] == []
    assert report["summary"]["w6_audit_cutover_ready"] is True
    assert report["summary"]["w6_audited_trial_count_remaining"] == 0


def test_cutover_horizon_has_no_blocker_when_all_components_clear(monkeypatch) -> None:
    _patch_ready_dependencies(
        monkeypatch,
        audit_report=_audit_report(audited_trial_count=30),
    )
    monkeypatch.setattr(
        report_mod,
        "build_seq_readiness_report",
        lambda rows, **kwargs: {
            "cutover_ready": True,
            "trusted_vector_trials": 120,
            "seq_shadow": {"seq_shadow_rows": 30},
            "thresholds": {
                "min_trusted_vector_trials": 120,
                "min_seq_shadow_rows": 30,
            },
            "cutover_blockers": [],
        },
    )

    report = report_mod.build_restart_readiness_report(
        _state(),
        [],
        require_w6_audit=True,
    )

    assert report["summary"]["cutover_horizon_clean_trials_remaining"] == 0
    assert report["summary"]["cutover_horizon_blocker"] is None
    assert report["summary"]["cutover_horizon_components"] == {
        "seq_trusted_vectors": 0,
        "seq_shadow_rows": 0,
        "w6_audited_trials": 0,
        "w6_alarm_clearance": 0,
    }


def test_require_w6_audit_uses_trailing_alarm_window(monkeypatch) -> None:
    _patch_ready_dependencies(monkeypatch)

    rows = []
    for trial_id, core_correct, audit_correct in [
        (1, 1, 3),
        (2, 3, 1),
        (3, 1, 1),
        (4, 1, 1),
        (5, 1, 1),
    ]:
        question_results = [
            {"qid": f"core-{trial_id}-{idx}", "partition": "core", "correct": idx < core_correct}
            for idx in range(4)
        ]
        question_results.extend(
            {
                "qid": f"audit-{trial_id}-{idx}",
                "partition": "audit",
                "correct": idx < audit_correct,
            }
            for idx in range(4)
        )
        rows.append({"trial_id": trial_id, "eval_details": {"question_results": question_results}})

    report = report_mod.build_restart_readiness_report(
        _state(),
        rows,
        require_w6_audit=True,
        min_w6_audited_trials=3,
    )

    assert report["restart_ready"] is True
    assert report["summary"]["w6_audit_cutover_ready"] is True
    assert report["summary"]["w6_alarm_window"] == 3
    assert report["summary"]["w6_alarm_window_trial_count"] == 3
    assert report["summary"]["w6_gaming_alarm"] is False
    assert report["summary"]["w6_potential_overfit_divergences"] == 0
    assert report["summary"]["w6_cumulative_gaming_alarm"] is True
    assert report["summary"]["w6_cumulative_potential_overfit_divergences"] == 1


def test_w6_cutover_horizon_counts_alarm_clearance_before_window_full(monkeypatch) -> None:
    _patch_ready_dependencies(monkeypatch)

    rows = []
    for trial_id, core_correct, audit_correct in [
        (1, 1, 3),
        (2, 3, 1),
        (3, 3, 3),
    ]:
        question_results = [
            {
                "qid": f"core-{trial_id}-{idx}",
                "partition": "core",
                "correct": idx < core_correct,
            }
            for idx in range(4)
        ]
        question_results.extend(
            {
                "qid": f"audit-{trial_id}-{idx}",
                "partition": "audit",
                "correct": idx < audit_correct,
            }
            for idx in range(4)
        )
        rows.append({"trial_id": trial_id, "eval_details": {"question_results": question_results}})

    report = report_mod.build_restart_readiness_report(
        _state(),
        rows,
        require_w6_audit=True,
        min_w6_audited_trials=5,
    )

    assert report["restart_ready"] is False
    assert report["summary"]["w6_audited_trial_count_remaining"] == 2
    assert report["summary"]["w6_alarm_clearance_clean_trials_required"] == 3
    assert report["summary"]["cutover_horizon_clean_trials_remaining"] == 3
    assert report["summary"]["cutover_horizon_blocker"] == "w6_alarm_clearance"
    assert report["summary"]["cutover_horizon_components"] == {
        "w6_audited_trials": 2,
        "w6_alarm_clearance": 3,
    }


def test_require_w6_audit_defaults_to_state_era_fence(monkeypatch) -> None:
    _patch_ready_dependencies(monkeypatch)
    state = _state()
    state["pareto_exclude_before_ts"] = 1782511631.0

    rows = []
    for trial_id, core_correct, audit_correct, timestamp in [
        (1, 1, 1, "2026-06-26T22:07:10+00:00"),
        (2, 2, 1, "2026-06-26T22:07:10.500000+00:00"),
        (3, 1, 1, "2026-06-26T22:07:11+00:00"),
        (4, 1, 1, "2026-06-27T00:00:00+00:00"),
        (5, 1, 1, "2026-06-27T01:00:00+00:00"),
    ]:
        question_results = [
            {
                "qid": f"core-{trial_id}-{idx}",
                "partition": "core",
                "correct": idx < core_correct,
            }
            for idx in range(2)
        ]
        question_results.extend(
            {
                "qid": f"audit-{trial_id}-{idx}",
                "partition": "audit",
                "correct": idx < audit_correct,
            }
            for idx in range(2)
        )
        rows.append(
            {
                "trial_id": trial_id,
                "timestamp": timestamp,
                "eval_details": {"question_results": question_results},
            }
        )

    report = report_mod.build_restart_readiness_report(
        state,
        rows,
        require_w6_audit=True,
        min_w6_audited_trials=3,
    )

    assert report["restart_ready"] is True
    assert report["summary"]["w6_era_exclude_before_ts"] == 1782511631.0
    assert report["summary"]["w6_all_raw_audited_trial_count"] == 5
    assert report["summary"]["w6_raw_audited_trial_count"] == 3
    assert report["summary"]["w6_era_excluded_audited_trial_count"] == 2
    assert report["summary"]["w6_era_excluded_audited_trial_ids"] == [1, 2]
    assert report["summary"]["w6_gaming_alarm"] is False
    assert report["summary"]["w6_cumulative_gaming_alarm"] is False


def test_restart_report_blocks_without_baseline_source(monkeypatch) -> None:
    monkeypatch.setattr(
        report_mod,
        "build_archive_authority_report",
        lambda state, rows: {"ok": True, "diagnostic": {"status": "match"}},
    )
    monkeypatch.setattr(
        report_mod,
        "build_baseline_authority_report",
        lambda state, rows: {"ok": False, "status": "no_events"},
    )
    monkeypatch.setattr(
        report_mod,
        "build_seq_readiness_report",
        lambda rows, **kwargs: _seq_ready(ready=True),
    )
    monkeypatch.setattr(
        report_mod,
        "build_snapshot_replay_diagnostic",
        lambda rows, events: SimpleNamespace(
            bounded_replay_readiness="current",
            event_count=1,
            status="archive_prefix_match",
            hash_status="match",
            latest_event=None,
            through_trial_id=10,
            policy_version="journal-archive-snapshot-v1",
            snapshot_hash="abc",
            parent_snapshot_hash="",
            tail_trial_count=0,
            tail_max_trial_id=None,
            journal_max_trial_id=10,
            post_snapshot_prefix_event_count=0,
            warnings=[],
        ),
    )
    monkeypatch.setattr(
        report_mod,
        "archive_payload_from_verified_snapshot",
        lambda rows, events: {"journal_max_trial_id": 10},
    )

    report = report_mod.build_restart_readiness_report({"trial_counter": 2}, [])

    assert report["restart_ready"] is False
    assert "no safe baseline startup source" in report["blockers"]
    assert report["summary"]["baseline_authority_source"] == "missing"


def test_cli_json_strict_returns_one_on_restart_blocker(tmp_path: Path, capsys, monkeypatch) -> None:
    state_path = tmp_path / "autopilot_state.json"
    journal_path = tmp_path / "autopilot_journal.jsonl"
    state_path.write_text("{}", encoding="utf-8")
    journal_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        report_mod,
        "build_restart_readiness_report",
        lambda state, rows, **kwargs: {
            "restart_ready": False,
            "blockers": ["blocked"],
        },
    )

    rc = report_mod.main(
        [
            "--state",
            str(state_path),
            "--journal",
            str(journal_path),
            "--json",
            "--strict",
        ]
    )
    out = json.loads(capsys.readouterr().out)

    assert rc == 1
    assert out["blockers"] == ["blocked"]


def test_cli_accepts_journal_directory_and_rollover_batches(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    state_path = tmp_path / "autopilot_state.json"
    journal_dir = tmp_path / "journal"
    journal_dir.mkdir()
    state_path.write_text("{}", encoding="utf-8")
    (journal_dir / "autopilot_journal.jsonl").write_text(
        json.dumps({"trial_id": 999}) + "\n",
        encoding="utf-8",
    )
    (journal_dir / "autopilot_journal_1.jsonl").write_text(
        json.dumps({"trial_id": 1000}) + "\n",
        encoding="utf-8",
    )

    observed: dict[str, list[int]] = {}

    def fake_report(state, rows, **kwargs):
        observed["trial_ids"] = [row["trial_id"] for row in rows]
        return {"restart_ready": True, "blockers": []}

    monkeypatch.setattr(report_mod, "build_restart_readiness_report", fake_report)

    rc = report_mod.main(
        [
            "--state",
            str(state_path),
            "--journal",
            str(journal_dir),
            "--json",
        ]
    )
    out = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert out["restart_ready"] is True
    assert observed["trial_ids"] == [999, 1000]
