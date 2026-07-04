from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "autopilot"))

import fable5_gate_report as report_mod  # noqa: E402


def test_build_report_blocks_on_underlying_fable5_gates(monkeypatch, tmp_path: Path) -> None:
    config = tmp_path / "classifier_config.yaml"
    config.write_text(
        """
xmas_routing:
  mode: "off"
  winner_table_path: ""
  require_complete_table: false
""",
        encoding="utf-8",
    )
    table = tmp_path / "xmas_winner_table.yaml"
    table.write_text("placeholder: true\n", encoding="utf-8")
    ab_root = tmp_path / "xmas_live_ab"
    run = ab_root / "run1"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(
        """
{
  "decision": {
    "status": "hold",
    "blockers": ["latency regression"]
  },
  "xmas_policy": "unknown_legacy",
  "score_delta_xmas_minus_baseline": -0.35,
  "latency_ratio_xmas_over_baseline": 16.18
}
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(report_mod, "validate_xmas_config", lambda path: [])
    monkeypatch.setattr(report_mod, "validate_xmas_table", lambda path, **kwargs: [])
    monkeypatch.setattr(
        report_mod,
        "xmas_quiet_window_report",
        lambda: {"ready": False, "blockers": ["active AutoPilot process(es): 123"]},
    )
    monkeypatch.setattr(
        report_mod,
        "ds_e1_clean_window_report",
        lambda: {
            "ready": False,
            "blockers": ["active AutoPilot process(es): 123 autopilot"],
        },
    )
    a9_manifest = tmp_path / "offline_reward_pairwise_expanded_gap_collection_manifest.json"
    monkeypatch.setattr(
        report_mod,
        "build_a9_collection_status",
        lambda path: {
            "ready": False,
            "status": "blocked",
            "manifest_path": str(a9_manifest),
            "manifest_schema_version": (
                "offline_reward_pairwise_collection_window.v1"
            ),
            "source_plan_decision": {"status": "expansion_plan_ready"},
            "batch_count": 9,
            "post_collection_step_count": 7,
            "autopilot_guard": {"refusal_exit_code": 75},
            "blockers": ["active AutoPilot process(es): 123 autopilot"],
            "warnings": [],
        },
    )
    monkeypatch.setattr(
        report_mod,
        "build_restart_readiness_report",
        lambda state, rows, **kwargs: {
            "restart_ready": False,
            "blockers": ["sequential verdict cutover readiness is blocked"],
            "archive_authority": {
                "journal_max_trial_id": 895,
                "state_trial_counter": 896,
            },
            "snapshot_replay": {
                "payload_journal_max_trial_id": 895,
            },
            "summary": {
                "seq_cutover_ready": False,
                "seq_trusted_vector_trials": 62,
                "seq_min_trusted_vector_trials": 120,
                "seq_trusted_vector_trials_remaining": 58,
                "seq_shadow_rows": 10,
                "seq_min_shadow_rows": 30,
                "seq_shadow_rows_remaining": 20,
                "snapshot_restart_readiness": "tail_fold_ready",
                "archive_source_surface_ok": True,
                "archive_source_surface_count": 6,
                "archive_source_surface_failed_count": 0,
                "w6_audit_cutover_ready": False,
                "w6_audited_trial_count": 34,
                "w6_min_audited_trials": 30,
                "w6_audited_trial_count_remaining": 0,
                "w6_alarm_clearance_clean_trials_required": 4,
                "w6_raw_audited_trial_count": 40,
                "w6_trusted_audited_trial_count": 39,
                "w6_untrusted_audited_trial_count": 1,
                "w6_untrusted_audited_trial_ids": [889],
                "w6_gaming_alarm": True,
                "w6_potential_overfit_divergences": 4,
                "cutover_horizon_clean_trials_remaining": 58,
                "cutover_horizon_blocker": "seq_trusted_vectors",
                "cutover_horizon_components": {
                    "seq_trusted_vectors": 58,
                    "seq_shadow_rows": 20,
                    "w6_audited_trials": 0,
                    "w6_alarm_clearance": 4,
                },
                "baseline_seed_append_ready": True,
                "baseline_seed_append_required": True,
                "baseline_seed_append_expect_trial_counter": 896,
                "baseline_seed_append_expect_journal_max_trial_id": 895,
            },
        },
    )

    report = report_mod.build_fable5_gate_report(
        state={"trial_counter": 896},
        journal_rows=[],
        phase_report={
            "ok": True,
            "status": "active",
            "trial_id": 896,
            "phase": "dispatch_action",
            "action_type": "deep_eval",
            "heartbeat_age_s": 4.0,
            "pid": 123,
            "pid_alive": True,
        },
        ds_e1_packet={
            "ready_for_profile_decision": False,
            "generated_at": "2026-06-20T00:00:00Z",
            "blockers": ["kv_size_measurements: missing"],
            "sections": [
                {
                    "key": "kv_size_measurements",
                    "status": "missing",
                    "details": {
                        "required_measurements": {
                            "frontdoor": [2048, 8192, 32768],
                            "worker_general": [2048, 8192],
                        },
                        "expected_csv_columns": [
                            "role",
                            "context_length",
                            "server_kv_size_mb",
                        ],
                        "searched_globs": ["orchestration/reports/ds_e1*kv*"],
                    },
                }
            ],
        },
        config_path=config,
        xmas_table_path=table,
        xmas_ab_root=ab_root,
        a9_collection_manifest=a9_manifest,
        include_tool_use_activation=False,
    )

    assert report["ready"] is False
    assert report["summary"]["ready"] is False
    assert report["summary"]["blocker_count"] == 7
    assert report["summary"]["blocked_sections"] == [
        "w4_w6_restart_cutover",
        "ds_e1_dynamic_stack",
        "a9_pairwise_collection",
        "xmas_production_path",
    ]
    assert report["summary"]["section_statuses"]["phase_health"] == "ready"
    assert report["summary"]["section_statuses"]["w4_w6_restart_cutover"] == "blocked"
    assert report["summary"]["section_statuses"]["a9_pairwise_collection"] == "blocked"
    assert report["summary"]["next_action_keys"] == [
        "append_baseline_seed_event",
        "continue_w4_w6_accrual",
        "run_ds_e1_kv_measurements",
        "collect_ri10_canary_arm_telemetry",
        "run_xmas_constrained_policy_ab",
        "run_a9_pairwise_collection_window",
    ]
    assert report["summary"]["active_next_action_keys"] == [
        "continue_w4_w6_accrual",
        "collect_ri10_canary_arm_telemetry",
    ]
    assert report["summary"]["blocked_next_action_keys"] == [
        "append_baseline_seed_event",
        "run_ds_e1_kv_measurements",
        "run_xmas_constrained_policy_ab",
        "run_a9_pairwise_collection_window",
    ]
    assert report["summary"]["restart_ready"] is False
    assert report["summary"]["phase_trial_id"] == 896
    assert report["summary"]["ds_e1_ready_for_profile_decision"] is False
    assert report["summary"]["ds_e1_clean_window_ready"] is False
    assert report["summary"]["ds_e1_clean_window_blockers"] == [
        "active AutoPilot process(es): 123 autopilot"
    ]
    assert report["summary"]["a9_collection_status"] == "blocked"
    assert report["summary"]["a9_collection_ready"] is False
    assert report["summary"]["a9_collection_batch_count"] == 9
    assert report["summary"]["a9_collection_blockers"] == [
        "active AutoPilot process(es): 123 autopilot"
    ]
    assert report["summary"]["xmas_mode"] == "off"
    assert report["summary"]["xmas_latest_ab_policy"] == "unknown_legacy"
    assert report["summary"]["xmas_latest_ab_decision_status"] == "hold"
    assert "w4_w6_restart_cutover: sequential verdict cutover readiness is blocked" in report[
        "blockers"
    ]
    assert "ds_e1_dynamic_stack: kv_size_measurements: missing" in report["blockers"]
    assert "a9_pairwise_collection: active AutoPilot process(es): 123 autopilot" in report[
        "blockers"
    ]
    assert "xmas_production_path: xmas_routing.mode is off; enforce remains default-off" in report[
        "blockers"
    ]
    assert (
        "xmas_production_path: latest X-MAS held-out A/B policy is "
        "unknown_legacy; required incumbent_constrained_cheapfirst_v2"
    ) in report["blockers"]
    assert "xmas_production_path: latest X-MAS held-out A/B decision is hold" in report[
        "blockers"
    ]
    xmas = [section for section in report["sections"] if section["key"] == "xmas_production_path"][0]
    assert xmas["details"]["latest_ab_decision_status"] == "hold"
    assert xmas["details"]["latest_ab_policy"] == "unknown_legacy"
    assert xmas["details"]["required_ab_policy"] == (
        "incumbent_constrained_cheapfirst_v2"
    )
    assert xmas["details"]["latest_ab_latency_ratio"] == 16.18
    assert xmas["details"]["quiet_window_ready"] is False
    assert xmas["details"]["quiet_window_blockers"] == [
        "active AutoPilot process(es): 123"
    ]
    assert report["sections"][0]["key"] == "phase_health"
    assert report["sections"][0]["status"] == "ready"
    restart = [
        section for section in report["sections"] if section["key"] == "w4_w6_restart_cutover"
    ][0]
    assert restart["details"]["durable_journal_max_trial_id"] == 895
    assert restart["details"]["state_trial_counter"] == 896
    assert restart["details"]["seq_trusted_vector_trials_remaining"] == 58
    assert restart["details"]["seq_shadow_rows_remaining"] == 20
    assert restart["details"]["archive_source_surface_ok"] is True
    assert restart["details"]["archive_source_surface_count"] == 6
    assert restart["details"]["archive_source_surface_failed_count"] == 0
    assert restart["details"]["w6_audited_trial_count_remaining"] == 0
    assert restart["details"]["w6_alarm_clearance_clean_trials_required"] == 4
    assert restart["details"]["snapshot_restart_readiness"] == "tail_fold_ready"
    assert restart["details"]["snapshot_payload_journal_max_trial_id"] == 895
    assert restart["details"]["baseline_seed_append_required"] is True
    assert restart["details"]["baseline_seed_append_expect_trial_counter"] == 896
    assert restart["details"]["baseline_seed_append_expect_journal_max_trial_id"] == 895
    assert restart["details"]["w6_untrusted_audited_trial_count"] == 1
    assert restart["details"]["w6_untrusted_audited_trial_ids"] == [889]
    assert restart["details"]["cutover_horizon_clean_trials_remaining"] == 58
    assert restart["details"]["cutover_horizon_blocker"] == "seq_trusted_vectors"
    assert restart["details"]["cutover_horizon_components"] == {
        "seq_trusted_vectors": 58,
        "seq_shadow_rows": 20,
        "w6_audited_trials": 0,
        "w6_alarm_clearance": 4,
    }
    assert [action["key"] for action in report["next_actions"]] == [
        "append_baseline_seed_event",
        "continue_w4_w6_accrual",
        "run_ds_e1_kv_measurements",
        "collect_ri10_canary_arm_telemetry",
        "run_xmas_constrained_policy_ab",
        "run_a9_pairwise_collection_window",
    ]
    seed_action = report["next_actions"][0]
    assert seed_action["status"] == "blocked"
    assert seed_action["blocked_by"] == [
        "active AutoPilot process; seed tool refuses live append"
    ]
    assert seed_action["evidence"] == {
        "baseline_seed_append_ready": True,
        "baseline_seed_append_required": True,
        "expect_trial_counter": 896,
        "expect_journal_max_trial_id": 895,
    }
    assert "baseline_authority_seed.py --append" in seed_action["command"]
    assert "--expect-trial-counter 896" in seed_action["command"]
    assert "--expect-journal-max-trial-id 895" in seed_action["command"]
    assert seed_action["follow_up"] == (
        "cd /mnt/raid0/llm/epyc-orchestrator && "
        "uv run python scripts/autopilot/restart_readiness_report.py "
        "--json --strict --require-seq-cutover --require-w6-audit"
    )
    assert report["next_actions"][1]["status"] == "active"
    assert report["next_actions"][1]["evidence"]["trusted_vectors_required"] == 120
    assert report["next_actions"][1]["evidence"]["trusted_vectors_remaining"] == 58
    assert report["next_actions"][1]["evidence"]["seq_shadow_rows_required"] == 30
    assert report["next_actions"][1]["evidence"]["seq_shadow_rows_remaining"] == 20
    assert report["next_actions"][1]["evidence"]["w6_audited_rows_required"] == 30
    assert report["next_actions"][1]["evidence"]["w6_audited_rows_remaining"] == 0
    assert (
        report["next_actions"][1]["evidence"][
            "w6_alarm_clearance_clean_trials_required"
        ]
        == 4
    )
    assert (
        report["next_actions"][1]["evidence"][
            "cutover_horizon_clean_trials_remaining"
        ]
        == 58
    )
    assert (
        report["next_actions"][1]["evidence"]["cutover_horizon_blocker"]
        == "seq_trusted_vectors"
    )
    assert "restart_readiness_report.py" in report["next_actions"][1]["command"]
    assert "--require-seq-cutover --require-w6-audit" in report["next_actions"][1]["command"]
    assert report["next_actions"][1]["follow_up"] == (
        "uv run python scripts/autopilot/fable5_gate_report.py --json --strict"
    )
    ds_e1 = [
        section for section in report["sections"] if section["key"] == "ds_e1_dynamic_stack"
    ][0]
    assert ds_e1["details"]["kv_required_measurements"]["frontdoor"] == [
        2048,
        8192,
        32768,
    ]
    assert ds_e1["details"]["kv_expected_csv_columns"] == [
        "role",
        "context_length",
        "server_kv_size_mb",
    ]
    assert "ds_e1_kv_measurements.sh --execute" in report["next_actions"][2]["command"]
    assert report["next_actions"][2]["status"] == "blocked"
    assert "$(date -u +%Y%m%dT%H%M%SZ)" in report["next_actions"][2]["follow_up"]
    assert "ds_e1_evidence_packet_20260620.md" not in report["next_actions"][2]["follow_up"]
    assert report["next_actions"][3]["command"] == (
        "uv run python scripts/analysis/ri10_canary_sample_report.py"
    )
    xmas_action = report["next_actions"][4]
    assert xmas_action["status"] == "blocked"
    assert xmas_action["blocked_by"] == ["active AutoPilot process(es): 123"]
    assert "latest X-MAS held-out A/B decision is hold" in xmas_action["evidence_blockers"]
    assert "xmas_live_ab.py" in xmas_action["command"]
    assert "<heldout_prompts.jsonl>" not in xmas_action["command"]
    assert xmas_action["prompt_manifest"] == (
        "benchmarks/results/runs/xmas_live_ab/20260618-heldout-resilient/prompts.jsonl"
    )
    assert xmas_action["required_policy"] == "incumbent_constrained_cheapfirst_v2"
    assert f"--prompts {xmas_action['prompt_manifest']}" in xmas_action["command"]
    assert "$(date -u +%Y%m%dT%H%M%SZ)-constrained-policy" in xmas_action["command"]
    a9_action = report["next_actions"][5]
    assert a9_action["status"] == "blocked"
    assert a9_action["blocked_by"] == [
        "active AutoPilot process(es): 123 autopilot"
    ]
    assert a9_action["manifest"] == str(a9_manifest)
    assert a9_action["batch_count"] == 9
    assert a9_action["post_collection_step_count"] == 7
    assert a9_action["source_plan_decision"] == {"status": "expansion_plan_ready"}
    assert "collect_offline_reward_pairwise_expanded_gap.sh" in a9_action["command"]
    assert "offline_reward_pairwise_collection_status.py" in a9_action["follow_up"]


def test_tool_use_activation_section_surfaces_missing_sentinel_env() -> None:
    section = report_mod.tool_use_activation_section(
        phase_report={"pid": 123},
        journal_rows=[
            {
                "trial_id": 1106,
                "eval_details": {
                    "total_tool_calls": 0,
                    "mean_tools_used": 0.0,
                    "tool_use_rate": 0.0,
                },
            }
        ],
        autopilot_env={},
        api_attest={
            "pid": 456,
            "flags": {
                "tools": True,
                "repl": True,
                "structured_tool_output": True,
            },
        },
        api_env={},
    )

    assert section.status == "attention"
    assert section.blockers == []
    assert section.details["activation_gaps"] == [
        "autopilot_env_missing_AUTOPILOT_TOOL_SENTINELS",
        "api_env_missing_AUTOPILOT_TOOL_SENTINELS",
        "latest_eval_total_tool_calls_zero",
    ]
    assert section.details["latest_tool_metrics"]["trial_id"] == 1106


def test_tool_use_next_action_requires_controlled_restart() -> None:
    phase = report_mod.GateSection(
        key="phase_health",
        status="ready",
        summary="active",
        blockers=[],
        details={"status": "active"},
    )
    tool_use = report_mod.GateSection(
        key="tool_use_activation",
        status="attention",
        summary="not active",
        blockers=[],
        details={
            "activation_gaps": [
                "autopilot_env_missing_AUTOPILOT_TOOL_SENTINELS",
                "api_env_missing_AUTOPILOT_TOOL_SENTINELS",
            ],
            "autopilot_tool_sentinels_enabled": False,
            "api_tool_sentinels_enabled": False,
            "api_tools_enabled": True,
            "api_repl_enabled": True,
            "latest_tool_metrics": {"trial_id": 1106, "total_tool_calls": 0},
        },
    )

    actions = report_mod.build_next_actions([phase, tool_use])

    assert actions == [
        {
            "key": "activate_tool_use_sentinel_lane",
            "priority": "P0",
            "status": "blocked",
            "reason": (
                "StrategyStore already exposes tool-use hints to the planner; "
                "the remaining gap is activating the API and AutoPilot "
                "tool-sentinel telemetry lane so tool use is measured."
            ),
            "requires": (
                "coordinated API reload plus AutoPilot restart at a trial "
                "boundary; this changes the active eval mix"
            ),
            "blocked_by": [
                "active AutoPilot process; wait for a controlled trial boundary"
            ],
            "evidence": {
                "activation_gaps": [
                    "autopilot_env_missing_AUTOPILOT_TOOL_SENTINELS",
                    "api_env_missing_AUTOPILOT_TOOL_SENTINELS",
                ],
                "autopilot_tool_sentinels_enabled": False,
                "api_tool_sentinels_enabled": False,
                "api_tools_enabled": True,
                "api_repl_enabled": True,
                "latest_tool_metrics": {
                    "trial_id": 1106,
                    "total_tool_calls": 0,
                },
            },
            "command": (
                "At a controlled trial boundary, reload the orchestrator API "
                "with AUTOPILOT_TOOL_SENTINELS=1, restart AutoPilot with "
                "AUTOPILOT_TOOL_SENTINELS=1 plus the existing W4/W6/planner "
                "env, then run AUTOPILOT_TOOL_SENTINELS=1 uv run python "
                "scripts/autopilot/gate3_tool_telemetry.py"
            ),
            "follow_up": report_mod.STRICT_FABLE5_GATE_COMMAND,
        }
    ]


def test_tool_use_next_action_waits_for_journal_when_lane_active() -> None:
    phase = report_mod.GateSection(
        key="phase_health",
        status="ready",
        summary="active",
        blockers=[],
        details={"status": "active"},
    )
    tool_use = report_mod.GateSection(
        key="tool_use_activation",
        status="attention",
        summary="waiting for first journaled tool telemetry",
        blockers=[],
        details={
            "activation_gaps": ["latest_eval_total_tool_calls_zero"],
            "autopilot_tool_sentinels_enabled": True,
            "api_tool_sentinels_enabled": True,
            "api_tools_enabled": True,
            "api_repl_enabled": True,
            "latest_tool_metrics": {"trial_id": 1107, "total_tool_calls": 0},
        },
    )
    ds_e1 = report_mod.GateSection(
        key="ds_e1_dynamic_stack",
        status="attention",
        summary="blocked",
        blockers=["kv missing"],
        details={},
    )

    actions = report_mod.build_next_actions([phase, tool_use, ds_e1])

    tool_action = actions[0]
    assert tool_action["key"] == "collect_tool_use_sentinel_journal_evidence"
    assert tool_action["status"] == "active"
    assert tool_action["blocked_by"] == []
    assert "sentinel-enabled AutoPilot eval finish" in tool_action["command"]
    assert actions[1]["key"] == "run_ds_e1_kv_measurements"


def test_phase_section_surfaces_eval_progress() -> None:
    section = report_mod.phase_section(
        {
            "ok": True,
            "status": "active",
            "trial_id": 902,
            "phase": "dispatch_action",
            "action_type": "deep_eval",
            "heartbeat_age_s": 4.0,
            "pid": 123,
            "pid_alive": True,
            "process_started_at_s": 1783021658.69,
            "require_current_code": True,
            "code_stale": True,
            "code_stale_paths": [{"path": "scripts/autopilot/autopilot.py"}],
            "eval_label": "T2",
            "eval_completed_questions": 200,
            "eval_total_questions": 500,
            "eval_correct_questions": 144,
            "eval_correct_pct": 72.0,
            "eval_concurrency": 1,
            "planner_hints_enabled": True,
            "seq_verdict_enabled": True,
            "w6_audit_accrual_enabled": True,
            "w6_audit_shadow_only": True,
            "w6_audit_n": "10",
            "w6_audit_every_n_trials": "1",
            "autopilot_planner_timeout": "600",
        }
    )

    assert section.status == "ready"
    assert "T2 200/500" in section.summary
    assert section.details["eval_completed_questions"] == 200
    assert section.details["eval_correct_pct"] == 72.0
    assert section.details["process_started_at_s"] == 1783021658.69
    assert section.details["require_current_code"] is True
    assert section.details["code_stale"] is True
    assert section.details["code_stale_paths"] == [
        {"path": "scripts/autopilot/autopilot.py"}
    ]
    assert section.details["planner_hints_enabled"] is True
    assert section.details["seq_verdict_enabled"] is True
    assert section.details["w6_audit_accrual_enabled"] is True
    assert section.details["w6_audit_n"] == "10"


def test_xmas_required_policy_hold_points_to_regression_diagnosis(
    monkeypatch, tmp_path: Path
) -> None:
    config = tmp_path / "classifier_config.yaml"
    config.write_text(
        """
xmas_routing:
  mode: "off"
  winner_table_path: "orchestration/xmas_winner_table.yaml"
  require_complete_table: true
""",
        encoding="utf-8",
    )
    table = tmp_path / "xmas_winner_table.yaml"
    table.write_text("placeholder: true\n", encoding="utf-8")
    ab_root = tmp_path / "xmas_live_ab"
    run = ab_root / "20260621T112005Z-constrained-policy"
    run.mkdir(parents=True)
    (run / "results.jsonl").write_text("{}\n", encoding="utf-8")
    (run / "summary.json").write_text(
        """
{
  "decision": {
    "status": "hold",
    "blockers": ["overall score delta -0.250 < required 0.050"]
  },
  "xmas_policy": "incumbent_constrained_cheapfirst_v2",
  "score_delta_xmas_minus_baseline": -0.25,
  "latency_ratio_xmas_over_baseline": 0.714
}
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(report_mod, "validate_xmas_config", lambda path: [])
    monkeypatch.setattr(
        report_mod, "validate_xmas_table", lambda path, **kwargs: []
    )

    section = report_mod.xmas_section(
        config_path=config,
        candidate_table_path=table,
        ab_root=ab_root,
        quiet_window={"ready": True, "blockers": []},
    )
    actions = report_mod.build_next_actions([section])

    assert section.status == "blocked"
    assert "latest X-MAS held-out A/B decision is hold" in section.blockers
    assert [action["key"] for action in actions] == [
        "diagnose_xmas_policy_regressions"
    ]
    action = actions[0]
    assert action["status"] == "ready"
    assert action["latest_ab_decision_status"] == "hold"
    assert action["latest_ab_score_delta"] == -0.25
    assert action["latest_ab_latency_ratio"] == 0.714
    assert action["latest_ab_results_path"] == str(run / "results.jsonl")
    assert "--summarize-results" in action["command"]
    assert str(run / "results.jsonl") in action["command"]


def test_ds_e1_section_surfaces_clean_window_blockers() -> None:
    section = report_mod.ds_e1_section(
        {
            "ready_for_profile_decision": False,
            "generated_at": "2026-06-20T00:00:00Z",
            "blockers": ["kv_size_measurements: missing"],
            "sections": [{"key": "kv_size_measurements", "status": "missing"}],
        },
        clean_window={
            "ready": False,
            "blockers": ["active AutoPilot process(es): 123 autopilot"],
        },
    )

    assert section.status == "blocked"
    assert section.blockers == ["kv_size_measurements: missing"]
    assert section.details["clean_window_ready"] is False
    assert section.details["clean_window_blockers"] == [
        "active AutoPilot process(es): 123 autopilot"
    ]


def test_w8_trajectory_section_surfaces_concentration_warning() -> None:
    section = report_mod.w8_trajectory_section(
        {
            "status": "progressing",
            "ok": True,
            "latest_trial_id": 1099,
            "snapshot_count": 155,
            "candidate_count": 41,
            "status_counts": {"active_recent_replay": 1},
            "open_requirements": ["replay_concentration_warning"],
            "recent_active_candidates": ["abc"],
            "stale_accumulating_candidates": ["def", "ghi"],
            "replay_concentration": {
                "warning": True,
                "warning_reason": "recent replay evidence is concentrated",
                "top_active_candidate": "abc",
                "top_active_attempt_share": 1.0,
            },
        }
    )

    assert section.status == "blocked"
    assert section.blockers == [
        "replay_concentration_warning: recent replay evidence is concentrated"
    ]
    assert section.details["stale_accumulating_candidate_count"] == 2
    assert section.details["replay_concentration"]["top_active_candidate"] == "abc"


def test_ds_e1_clean_window_report_surfaces_measurement_port(
    monkeypatch,
) -> None:
    monkeypatch.setattr(report_mod, "_pgrep", lambda pattern: [])
    monkeypatch.setattr(report_mod, "_pgrep_exact", lambda name: [])
    monkeypatch.setattr(report_mod, "_tcp_port_accepting", lambda port: True)

    report = report_mod.ds_e1_clean_window_report()

    assert report["ready"] is False
    assert report["measurement_port"] == 8194
    assert report["measurement_port_in_use"] is True
    assert report["blockers"] == [
        "measurement port 8194 is already accepting connections"
    ]


def test_xmas_section_accepts_promote_candidate_ab(monkeypatch, tmp_path: Path) -> None:
    config = tmp_path / "classifier_config.yaml"
    config.write_text(
        """
xmas_routing:
  mode: "enforce"
  winner_table_path: "xmas_winner_table.yaml"
  require_complete_table: true
""",
        encoding="utf-8",
    )
    table = tmp_path / "xmas_winner_table.yaml"
    table.write_text("placeholder: true\n", encoding="utf-8")
    ab_root = tmp_path / "ab"
    run = ab_root / "run"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(
        '{"decision": {"status": "promote_candidate", "blockers": []}, '
        '"xmas_policy": "incumbent_constrained_cheapfirst_v2"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(report_mod, "validate_xmas_config", lambda path: [])
    monkeypatch.setattr(report_mod, "validate_xmas_table", lambda path, **kwargs: [])

    section = report_mod.xmas_section(
        config_path=config,
        candidate_table_path=table,
        ab_root=ab_root,
        quiet_window={"ready": True, "blockers": []},
    )

    assert section.status == "ready"
    assert section.blockers == []
    assert section.details["latest_ab_decision_status"] == "promote_candidate"
    assert section.details["latest_ab_policy"] == (
        "incumbent_constrained_cheapfirst_v2"
    )
    assert section.details["latest_ab_ready"] is True


def test_xmas_section_promote_candidate_off_mode_waits_for_enablement(
    monkeypatch, tmp_path: Path
) -> None:
    config = tmp_path / "classifier_config.yaml"
    config.write_text(
        """
xmas_routing:
  mode: "off"
  winner_table_path: "xmas_winner_table.yaml"
  require_complete_table: true
""",
        encoding="utf-8",
    )
    table = tmp_path / "xmas_winner_table.yaml"
    table.write_text("placeholder: true\n", encoding="utf-8")
    ab_root = tmp_path / "ab"
    run = ab_root / "run"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(
        '{"decision": {"status": "promote_candidate", "blockers": []}, '
        '"xmas_policy": "incumbent_constrained_cheapfirst_v2"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(report_mod, "validate_xmas_config", lambda path: [])
    monkeypatch.setattr(report_mod, "validate_xmas_table", lambda path, **kwargs: [])

    section = report_mod.xmas_section(
        config_path=config,
        candidate_table_path=table,
        ab_root=ab_root,
        quiet_window={"ready": True, "blockers": []},
    )

    assert section.status == "blocked"
    assert section.blockers == ["xmas_routing.mode is off; enforce remains default-off"]
    assert section.details["latest_ab_ready"] is True
    assert section.details["latest_ab_decision_status"] == "promote_candidate"


def test_xmas_section_blocks_promote_candidate_from_legacy_policy(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = tmp_path / "classifier_config.yaml"
    config.write_text(
        """
xmas_routing:
  mode: "enforce"
  winner_table_path: "xmas_winner_table.yaml"
  require_complete_table: true
""",
        encoding="utf-8",
    )
    table = tmp_path / "xmas_winner_table.yaml"
    table.write_text("placeholder: true\n", encoding="utf-8")
    ab_root = tmp_path / "ab"
    run = ab_root / "run"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(
        '{"decision": {"status": "promote_candidate", "blockers": []}, '
        '"xmas_policy": "unknown_legacy"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(report_mod, "validate_xmas_config", lambda path: [])
    monkeypatch.setattr(report_mod, "validate_xmas_table", lambda path, **kwargs: [])

    section = report_mod.xmas_section(
        config_path=config,
        candidate_table_path=table,
        ab_root=ab_root,
        quiet_window={"ready": True, "blockers": []},
    )

    assert section.status == "blocked"
    assert section.blockers == [
        "latest X-MAS held-out A/B policy is "
        "unknown_legacy; required incumbent_constrained_cheapfirst_v2"
    ]


def test_xmas_next_action_ready_when_only_evidence_is_missing() -> None:
    section = report_mod.GateSection(
        key="xmas_production_path",
        status="blocked",
        summary="blocked",
        blockers=["latest X-MAS held-out A/B decision is hold"],
        details={"quiet_window_ready": True, "quiet_window_blockers": []},
    )

    actions = report_mod.build_next_actions([section])

    assert actions == [
        {
            "key": "run_xmas_constrained_policy_ab",
            "priority": "P0",
            "status": "ready",
            "reason": (
                "X-MAS enforce needs a fresh held-out A/B carrying "
                "incumbent_constrained_cheapfirst_v2 and a promote_candidate verdict."
            ),
            "requires": (
                "attested quiet window; runner preflight refuses AutoPilot "
                "and competing benchmark coordinators"
            ),
            "blocked_by": [],
            "evidence_blockers": ["latest X-MAS held-out A/B decision is hold"],
            "prompt_manifest": (
                "benchmarks/results/runs/xmas_live_ab/"
                "20260618-heldout-resilient/prompts.jsonl"
            ),
            "required_policy": "incumbent_constrained_cheapfirst_v2",
            "command": (
                "cd /mnt/raid0/llm/epyc-orchestrator && "
                "uv run python scripts/benchmark/xmas_live_ab.py "
                "--prompts benchmarks/results/runs/xmas_live_ab/"
                "20260618-heldout-resilient/prompts.jsonl "
                "--reps 2 --host-quiet-confirmed "
                "--output benchmarks/results/runs/xmas_live_ab/"
                "$(date -u +%Y%m%dT%H%M%SZ)-constrained-policy"
            ),
        }
    ]


def test_xmas_next_action_enablement_when_repaired_ab_passed() -> None:
    section = report_mod.GateSection(
        key="xmas_production_path",
        status="blocked",
        summary="blocked",
        blockers=["xmas_routing.mode is off; enforce remains default-off"],
        details={
            "latest_ab_ready": True,
            "latest_ab_policy": "incumbent_constrained_cheapfirst_v2",
            "latest_ab_decision_status": "promote_candidate",
            "latest_ab_summary_path": "benchmarks/results/runs/xmas_live_ab/run/summary.json",
            "latest_ab_results_path": "benchmarks/results/runs/xmas_live_ab/run/results.jsonl",
            "latest_ab_score_delta": 0.1,
            "latest_ab_latency_ratio": 0.938,
        },
    )

    actions = report_mod.build_next_actions([section])

    assert len(actions) == 1
    action = actions[0]
    assert action["key"] == "decide_xmas_enforce_enablement"
    assert action["status"] == "ready"
    assert action["blocked_by"] == []
    assert action["evidence_blockers"] == [
        "xmas_routing.mode is off; enforce remains default-off"
    ]
    assert action["latest_ab_decision_status"] == "promote_candidate"
    assert action["required_policy"] == "incumbent_constrained_cheapfirst_v2"
    assert "--summarize-results benchmarks/results/runs/xmas_live_ab/run/results.jsonl" in action["command"]


def test_a9_next_action_ready_when_collection_window_clear() -> None:
    section = report_mod.GateSection(
        key="a9_pairwise_collection",
        status="ready",
        summary="ready",
        blockers=[],
        details={
            "ready": True,
            "status": "ready",
            "manifest_path": "/tmp/a9_manifest.json",
            "batch_count": 9,
            "post_collection_step_count": 7,
            "source_plan_decision": {"status": "expansion_plan_ready"},
        },
    )

    actions = report_mod.build_next_actions([section])

    assert len(actions) == 1
    action = actions[0]
    assert action["key"] == "run_a9_pairwise_collection_window"
    assert action["priority"] == "P1"
    assert action["status"] == "ready"
    assert action["blocked_by"] == []
    assert action["manifest"] == "/tmp/a9_manifest.json"
    assert action["batch_count"] == 9
    assert action["post_collection_step_count"] == 7
    assert action["source_plan_decision"] == {"status": "expansion_plan_ready"}
    assert "collect_offline_reward_pairwise_expanded_gap.sh" in action["command"]
    assert "offline_reward_pairwise_collection_status.py" in action["follow_up"]


def test_w8_next_action_when_restart_ready_without_promotion_finalization() -> None:
    sections = [
        report_mod.GateSection(
            key="phase_health",
            status="ready",
            summary="active",
            blockers=[],
            details={"status": "active"},
        ),
        report_mod.GateSection(
            key="w4_w6_restart_cutover",
            status="ready",
            summary="ready",
            blockers=[],
            details={
                "w8_promotion_status": "pending_fresh_eval",
                "w8_open_requirements": [
                    "pending_fresh_eval_queued",
                    "combined_E_below_required",
                    "fresh_promotion_eval_required",
                ],
                "w8_pending_candidate": "candidate-a",
                "w8_pending_source_trial_id": 41,
                "w8_pending_attempts": 1,
                "w8_last_blocked_reason": None,
                "w8_latest_seq_trial_id": 40,
                "w8_latest_combined_E": 32.0,
                "w8_latest_required_E": 100.0,
                "w8_latest_fresh_eval": False,
                "w8_latest_baseline_reference_state": "fresh",
            },
        ),
    ]

    actions = report_mod.build_next_actions(sections)

    assert actions[0]["key"] == "collect_w8_promotion_eval_evidence"
    assert actions[0]["status"] == "active"
    assert actions[0]["evidence"]["w8_promotion_status"] == "pending_fresh_eval"
    assert actions[0]["evidence"]["open_requirements"] == [
        "pending_fresh_eval_queued",
        "combined_E_below_required",
        "fresh_promotion_eval_required",
    ]
    assert actions[0]["evidence"]["pending_candidate"] == "candidate-a"
    assert actions[0]["evidence"]["latest_seq_trial_id"] == 40
    assert actions[0]["evidence"]["latest_combined_E"] == 32.0
    assert actions[0]["evidence"]["latest_required_E"] == 100.0
    assert actions[0]["evidence"]["latest_fresh_eval"] is False
    assert actions[0]["evidence"]["latest_baseline_reference_state"] == "fresh"


def test_quiet_window_process_matcher_ignores_script_names_in_prompts() -> None:
    planner_line = (
        "123 claude -p Use scripts/benchmark/seed_specialist_routing.py "
        "only when the clean window is approved"
    )
    real_line = (
        "456 uv run python scripts/benchmark/seed_specialist_routing.py "
        "--dry-run"
    )

    assert not report_mod._process_line_matches_pattern(
        planner_line,
        "seed_specialist_routing.py",
    )
    assert report_mod._process_line_matches_pattern(
        real_line,
        "seed_specialist_routing.py",
    )


def test_render_markdown_surfaces_section_details() -> None:
    rendered = report_mod.render_markdown(
        {
            "ready": False,
            "blockers": ["gate: blocked"],
            "next_actions": [
                {
                    "key": "run_gate",
                    "priority": "P0",
                    "status": "blocked",
                    "reason": "needs evidence",
                    "blocked_by": ["window busy"],
                    "command": "python3 gate.py --strict",
                }
            ],
            "sections": [
                {
                    "key": "gate",
                    "status": "blocked",
                    "summary": "blocked",
                    "blockers": ["blocked"],
                    "details": {"count": 1},
                }
            ],
        }
    )

    assert "# Fable5 Gate Report" in rendered
    assert "Ready: false" in rendered
    assert "- gate: blocked" in rendered
    assert "## Next Actions" in rendered
    assert "### run_gate" in rendered
    assert "- Command: `python3 gate.py --strict`" in rendered
    assert "### gate" in rendered
    assert '`count`: 1' in rendered


def test_cli_strict_returns_one_when_gate_blocks(tmp_path: Path, monkeypatch, capsys) -> None:
    state = tmp_path / "state.json"
    journal = tmp_path / "journal.jsonl"
    phase = tmp_path / "phase.json"
    state.write_text('{"trial_counter": 1}\n', encoding="utf-8")
    journal.write_text("", encoding="utf-8")
    phase.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(report_mod, "_load_jsonl", lambda path: [])
    phase_kwargs = {}

    def fake_phase_health(**kwargs):
        phase_kwargs.update(kwargs)
        return {"ok": False, "blockers": ["stale"]}

    monkeypatch.setattr(report_mod, "build_phase_health_report", fake_phase_health)
    monkeypatch.setattr(report_mod, "build_ds_e1_packet", lambda: {"ready_for_profile_decision": True, "blockers": [], "sections": []})
    monkeypatch.setattr(
        report_mod,
        "build_fable5_gate_report",
        lambda **kwargs: {
            "ready": False,
            "blockers": ["phase_health: stale"],
            "sections": [],
        },
    )

    rc = report_mod.main(
        [
            "--state",
            str(state),
            "--journal",
            str(journal),
            "--phase",
            str(phase),
            "--json",
            "--strict",
        ]
    )

    assert rc == 1
    assert phase_kwargs["require_current_code"] is True
    assert "phase_health: stale" in capsys.readouterr().out


def test_cli_writes_json_and_markdown_outputs(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    state = tmp_path / "state.json"
    journal = tmp_path / "journal.jsonl"
    phase = tmp_path / "phase.json"
    out_json = tmp_path / "reports" / "fable5_gate.json"
    out_md = tmp_path / "reports" / "fable5_gate.md"
    state.write_text('{"trial_counter": 1}\n', encoding="utf-8")
    journal.write_text("", encoding="utf-8")
    phase.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(report_mod, "_load_jsonl", lambda path: [])
    monkeypatch.setattr(
        report_mod,
        "build_phase_health_report",
        lambda **kwargs: {"ok": False, "blockers": ["stale"]},
    )
    monkeypatch.setattr(
        report_mod,
        "build_ds_e1_packet",
        lambda: {"ready_for_profile_decision": True, "blockers": [], "sections": []},
    )
    monkeypatch.setattr(
        report_mod,
        "build_fable5_gate_report",
        lambda **kwargs: {
            "ready": False,
            "blockers": ["phase_health: stale"],
            "next_actions": [],
            "sections": [
                {
                    "key": "phase_health",
                    "status": "blocked",
                    "summary": "stale",
                    "blockers": ["stale"],
                    "details": {},
                }
            ],
        },
    )

    rc = report_mod.main(
        [
            "--state",
            str(state),
            "--journal",
            str(journal),
            "--phase",
            str(phase),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
            "--strict",
        ]
    )

    assert rc == 1
    assert "# Fable5 Gate Report" in capsys.readouterr().out
    assert '"ready": false' in out_json.read_text(encoding="utf-8")
    assert "- phase_health: stale" in out_md.read_text(encoding="utf-8")
