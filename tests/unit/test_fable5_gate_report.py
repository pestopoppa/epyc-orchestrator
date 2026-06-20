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
                "seq_shadow_rows": 10,
                "snapshot_restart_readiness": "tail_fold_ready",
                "w6_audit_cutover_ready": False,
                "w6_audited_trial_count": 34,
                "w6_min_audited_trials": 30,
                "w6_gaming_alarm": True,
                "w6_potential_overfit_divergences": 4,
                "baseline_seed_append_ready": True,
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
            "sections": [{"key": "kv_size_measurements", "status": "missing"}],
        },
        config_path=config,
        xmas_table_path=table,
        xmas_ab_root=ab_root,
    )

    assert report["ready"] is False
    assert "w4_w6_restart_cutover: sequential verdict cutover readiness is blocked" in report[
        "blockers"
    ]
    assert "ds_e1_dynamic_stack: kv_size_measurements: missing" in report["blockers"]
    assert "xmas_production_path: xmas_routing.mode is off; enforce remains default-off" in report[
        "blockers"
    ]
    assert (
        "xmas_production_path: latest X-MAS held-out A/B policy is "
        "unknown_legacy; required incumbent_constrained_v1"
    ) in report["blockers"]
    assert "xmas_production_path: latest X-MAS held-out A/B decision is hold" in report[
        "blockers"
    ]
    xmas = [section for section in report["sections"] if section["key"] == "xmas_production_path"][0]
    assert xmas["details"]["latest_ab_decision_status"] == "hold"
    assert xmas["details"]["latest_ab_policy"] == "unknown_legacy"
    assert xmas["details"]["required_ab_policy"] == "incumbent_constrained_v1"
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
    assert restart["details"]["snapshot_restart_readiness"] == "tail_fold_ready"
    assert restart["details"]["snapshot_payload_journal_max_trial_id"] == 895
    assert [action["key"] for action in report["next_actions"]] == [
        "continue_w4_w6_accrual",
        "run_ds_e1_kv_measurements",
        "collect_ri10_canary_arm_telemetry",
        "run_xmas_constrained_policy_ab",
    ]
    assert report["next_actions"][0]["status"] == "active"
    assert "restart_readiness_report.py" in report["next_actions"][0]["command"]
    assert "--require-seq-cutover --require-w6-audit" in report["next_actions"][0]["command"]
    assert report["next_actions"][0]["follow_up"] == (
        "python3 scripts/autopilot/fable5_gate_report.py --json --strict"
    )
    assert "ds_e1_kv_measurements.sh --execute" in report["next_actions"][1]["command"]
    assert report["next_actions"][1]["status"] == "blocked"
    assert report["next_actions"][2]["command"] == (
        "python3 scripts/analysis/ri10_canary_sample_report.py"
    )
    xmas_action = report["next_actions"][3]
    assert xmas_action["status"] == "blocked"
    assert xmas_action["blocked_by"] == ["active AutoPilot process(es): 123"]
    assert "latest X-MAS held-out A/B decision is hold" in xmas_action["evidence_blockers"]
    assert "xmas_live_ab.py" in xmas_action["command"]
    assert "<heldout_prompts.jsonl>" not in xmas_action["command"]
    assert xmas_action["prompt_manifest"] == (
        "benchmarks/results/runs/xmas_live_ab/20260618-heldout-resilient/prompts.jsonl"
    )
    assert xmas_action["required_policy"] == "incumbent_constrained_v1"
    assert f"--prompts {xmas_action['prompt_manifest']}" in xmas_action["command"]
    assert "$(date -u +%Y%m%dT%H%M%SZ)-constrained-policy" in xmas_action["command"]


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
            "eval_label": "T2",
            "eval_completed_questions": 200,
            "eval_total_questions": 500,
            "eval_correct_questions": 144,
            "eval_correct_pct": 72.0,
            "eval_concurrency": 1,
        }
    )

    assert section.status == "ready"
    assert "T2 200/500" in section.summary
    assert section.details["eval_completed_questions"] == 200
    assert section.details["eval_correct_pct"] == 72.0


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
        '"xmas_policy": "incumbent_constrained_v1"}\n',
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
    assert section.details["latest_ab_policy"] == "incumbent_constrained_v1"


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
        "unknown_legacy; required incumbent_constrained_v1"
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
                "incumbent_constrained_v1 and a promote_candidate verdict."
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
            "required_policy": "incumbent_constrained_v1",
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
    monkeypatch.setattr(report_mod, "build_phase_health_report", lambda path: {"ok": False, "blockers": ["stale"]})
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
        lambda path: {"ok": False, "blockers": ["stale"]},
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
