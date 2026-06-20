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
        "build_restart_readiness_report",
        lambda state, rows, **kwargs: {
            "restart_ready": False,
            "blockers": ["sequential verdict cutover readiness is blocked"],
            "summary": {
                "seq_cutover_ready": False,
                "seq_trusted_vector_trials": 62,
                "seq_shadow_rows": 10,
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
    assert "xmas_production_path: latest X-MAS held-out A/B decision is hold" in report[
        "blockers"
    ]
    xmas = [section for section in report["sections"] if section["key"] == "xmas_production_path"][0]
    assert xmas["details"]["latest_ab_decision_status"] == "hold"
    assert xmas["details"]["latest_ab_latency_ratio"] == 16.18
    assert report["sections"][0]["key"] == "phase_health"
    assert report["sections"][0]["status"] == "ready"


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
        '{"decision": {"status": "promote_candidate", "blockers": []}}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(report_mod, "validate_xmas_config", lambda path: [])
    monkeypatch.setattr(report_mod, "validate_xmas_table", lambda path, **kwargs: [])

    section = report_mod.xmas_section(
        config_path=config,
        candidate_table_path=table,
        ab_root=ab_root,
    )

    assert section.status == "ready"
    assert section.blockers == []
    assert section.details["latest_ab_decision_status"] == "promote_candidate"


def test_render_markdown_surfaces_section_details() -> None:
    rendered = report_mod.render_markdown(
        {
            "ready": False,
            "blockers": ["gate: blocked"],
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
