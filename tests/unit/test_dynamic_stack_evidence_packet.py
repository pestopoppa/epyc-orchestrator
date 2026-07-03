"""Tests for the read-only Dynamic Stack DS-E1 evidence packet."""

from __future__ import annotations

from pathlib import Path

import scripts.server.dynamic_stack_evidence_packet as packet_mod


def test_stack_roster_section_packages_live_generated_roles(tmp_path: Path) -> None:
    priors = tmp_path / "stack_priors.yaml"
    priors.write_text(
        """
status: compiled
compiled_at: "2026-06-20T00:00:00Z"
source_artifacts:
  registry:
    repo_commit: abc1234
roles:
  frontdoor:
    deployment_status: live_stack
    model_id: qwen-frontdoor
    serving:
      endpoint: http://localhost:8070
      ports: [8070]
      tier: hot
      effective_context_tokens: 32768
    priors:
      throughput_tps: 24.3
    model:
      mem_gb: 37
  retired:
    deployment_status: retired
    model_id: old
""",
        encoding="utf-8",
    )

    section = packet_mod.stack_roster_section(priors)

    assert section.status == "ready"
    assert section.details["compiled_at"] == "2026-06-20T00:00:00Z"
    assert section.details["source_commit"] == "abc1234"
    assert [row["role"] for row in section.details["roles"]] == ["frontdoor"]
    assert section.details["roles"][0]["effective_context_tokens"] == 32768


def test_ds5_manifest_section_flags_stale_compile(tmp_path: Path) -> None:
    manifest = tmp_path / "MODEL_MANIFEST.md"
    manifest.write_text(
        "compiled at `2026-06-14T00:00:00Z`\n",
        encoding="utf-8",
    )
    priors = tmp_path / "stack_priors.yaml"
    priors.write_text('compiled_at: "2026-06-20T00:00:00Z"\n', encoding="utf-8")

    section = packet_mod.ds5_manifest_section(manifest, priors)

    assert section.status == "stale"
    assert section.details["manifest_compiled_at"] == "2026-06-14T00:00:00Z"
    assert section.details["stack_priors_compiled_at"] == "2026-06-20T00:00:00Z"


def test_ri10_section_reports_config_but_missing_decision_data(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "classifier_config.yaml"
    config.write_text(
        """
factual_risk:
  mode: canary
  canary_ratio: 0.25
  canary_roles: [frontdoor]
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(packet_mod, "ORCH_ROOT", tmp_path)

    section = packet_mod.ri10_canary_section(config)

    assert section.status == "missing_data"
    assert section.details["mode"] == "canary"
    assert section.details["canary_ratio"] == 0.25
    assert section.details["canary_roles"] == ["frontdoor"]


def test_ri10_section_flags_report_with_insufficient_arm_data(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "classifier_config.yaml"
    config.write_text(
        """
factual_risk:
  mode: canary
  canary_ratio: 0.25
  canary_roles: [frontdoor]
""",
        encoding="utf-8",
    )
    report_dir = tmp_path / "orchestration" / "reports"
    report_dir.mkdir(parents=True)
    report_path = report_dir / "ri10_canary_sample_report_20260620.json"
    report_path.write_text(
        """
{
  "sample_count_ready": true,
  "canary_decision_ready": false,
  "high_risk_rows_since_canary_start": 490,
  "decision_reason": "high-risk samples exist, but enforce/shadow canary arm telemetry is not observable"
}
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(packet_mod, "ORCH_ROOT", tmp_path)

    section = packet_mod.ri10_canary_section(config)

    assert section.status == "insufficient_data"
    assert section.details["report_path"] == str(report_path)
    assert section.details["report_summary"]["high_risk_rows_since_canary_start"] == 490


def test_ri10_section_requires_decision_grade_arm_report(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "classifier_config.yaml"
    config.write_text(
        """
factual_risk:
  mode: canary
  canary_ratio: 0.25
  canary_roles: [frontdoor]
""",
        encoding="utf-8",
    )
    report_dir = tmp_path / "orchestration" / "reports"
    report_dir.mkdir(parents=True)
    report_path = report_dir / "ri10_canary_sample_report_20260703.json"
    report_path.write_text(
        """
{
  "sample_count_ready": true,
  "canary_arm_sample_count_ready": false,
  "canary_arm_balance_ready": false,
  "canary_decision_ready": false,
  "high_risk_rows_since_canary_start": 463,
  "evaluable_canary_arm_high_risk_rows": 19,
  "canary_role_missing_factual_risk_mode_high_risk_rows": 444,
  "canary_role_factual_risk_modes_since_canary_start": {
    "enforce": 1,
    "shadow": 18,
    "<missing>": 444
  },
  "canary_arm_counts_since_canary_start": {
    "enforce_high_risk": 1,
    "shadow_high_risk": 18
  },
  "decision_reason": "only 19 high-risk rows have observable enforce/shadow canary arms; gate requires 50"
}
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(packet_mod, "ORCH_ROOT", tmp_path)

    section = packet_mod.ri10_canary_section(config)

    assert section.status == "insufficient_data"
    assert section.details["report_path"] == str(report_path)
    assert section.details["report_summary"]["evaluable_canary_arm_high_risk_rows"] == 19
    assert (
        section.details["report_summary"][
            "canary_role_missing_factual_risk_mode_high_risk_rows"
        ]
        == 444
    )


def test_ri10_section_surfaces_current_canary_role_scope_starvation(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "classifier_config.yaml"
    config.write_text(
        """
factual_risk:
  mode: canary
  canary_ratio: 0.25
  canary_roles: [frontdoor]
""",
        encoding="utf-8",
    )
    report_dir = tmp_path / "orchestration" / "reports"
    report_dir.mkdir(parents=True)
    report_path = report_dir / "ri10_canary_sample_report_20260703.json"
    report_path.write_text(
        """
{
  "sample_count_ready": true,
  "canary_decision_ready": false,
  "high_risk_rows_since_canary_start": 464,
  "telemetry_health_start": "2026-06-20",
  "high_risk_rows_since_telemetry_health_start": 20,
  "canary_role_high_risk_rows_since_telemetry_health_start": 2,
  "non_canary_role_high_risk_rows_since_telemetry_health_start": 18,
  "telemetry_producer_currently_healthy": true,
  "telemetry_canary_role_scope_starved": true,
  "telemetry_collection_blocker": "canary_role_scope_starved",
  "telemetry_collection_reason": "current factual-risk telemetry is populated"
}
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(packet_mod, "ORCH_ROOT", tmp_path)

    section = packet_mod.ri10_canary_section(config)

    assert section.status == "insufficient_data"
    assert "canary_roles are starving" in section.summary
    assert section.details["report_path"] == str(report_path)
    assert (
        section.details["report_summary"]["telemetry_collection_blocker"]
        == "canary_role_scope_starved"
    )
    assert (
        section.details["report_summary"][
            "non_canary_role_high_risk_rows_since_telemetry_health_start"
        ]
        == 18
    )


def test_kv_measurement_section_flags_missing_series(tmp_path: Path) -> None:
    section = packet_mod.kv_measurement_section(root=tmp_path, patterns=("missing*",))

    assert section.status == "missing"
    assert section.details["required_contexts"] == ["2K", "8K", "32K"]
    assert section.details["required_measurements"]["frontdoor"] == [
        2048,
        8192,
        32768,
    ]
    assert section.details["required_measurements"]["architect_general"] == [2048, 8192]
    assert "server_kv_size_mb" in section.details["expected_csv_columns"]
    assert "ds_e1_kv_measurements.sh --execute" in section.details["producer_command"]


def test_kv_measurement_section_finds_recursive_relative_candidates(tmp_path: Path) -> None:
    target = tmp_path / ".." / "epyc-inference-research" / "data" / "dynamic_stack" / "run1"
    target.mkdir(parents=True)
    artifact = target / "kv_2k_8k_32k.json"
    artifact.write_text("{}", encoding="utf-8")

    section = packet_mod.kv_measurement_section(
        root=tmp_path,
        patterns=("../epyc-inference-research/data/dynamic_stack/**/kv*",),
    )

    assert section.status == "incomplete"
    assert section.details["paths"] == [str(artifact.resolve())]
    assert section.details["missing_measurements"]["frontdoor"] == [2048, 8192, 32768]


def test_kv_measurement_section_rejects_partial_csv(tmp_path: Path) -> None:
    artifact = tmp_path / "kv_measurements.csv"
    artifact.write_text(
        "\n".join(
            [
                "role,model_id,model_path,context_length,max_context,ctk,ctv,hadamard,status,rss_load_mb,rss_after_prefill_mb,server_kv_size_mb,prompt_tokens,prompt_tps,log_file,notes",
                "frontdoor,qwen-frontdoor,/models/frontdoor.gguf,2048,32768,q4_0,f16,yes,ok,100,120,512,1500,20,frontdoor.log,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    section = packet_mod.kv_measurement_section(root=tmp_path, patterns=("kv*.csv",))

    assert section.status == "incomplete"
    assert section.details["observed_measurements"] == {"frontdoor": [2048]}
    assert section.details["missing_measurements"]["frontdoor"] == [8192, 32768]
    assert section.details["missing_measurements"]["architect_general"] == [2048, 8192]


def test_kv_measurement_section_accepts_complete_successful_csv(tmp_path: Path) -> None:
    artifact = tmp_path / "kv_measurements.csv"
    rows = [
        "role,model_id,model_path,context_length,max_context,ctk,ctv,hadamard,status,rss_load_mb,rss_after_prefill_mb,server_kv_size_mb,prompt_tokens,prompt_tps,log_file,notes"
    ]
    for role, contexts in packet_mod.REQUIRED_KV_MEASUREMENTS.items():
        for context in sorted(contexts):
            rows.append(
                f"{role},model,/models/{role}.gguf,{context},32768,q4_0,f16,yes,ok,100,120,512,1500,20,{role}.log,"
            )
    artifact.write_text("\n".join(rows) + "\n", encoding="utf-8")

    section = packet_mod.kv_measurement_section(root=tmp_path, patterns=("kv*.csv",))

    assert section.status == "ready"
    assert "missing_measurements" not in section.details
    assert section.details["observed_measurements"]["frontdoor"] == [2048, 8192, 32768]


def test_kv_measurement_section_rejects_zero_kv_size(tmp_path: Path) -> None:
    artifact = tmp_path / "kv_measurements.csv"
    rows = [
        "role,model_id,model_path,context_length,max_context,ctk,ctv,hadamard,status,rss_load_mb,rss_after_prefill_mb,server_kv_size_mb,prompt_tokens,prompt_tps,log_file,notes"
    ]
    for role, contexts in packet_mod.REQUIRED_KV_MEASUREMENTS.items():
        for context in sorted(contexts):
            kv_size = "0" if role == "frontdoor" and context == 2048 else "512"
            rows.append(
                f"{role},model,/models/{role}.gguf,{context},32768,q4_0,f16,yes,ok,100,120,{kv_size},1500,20,{role}.log,"
            )
    artifact.write_text("\n".join(rows) + "\n", encoding="utf-8")

    section = packet_mod.kv_measurement_section(root=tmp_path, patterns=("kv*.csv",))

    assert section.status == "incomplete"
    assert section.details["missing_measurements"]["frontdoor"] == [2048]
    assert section.details["failed_rows"][0]["reason"] == "measurement_not_successful"


def test_render_markdown_surfaces_blockers() -> None:
    packet = {
        "generated_at": "2026-06-20T00:00:00Z",
        "ready_for_profile_decision": False,
        "blockers": ["ri10_canary: missing"],
        "sections": [
            {
                "key": "ri10_canary",
                "status": "missing_data",
                "summary": "missing",
                "details": {"mode": "canary"},
            }
        ],
    }

    rendered = packet_mod.render_markdown(packet)

    assert "Ready for DS-7/DS-6 profile decision: false" in rendered
    assert "- ri10_canary: missing" in rendered
    assert "### ri10_canary" in rendered
