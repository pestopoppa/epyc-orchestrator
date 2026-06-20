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


def test_ri10_section_reports_config_but_missing_decision_data(tmp_path: Path) -> None:
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

    section = packet_mod.ri10_canary_section(config)

    assert section.status == "missing_data"
    assert section.details["mode"] == "canary"
    assert section.details["canary_ratio"] == 0.25
    assert section.details["canary_roles"] == ["frontdoor"]


def test_kv_measurement_section_flags_missing_series(tmp_path: Path) -> None:
    section = packet_mod.kv_measurement_section(root=tmp_path, patterns=("missing*",))

    assert section.status == "missing"
    assert section.details["required_contexts"] == ["2K", "8K", "32K"]


def test_kv_measurement_section_finds_recursive_relative_candidates(tmp_path: Path) -> None:
    target = tmp_path / ".." / "epyc-inference-research" / "data" / "dynamic_stack" / "run1"
    target.mkdir(parents=True)
    artifact = target / "kv_2k_8k_32k.json"
    artifact.write_text("{}", encoding="utf-8")

    section = packet_mod.kv_measurement_section(
        root=tmp_path,
        patterns=("../epyc-inference-research/data/dynamic_stack/**/kv*",),
    )

    assert section.status == "candidate"
    assert section.details["paths"] == [str(artifact.resolve())]


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
