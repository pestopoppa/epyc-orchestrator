from __future__ import annotations

from pathlib import Path

import pytest

from scripts.analysis import ri10_canary_request_plan as plan


def _write_config(path: Path, *, mode: str = "canary") -> None:
    path.write_text(
        f"""
factual_risk:
  mode: {mode}
  canary_ratio: 0.25
  canary_roles: [frontdoor, worker_general]
  threshold_low: 0.3
  threshold_high: 0.7
  role_adjustments:
    tier_1: 0.727978
    tier_2: 0.824178
    tier_3: 1.0
  feature_weights:
    has_date_question: 0.15
    has_entity_question: 0.15
    has_citation_request: 0.10
    claim_density: 0.25
    factual_keyword_ratio: 0.20
    uncertainty_markers: 0.15
""",
        encoding="utf-8",
    )


def test_build_plan_balances_expected_canary_arms(tmp_path: Path) -> None:
    config = tmp_path / "classifier_config.yaml"
    _write_config(config)

    out = plan.build_plan(
        config_path=config,
        roles=["frontdoor"],
        per_role_per_arm=2,
        max_candidates=10_000,
    )

    assert out["request_count"] == 4
    assert out["selected_roles"] == ["frontdoor"]
    assert out["prompt_risk"]["frontdoor"]["risk_band"] == "high"
    assert [item["expected_factual_risk_mode"] for item in out["requests"]].count("enforce") == 2
    assert [item["expected_factual_risk_mode"] for item in out["requests"]].count("shadow") == 2

    for item in out["requests"]:
        payload = item["payload"]
        assert payload["force_role"] == "frontdoor"
        assert payload["force_mode"] == "direct"
        assert payload["request_id"] == item["request_id"]
        assert payload["request_priority"] == "background"
        assert payload["workload_class"] == "campaign"
        assert (
            plan.get_mode(
                plan.load_factual_risk_config(config),
                role="frontdoor",
                sample_key=item["request_id"],
            )
            == item["expected_factual_risk_mode"]
        )


def test_build_plan_rejects_non_canary_config(tmp_path: Path) -> None:
    config = tmp_path / "classifier_config.yaml"
    _write_config(config, mode="shadow")

    with pytest.raises(ValueError, match="must be canary"):
        plan.build_plan(config_path=config, roles=["frontdoor"])


def test_build_plan_rejects_non_high_prompt(tmp_path: Path) -> None:
    config = tmp_path / "classifier_config.yaml"
    _write_config(config)

    with pytest.raises(ValueError, match="not high factual-risk"):
        plan.build_plan(
            config_path=config,
            roles=["frontdoor"],
            prompt="hello",
        )
