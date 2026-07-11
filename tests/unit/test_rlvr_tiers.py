"""Tests for AP-27 deterministic RLVR tier reward contracts."""

from __future__ import annotations

from types import SimpleNamespace

from src.autopilot_core.rlvr_tiers import (
    RLVR_REWARD_POLICY,
    rlvr_reward_from_result,
    spec_for_rlvr_tier,
)


def _result(**kw) -> SimpleNamespace:
    base = {
        "tier": 1,
        "quality": 0.0,
        "reliability": 1.0,
        "ece": 0.0,
        "auroc": 1.0,
        "question_results": [],
    }
    base.update(kw)
    return SimpleNamespace(**base)


def test_t0_reward_is_binary_state_match() -> None:
    passed = rlvr_reward_from_result(_result(tier=0, quality=3.0, reliability=1.0))
    failed = rlvr_reward_from_result(_result(tier=0, quality=2.7, reliability=1.0))

    assert passed.policy == RLVR_REWARD_POLICY
    assert passed.reward_signal == "binary_outcome"
    assert passed.reward == 1.0
    assert passed.ready_for_training
    assert failed.reward == 0.0


def test_t1_reward_uses_calibration_and_discrimination() -> None:
    good = rlvr_reward_from_result(
        _result(tier=1, quality=2.4, reliability=0.9, ece=0.05, auroc=0.85)
    )
    poorly_calibrated = rlvr_reward_from_result(
        _result(tier=1, quality=2.4, reliability=0.9, ece=0.45, auroc=0.55)
    )

    assert good.reward_signal == "calibrated_continuous"
    assert good.ready_for_training
    assert good.reward > poorly_calibrated.reward
    assert good.components == {
        "accuracy": 0.7999999999999999,
        "reliability": 0.9,
        "calibration": 0.95,
        "discrimination": 0.85,
    }


def test_t1_reward_reports_missing_calibration_blockers() -> None:
    reward = rlvr_reward_from_result(_result(tier=1, quality=2.0, ece=float("nan"), auroc=0.0))

    assert reward.reward_signal == "calibrated_continuous"
    assert not reward.ready_for_training
    assert reward.blockers == ("ece_missing", "auroc_missing_or_degenerate")


def test_t2_reward_requires_process_rows_and_penalizes_broken_process() -> None:
    clean = rlvr_reward_from_result(
        _result(
            tier=2,
            quality=2.0,
            reliability=0.75,
            ece=0.1,
            auroc=0.8,
            question_results=[
                {"qid": "q1", "correct": True},
                {"qid": "q2", "correct": False, "partial": True},
            ],
        )
    )
    missing = rlvr_reward_from_result(
        _result(tier=2, quality=2.0, reliability=0.75, ece=0.1, auroc=0.8)
    )

    assert clean.reward_signal == "process_attributed"
    assert clean.metrics["process_integrity"] == 0.5
    assert clean.ready_for_training
    assert not missing.ready_for_training
    assert missing.blockers == ("question_results_missing",)


def test_unknown_higher_tier_uses_process_contract() -> None:
    spec = spec_for_rlvr_tier(3)
    reward = rlvr_reward_from_result(
        _result(
            tier=3,
            quality=3.0,
            reliability=1.0,
            ece=0.0,
            auroc=1.0,
            question_results=[{"qid": "q1", "correct": True}],
        )
    )

    assert spec.reward_signal == "process_attributed"
    assert reward.ready_for_training
    assert reward.reward == 1.0
