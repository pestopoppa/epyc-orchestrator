"""RTG-09: wall-clock task-duration is the PRIMARY speed axis in compute_reward.

Precondition for DAR-5 (decision-aware-routing.md): the reward must carry a
speed axis that prices orchestration/tool overhead, not just tokens/sec (which
is gameable through tool calls and blind to it -- DAR handoff measured median
wall/model-compute overhead 1.60x, p90 9.09x over 19,433 tasks).

These tests exercise `compute_reward`'s new `task_duration_s` parameter and
`ScoringConfig.baseline_duration_by_role` in isolation, mirroring the shape of
`test_q_reward_role_key.py` (the sibling regression suite for the tokens/sec
axis's own "silent miss" defect).
"""

from __future__ import annotations

import logging

from orchestration.repl_memory.q_reward import compute_reward
from orchestration.repl_memory.q_scorer import ScoringConfig


class _Entry:
    def __init__(self, outcome: str, data: dict) -> None:
        self.outcome = outcome
        self.data = data
        self.event_type = None


_DURATION_BASELINE = {
    "worker_general": {"p50_s": 10.0, "p90_s": 50.0},
}


def _config(**overrides) -> ScoringConfig:
    overrides.setdefault("baseline_duration_by_role", _DURATION_BASELINE)
    # Isolate the duration dimension: zero every other cost dimension so the
    # reward delta is attributable to duration alone.
    overrides.setdefault("cost_penalty_lambda", 0.0)
    overrides.setdefault("cost_lambda_quality_gap", 0.0)
    overrides.setdefault("cost_lambda_memory", 0.0)
    overrides.setdefault("baseline_tps_by_role", {})
    return ScoringConfig(**overrides)


def _reward(task_duration_s, config=None, role="worker_general"):
    data = {"producer_role": role}
    return compute_reward(
        _Entry("success", data), [], [], None, data,
        config=config or _config(), task_duration_s=task_duration_s,
    )


def test_at_p50_no_duration_penalty():
    """Wall-clock exactly at the role's measured p50 -> zero duration penalty."""
    assert _reward(10.0) == 1.0


def test_faster_than_p50_no_duration_penalty():
    assert _reward(2.0) == 1.0


def test_between_p50_and_p90_graded_penalty():
    """Halfway between p50 (10s) and p90 (50s) -> half of cost_lambda_duration."""
    cfg = _config(cost_lambda_duration=0.20)
    r = _reward(30.0, config=cfg)  # (30-10)/(50-10) = 0.5
    assert r == 1.0 - 0.5 * 0.20


def test_at_p90_full_weight_penalty():
    cfg = _config(cost_lambda_duration=0.20)
    r = _reward(50.0, config=cfg)
    assert abs(r - (1.0 - 0.20)) < 1e-9


def test_beyond_p90_saturates_does_not_exceed_full_weight():
    """Far past p90 must not exceed the dimension's own weight (graded, capped)."""
    cfg = _config(cost_lambda_duration=0.20)
    r_at_p90 = _reward(50.0, config=cfg)
    r_way_past = _reward(5000.0, config=cfg)
    assert r_way_past == r_at_p90


def test_correctness_gated_no_penalty_on_failure():
    cfg = _config(cost_lambda_duration=0.20)
    data = {"producer_role": "worker_general"}
    r = compute_reward(
        _Entry("failure", data), [], [], None, data,
        config=cfg, task_duration_s=5000.0,
    )
    assert r == -0.5  # failure_reward, no cost dimension applied at all


def test_missing_task_duration_arg_skips_dimension_and_warns(caplog):
    """Caller not wiring task_duration_s must not silently vanish the axis."""
    from orchestration.repl_memory import q_reward

    q_reward._warned_missing_duration_arg = False
    cfg = _config(cost_lambda_duration=0.20)
    with caplog.at_level(logging.WARNING):
        r = _reward(None, config=cfg)
    assert r == 1.0  # dimension skipped, not defaulted to a penalty
    assert any("did not pass task_duration_s" in rec.message for rec in caplog.records)


def test_role_missing_duration_baseline_skips_and_warns_not_silent(caplog):
    """A role absent from baseline_duration_by_role must warn, not silently
    award full reward on the duration axis -- the same defect shape as the
    role-key/baseline_tps miss this file's sibling regresses against."""
    from orchestration.repl_memory import q_reward

    q_reward._warned_unpriced_duration_roles.clear()
    cfg = _config(cost_lambda_duration=0.20)
    with caplog.at_level(logging.WARNING):
        r = _reward(5000.0, config=cfg, role="brand_new_unregistered_role")
    assert r == 1.0
    assert any(
        "no baseline_duration_by_role entry" in rec.message for rec in caplog.records
    )


def test_known_unpriced_roles_do_not_warn_on_duration(caplog):
    from orchestration.repl_memory import q_reward

    q_reward._warned_unpriced_duration_roles.clear()
    cfg = _config(cost_lambda_duration=0.20)
    with caplog.at_level(logging.WARNING):
        _reward(5000.0, config=cfg, role="mock")
    assert not [
        r for r in caplog.records if "no baseline_duration_by_role" in r.message
    ]


def test_cost_lambda_duration_outweighs_demoted_tokens_per_sec_default():
    """RTG-09 contract: duration (primary) > tokens/sec (secondary) by default."""
    cfg = ScoringConfig()
    assert cfg.cost_lambda_duration > cfg.cost_penalty_lambda


def test_degenerate_baseline_p90_equals_p50_falls_back_to_ratio():
    """A low-n role whose p50/p90 collapsed to the same value must not divide
    by zero -- falls back to a plain ratio-past-p50 formula."""
    cfg = _config(
        cost_lambda_duration=0.20,
        baseline_duration_by_role={"worker_general": {"p50_s": 10.0, "p90_s": 10.0}},
    )
    # 2x the baseline -> ratio 1.0 past p50, capped at full weight.
    r = _reward(20.0, config=cfg)
    assert abs(r - (1.0 - 0.20)) < 1e-9
