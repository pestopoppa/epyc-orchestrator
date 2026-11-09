"""SEQ-4 replay: allocators over a synthetic pool (no journal, no inference)."""
import pytest

from scripts.analysis.sequential_allocation_replay import ALLOCATORS, Arm, replay, run_all


def _pool():
    good = Arm(("good", "c"), [0.9] * 30, [f"2026-01-01T00:00:{i:02d}" for i in range(30)])
    bad = [Arm((f"bad{j}", "c"), [-0.3] * 3, [f"2026-01-02T00:0{j}:0{i}" for i in range(3)]) for j in range(4)]
    return [*bad, good]


def test_leader_reaches_confirm_and_never_exceeds_recorded_trials():
    arms = _pool()
    out = replay(arms, "leader_auc")
    assert out["confirmed"] == 1 and out["pulls_to_first_confirm"] is not None
    assert out["pulls"] <= sum(len(a.zs) for a in arms)
    assert arms[-1].pos == 0  # input arms are not mutated


def test_chronological_budget_rule_stops_flat_arm():
    flat = Arm(("flat", "c"), [0.0] * 20, [f"2026-01-01T00:00:{i:02d}" for i in range(20)])
    out = replay([flat], "chronological")
    assert out["confirmed"] == 0 and out["pulls"] == 8 and out["endpoint_max_E"] == pytest.approx(1.0)
    ucb = replay([flat], "ucb")
    assert ucb["pulls"] == 20  # relaxed stop rule: bandit may keep pulling a flat arm


def test_thompson_is_seeded_and_report_shape():
    a = replay(_pool(), "thompson", seed=7)
    b = replay(_pool(), "thompson", seed=7)
    assert a == b
    rep = run_all(_pool(), thompson_seeds=(0, 1))
    assert {r["allocator"] for r in rep["results"]} == set(ALLOCATORS)
    assert rep["n_arms"] == 5 and rep["arms_with_one_trial"] == 0
    with pytest.raises(ValueError):
        replay(_pool(), "nope")
