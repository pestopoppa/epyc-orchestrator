"""HIL-17 metrics and k-sample aggregator: hand-computed expectations (scripted judges, no model)."""

import json
import math

import pytest

from src.trace.attribution.aggregate import (
    AggregatorConfig,
    aggregate_case,
    parse_steps,
    vote_share,
)
from src.trace.attribution.case import consensus_from_labels
from src.trace.attribution.judge import JudgeSpec, ScriptedJudge
from src.trace.attribution.metrics import (
    ndcg,
    pick_accuracy,
    random_ndcg_expected,
    random_ndcg_mc,
)
from src.trace.attribution.schema import AgentEvent, AgentTrajectory, EventType

D2 = 1 / math.log2(3)  # discount at rank 2 = 0.63093
SUM_D5 = 1 + D2 + 0.5 + 1 / math.log2(5) + 1 / math.log2(6)  # 2.94846


def test_ndcg_exp_and_linear_hand_computed():
    rel = {3: 1.0, 5: 0.5}
    pred = {5: 0.9, 3: 0.4}  # ranks 5 first, the top-consensus step second
    # exp: DCG = (2^.5-1)/1 + (2^1-1)*D2 ; IDCG = 1 + (2^.5-1)*D2
    assert ndcg(pred, rel, 5, "exp") == pytest.approx(0.8285978, abs=1e-6)
    # linear: DCG = .5 + 1*D2 ; IDCG = 1 + .5*D2
    assert ndcg(pred, rel, 5, "linear") == pytest.approx(0.8597187, abs=1e-6)
    assert ndcg({3: 1.0, 5: 0.5}, rel, 5) == pytest.approx(1.0)


def test_ties_charged_at_expected_dcg():
    rel = {2: 1.0}
    assert ndcg({1: 1.0, 2: 1.0}, rel, 5, tie="expected") == pytest.approx(
        0.5 * (1 + D2)
    )  # 0.815465
    assert ndcg({1: 1.0, 2: 1.0}, rel, 5, tie="index") == pytest.approx(D2)


def test_cutoff_and_undefined():
    pred = {i: 10.0 - i for i in range(7)}  # step 5 lands at rank 6
    assert ndcg(pred, {5: 1.0}, 5) == 0.0
    assert ndcg(pred, {5: 1.0}, None) == pytest.approx(1 / math.log2(7))
    assert ndcg(pred, {}, 5) is None
    assert ndcg({}, {1: 1.0}, 5) == 0.0


def test_random_baseline_expected_is_exact_and_mc_agrees():
    steps = list(range(10))
    exp = random_ndcg_expected(steps, {4: 1.0}, 5)
    assert exp == pytest.approx(0.1 * SUM_D5)  # 0.294846
    assert random_ndcg_mc(steps, {4: 1.0}, 5, draws=20000, seed=1) == pytest.approx(exp, abs=0.01)
    # fewer steps than K: every step is ranked
    assert random_ndcg_expected([0, 1], {0: 1.0}, 5) == pytest.approx(0.5 * (1 + D2))
    assert random_ndcg_expected(steps, {}, 5) is None


def test_pick_accuracy():
    rel = {4: 1.0, 9: 1 / 3}
    assert pick_accuracy(4, rel) == {"abstain": False, "exact": True, "within1": True, "hit": True}
    assert pick_accuracy(5, rel) == {
        "abstain": False,
        "exact": False,
        "within1": True,
        "hit": False,
    }
    assert pick_accuracy(9, rel) == {
        "abstain": False,
        "exact": False,
        "within1": False,
        "hit": True,
    }
    assert pick_accuracy(None, rel)["abstain"] is True


def test_consensus_and_vote_share():
    assert consensus_from_labels([[2, 5], [2], [2, 7, 7]]) == pytest.approx(
        {2: 1.0, 5: 1 / 3, 7: 1 / 3}
    )
    vs = vote_share([{1, 2}, {2}, None, {2, 3}, {1}])  # the unparsed draw stays in the denominator
    assert vs == pytest.approx({1: 0.4, 2: 0.6, 3: 0.2})


def test_parse_steps():
    assert parse_steps(
        '```json\n{"steps": [{"step": 2}, {"step": "3"}, {"step": 99}]}\n```', {1, 2, 3}
    ) == ({2, 3}, 1)
    assert parse_steps("no json here", {1}) == (None, 0)
    assert parse_steps('{"steps": []}', {1}) == (set(), 0)


def _traj(n: int) -> AgentTrajectory:
    return AgentTrajectory(
        "t",
        goal="g",
        events=[
            AgentEvent(f"e{i}", "t", EventType.AGENT_STEP, "a", step_index=i, output=f"o{i}")
            for i in range(n)
        ],
    )


def _seq(answers):
    it = iter(answers)

    def responder(req):
        if req.temperature == 0.0:
            return json.dumps({"steps": [{"step": 4}, {"step": 3}]})
        nxt = next(it)
        return nxt if isinstance(nxt, str) else json.dumps({"steps": [{"step": s} for s in nxt]})

    return responder


def test_aggregator_hand_computed():
    a = ScriptedJudge(_seq([[2], [2], [3], [2, 3], "garbage"]), 100, 10)
    b = ScriptedJudge(_seq([[3], [2], [2], [4], [2]]), 120, 12)
    fams = [JudgeSpec("A", a, "fam-a", True), JudgeSpec("B", b, "fam-b", False)]
    res = aggregate_case(
        "t", _traj(6), fams, {3: 1.0}, AggregatorConfig(k=5, temperature=0.7, seed_base=0)
    )
    # 10 draws: step 2 named 6x, step 3 3x, step 4 1x
    assert res.vote_share == pytest.approx({2: 0.6, 3: 0.3, 4: 0.1})
    assert res.ndcg["ksample_exp"] == pytest.approx(D2)  # gold at rank 2
    assert res.ndcg["temp0_exp"] == pytest.approx(0.5 * (1 + D2))  # temp-0 names {4,3}: a tie
    assert res.ndcg["random_expected_exp"] == pytest.approx(SUM_D5 / 6)  # 0.491410
    assert res.temp0_steps == {3, 4}
    assert res.per_family["A"]["parse_failures"] == 1 and res.per_family["A"]["quantised"] is True
    assert res.tokens_ksample.total == 5 * 110 + 5 * 132 and res.tokens_ksample.calls == 10
    assert res.tokens_temp0.total == 110
    assert [r.seed for r in a.calls[:5]] == [0, 1, 2, 3, 4] and [r.seed for r in b.calls] == [
        5,
        6,
        7,
        8,
        9,
    ]
    assert all(r.temperature == 0.7 for r in b.calls) and a.calls[-1].temperature == 0.0
