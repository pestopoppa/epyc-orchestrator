"""HIL-17 ladder rungs, escalation policy and per-rung token accounting (scripted judge, no model)."""

import json
import re

import pytest

from src.trace.attribution.judge import ScriptedJudge, extract_json, OpenAICompatJudge
from src.trace.attribution.ladder import (
    LadderConfig,
    majority,
    rung0_l2,
    rung1_whole,
    rung2_bisect,
    run_ladder,
)
from src.trace.attribution.prompts import PromptConfig, localize_request
from src.trace.attribution.report import rung_table
from src.trace.attribution.case import AttributionCase
from src.trace.attribution.schema import AgentEvent, AgentTrajectory, EventType
from src.trace.l2.detectors import DetectorConfig

PC = PromptConfig()


def traj(n: int, error_at: int | None = None) -> AgentTrajectory:
    ev = [AgentEvent("start", "t", EventType.RUN_START, "user", output="task")]
    for i in range(n):
        ev.append(
            AgentEvent(
                f"e{i}",
                "t",
                EventType.TOOL_CALL,
                "worker",
                step_index=i,
                input={"path": f"f{i}"},
                output=f"content {i}",
                error="error" if i == error_at else None,
                metadata={"tool": "read_file"},
            )
        )
    return AgentTrajectory("t", goal="fix the bug", events=ev)


def judge(localize=None, target=None, adjudicate=max):
    """localize -> fixed step; bisect -> half holding ``target``; adjudicate -> ``adjudicate(a, b)``."""

    def responder(req):
        if req.purpose == "localize":
            return localize if isinstance(localize, str) else json.dumps({"step_index": localize})
        if req.purpose == "bisect":
            a, b = map(int, re.search(r"UPPER HALF \(steps (\d+)-(\d+)\)", req.user).groups())
            return json.dumps({"half": "upper" if a <= target <= b else "lower"})
        if req.purpose == "adjudicate":
            i, j = map(int, re.findall(r"CANDIDATE [AB] = step (\d+)", req.user))
            return json.dumps({"step": adjudicate(i, j)})
        raise AssertionError(req.purpose)

    return ScriptedJudge(responder, 100, 10)


def test_rung0_is_free_and_names_first_visible_deviation():
    r = rung0_l2(traj(6, error_at=3), DetectorConfig())
    assert (r.step, r.detail["category"], r.tokens.calls, r.tokens.total) == (3, "tool_error", 0, 0)
    assert rung0_l2(traj(6), DetectorConfig()).step is None


@pytest.mark.parametrize(
    "reply,step,reason",
    [
        (json.dumps({"step_index": 4}), 4, None),
        ('Sure! ```json\n{"step_index": "2"}\n```', 2, None),
        (json.dumps({"step_index": None}), None, "judge_returned_null"),
        (json.dumps({"step_index": 40}), None, "out_of_range:40"),
        ("I think step 3", None, "parse_failure"),
    ],
)
def test_rung1_parsing(reply, step, reason):
    r = rung1_whole(traj(6), judge(localize=reply), PC, None)
    assert (r.step, r.reason, r.tokens.calls, r.tokens.total) == (step, reason, 1, 110)


@pytest.mark.parametrize("n,target,calls", [(8, 5, 3), (8, 0, 3), (5, 4, 2), (5, 0, 3), (1, 0, 0)])
def test_rung2_bisection_converges_in_hand_counted_probes(n, target, calls):
    # n=5, target 4: [0..2|3..4] lower, [3|4] lower -> 2 probes; target 0: [0..2|..] upper, [0..1|2] upper, [0|1] upper -> 3
    r = rung2_bisect(traj(n), judge(target=target), PC, None)
    assert (r.step, r.tokens.calls, r.tokens.total) == (target, calls, 110 * calls)


def test_rung2_parse_failure_abstains():
    j = ScriptedJudge(lambda req: "dunno")
    r = rung2_bisect(traj(8), j, PC, None)
    assert r.step is None and r.reason == "parse_failure" and r.tokens.calls == 1


def test_majority():
    assert majority([None, None, None]) == (None, False)
    assert majority([3, None, None]) == (3, False)
    assert majority([3, 3, 5]) == (3, False)
    assert majority([None, 2, 5]) == (None, True)
    assert majority([3, 2, 5]) == (None, True)


def test_ladder_adjudicates_only_on_disagreement_and_charges_each_rung():
    # rung0 = 3 (error), rung1 = 2, rung2 = 5 -> no majority -> tournament 3 v 2 -> 3, 3 v 5 -> 5
    j = judge(localize=2, target=5, adjudicate=max)
    res = run_ladder("c", traj(8, error_at=3), j, LadderConfig())
    steps = {k: v.step for k, v in res.rungs.items()}
    assert steps == {"rung0_l2": 3, "rung1_whole": 2, "rung2_bisect": 5, "rung3_adjudicate": 5}
    assert res.final_step == 5 and res.decided_by == "rung3"
    assert res.rungs["rung3_adjudicate"].detail["bouts"] == [[3, 2, 3], [3, 5, 5]]
    calls = {k: v.tokens.calls for k, v in res.rungs.items()}
    assert calls == {"rung0_l2": 0, "rung1_whole": 1, "rung2_bisect": 3, "rung3_adjudicate": 2}
    assert res.tokens_total().total == 6 * 110


def test_ladder_majority_skips_adjudication():
    res = run_ladder("c", traj(8, error_at=3), judge(localize=3, target=6), LadderConfig())
    assert res.final_step == 3 and res.decided_by == "rung0+rung1"
    assert (
        res.rungs["rung3_adjudicate"].ran is False
        and res.rungs["rung3_adjudicate"].tokens.calls == 0
    )


def test_escalate_policy_stops_when_rule_and_whole_trace_agree():
    res = run_ladder(
        "c", traj(8, error_at=3), judge(localize=3, target=6), LadderConfig(policy="escalate")
    )
    assert res.final_step == 3 and res.rungs["rung2_bisect"].ran is False
    assert res.tokens_total().calls == 1
    # rule silent -> bisection runs; rung1 and rung2 agree -> no adjudication
    res = run_ladder("c", traj(8), judge(localize=6, target=6), LadderConfig(policy="escalate"))
    assert (
        res.rungs["rung2_bisect"].ran
        and res.final_step == 6
        and res.rungs["rung3_adjudicate"].ran is False
    )


def test_rung_table_accuracy_and_tokens():
    cases = [
        AttributionCase("a", traj(8, error_at=3), {5: 1.0}, {"worker"}),
        AttributionCase("b", traj(8, error_at=3), {3: 1.0}, {"worker"}),
    ]
    res = [
        run_ladder("a", cases[0].trajectory, judge(localize=2, target=5)),
        run_ladder("b", cases[1].trajectory, judge(localize=3, target=6)),
    ]
    rows = {r["rung"]: r for r in rung_table(cases, res)}
    assert rows["rung0_l2"]["exact"] == 0.5 and rows["rung0_l2"]["total_tokens"] == 0
    assert rows["rung1_whole"]["exact"] == 0.5 and rows["rung1_whole"]["within1"] == 0.5
    assert rows["rung2_bisect"]["exact"] == 0.5 and rows["rung2_bisect"]["calls"] == 6
    assert rows["rung3_adjudicate"]["ran_on"] == 1 and rows["rung3_adjudicate"]["exact"] == 1.0
    assert rows["ladder_final"]["exact"] == 1.0 and rows["ladder_final"]["agent_acc"] == 1.0
    assert rows["ladder_final"]["total_tokens"] == (6 + 4) * 110


def test_reference_answer_rendered_only_on_request():
    t = traj(3)
    assert "Correct answer" not in localize_request(t, PromptConfig(), "42").user
    assert "Correct answer: 42" in localize_request(t, PromptConfig(with_reference=True), "42").user


def test_extract_json_and_local_only_client():
    assert extract_json('prose {"a": 1} tail') == {"a": 1}
    assert extract_json("") is None
    OpenAICompatJudge.local(8080, "m")  # constructing never contacts the server
    with pytest.raises(ValueError):
        OpenAICompatJudge("http://10.0.0.5:8080/v1", "m")
