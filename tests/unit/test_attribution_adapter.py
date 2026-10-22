"""HIL-17 AgentDebugX-schema adapter, Who&When Pro loader and the ladder CLI dry-run (zero inference)."""

import json
import sqlite3
from collections import Counter
from pathlib import Path

import pytest

from src.trace.attribution import whowhen_pro as WP
from src.trace.attribution.adapter import from_l2_trace, from_recorded_run, to_l2_trace
from src.trace.attribution.prompts import PromptConfig, localize_request, multi_request
from src.trace.attribution.schema import (
    ADX_EVENT_FIELDS,
    ADX_TRAJECTORY_FIELDS,
    AgentTrajectory,
    EventType,
    read_jsonl,
    write_jsonl,
)
from src.trace.l2.detectors import detect
from src.trace.l2.model import L2Trace

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests" / "fixtures" / "attribution" / "whowhen_pro_tiny.jsonl"


# ------------------------------------------------------------------------------------ schema pin
def test_schema_pins_agentdebugx_052_field_sets():
    # agentdebug/schema/models.py in agentdebugx 0.5.2 (field order irrelevant, set must match)
    assert set(ADX_EVENT_FIELDS) == {
        "event_id",
        "trace_id",
        "parent_event_id",
        "agent_name",
        "event_type",
        "module",
        "step_index",
        "timestamp",
        "input",
        "output",
        "error",
        "duration_ms",
        "metadata",
        "artifacts",
    }
    assert set(ADX_TRAJECTORY_FIELDS) == {
        "trace_id",
        "task_id",
        "goal",
        "framework",
        "started_at",
        "ended_at",
        "metadata",
        "events",
    }
    assert EventType.TOOL_CALL.value == "tool.call" and EventType.RUN_END.value == "run.end"
    assert len(EventType) == 16


def _l2(**over) -> L2Trace:
    d = {
        "trace_id": "r1",
        "task_key": "task-a",
        "outcome": "fail",
        "role": "worker",
        "steps": [
            {
                "idx": 0,
                "tool": "read_file",
                "args": {"path": "a.py"},
                "observation": "alpha",
                "exit_status": 0,
            },
            {
                "idx": 1,
                "tool": "read_file",
                "args": {"path": "b.py"},
                "observation": "beta",
                "exit_status": 2,
            },
            {"idx": 2, "tool": "run", "args": {}, "observation": "same", "exit_status": 0},
            {"idx": 3, "tool": "run", "args": {}, "observation": "same", "exit_status": 0},
            {"idx": 4, "tool": "run", "args": {}, "observation": "same", "exit_status": 0},
            {"idx": 5, "tool": "bad", "args_raw": "{not json", "truncated": True},
            {"idx": 6, "kind": "final", "is_final": True},
        ],
    }
    d.update(over)
    return L2Trace.from_dict(d)


def test_l2_trace_roundtrip_preserves_detections_and_index_space():
    t = _l2()
    traj = from_l2_trace(t, goal="g")
    assert [e.step_index for e in traj.events] == list(range(7))
    assert (
        traj.events[1].error == "exit_status=2" and traj.events[6].event_type == EventType.RUN_END
    )
    assert traj.metadata["outcome"] == "fail" and traj.framework == "epyc-orchestrator"
    key = lambda ds: sorted((d.category, d.step_idx) for d in ds)  # noqa: E731
    assert key(detect(to_l2_trace(traj))) == key(detect(t))
    assert {c for c, _ in key(detect(t))} >= {
        "tool_error",
        "loop",
        "truncation",
        "invalid_invocation",
    }


def test_trajectory_json_roundtrip(tmp_path):
    traj = from_l2_trace(_l2())
    d = traj.to_dict()
    assert set(d) == set(ADX_TRAJECTORY_FIELDS) and set(d["events"][0]) == set(ADX_EVENT_FIELDS)
    p = tmp_path / "t.jsonl"
    assert write_jsonl([traj, traj], p) == 2
    back = list(read_jsonl(p))
    assert len(back) == 2 and back[0].to_dict() == d


def test_recorded_run_adapter_interleaves_turns_and_reads_cas(tmp_path):
    from src.trace.replay.cas import CAS

    cas = CAS(tmp_path / "cas")
    db = tmp_path / "trace.sqlite"
    conn = sqlite3.connect(db)
    conn.execute(
        "create table event (id integer primary key, category text, session_id text, detail_json text)"
    )
    rows = [
        (
            "model_turn",
            {
                "run_id": "R",
                "turn": 0,
                "role": "coder",
                "prompt_cas": cas.put("P0"),
                "output_cas": cas.put("call read"),
                "step_mark": 0,
                "chain": "c0",
            },
        ),
        (
            "tool_step",
            {
                "run_id": "R",
                "idx": 0,
                "kind": "tool_call",
                "tool": "read_file",
                "ok": True,
                "args_cas": cas.put('{"path": "a"}'),
                "output_cas": cas.put("alpha"),
            },
        ),
        (
            "tool_step",
            {
                "run_id": "R",
                "idx": 1,
                "kind": "tool_call",
                "tool": "run",
                "ok": False,
                "error_class": "exception",
                "args_cas": cas.put("{}"),
                "output_cas": cas.put("boom"),
            },
        ),
        (
            "model_turn",
            {
                "run_id": "R",
                "turn": 1,
                "role": "coder",
                "prompt_cas": cas.put("P1"),
                "output_cas": cas.put("done"),
                "step_mark": 2,
                "chain": "c1",
            },
        ),
        ("tool_step", {"run_id": "R", "idx": 2, "kind": "final", "ok": True}),
    ]
    for cat, rec in rows:
        conn.execute(
            "insert into event (category, session_id, detail_json) values (?,?,?)",
            (cat, "R", json.dumps(rec)),
        )
    conn.commit()
    conn.close()
    traj = from_recorded_run(db, "R", tmp_path / "cas", goal="g", outcome="fail")
    kinds = [(e.step_index, e.event_type.value, e.output) for e in traj.events]
    assert kinds == [
        (0, "llm.response", "call read"),
        (1, "tool.call", "alpha"),
        (2, "tool.call", "boom"),
        (3, "llm.response", "done"),
        (4, "run.end", None),
    ]
    assert traj.events[1].input == {"path": "a"} and traj.events[2].error == "exception"
    assert traj.events[1].parent_event_id == traj.events[0].event_id
    assert "prompt" not in traj.events[0].input
    assert (
        from_recorded_run(db, "R", tmp_path / "cas", include_prompts=True).events[0].input["prompt"]
        == "P0"
    )
    assert (
        from_recorded_run(db, "R", None).events[1].output is None
    )  # no CAS: unobserved, not empty


# ------------------------------------------------------------------------------------ Who&When Pro
def test_pro_loader_resolves_gold_per_framework():
    skips = Counter()
    cases = {c.case_id: c for c in WP.load_cases(FIX, skips)}
    assert skips == Counter({"gold_unresolved:smolagents": 1})
    want = {
        "alfworld_alfagent_9001": ({2: 1.0}, set()),  # step_number 3 = third action
        "gaia_smolagents_9002": ({0: 1.0}, set()),  # step 0 = the planning entry
        "gaia_magentic_9003": ({1: 1.0}, {"websurfer"}),  # "0.1" = round 0 position 1
        "math_mathchat_9004": ({3: 1.0}, {"assistant"}),
        "math_debate_9005": ({2: 1.0, 3: 1.0}, {"agent_0", "agent_1"}),
        "code_macnet_9007": ({1: 1.0}, {"node_1"}),
        "code_metagpt_9008": ({1: 1.0}, {"engineer"}),
    }
    assert {k: (c.relevance, c.gold_agents) for k, c in cases.items()} == want
    assert cases["gaia_smolagents_9002"].trajectory.by_step()[2].error == "Traceback: NameError"
    assert (
        cases["gaia_smolagents_9002"].trajectory.by_step()[1].error is None
    )  # "None" string is no error
    assert cases["math_mathchat_9004"].trajectory.metadata["final_answer"] == "6"


def test_pro_labels_never_reach_a_prompt():
    for c in WP.load_cases(FIX):
        blob = c.trajectory.to_json()
        for leak in ("is_injected", "is_replayed", "injection_via", "is_live", "ground_truth"):
            assert leak not in blob, (c.case_id, leak)
        for req in (
            localize_request(c.trajectory, PromptConfig(), c.reference),
            multi_request(c.trajectory, PromptConfig(), c.reference),
        ):
            assert "Correct answer" not in req.user


def test_pro_slice_is_deterministic_and_stratified():
    cases = list(WP.load_cases(FIX))
    s1 = WP.select_slice(cases, 1, seed=3)
    assert [c.case_id for c in s1] == [
        c.case_id for c in WP.select_slice(list(reversed(cases)), 1, seed=3)
    ]
    assert Counter(c.meta["framework"] for c in s1) == Counter(
        {fw: 1 for fw in {c.meta["framework"] for c in cases}}
    )
    assert [c.case_id for c in WP.select_slice(cases, 5, frameworks=["debate", "metagpt"])] == [
        "math_debate_9005",
        "code_metagpt_9008",
    ]  # grouped by framework name, then id
    assert WP.select_slice(cases, 5, max_steps=3) and all(
        len(c.trajectory.step_indices()) <= 3 for c in WP.select_slice(cases, 5, max_steps=3)
    )


# ------------------------------------------------------------------------------------ CLI
def _cli():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "attribution_ladder", ROOT / "scripts" / "trace" / "attribution_ladder.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_dry_run_writes_rung_and_ndcg_tables(tmp_path):
    out, adx = tmp_path / "r.json", tmp_path / "adx.jsonl"
    assert _cli().main(["--dry-run", "--out", str(out), "--export-adx", str(adx)]) == 0
    rep = json.loads(out.read_text())
    assert rep["mock"] is True and rep["n_cases"] == 7 and rep["n_labelled"] == 7
    assert [r["rung"] for r in rep["rungs"]] == [
        "rung0_l2",
        "rung1_whole",
        "rung2_bisect",
        "rung3_adjudicate",
        "ladder_final",
    ]
    assert rep["rungs"][0]["total_tokens"] == 0 and rep["rungs"][1]["calls"] == 7
    agg = rep["aggregate"]
    assert (
        agg["N"] == 7
        and agg["k_per_family"] == 5
        and set(agg["per_family"]) == {"fake-heuristic-a", "fake-heuristic-b"}
    )
    assert agg["ksample_exp_N"] == 7 and agg["random_expected_exp"] is not None
    assert rep["adx_schema"]["version"] == "0.5.2" and rep["with_reference"] is False
    assert (tmp_path / "r.md").read_text().startswith("# Attribution ladder")
    assert len(list(read_jsonl(adx))) == 7


def test_cli_real_run_refuses_without_window(tmp_path, monkeypatch):
    monkeypatch.delenv("WINDOW_CONFIRMED", raising=False)
    with pytest.raises(SystemExit, match="WINDOW_CONFIRMED"):
        _cli().main(["--pro", str(FIX), "--port-a", "1", "--out", str(tmp_path / "r.json")])
    assert not (tmp_path / "r.json").exists()


def test_cli_export_only_needs_no_judge(tmp_path, monkeypatch):
    monkeypatch.delenv("WINDOW_CONFIRMED", raising=False)
    adx = tmp_path / "adx.jsonl"
    assert (
        _cli().main(
            [
                "--pro",
                str(FIX),
                "--export-only",
                "--export-adx",
                str(adx),
                "--out",
                str(tmp_path / "r.json"),
            ]
        )
        == 0
    )
    assert isinstance(next(read_jsonl(adx)), AgentTrajectory)
