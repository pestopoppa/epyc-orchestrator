"""HIL-18: deterministic step-type key and Kulczynski2 ranking. Constructed replays: these test the
formula, key and plumbing, not the usefulness of the ranking on real traces."""
from __future__ import annotations

import json

import pytest

from src.trace import store as trace_store
from src.trace.l2.model import L2Step, L2Trace
from src.trace.spectrum import (UNKNOWN, eligible_for_spectrum, exit_status, kulczynski2, l2_agreement, overlap_with_human,
                                rank_task, relabel_ambiguous, step_key)
from src.trace.spectrum_replay import RunOutcome, build_report, replay_task, traces_for_manifest


def S(idx, tool, err=False, **kw):
    return L2Step(idx=idx, tool=tool, error=err, **kw)


def T(rid, outcome, steps, role="worker"):
    return L2Trace(trace_id=rid, task_key="t1", outcome=outcome, role=role, steps=steps)


class TestKey:
    def test_status_precedence(self):
        assert exit_status(S(0, "a", exit_status=0, err=True)) == "exit:0"      # exit code wins
        assert exit_status(S(0, "a", True, error_class="Timeout")) == "error:Timeout"
        assert exit_status(S(0, "a", False)) == "ok"
        assert exit_status(L2Step(idx=0, tool="a")) == UNKNOWN                  # nothing captured: ambiguous, never guessed

    def test_key_shape(self):
        assert step_key(S(0, "read_file"), "coder") == ("coder", "read_file", "ok")
        assert step_key(L2Step(idx=1, kind="final", error=False), None) == ("?", "<final>", "ok")


class TestFormula:
    def test_k2_values(self):
        assert kulczynski2(10, 0, 0) == 1.0
        assert kulczynski2(0, 5, 5) == 0.0
        assert kulczynski2(5, 5, 5) == pytest.approx(0.5)
        assert kulczynski2(10, 0, 10) == pytest.approx(0.75)       # 0.5*(1 + 0.5)
        assert kulczynski2(4, 6, 0) == pytest.approx(0.5 * (0.4 + 1.0))

    def test_ranking_puts_failure_only_key_first_and_is_deterministic(self):
        bad = [S(0, "read_file"), S(1, "run_tests", True)]
        good = [S(0, "read_file"), S(1, "run_tests", False)]
        traces = [T(f"f{i}", "fail", bad) for i in range(6)] + [T(f"p{i}", "pass", good) for i in range(6)]
        rk = rank_task(traces)
        assert rk.n_fail == rk.n_pass == 6 and rk.usable
        assert rk.ranking[0].key[1:] == ("run_tests", "error:error") and rk.ranking[0].score == 1.0
        shared = next(r for r in rk.ranking if r.key[1:] == ("read_file", "ok"))
        assert shared.score == pytest.approx(0.75)
        assert [r.key for r in rank_task(list(reversed(traces))).ranking] == [r.key for r in rk.ranking]

    def test_unknown_outcome_excluded(self):
        rk = rank_task([T("a", "fail", [S(0, "x", True)]), T("b", "unknown", [S(0, "y", True)])])
        assert rk.n_runs == 1 and all(r.key[1] != "y" for r in rk.ranking)

    def test_eligibility_strictly_between(self):
        assert eligible_for_spectrum(5, 15) and not eligible_for_spectrum(0, 20) and not eligible_for_spectrum(20, 0)
        assert not eligible_for_spectrum(1, 10)

    def test_ambiguous_flagged_and_relabelled_only_where_unknown(self):
        steps = [L2Step(idx=0, tool="a"), S(1, "b", False)]
        traces = [T("f", "fail", steps), T("p", "pass", [S(0, "b", False)])]
        rk = rank_task(traces)
        assert [r.key[1] for r in rk.ranking if r.ambiguous] == ["a"]
        calls = []
        out = relabel_ambiguous(traces, lambda t, s: calls.append(s.tool) or True)
        assert calls == ["a"] and out[0].steps[0].error is True and out[0].steps[1].error is False
        assert traces[0].steps[0].error is None                                   # input not mutated


def test_l2_agreement_counts_unclassified_and_category_hits():
    # failing runs repeat the same call with no state change -> loop; the key of the flagged step is top-ranked
    loop = [S(0, "grep", False, state_digest="s"), S(1, "grep", False, state_digest="s"), S(2, "grep", False, state_digest="s")]
    ok = [S(0, "read", False, state_digest="r")]
    traces = [T("f1", "fail", loop), T("f2", "fail", loop), T("f3", "fail", ok), T("p1", "pass", ok)]
    rk = rank_task(traces)
    a = l2_agreement(traces, rk, top_m=1)
    assert a["n_failed_runs"] == 3 and a["unclassified"] == 1
    assert a["by_category"]["loop"]["failed_runs_flagged"] == 2 and a["by_category"]["loop"]["top_key_at_flagged_step"] == 2


def test_human_overlap():
    rk = rank_task([T("f", "fail", [S(0, "a", True)]), T("p", "pass", [S(0, "b", False)])])
    o = overlap_with_human(rk, [("worker", "a", "error:error")], top_m=1)
    assert o["hits"] == 1


# --------------------------------------------------------------------------- wrapper
def _store(tmp_path, runs):
    """runs: rid -> list of tool names; written as tool_step rows."""
    from src.trace.step_events import StepEventLog, trace_store_sink
    db = tmp_path / "e.sqlite"
    trace_store.ensure_schema(db).close()
    for rid, tools in runs.items():
        log = StepEventLog(rid, role="worker", sink=trace_store_sink(db))
        for name, err in tools:
            log.observe_call(name, {"i": rid}, "[ERROR] x" if err else "fine")
    return db


def test_wrapper_end_to_end_and_join_accounting(tmp_path):
    task = {"task_key": "t1", "prompt": "p", "expected": "42"}
    runs = {}

    def runner(task, rid, i):
        runs[rid] = [("read", False), ("calc", i % 2 == 0)]
        return RunOutcome(final_answer="41" if i % 2 == 0 else "42") if i != 7 else RunOutcome(error="HTTPError")
    man = tmp_path / "m.jsonl"
    ents = replay_task(task, 24, runner, lambda t, a: a == t["expected"], man)
    assert sum(e["outcome"] == "unknown" for e in ents) == 1 and len(man.read_text().splitlines()) == 24
    missing = ents[-1]["run_id"]
    db = _store(tmp_path, {k: v for k, v in runs.items() if k != missing})
    per_task, acct = traces_for_manifest(db, ents)
    assert acct["replays_started"] == 24 and acct["without_steps"] == [missing] and acct["outcome_unknown"] == 1
    rep = build_report(per_task, acct, min_replays=20)
    assert rep["tasks_reported"] == 1 and not rep["tasks_skipped"]          # 22 labelled replays with steps
    top = rep["tasks"][0]["ranking"]["top"][0]
    assert top["key"] == ["worker", "calc", "error:result_marker"] and top["k2"] == 1.0
    assert json.loads(json.dumps(rep))["schema"] == "spectrum_report/v1"
