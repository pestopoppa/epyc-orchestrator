"""Probe v2 harness (scripts/benchmark/codemode_ab): transcripts, failure enum, v2 arms, discovery cap fix, smoke gate, scorer. No model, no server."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path

import pytest

from scripts.benchmark.codemode_ab import arms, run_probe, score_v2, smoke_check_v2
from scripts.benchmark.codemode_ab.mock import MockLLM
from scripts.benchmark.codemode_ab.tasks import build_tasks
from scripts.benchmark.codemode_ab.validate import repair_tool_names
from scripts.benchmark.codemode_ab.world import World

ORCH = Path(__file__).resolve().parents[2]
TMP = Path("/mnt/raid0/llm/tmp/codemode-rel-test")


@pytest.fixture
def work():
    d = TMP / uuid.uuid4().hex
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def selectors(work):
    from argparse import Namespace

    from src.prompt_builders import tool_selector as ts

    template = World(work / "template", ORCH)
    sel = run_probe.build_selectors(Namespace(mock=True), template, None)
    run_probe.register_native_picker(sel, template)
    os.environ["ORCHESTRATOR_TOOL_SELECTOR_LOG"] = ""
    yield sel
    ts.set_native_selector_provider(None)
    os.environ.pop("ORCHESTRATOR_TOOL_SELECTOR_LOG", None)


def task(tid="N-CSV-n5-s0"):
    return next(t for t in build_tasks([0]) if t.task_id == tid)


def episode(work, selectors, arm, tid="N-CSV-n5-s0", seed=1):
    t = task(tid)
    w = World(work / f"{arm}_{tid}", ORCH)
    t.setup(w)  # MockLLM is privileged: it reads the sandbox data directly (run_episode sets the same world up again)
    return arms.run_episode(t, arm, w, MockLLM(t, w, seed), seed, selectors=selectors)


# --- failure enum ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("kw,want", [
    (dict(code="", error="", is_final=False, wrapped=False), "no_code"),
    (dict(code="open('x')", error="Dangerous operation not allowed: open().", is_final=False, wrapped=False), "ast_refused"),
    (dict(code="x(", error="SyntaxError: invalid", is_final=False, wrapped=False), "syntax"),
    (dict(code="foo()", error="NameError: name 'foo' is not defined", is_final=False, wrapped=False), "name_error"),
    (dict(code="CALL('x')", error="ValueError: Unknown tool: x", is_final=False, wrapped=False), "name_error"),
    (dict(code="peek(1)", error="ValueError: peek(n, file_path) needs a file_path", is_final=False, wrapped=False), "tool_arg_error"),
    (dict(code="f()", error="PermissionError: not available", is_final=False, wrapped=False), "tool_error"),
    (dict(code="f()", error="ZeroDivisionError", is_final=False, wrapped=False), "other_error"),
    (dict(code="FINAL('x')", error="", is_final=True, wrapped=False), "explicit_final"),
    (dict(code="FINAL(list_dir('d'))", error="", is_final=True, wrapped=True), "implicit_final"),
    (dict(code="print(1)", error="", is_final=False, wrapped=False), "ok"),
])
def test_classify_turn(kw, want):
    assert arms.classify_turn(**kw) == want and want in arms.FAILURE_REASONS


# --- arm environments ----------------------------------------------------------------------------------------
def test_arm_env_sets_and_restores_and_never_leaks(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_REPL_TURN_MEMORY", "9")
    with arms.arm_env("B1"):
        assert os.environ["ORCHESTRATOR_REPL_EXPLICIT_FINAL"] == "1" and os.environ["ORCHESTRATOR_CODEMODE_PROMPT"] == "v2"
        assert "ORCHESTRATOR_REPL_TURN_MEMORY" not in os.environ  # B1 has no memory even if the shell sets it
    assert os.environ["ORCHESTRATOR_REPL_TURN_MEMORY"] == "9" and "ORCHESTRATOR_REPL_EXPLICIT_FINAL" not in os.environ
    with arms.arm_env("C3"):
        assert os.environ["ORCHESTRATOR_REPL_TURN_MEMORY"] == "4" and os.environ["ORCHESTRATOR_CODEMODE_STUBS"] == "1"
        assert os.environ["ORCHESTRATOR_STRUCTURED_READ_ONLY"] == "1" and os.environ["ORCHESTRATOR_TOOL_SELECTOR"] == "embed"
    with arms.arm_env("B"):
        assert all(k not in os.environ for k in arms._ALL_FLAGS) or os.environ.get("ORCHESTRATOR_REPL_TURN_MEMORY") is None


def test_v2_turn_cap_matches_arm_a_and_v1_unchanged():
    assert arms.MAX_TURNS_V2 == arms.MAX_TURNS_A == 40 and arms.MAX_TURNS_REPL == 12
    assert arms.ARMS_V2 == ("A_pick", "B1", "B2", "C3") and arms.ARMS_V1 == ("A", "B", "C1", "C2")


# --- name repair ---------------------------------------------------------------------------------------------------
def test_repair_tool_names():
    names = ["syn_sku_price", "read_file"]
    code, rep = repair_tool_names('x = CALL("syn_sku_pric", sku="a")\nTOOL(\'read_fil\', path="p")\nCALL("zzz")', names)
    assert 'CALL("syn_sku_price"' in code and "TOOL('read_file'" in code and 'CALL("zzz")' in code
    assert [r["to"] for r in rep] == ["syn_sku_price", "read_file"]
    assert repair_tool_names('CALL("syn_sku_price")', names) == ('CALL("syn_sku_price")', [])


# --- discovery cap fix ----------------------------------------------------------------------------------------------
def test_syn_sku_price_is_visible_inside_every_v2_cap(work):
    w = World(work / "w", ORCH)
    v1 = json.dumps([{"name": t["name"], "description": t["description"], "parameters": t["parameters"]} for t in w.tool_dicts()])
    assert v1.index("syn_sku_price") > arms.TOOL_RESULT_CAP and v1.index("syn_sku_price") > 1500  # the v1 failure, reproduced
    compact = json.dumps(arms.compact_tools(w))
    assert compact.index("syn_sku_price") < arms.V2_TOOL_RESULT_CAP and len(compact) < arms.V2_TOOL_RESULT_CAP
    names_only = str([t["name"] for t in arms.compact_tools(w)])
    assert len(names_only) < arms.V2_PREVIEW_CHARS and "syn_sku_price" in names_only and "syn_ledger_post" in names_only


@pytest.mark.repl_in_process
def test_repl_list_tools_is_compact_only_for_v2_arms(work):
    w = World(work / "r", ORCH)
    assert "parameters" in arms.make_repl(w).execute("print(list_tools()[0].keys())").output
    out = arms.make_repl(World(work / "r2", ORCH), compact_list_tools=True).execute("print(sorted(list_tools()[0].keys()))").output
    assert "['description', 'name']" in out


# --- episodes (mock) --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("arm", arms.ARMS)
def test_every_arm_returns_a_transcript_and_reason_enum(work, selectors, arm):
    rec = episode(work, selectors, arm)
    tr = rec["_transcript"]
    assert tr and rec["transcript_turns"] == len(tr) and set(rec["turn_failure_reasons"]) <= set(arms.FAILURE_REASONS)
    assert all(len(t["prompt_hash"]) == 16 for t in tr) and "prompt" in tr[0]
    assert all("failure_reason" in t for t in tr)
    assert not any(k in os.environ for k in ("ORCHESTRATOR_REPL_EXPLICIT_FINAL", "ORCHESTRATOR_CODEMODE_PROMPT", "ORCHESTRATOR_CODEMODE_STUBS"))


@pytest.mark.parametrize("arm", ["B1", "B2", "C3"])
@pytest.mark.repl_in_process
def test_v2_repl_arms_serve_the_v2_prompt_and_end_explicitly(work, selectors, arm):
    rec = episode(work, selectors, arm)
    first = rec["_transcript"][0]["prompt"]
    for bad in smoke_check_v2.BANNED_IN_V2_PROMPT:
        assert bad not in first
    assert "SANDBOX CONTRACT" in first and rec["final_kind"] == "explicit" and rec["max_turns"] == 40


def test_b0_keeps_the_v1_prompt_and_cap(work, selectors):
    rec = episode(work, selectors, "B")
    assert "STOP generating after the CALL" in rec["_transcript"][0]["prompt"] and rec["max_turns"] == 12


def test_b2_and_c3_carry_turn_memory_b1_does_not(work, selectors):
    # seed 3 -> the mock's first snippet fails (unknown tool), so turn 2 has history to show
    b1, b2, c3 = (episode(work, selectors, a, seed=3) for a in ("B1", "B2", "C3"))
    assert "## Previous Turns" not in b1["_transcript"][1]["prompt"]
    for rec in (b2, c3):
        p2 = rec["_transcript"][1]["prompt"]
        assert "## Previous Turns" in p2 and "no_such_tool" in p2 and "## Last Error" not in p2
        assert p2.index("## Previous Turns") > p2.index("## Task")  # appended after every other section


@pytest.mark.repl_in_process
def test_c3_binds_picked_tools_and_renders_stubs(work, selectors):
    rec = episode(work, selectors, "C3", tid="N-QUOTE-s0")
    assert "def " in rec["_transcript"][0]["prompt"] and "list_tools()" in rec["_transcript"][0]["prompt"]
    assert rec["selector"]["variant"] == "hardened_embed" and "read_file" in rec["selector"]["core"]
    assert rec["success"] is True


def test_a_pick_goes_through_the_production_picker(work, selectors):
    rec = episode(work, selectors, "A_pick", tid="N-QUOTE-s0")
    sent = rec["_transcript"][0]["tools_sent"]
    assert sent[:3] == ["list_tools", "search_tools", "describe_tool"] and "read_file" in sent and len(sent) < 20
    assert rec["selector"]["variant"] == "hardened_embed" and rec["selector"]["proxy_recall"] is not None


# --- runner + smoke gate ---------------------------------------------------------------------------------------------------
def test_runner_writes_transcripts_and_smoke_passes_then_fails_on_tamper(work):
    out = work / "out"
    rc = run_probe.main(["--mock", "--out", str(out), "--arm-set", "v2", "--seeds", "0", "--stages", "1", "--limit", "12",
                         "--work", str(work / "wk")])
    assert rc == 0
    assert smoke_check_v2.check(out, "smoke", min_episodes=12) == []
    eps = smoke_check_v2.episodes(out, "smoke")
    assert all((out / e["transcript_path"]).is_file() for e in eps)
    victim = next(e for e in eps if e["arm"] == "B1")
    (out / victim["transcript_path"]).unlink()
    assert any("no transcript file" in p for p in smoke_check_v2.check(out, "smoke"))


def test_smoke_flags_implicit_final_and_stale_prompt(work):
    out = work / "o2"
    (out / "transcripts").mkdir(parents=True)
    tp = out / "transcripts" / "u.jsonl"
    tp.write_text(json.dumps({"turn": 0, "prompt": "... STOP generating after the CALL ..."}) + "\n")
    row = {"kind": "episode", "mode": "smoke", "unit_id": "m|B1|t", "arm": "B1", "end_reason": "final", "final_kind": "implicit",
           "transcript_path": "transcripts/u.jsonl", "transcript_turns": 1, "max_turns": 12, "turn_failure_reasons": {"implicit_final": 1}}
    (out / "results.jsonl").write_text(json.dumps(row) + "\n")
    probs = " | ".join(smoke_check_v2.check(out, "smoke"))
    assert "without an explicit FINAL" in probs and "STOP generating" in probs and "turn cap 12" in probs


def test_runner_default_is_v1_and_v2_set_expands(capsys):
    assert run_probe.main(["--dry-run", "--out", "/tmp/x", "--stages", "1", "--seeds", "0"]) == 0
    assert json.loads(capsys.readouterr().out)["arms"] == list(arms.ARMS_V1)
    assert run_probe.main(["--dry-run", "--out", "/tmp/x", "--stages", "1", "--seeds", "0", "--arm-set", "v2"]) == 0
    assert json.loads(capsys.readouterr().out)["arms"] == ["A", "B", "A_pick", "B1", "B2", "C3"]
    assert run_probe.main(["--dry-run", "--out", "/tmp/x", "--arms", "A,ZZ"]) == 3


# --- scorer -------------------------------------------------------------------------------------------------------------------------
def _rec(model, arm, tid, seed, ok, tok=1000, **kw):
    r = {"kind": "episode", "mode": "full", "model": model, "arm": arm, "task_id": tid, "seed": seed, "n": 5, "stage": 1, "tier": "static",
         "success": ok, "in_tokens": tok, "out_tokens": 0, "wall_s": 1.0, "turns": 2, "failed_snippet_rate": 0.0, "param_accuracy": 1.0,
         "duplicate_writes": 0, "end_reason": "final", "turn_failure_reasons": {"ok": 1, "explicit_final": 1}}
    r.update(kw)
    return r


def test_score_v2_contrasts_futility_and_paired_success_tokens():
    recs = []
    for i in range(12):
        tid = f"t{i}"
        for arm, ok, tok in (("B", i % 2 == 0, 1000), ("B1", i % 2 == 0, 1000), ("A_pick", True, 1000), ("C3", True, 500), ("A", True, 1000), ("B2", True, 900)):
            recs.append(_rec("m35", arm, tid, i % 3, ok, tok))
    res = score_v2.evaluate(recs)
    assert res["schema"].endswith("v2") and "m35|Q1a|n_le_20" in res["cells"] and "m35|Q3|n_le_20" in res["cells"]
    assert res["interim"]["n20"]["stop_on_futility"] is True  # B1 == B on every pair: |d| < 0.05, the diagnosis was wrong
    ps = res["paired_success_tokens"]["m35|Q3|n_le_20"]
    assert ps["n_pairs_both_success"] == 12 and ps["ln_ratio_median"] < -0.5
    assert res["end_reasons"]["B1"]["ok"] == 12
    assert res["interim"]["n20"]["qp_futile_by_model"] == {"m35": True}  # A_pick == A on every pair: the picker adds nothing here
    assert score_v2.score.CONTRASTS == score_v2.score.CONTRASTS and "P1" in score_v2.score.CONTRASTS  # v1 contrasts restored


def test_score_v2_harm_stops_for_the_picker():
    recs = []
    for i in range(60):
        recs.append(_rec("m", "A", f"t{i}", i % 3, True))
        sel = {"proxy_recall": 0.5, "recovery_fired": i < 10, "elapsed_ms": 900.0 if i % 10 == 0 else 20.0, "breaker_opens": 3}
        recs.append(_rec("m", "A_pick", f"t{i}", i % 3, i % 3 != 0, selector=sel))
    h = score_v2.harm_check(recs, "m")
    assert h["harm"] and all(h["checks"].values())
    ok = [_rec("m", "A", f"t{i}", 0, True) for i in range(60)] + [
        _rec("m", "A_pick", f"t{i}", 0, True, selector={"proxy_recall": 1.0, "recovery_fired": False, "elapsed_ms": 30.0, "breaker_opens": 0})
        for i in range(60)]
    assert score_v2.harm_check(ok, "m")["harm"] is False
