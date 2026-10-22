"""HIL-LB1: Locals loop-detector port, grace window and test-after-edit exemption (hand-made fixtures).

Detector unit tests, not recall: the labelled scoring is scripts/trace/hil_lb1_loop_score.py.
"""
from __future__ import annotations

import json
import random

from src.trace.l2 import loop_detectors as LD
from src.trace.l2.detectors import DetectorConfig, detect
from src.trace.l2.model import L2Step, L2Trace


def _looping_ref(toks, reps=3, max_period=256, min_span=24, short_span=128):
    """Verbatim Locals locals/server.py:_looping @0d39e16 (reference for the port)."""
    n = len(toks)
    if n < min_span:
        return False
    for p in range(1, 8):
        if n >= 32 and toks[n - 32 + p:] == toks[n - 32:n - p]:
            span = max(reps * p, min_span, short_span)
            return span <= n and toks[n - span + p:] == toks[n - span:n - p]
    last = toks[-1]
    for p in range(8, min(max_period, n // reps) + 1):
        if toks[n - 1 - p] != last:
            continue
        span = max(reps * p, min_span, short_span if p < 8 else 0)
        if span <= n and toks[n - span + p:] == toks[n - span:n - p]:
            return True
    return False


def _seq(seed: int, n: int, vocab: int) -> list[int]:
    r = random.Random(seed)
    out: list[int] = []
    while len(out) < n:
        if r.random() < 0.3 and len(out) > 10:      # paste a repeated block to create periods
            p = r.randint(1, min(60, len(out)))
            block = out[-p:]
            out.extend(block * r.randint(1, 5))
        else:
            out.append(r.randrange(vocab))
    return out[:n]


def test_port_matches_locals_reference_on_every_prefix():
    for seed in range(40):
        toks = _seq(seed, 400, vocab=7 if seed % 2 else 50)
        pos: dict = {}
        for n in range(1, len(toks) + 1):
            if n >= 2:
                pos.setdefault(toks[n - 2], []).append(n - 2)
            for prof in (LD.TOOLS, LD.IN_GRAMMAR, LD.NO_TOOLS):
                ref = _looping_ref(toks[:n], prof.reps, prof.max_period, prof.min_span, prof.short_span)
                assert LD.looping_at(toks, n, prof.reps, prof.max_period, prof.min_span, prof.short_span) == ref
                assert LD.looping_at(toks, n, prof.reps, prof.max_period, prof.min_span, prof.short_span,
                                     _pos=pos) == ref


def test_sentence_loop_fires_and_varied_text_does_not():
    loop = LD.regex_tokenize("I will execute the edit now.\n" * 10)
    assert LD.looping(loop)
    prose = LD.regex_tokenize(" ".join(f"word{i} and thing{i * 7}." for i in range(80)))
    assert not LD.looping(prose)
    assert LD.first_period_fire(prose, LD.TOOLS) is None


def test_short_block_needs_short_span():
    # "0, " is a 3-token block: under TOOLS it must repeat over >= 128 tokens.
    few = LD.regex_tokenize("[" + "0, " * 20)       # 60 tokens of period 3
    many = LD.regex_tokenize("[" + "0, " * 60)      # 180 tokens
    assert not LD.looping(few)
    assert LD.looping(many)


def test_grace_window_lets_an_ending_repetition_through_and_stops_a_runaway():
    row = "  [0, 0, 0, 0, 0, 0, 0, 0],\n"
    ending = LD.regex_tokenize("[\n" + row * 40 + "  [1]\n]")
    assert LD.first_period_fire(ending, LD.IN_GRAMMAR) is None
    runaway = LD.regex_tokenize("[\n" + row * 400)
    n = LD.first_period_fire(runaway, LD.IN_GRAMMAR)
    assert n is not None
    # it started looking like a loop at some m and stopped exactly LOOP_GRACE tokens later
    m = next(k for k in range(1, n + 1) if LD.looping_at(runaway, k, 16, 256, 256, 256))
    assert n - m == LD.LOOP_GRACE
    # without grace (TOOLS) the same runaway is cut far earlier
    assert LD.first_period_fire(runaway, LD.TOOLS) < n


def test_grace_resets_when_repetition_stops():
    row = LD.regex_tokenize("  [0, 0, 0, 0, 0, 0, 0, 0],\n")
    noise = LD.regex_tokenize(" ".join(f"x{i}" for i in range(40)) + "\n")
    # periodic stretches of ~900 tokens separated by noise: never 1,024 consecutive looping tokens
    toks = []
    for _ in range(4):
        toks += row * 45 + noise
    assert LD.first_period_fire(toks, LD.IN_GRAMMAR) is None


def test_td1_shaped_legit_reply_is_cut_by_current_guard_not_by_in_grammar():
    from src.llm_primitives.inference import _detect_streaming_repetition

    # 24 confident answers, each one of two shapes in a fixed non-periodic order (TD-1's case:
    # near-identical objects, one per question).
    order = [int(c) for c in "011010011100101101001101"]
    objs = [{"noul": False, "probabilities": {"A": 1.0 - o, "B": float(o)}, "confidence": 1.0}
            for o in order]
    text = json.dumps({"answers": objs}, indent=2)
    assert _detect_streaming_repetition(text)            # TD-1d.2: the free-form guard cuts it
    toks = LD.regex_tokenize(text)
    for prof in (LD.IN_GRAMMAR, LD.NO_TOOLS):
        assert LD.first_period_fire(toks, prof) is None


def test_paraphrase_detector_gemma_example():
    phrases = ["I'll use the read tool.", "Let's read the file.", "I'll proceed with the read call."]
    text = "\n".join(phrases[i % 3] for i in range(30))
    assert LD.line_looping(text)
    assert LD.first_line_fire(text) == 30
    assert not LD.line_looping("\n".join(phrases[i % 3] for i in range(29)))
    varied = "\n".join(f"Step {i}: I'll read file number {i}." for i in range(40))
    assert not LD.line_looping(varied)
    assert LD.line_looping(varied, normalize=True)       # EPYC extension: digits dropped
    short = "\n".join(["ok"] * 50)                        # lines under min_len never count
    assert not LD.line_looping(short)


def _u(tool, **args):
    return LD.CallUnit(tool, args)


def test_test_after_edit_is_exempt():
    units = [_u("Bash", command="pytest -q tests/a.py", description="run"),
             _u("Edit", file_path="a.py", old_string="x", new_string="y"),
             _u("Bash", command="pytest -q tests/a.py", description="rerun after edit")]
    assert LD.classify_call(units[0]) == "verifying"
    assert LD.classify_call(units[1]) == "mutating"
    assert LD.test_after_edit_exempt(units) == [False, False, True]
    assert LD.repeated_calls(units) == [2]                # description is a label, not identity
    assert LD.repeated_calls(units, LD.test_after_edit_exempt(units)) == []


def test_rerun_without_edit_and_repeated_edit_are_not_exempt():
    t = _u("Bash", command="pytest -q")
    e = _u("Edit", file_path="a.py", old_string="x", new_string="y")
    assert LD.test_after_edit_exempt([t, t]) == [False, False]
    assert LD.test_after_edit_exempt([e, t, e, t]) == [False, False, False, True]
    assert LD.repeated_calls([e, t, e, t], LD.test_after_edit_exempt([e, t, e, t])) == [2]


def test_text_environment_verbs():
    look = _u("env_action", action="examine drawer 1")
    take = _u("env_action", action="take pencil 1 from drawer 1")
    go = _u("env_action", action="go to drawer 1")
    assert LD.classify_call(look) == "verifying"
    assert LD.classify_call(take) == "mutating"
    assert LD.classify_call(go) == "other"
    assert LD.test_after_edit_exempt([look, take, look, go, go]) == [False, False, True, False, False]


def test_action_stream_drops_exempt_calls():
    units = [_u("Bash", command="make test"), _u("Write", file_path="f", content="1"),
             _u("Bash", command="make test")]
    text, kept = LD.action_stream(units, LD.test_after_edit_exempt(units))
    assert kept == [0, 1]
    assert text.splitlines()[0] == "make test"


def test_edit_test_cycle_stream_is_not_a_token_period_loop_once_exempt():
    units = []
    for i in range(12):
        units.append(_u("Edit", file_path="a.py", old_string=f"v{i}", new_string=f"v{i + 1}"))
        units.append(_u("Bash", command="python -m pytest -q tests/unit/test_a.py"))
    ex = LD.test_after_edit_exempt(units)
    assert sum(ex) == 11
    assert LD.first_period_fire(LD.regex_tokenize(LD.action_stream(units, ex)[0]), LD.TOOLS) is None


def test_existing_l2_loop_detector_is_untouched_by_this_module():
    steps = [L2Step(idx=i, tool="env_action", args={"action": "go to cabinet 2"}, observation="nothing")
             for i in range(3)]
    t = L2Trace(trace_id="t", task_key="t", steps=steps)
    assert [d.category for d in detect(t, DetectorConfig())] == ["loop"]


def _load_script():
    import importlib.util
    from pathlib import Path

    p = Path(__file__).resolve().parents[2] / "scripts" / "trace" / "hil_lb1_loop_score.py"
    spec = importlib.util.spec_from_file_location("hil_lb1_loop_score", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_scoring_script_on_synthetic_whowhen_rows(tmp_path):
    S = _load_script()

    def row(rid, mode, actions):
        traj = [{"kind": "user", "content": "task"}] + [
            {"kind": "action", "step_number": i + 1, "reasoning": "", "action": a, "observation": "ok",
             "is_final_answer": False} for i, a in enumerate(actions)]
        return json.dumps({"id": rid, "framework": "alfagent", "trajectory": json.dumps(traj),
                           "ground_truth": json.dumps({"mode": mode, "step": 2, "agent": None})})

    loop = ["go to desk 1"] + ["go to cabinet 2"] * 40
    clean = [f"go to drawer {i}" for i in range(1, 42)]
    f = tmp_path / "ww.jsonl"
    f.write_text(row("a", "A.4", loop) + "\n" + row("b", "A.3", clean) + "\n")
    tab = S.whowhen_table(S.whowhen_rows(f, LD.regex_tokenize))
    allp = tab["populations"]["all"]
    assert allp["l2_loop_existing"]["tp"] == 1 and allp["l2_loop_existing"]["fp"] == 0
    assert allp["token_period_stream"]["tp"] == 1 and allp["token_period_stream"]["fp"] == 0
    assert allp["in_grammar_stream_exempt"]["tp"] == 0
    assert tab["positives_fire_timing"]["l2_loop_existing"] == {"at_or_after_injected_step": 1}
    fx = S.constrained_table(LD.regex_tokenize)
    assert fx["runaway_identical_obj_to_limit"]["in_grammar_fire_token"] is not None
    assert fx["td1_24q_varying"]["in_grammar_fire_token"] is None
