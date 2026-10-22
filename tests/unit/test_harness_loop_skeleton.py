"""HIL-6 option B sibling-loop skeleton: substrate facade, attempt ledger, pilot validator, driver.

Everything runs on fakes: no model, no eval runner, no scheduler. The last tests assert that the
package is not wired to any runtime entrypoint.
"""

from __future__ import annotations

import os
import re
from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest

from src.harness_loop import substrate
from src.harness_loop.attempt_ledger import AlphaBudgetExhausted, AttemptLedger
from src.harness_loop.driver import (
    ARM_PIN_FIELDS,
    CYCLE_EVENT,
    DECOY_CANARY,
    CycleSpec,
    PilotDesignIncomplete,
    ResultRefused,
    SiblingLoopDriver,
    TripwireTripped,
)
from src.harness_loop.finalize import CARRIER_ARM, finalize_run, observation
from src.harness_loop.interfaces import (
    ArmResult,
    ArmSpec,
    Candidate,
    IncumbentWinsTies,
    ProposalResult,
    ProposerView,
    SplitRef,
    TokenCount,
    TraceRef,
    VerdictMatrix,
    outcome_keys_in,
    policy_document_reason,
    sha256_text,
)
from src.harness_loop.pilot import CHECKLIST, STREAM_POSITIONS, PilotDesign, unmet_requirements

ROOT = Path(__file__).resolve().parents[2]
PINS = {"tool_surface_sha256": "a" * 64, "served_context_window": 65536,
        "compaction_policy_sha256": "c" * 64, "wall_time_budget_s": 900.0, "turn_budget": 12,
        "final_result_selection_rule": "last FINAL()"}
INCUMBENT = "Summarise the session.\nKeep tool outputs short.\n"


def _empty_harness():
    import numpy as np

    from src.autopilot_core.mutation_guards import _HarnessContent

    empty = np.empty(0, dtype=np.uint64)
    return _HarnessContent(shingles=empty, tokens=empty, token_files=np.empty(0, np.int64))


_EMPTY_HARNESS = _empty_harness()


def _split(sid, role, n0, n1):
    return SplitRef(sid, role, frozenset(f"q{i}" for i in range(n0, n1)), arrived_at="2026-10-22")


def _design(**over):
    base = dict(
        feature_id="HS4-3.6-summary-prompt", home="3.6", primary_metric="accuracy",
        primary_unit="fraction", primary_direction="higher_better",
        train=_split("T1", "train", 0, 10), validation=_split("V", "validation", 10, 20),
        arrival=_split("T2", "arrival", 20, 30), test=_split("TEST", "test", 30, 40),
        cycles=2, stopping_rule="2 cycles; stop early on 2 consecutive KEEP_BASE",
        local_editor_model="qwen3.6-35b-a3b", reference_editor_model="claude-opus-5-5",
        judge_model="cloud:sonnet-5-5-low", matched_rollout_baseline=True, unmodified_control=True,
        verification_arms=frozenset({"with", "none", "self_judged"}), runs=2, mde=0.05,
        paired_n=200, expected_discordant=40, min_win_share=0.65,
        token_seats=frozenset({"proposer", "judge", "solver"}), headroom_note="expected-null risk noted",
        exclusions_declared=True, regression_set_id="hil-regress-v0", per_role_arm_set=True,
        context_policy_controls=frozenset({"no_management", "wording_only"}),
    )
    base.update(over)
    return PilotDesign(**base)


# ── substrate facade ────────────────────────────────────────────────────────


def test_open_loop_journal_refuses_live_and_implicit_dirs(tmp_path):
    from scripts.autopilot.experiment_journal import DEFAULT_JOURNAL_DIR

    with pytest.raises(substrate.LiveSubstrateRefused):
        substrate.open_loop_journal(None)  # type: ignore[arg-type]
    with pytest.raises(substrate.LiveSubstrateRefused):
        substrate.open_loop_journal(DEFAULT_JOURNAL_DIR)
    foreign = tmp_path / "ap"
    foreign.mkdir()
    (foreign / "autopilot_journal.jsonl").write_text("")
    with pytest.raises(substrate.LiveSubstrateRefused):
        substrate.open_loop_journal(foreign)
    with pytest.raises(substrate.LiveSubstrateRefused):
        substrate.append_rejection(foreign, target="x", mutation_type="m", artifact_kind="prompt",
                                   rejecting_gate="g")


def test_open_loop_journal_owns_a_fresh_dir_and_reopens(tmp_path):
    j = substrate.open_loop_journal(tmp_path / "loop")
    assert (tmp_path / "loop" / substrate.LOOP_JOURNAL_MARKER).is_file()
    j.append_ledger_event({"type": "hil_probe", "n": 1})
    j2 = substrate.open_loop_journal(tmp_path / "loop")
    assert [e["n"] for e in j2.ledger_events("hil_probe")] == [1]
    assert j2.all_entries() == []  # no AutoPilot trial rows, no trace emission


def test_fence_helpers_arm_and_refuse_disarm():
    assert substrate.fenced_eval_request({"prompt": "x"}) == {"prompt": "x", "eval_fence": True}
    with pytest.raises(ValueError):
        substrate.fenced_eval_request({"eval_fence": False})
    substrate.require_fence_active({"state": "active"})
    for bad in ({"state": "unarmed"}, {}, None):
        with pytest.raises(ValueError):
            substrate.require_fence_active(bad)


def test_leakage_seam_fails_closed_without_vocabulary(tmp_path, monkeypatch):
    # S2 landed: MHS-3 is the real guard and keeps its fail-closed contract.
    monkeypatch.setenv("AUTOPILOT_EVAL_ID_VOCAB_SOURCES", str(tmp_path / "absent_pool.jsonl"))
    reason = substrate.leakage_reason("anything", harness=_EMPTY_HARNESS)
    assert reason.startswith("eval_leakage_vocabulary_unavailable:missing_eval_id_source")


def test_fingerprint_is_the_autopilot_implementation():
    from src.autopilot_core import infra_fingerprint

    assert substrate.compare_infra_fingerprints is infra_fingerprint.compare_infra_fingerprints
    assert substrate.compare_infra_fingerprints(None, None)["status"] == substrate.UNVERIFIED


# ── attempt ledger (item 21) ─────────────────────────────────────────────────


def test_attempt_ledger_reserves_alpha_and_resumes(tmp_path):
    j = substrate.open_loop_journal(tmp_path / "loop")
    led = AttemptLedger(j, alpha_budget=0.05)
    a = led.open("c1", 0.02)
    led.close(a, "keep_base")
    led.open("c2", 0.02)  # interrupted: never closed
    j2 = substrate.open_loop_journal(tmp_path / "loop")
    led2 = AttemptLedger(j2, alpha_budget=0.05)
    assert led2.next_index() == 2
    assert led2.alpha_spent() == pytest.approx(0.04)
    assert [x.outcome for x in led2.attempts()] == ["keep_base", None]
    with pytest.raises(AlphaBudgetExhausted):
        led2.open("c3", 0.02)
    with pytest.raises(ValueError):
        led2.close(0, "keep")
    with pytest.raises(KeyError):
        led2.close(9, "keep")


# ── pilot validator ──────────────────────────────────────────────────────────


def test_complete_design_has_no_unmet_requirements():
    assert unmet_requirements(_design()) == []
    assert sorted(CHECKLIST) == list(range(1, 23))


def test_pilot_requirements_are_enforced():
    keys = {k for k, _ in unmet_requirements(_design(cycles=1, judge_model="qwen3.6-27b"))}
    assert {"P1", "P3"} <= keys
    keys = {k for k, _ in unmet_requirements(_design(reference_editor_model="qwen3.6-35b-a3b"))}
    assert "P2" in keys
    keys = {k for k, _ in unmet_requirements(_design(home="3.1"))}
    assert "I20" in keys
    assert unmet_requirements(_design(home="3.1", stream_positions=STREAM_POSITIONS)) == []
    keys = {k for k, _ in unmet_requirements(_design(context_policy_controls=frozenset()))}
    assert "I8" in keys
    keys = {k for k, _ in unmet_requirements(_design(arrival=_split("T2", "arrival", 5, 15)))}
    assert "I3" in keys
    assert {k for k, _ in unmet_requirements(_design(home="3.7"))} >= {"home"}


# ── proposer view (item 7) and judge matrix (item 17) ─────────────────────────


def test_view_refuses_policy_documents_and_unmutable_homes():
    assert policy_document_reason("agents/AGENTS.md")
    assert policy_document_reason(".claude/skills/x/notes.md")
    assert policy_document_reason("orchestration/prompts/compaction.md") is None
    with pytest.raises(ValueError):
        ProposerView("f", "3.6", "AGENTS.md", "prompt", INCUMBENT)
    with pytest.raises(ValueError):
        ProposerView("f", "3.7", "orchestration/prompts/x.md", "prompt", INCUMBENT)
    with pytest.raises(ValueError):
        ProposerView("f", "3.6", "p.md", "prompt", INCUMBENT, traces=(TraceRef("t", "worker"),))


def test_outcome_keys_are_detected_at_any_depth():
    assert outcome_keys_in({"a": [{"b": {"Score": 1}}], "passed": True}) == ["a[0].b.Score", "passed"]
    assert outcome_keys_in({"events": ["tool_error"], "role": "worker"}) == []


def test_verdict_matrix_completeness_and_argmax():
    m = VerdictMatrix("j", {"a": {"q1": "credible", "q2": "credible"},
                            "b": {"q1": "credible", "q2": "not_credible"}}, chosen="b")
    m.validate(["a", "b"], ["q1", "q2"])
    assert m.choice_is_argmax() is False
    with pytest.raises(ValueError):
        VerdictMatrix("j", {"a": {"q1": ""}}).validate(["a"], ["q1"])


# ── driver on fakes ───────────────────────────────────────────────────────────


class FakeProposer:
    model_id = "qwen3.6-35b-a3b"

    def __init__(self, texts):
        self.texts = texts
        self.views = []

    def propose(self, view, *, max_candidates):
        self.views.append(view)
        cands = tuple(
            Candidate(f"c{len(self.views)}-{i}", view.feature_id, view.home, view.target_path,
                      view.target_kind, view.incumbent_text, t, self.model_id)
            for i, t in enumerate(self.texts[:max_candidates]))
        return ProposalResult(cands, (TokenCount(100, 20, 0, "proposer"),))


class FakeEvaluator:
    """Score = fraction of 'short' occurrences, capped: deterministic, no model."""

    def __init__(self, texts_by_sha):
        self.texts = texts_by_sha
        self.calls = []

    def evaluate(self, arm, split):
        self.calls.append((arm.arm_id, split.split_id))
        text = self.texts[arm.content_sha256]
        value = min(1.0, 0.5 + 0.1 * text.count("short"))
        return ArmResult(arm.arm_id, split.split_id, "accuracy", value, "fraction", "higher_better",
                         scored=len(split.qids), attempted=len(split.qids),
                         tokens=(TokenCount(1000, 50, 200, "solver"),), treatment_fired=len(split.qids),
                         fence_state="active")


def _driver(tmp_path, texts, *, leakage=lambda t, **_: None, judge=None, design=None):
    j = substrate.open_loop_journal(tmp_path / "loop")
    pool = [INCUMBENT, *texts, *(INCUMBENT + t for t in texts)]
    ev = FakeEvaluator({sha256_text(t): t for t in pool})
    prop = FakeProposer(texts)
    drv = SiblingLoopDriver(design=design or _design(), journal=j, journal_dir=tmp_path / "loop",
                            proposer=prop, evaluator=ev, leakage=leakage, judge=judge,
                            answer_markers=("GOLD-ANSWER-",))
    return drv, prop, ev, j


def _spec(cycle, text, tune):
    inc = ArmSpec(f"inc{cycle}", "incumbent", "HS4-3.6-summary-prompt",
                  "orchestration/prompts/compaction.md", sha256_text(text), dict(PINS))
    return CycleSpec(cycle, tune, inc, text, dict(PINS), max_candidates=3)


def _view(text):
    return ProposerView("HS4-3.6-summary-prompt", "3.6", "orchestration/prompts/compaction.md",
                        "prompt", text, traces=(TraceRef("t1", "worker", ("truncation",), "..."),),
                        selection_rule="first 20 traces with any L2 event, by trace id")


def test_driver_refuses_an_incomplete_design(tmp_path):
    with pytest.raises(PilotDesignIncomplete):
        _driver(tmp_path, [], design=_design(cycles=1))


def test_two_cycles_with_arriving_split(tmp_path):
    better = INCUMBENT + "Prefer short short answers.\n"
    drv, prop, ev, j = _driver(tmp_path, ["Prefer short short answers.\n"])
    prop.texts = [better]
    ev.texts[sha256_text(better)] = better
    d = drv.design
    out1 = drv.run_cycle(_view(INCUMBENT), _spec(1, INCUMBENT, d.train))
    assert out1.outcome == "keep" and out1.committed_text == better
    assert out1.arrival is not None and out1.arrival.split_id == "T2"
    # selection on validation only; arrival evaluated once, for the committed arm, no tuning
    assert [s for _, s in ev.calls] == ["V", "V", "T2"]
    # cycle 2 re-tunes on T1+T2 from the cycle-1 harness; nothing beats it -> KEEP_BASE
    prop.texts = [better.replace("short short", "brief")]
    ev.texts[sha256_text(prop.texts[0])] = prop.texts[0]
    t12 = SplitRef("T1+T2", "train", d.train.qids | d.arrival.qids)
    with pytest.raises(ResultRefused):  # T2 is the arrival split: tuning on it must be refused
        drv.run_cycle(_view(better), _spec(2, better, t12))
    out2 = drv.run_cycle(_view(better), _spec(2, better, d.train))
    assert out2.outcome == "keep_base" and out2.committed_text == better
    events = j.ledger_events(CYCLE_EVENT)
    assert [e["outcome"] for e in events] == ["keep", "keep_base"]
    assert events[0]["harness_bytes"] == len(better.encode())
    assert events[0]["tokens_by_seat"] == {"proposer": 120, "solver": 3 * 1250}
    assert all(e["decision_grade"] == "observation" for e in events)
    assert [a.outcome for a in drv.ledger.attempts()] == ["keep", "keep_base"]


def test_default_leakage_seam_fails_closed_to_keep_base(tmp_path, monkeypatch):
    # The default driver leakage check is the real MHS-3 guard (S2); with no vocabulary it
    # refuses every candidate, and the stalled round is KEEP_BASE, not a failure.
    monkeypatch.setenv("AUTOPILOT_EVAL_ID_VOCAB_SOURCES", str(tmp_path / "absent_pool.jsonl"))
    drv, _, _, _ = _driver(tmp_path, ["Prefer short answers.\n"])
    drv.leakage = partial(substrate.leakage_reason, harness=_EMPTY_HARNESS)
    out = drv.run_cycle(_view(INCUMBENT), _spec(1, INCUMBENT, drv.design.train))
    assert out.outcome == "keep_base"
    (reason,) = set(out.rejected.values())
    assert reason.startswith("eval_leakage_vocabulary_unavailable:missing_eval_id_source")
    recs = substrate.load_rejections(tmp_path / "loop")
    assert len(recs) == 1 and recs[0]["gate_detail"].startswith("eval_leakage_vocabulary_unavailable")


def _fixture_pool(tmp_path):
    import json

    rows = [{"id": f"fixture_story_{i:04d}", "suite": "fixturebench",
             "prompt": f"Which cartographer named Ignatius Pemberton{i} charted the Vorlaxian "
                       f"archipelago using a heliograph relay in winter expedition number {i}?",
             "expected": f"Quillon Varnesbury{i}"} for i in range(1, 6)]
    pool = tmp_path / "question_pool.jsonl"
    pool.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return pool, rows


def test_leakage_seam_is_promptforge_mhs3_over_the_added_text(tmp_path, monkeypatch):
    from src.autopilot_core import mutation_guards as mg

    pool, rows = _fixture_pool(tmp_path)
    vocab = mg.load_eval_id_vocabulary(((pool, True),))
    assert vocab.available
    check = partial(substrate.leakage_reason, vocabulary=vocab, harness=_EMPTY_HARNESS)
    # A named eval instance in the added text is refused...
    assert check(INCUMBENT + "See fixture_story_0003 first.\n", original=INCUMBENT).startswith(
        "eval_instance_leakage")
    # ...but the same id already in the incumbent is not blamed on the candidate.
    seeded = INCUMBENT + "See fixture_story_0003 first.\n"
    assert check(seeded + "Be brief.\n", original=seeded) is None
    # Verbatim eval content and suite keying are refused; plain advice is clean.
    copied = INCUMBENT + rows[1]["prompt"] + "\n"
    assert check(copied, original=INCUMBENT).startswith("eval_content_ngram_overlap")
    assert check(INCUMBENT + 'if suite == "fixturebench": answer tersely\n',
                 original=INCUMBENT).startswith("eval_suite_special_casing")
    assert check(INCUMBENT + "Prefer short sentences.\n", original=INCUMBENT) is None
    # Same verdicts as calling the guards the way PromptForge does.
    added = mg._added_text(INCUMBENT, copied)
    assert check(copied, original=INCUMBENT) == mg.eval_leakage_reason(
        added, vocab, original=INCUMBENT, harness=_EMPTY_HARNESS)


def test_driver_passes_original_and_rendered_traces_to_leakage(tmp_path):
    seen = []

    def spy(text, **kw):
        seen.append((text, kw))
        return None

    drv, _, _, _ = _driver(tmp_path, ["Prefer short answers.\n"], leakage=spy)
    drv.run_cycle(_view(INCUMBENT), _spec(1, INCUMBENT, drv.design.train))
    (text, kw), = seen
    assert kw["original"] == INCUMBENT and text != INCUMBENT
    from src.autopilot_core.mutation_guards import _trace_blocks

    assert [label for label, _ in _trace_blocks(kw["trace_context"])] == [
        "[1] trace=t1 role=worker"]


def test_tripwire_blocks_decoy_canary_and_answer_markers(tmp_path):
    drv, prop, _, _ = _driver(tmp_path, [])
    for bad in (INCUMBENT + DECOY_CANARY, INCUMBENT + "GOLD-ANSWER-42"):
        with pytest.raises(TripwireTripped):
            drv.run_cycle(_view(bad), _spec(1, bad, drv.design.train))
    assert prop.views == []  # the proposer never saw it


def test_unarmed_fence_or_missing_pins_refuse_the_result(tmp_path):
    drv, _, ev, _ = _driver(tmp_path, [])
    orig = ev.evaluate
    ev.evaluate = lambda arm, split: replace(orig(arm, split), fence_state="unarmed")
    with pytest.raises(ValueError):
        drv.run_cycle(_view(INCUMBENT), _spec(1, INCUMBENT, drv.design.train))
    ev.evaluate = orig
    spec = _spec(1, INCUMBENT, drv.design.train)
    bad = replace(spec, incumbent=replace(spec.incumbent, pins={}))
    with pytest.raises(ResultRefused):
        drv.run_cycle(_view(INCUMBENT), bad)


def test_judge_matrix_is_validated_and_argmax_checked(tmp_path):
    class J:
        model_id = "cloud:sonnet-5-5-low"

        def judge(self, cands, instances):
            cells = {c.candidate_id: {i: "credible" for i in instances} for c in cands}
            return VerdictMatrix(self.model_id, cells, chosen=cands[0].candidate_id,
                                 tokens=(TokenCount(10, 5, 0, "judge"),))

    drv, _, _, _ = _driver(tmp_path, ["Prefer short answers.\n"], judge=J())
    out = drv.run_cycle(_view(INCUMBENT), _spec(1, INCUMBENT, drv.design.train))
    assert out.judge_argmax_ok is True and out.tokens_by_seat["judge"] == 15


def test_final_test_touches_the_test_split_once(tmp_path):
    drv, _, _, _ = _driver(tmp_path, [])
    arm = _spec(1, INCUMBENT, drv.design.train).incumbent
    assert drv.final_test(arm).split_id == "TEST"
    with pytest.raises(ResultRefused):
        drv.final_test(arm)


def test_incumbent_wins_ties():
    r = lambda v, d="higher_better": ArmResult("a", "V", "m", v, "u", d, 1, 1)  # noqa: E731
    assert not IncumbentWinsTies().decide(r(0.5), r(0.5)).keep
    assert IncumbentWinsTies(0.01).decide(r(0.5), r(0.52)).keep
    assert IncumbentWinsTies().decide(r(2.0, "lower_better"), r(1.0, "lower_better")).keep
    assert IncumbentWinsTies().decide(r(0.5), r(0.9)).claim_grade == "observation"


# ── finalize (VB-HIL-WRITE) ───────────────────────────────────────────────────


def test_arm_pin_fields_match_the_sidecar():
    from scripts.benchmark import hil_ab_sidecar

    assert tuple(ARM_PIN_FIELDS) == tuple(hil_ab_sidecar.ARM_PIN_FIELDS)


def _obs_pair(tmp_path):
    prompt = tmp_path / "prompt.md"
    prompt.write_text(INCUMBENT)
    inc = ArmSpec("inc", "incumbent", "HS4-3.6", str(prompt), sha256_text(INCUMBENT), dict(PINS))
    cand = ArmSpec("c1", "candidate", "HS4-3.6", str(prompt), "f" * 64, dict(PINS))
    res = lambda a, v: ArmResult(a, "V", "accuracy", v, "fraction", "higher_better", 10, 10)  # noqa: E731
    common = dict(objective="summary_accuracy", suite_id="hil-fixture", suite_version="v0",
                  split="V", native_verdict="observed", prompt_param_path=str(prompt),
                  harness_pin="orch@test", config_sha256="c" * 64,
                  identities={"model": "fixture-model"})
    return [observation(res("inc", 0.5), inc, **common), observation(res("c1", 0.7), cand, **common)]


def test_finalize_projects_and_calls_the_hook_once(tmp_path):
    obs = _obs_pair(tmp_path)
    assert [o["arm"] for o in obs] == ["BASELINE", "CANDIDATE"]
    assert set(CARRIER_ARM.values()) == {"BASELINE", "CANDIDATE"}
    calls = []
    results = tmp_path / "results.jsonl"
    with pytest.raises(FileNotFoundError):
        finalize_run(tmp_path / "out", run_id="r", results_path=results, observations=obs,
                     pins=PINS, finalize_fn=lambda *a, **k: calls.append(k))
    results.write_text("{}\n")
    with pytest.raises(ValueError):
        finalize_run(tmp_path / "out", run_id="r", results_path=results, observations=obs + obs[:1],
                     pins=PINS, finalize_fn=lambda *a, **k: calls.append(k))
    finalize_run(tmp_path / "out", run_id="r", results_path=results, observations=obs, pins=PINS,
                 finalize_fn=lambda *a, **k: calls.append(k) or Path("x"))
    assert len(calls) == 1 and calls[0]["protocol_id"] is None


_EPYC_ROOT = Path(os.environ.get("EPYC_ROOT", "/workspace"))


@pytest.mark.skipif(not (_EPYC_ROOT / "scripts/vidya/adapters/harness_improvement.py").is_file(),
                    reason=f"root carrier absent under {_EPYC_ROOT} (set EPYC_ROOT)")
def test_finalize_rows_are_accepted_by_the_root_carrier(tmp_path):
    obs = _obs_pair(tmp_path)
    results = tmp_path / "results.jsonl"
    results.write_text('{"item": 1}\n')
    out = finalize_run(tmp_path / "out", run_id="hil-skeleton-fixture", results_path=results,
                       observations=obs, pins=PINS, epyc_root=_EPYC_ROOT)
    assert len(out.read_text().splitlines()) == 2


# ── not wired to any runtime entrypoint ───────────────────────────────────────


def test_package_has_no_entrypoint_and_no_importer():
    # HIL-6j (operator lift 2026-10-22, pilot only): pilot_run.py is the ONE entrypoint, and it
    # refuses every design but the HIL-6i pilot (tests/unit/test_harness_loop_pilot_run.py).
    pkg = ROOT / "src" / "harness_loop"
    assert not (pkg / "__main__.py").exists()
    for f in pkg.glob("*.py"):
        if f.name == "pilot_run.py":
            continue
        assert not re.search(r"__name__\s*==\s*[\"']__main__", f.read_text()), f
    pat = re.compile(r"\b(?:from|import)\s+src\.harness_loop\b|src/harness_loop")
    importers = []
    for base in ("src", "scripts", "orchestration"):
        for f in (ROOT / base).rglob("*"):
            if f.is_dir() or pkg in f.parents or f.suffix not in {".py", ".sh", ".yaml", ".yml", ".toml", ".json"}:
                continue
            try:
                if pat.search(f.read_text(errors="ignore")):
                    importers.append(str(f.relative_to(ROOT)))
            except OSError:
                continue
    assert importers == []


def test_driver_refuses_outcome_named_trace_events(tmp_path):
    # outcome_keys_in scans keys; an EVENT named like an outcome is a value and must be refused too.
    drv, prop, _, _ = _driver(tmp_path, ["Prefer short answers.\n"])
    view = ProposerView("HS4-3.6-summary-prompt", "3.6", "orchestration/prompts/compaction.md",
                        "prompt", INCUMBENT, traces=(TraceRef("t1", "worker", ("passed",), "..."),),
                        selection_rule="first 20 traces with any L2 event, by trace id")
    with pytest.raises(TripwireTripped):
        drv.run_cycle(view, _spec(1, INCUMBENT, drv.design.train))
    assert prop.views == []


def test_paired_final_test_is_one_touch_for_all_arms(tmp_path):
    drv, _, _, j = _driver(tmp_path, [])
    spec = _spec(1, INCUMBENT, drv.design.train)
    committed, control = drv.final_test_paired(spec.incumbent, spec.incumbent)
    assert committed.split_id == control.split_id == "TEST"
    from src.harness_loop.driver import TEST_EVENT

    (event,) = j.ledger_events(TEST_EVENT)
    assert len(event["arm_ids"]) == 2
    with pytest.raises(ResultRefused):
        drv.final_test(spec.incumbent)


def test_paired_win_share_is_the_exact_sign_test():
    from src.harness_loop.interfaces import PairedWinShare, sign_test_p

    def arm(outcomes):
        return ArmResult("a", "V", "pass", None, "fraction", "higher_better", len(outcomes),
                         len(outcomes), item_outcomes=outcomes)

    # 84 discordant pairs at alpha 0.00625: 54 wins is the smallest significant count.
    assert sign_test_p(54, 84) <= 0.00625 < sign_test_p(53, 84)
    rule = PairedWinShare(alpha=0.00625, min_win_share=54 / 84)

    def pair(wins, losses, ties=300):
        inc = {f"w{i}": False for i in range(wins)} | {f"l{i}": True for i in range(losses)}
        cand = {f"w{i}": True for i in range(wins)} | {f"l{i}": False for i in range(losses)}
        tied = {f"t{i}": True for i in range(ties)}
        return arm(inc | tied), arm(cand | tied)

    inc, cand = pair(54, 30)
    assert rule.decide(inc, cand).keep
    assert not rule.decide(*pair(53, 31)).keep  # same D = 84, one win short
    inc = inc.item_outcomes
    cand = cand.item_outcomes
    # Ties go to the incumbent; unscored items are excluded, not counted as losses.
    assert not rule.decide(arm(inc), arm(inc)).keep
    assert rule.decide(arm(inc), arm(dict(cand, t0=None))).keep
    assert rule.decide(arm({}), arm({})).reason == "no_paired_items"
