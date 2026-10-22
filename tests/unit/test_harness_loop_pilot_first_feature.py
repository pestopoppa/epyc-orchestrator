"""HIL-6i: the first-feature pilot design instance is complete, consistent and build-only.

Pure: no model, no pool read, no process. Splits are drawn from synthetic rows that carry the
real suite names; the real draw happens once at freeze over the epyc-inference-research pool.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
import yaml

from src.harness_loop.driver import PilotDesignIncomplete, SiblingLoopDriver
from src.harness_loop.interfaces import MEMORY_ACCUMULATING_HOMES, MUTABLE_HOMES, splits_disjoint
from src.harness_loop.pilot import model_family, unmet_requirements
from src.harness_loop.pilots import first_feature as ff

ROOT = Path(__file__).resolve().parents[2]
_EPYC_ROOT = Path(os.environ.get("EPYC_ROOT", "/workspace"))


def _rows(per_suite: int = 260, *, extra=()):
    rows = [{"__pool_metadata__": True, "corpus_version": "synthetic"}]
    for suite in (*ff.STRATA, "hotpotqa", "gpqa_diamond"):
        for i in range(per_suite):
            rows.append({"id": f"{suite}_{i:04d}", "suite": suite, "prompt": "q", "context": "",
                         "image_path": None})
    rows.extend(extra)
    return rows


@pytest.fixture(scope="module")
def splits():
    return ff.build_splits(_rows(), frozen_at="2026-10-22T00:00:00Z")


# ── completeness ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("cycle", [1, 2])
def test_design_has_no_unmet_requirements(splits, cycle):
    assert unmet_requirements(ff.first_feature_design(splits, cycle=cycle)) == []


def test_driver_accepts_the_design(splits, tmp_path):
    from src.harness_loop import substrate

    class _Nope:
        model_id = ff.LOCAL_EDITOR_MODEL

        def propose(self, view, *, max_candidates):  # pragma: no cover - never called
            raise AssertionError("no proposal in a build-only test")

        def evaluate(self, arm, split):  # pragma: no cover - never called
            raise AssertionError("no evaluation in a build-only test")

    j = substrate.open_loop_journal(tmp_path / "loop")
    try:
        SiblingLoopDriver(design=ff.first_feature_design(splits), journal=j,
                          journal_dir=tmp_path / "loop", proposer=_Nope(), evaluator=_Nope())
    except PilotDesignIncomplete as exc:  # pragma: no cover - the assertion above covers it
        pytest.fail(str(exc))


# ── home and target ─────────────────────────────────────────────────────────


def test_home_is_delegation_and_not_memory_accumulating(splits):
    d = ff.first_feature_design(splits)
    assert d.home == ff.HOME == "3.5"
    assert MUTABLE_HOMES[d.home] == "delegation"
    assert d.home not in MEMORY_ACCUMULATING_HOMES and d.stream_positions == ()
    assert d.context_policy_controls == frozenset()  # item 8 binds home 3.6 only
    assert d.feature_id == ff.FEATURE_ID


def test_target_is_live_resolver_prompt_with_its_protected_layer():
    target = ROOT / ff.TARGET_PATH
    text = target.read_text()
    for ph in ff.PLACEHOLDERS:
        assert text.count(ph) == 1, ph
    for marker in ff.GRAMMAR_MARKERS:
        assert marker in text
    assert text.rstrip().endswith(ff.FINAL_ANCHOR)
    review = (ROOT / "src/prompt_builders/review.py").read_text()
    assert '"architect_investigate"' in review
    assert f"question[:{ff.QUESTION_CAP_CHARS}]" in review
    assert f"context[:{ff.EVIDENCE_CAP_CHARS}]" in review
    stage = (ROOT / "src/api/routes/chat_pipeline/delegation_stage.py").read_text()
    assert f'str(initial_role) == "{ff.CONSUMER_ROLE}"' in stage
    features = (ROOT / "src/features.py").read_text()
    assert re.search(r'FeatureSpec\("architect_delegation", False, True,', features)


def test_protected_layer_accepts_wording_and_refuses_policy_edits():
    ref = (ROOT / ff.TARGET_PATH).read_text()
    assert ff.protected_layer_reason(ref, ref) is None
    reworded = ref.replace("This pass is for discovery, not for prescribing a fix.",
                           "Use this pass to discover what matters; do not prescribe a fix.")
    assert reworded != ref and ff.protected_layer_reason(ref, reworded) is None
    assert ff.protected_layer_reason(ref, ref.replace("{question}", "")) == \
        "protected_layer:placeholder:{question}"
    assert ff.protected_layer_reason(ref, ref.replace("Decision:", "Answer:")) == \
        "protected_layer:final_anchor"
    rerouted = ref.replace("to:worker_general", "to:architect_critic")
    assert ff.protected_layer_reason(ref, rerouted) == "protected_layer:role_binding"
    assert ff.protected_layer_reason(ref, ref + "\n{gold_answer}\nDecision:").startswith(
        "protected_layer:new_placeholder")
    screen = ff.make_candidate_screen(ref, leakage=lambda t: None)
    assert screen(reworded) is None and screen(rerouted) == "protected_layer:role_binding"


def test_default_screen_is_the_fail_closed_mhs3_guard(tmp_path, monkeypatch):
    # Seam S2 landed: the default leakage check is the real MHS-3 guard; with no eval-id
    # vocabulary it refuses every text.
    monkeypatch.setenv("AUTOPILOT_EVAL_ID_VOCAB_SOURCES", str(tmp_path / "absent_pool.jsonl"))
    ref = (ROOT / ff.TARGET_PATH).read_text()
    reason = ff.make_candidate_screen(ref)(ref, original=ref)
    assert reason.startswith("eval_leakage_vocabulary_unavailable:missing_eval_id_source")


# ── splits ──────────────────────────────────────────────────────────────────


def test_splits_are_disjoint_sized_and_deterministic(splits):
    parts = (splits.train, splits.validation, splits.arrival, splits.arrival_2, splits.test)
    assert splits_disjoint(*parts)
    sizes = dict(ff.PER_STRATUM)
    k = len(ff.STRATA)
    assert [len(p.qids) for p in parts] == [sizes[r] * k for r in
                                            ("train", "validation", "arrival", "arrival_2", "test")]
    assert len(splits.validation.qids) == ff.PAIRED_N == 420
    assert len(splits.test.qids) >= 200  # P-QUAL-PROMO fresh-draw floor
    again = ff.build_splits(list(reversed(_rows())), frozen_at="2026-10-22T00:00:00Z")
    assert again == splits
    other = ff.build_splits(_rows(), seed="another-seed")
    assert other.manifest_sha256 != splits.manifest_sha256
    for p in parts:
        assert all(q.split("_")[0] != "hotpotqa" and not q.startswith("gpqa") for q in p.qids)
    assert splits.arrival.arrived_at == "" and splits.arrival_2.arrived_at == ""  # sealed


def test_draw_exclusions_and_short_strata_refuse():
    long_q = {"id": "math_long", "suite": "math", "prompt": "x" * (ff.QUESTION_CAP_CHARS + 1),
              "context": "", "image_path": None}
    long_ctx = {"id": "math_ctx", "suite": "math", "prompt": "q",
                "context": "x" * (ff.EVIDENCE_CAP_CHARS + 1), "image_path": None}
    img = {"id": "math_img", "suite": "math", "prompt": "q", "context": "", "image_path": "a.png"}
    s = ff.build_splits(_rows(extra=(long_q, long_ctx, img)), exclude_qids={"math_0000", "mmlu_pro_0001"})
    every = set().union(s.train.qids, s.validation.qids, s.arrival.qids, s.arrival_2.qids, s.test.qids)
    assert not every & {"math_long", "math_ctx", "math_img", "math_0000", "mmlu_pro_0001"}
    need = sum(n for _, n in ff.PER_STRATUM)
    with pytest.raises(ValueError, match="stratum"):
        ff.build_splits(_rows(per_suite=need - 1))


def test_cycle_two_retunes_on_t1_plus_t2_with_a_fresh_arrival(splits):
    d1 = ff.first_feature_design(splits, cycle=1)
    d2 = ff.first_feature_design(splits, cycle=2)
    assert d1.arrival == splits.arrival and d2.arrival == splits.arrival_2
    assert d2.train.qids == splits.train.qids | splits.arrival.qids
    assert splits_disjoint(d2.train, d2.validation, d2.arrival, d2.test)
    assert d1.validation == d2.validation and d1.test == d2.test
    with pytest.raises(ValueError):
        ff.first_feature_design(splits, cycle=3)


# ── power, runs, alpha ───────────────────────────────────────────────────────


def test_power_statement_is_internally_consistent(splits):
    d = ff.first_feature_design(splits)
    assert d.runs >= 2 and d.cycles == 2 and d.mde == ff.MDE
    assert d.paired_n == 420 and d.expected_discordant == 84
    assert ff.ATTEMPTS_PLANNED == 8 and ff.ALPHA_PER_ATTEMPT * ff.ATTEMPTS_PLANNED == pytest.approx(d.alpha_budget)
    k = ff.CRITICAL_WINS
    assert k == 54 and d.min_win_share == pytest.approx(54 / 84)
    # exact one-sided sign test: k passes alpha, k-1 does not
    from math import comb
    tail = lambda j: sum(comb(84, i) for i in range(j, 85)) / 2 ** 84  # noqa: E731
    assert tail(k) <= ff.ALPHA_PER_ATTEMPT < tail(k - 1)
    # the MDE is powered at the assumed discordance; a 5 pp gain is not (null uninformative)
    q = lambda delta: (1 + delta / ff.ASSUMED_DISCORDANCE) / 2  # noqa: E731
    assert ff.sign_test_power(84, k, q(ff.MDE)) >= 0.8
    assert ff.sign_test_power(84, k, q(0.05)) < 0.5
    # near-ceiling (M9): above this incumbent pass share no candidate can reach k wins
    assert ff.PAIRED_N * (1 - ff.CEILING_PASS_SHARE) == pytest.approx(k)
    assert 0.85 < ff.CEILING_PASS_SHARE < 0.9


# ── M9 extras and the notes ──────────────────────────────────────────────────


def test_m9_extras_are_declared(splits):
    notes = ff.first_feature_design(splits).notes
    for key in ("m9.near_ceiling_power", "m9.zero_adoption", "m9.reference_trajectory_arm",
                "m9.offline_then_freeze", "m9.local_editor_prompt_port",
                "m9.planner_self_modification", "m9.regression_set"):
        assert notes.get(key, "").strip(), key
    assert ff.REGRESSION_SET_ID in notes["m9.regression_set"]
    assert ff.EDITOR_PROMPT_SOURCE_SHA256 in notes["m9.local_editor_prompt_port"]
    assert notes["splits.manifest_sha256"] == splits.manifest_sha256


def test_editor_prompt_source_pin_matches_the_installed_package():
    gepa_ip = pytest.importorskip("gepa.strategies.instruction_proposal")
    import hashlib

    tpl = gepa_ip.InstructionProposalSignature.default_prompt_template
    assert hashlib.sha256(tpl.encode("utf-8")).hexdigest() == ff.EDITOR_PROMPT_SOURCE_SHA256


# ── P2 / P3 and the ids against the orchestrator config ──────────────────────


def test_p3_judge_is_cross_model_and_cross_family(splits):
    d = ff.first_feature_design(splits)
    assert d.judge_model not in {d.local_editor_model, d.reference_editor_model}
    fam = model_family(d.judge_model)
    assert fam == "anthropic"
    assert model_family(d.local_editor_model) == "qwen"
    assert model_family(d.reference_editor_model) == "openai"
    assert fam != model_family(d.local_editor_model)
    assert fam != model_family(d.reference_editor_model)  # stronger than the validator
    assert model_family(ff.CONSUMER_MODEL) != fam
    # the alternative judge would share the reference editor's family, so it is not used
    assert model_family(ff.CROSS_FAMILY_JUDGE_OPTION) == model_family(d.reference_editor_model)


def test_ids_resolve_in_the_orchestrator_config(splits):
    judges = yaml.safe_load((ROOT / "orchestration/cloud_judges.yaml").read_text())["judges"]
    j = judges[ff.JUDGE_MODEL.removeprefix("cloud:")]
    assert j["model"] == ff.JUDGE_PINNED_MODEL and j["transport"] == "claude_cli"
    assert judges[ff.CROSS_FAMILY_JUDGE_OPTION.removeprefix("cloud:")]["model"] == ff.REFERENCE_EDITOR_MODEL
    roles = yaml.safe_load((ROOT / "orchestration/model_registry.yaml").read_text())["roles"]
    live = {r: (v.get("model") or {}).get("name") for r, v in roles.items()
            if isinstance(v, dict) and not v.get("deprecated")}
    assert live["frontdoor"] == ff.LOCAL_EDITOR_MODEL
    assert live[ff.CONSUMER_ROLE] == ff.CONSUMER_MODEL
    assert ff.SERVED_LOCAL_MODELS <= set(live.values())
    d = ff.first_feature_design(splits)
    assert d.local_editor_model in d.served_local_models
    assert d.reviewer_model == ""  # not a reviewer/critic mechanism


@pytest.mark.skipif(not (_EPYC_ROOT / ff.DESIGN_DOC).is_file(),
                    reason=f"design doc absent under {_EPYC_ROOT} (set EPYC_ROOT)")
def test_design_document_names_the_same_home_and_feature():
    doc = (_EPYC_ROOT / ff.DESIGN_DOC).read_text()
    assert ff.FEATURE_ID in doc and ff.TARGET_PATH in doc
    assert "§3.5" in doc and ff.REGRESSION_SET_ID in doc


def test_pilot_module_has_no_entrypoint():
    src = (ROOT / "src/harness_loop/pilots/first_feature.py").read_text()
    assert not re.search(r"__name__\s*==\s*[\"']__main__", src)
    assert not (ROOT / "src/harness_loop/pilots/__main__.py").exists()
    for banned in ("httpx", "requests", "subprocess", "socket", "open("):
        assert banned not in src, banned


def test_pilot_and_local_editor_cite_the_same_ported_source():
    # HIL-6h landed the port itself (editor.LOCAL_EDITOR_META_PROMPT_SHA256); the pilot's M9
    # "local-editor port" must name the same published source the editor was ported from.
    from src.harness_loop import editor

    assert ff.EDITOR_PROMPT_SOURCE_SHA256 == editor.GEPA_SOURCE_PROMPT_SHA256
    assert editor.LOCAL_EDITOR_META_PROMPT.sha256 == editor.LOCAL_EDITOR_META_PROMPT_SHA256


def test_declared_decision_rule_is_the_paired_sign_test_at_the_attempt_alpha():
    from src.harness_loop.interfaces import PairedWinShare, sign_test_p

    assert isinstance(ff.DECISION_RULE, PairedWinShare)
    assert ff.DECISION_RULE.alpha == ff.ALPHA_PER_ATTEMPT
    assert ff.DECISION_RULE.min_win_share == ff.MIN_WIN_SHARE
    k, d = ff.CRITICAL_WINS, ff.EXPECTED_DISCORDANT
    assert sign_test_p(k, d) <= ff.ALPHA_PER_ATTEMPT < sign_test_p(k - 1, d)
