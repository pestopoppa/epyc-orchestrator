#!/usr/bin/env python3
"""Unit tests for scripts/analysis/run_paired_ab.py (paired A/B driver).

Coverage is entirely INFERENCE-FREE: arm-config parsing (flag/params/control),
corpus resolution from a JSONL manifest fixture (+ suite filter, seed-deterministic
order, dataset hash), grader dispatch (a FAKE generic grader AND the REAL EV-12
patch_verifier against a small unified-diff fixture), and the McNemar/Wilson paired
wiring on synthetic paired outcomes with concrete expected values. The env-gated
execution bridge (placement-queue inference) is never exercised.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# ── load the runner module by path (robust; no scripts.* package needed) ──────
_MODULE_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "analysis" / "run_paired_ab.py"
)
_SPEC = importlib.util.spec_from_file_location("run_paired_ab", _MODULE_PATH)
runner = importlib.util.module_from_spec(_SPEC)
sys.modules["run_paired_ab"] = runner  # register before exec (dataclasses)
_SPEC.loader.exec_module(runner)

# paired_stats (McNemar + profile gate) via the runner's own loader.
ps = runner._load_paired_stats()
QuestionOutcome = ps.QuestionOutcome
PairedComparisonMismatchError = ps.PairedComparisonMismatchError


# ── real unified-diff fixtures (self-contained; mirror test_patch_verifier) ───

CLEAN_DIFF = (
    "--- a/mod.py\n"
    "+++ b/mod.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def foo():\n"
    "-    return 1\n"
    "+    return 2\n"
)

NON_APPLYING_DIFF = (
    "--- a/mod.py\n"
    "+++ b/mod.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def foo():\n"
    "-    return 999\n"
    "+    return 2\n"
)

BASE_TREE = {"mod.py": "def foo():\n    return 1\n"}


# --------------------------------------------------------------------------- #
# Arm-config parsing
# --------------------------------------------------------------------------- #
def test_parse_arms_control_defaults_baseline_second():
    candidate, baseline = runner.parse_arms("edit_transaction,baseline")
    assert candidate.name == "edit_transaction"
    assert baseline.name == "baseline"
    assert baseline.is_baseline is True
    assert candidate.is_baseline is False
    # No specs -> both are control arms (empty flags/params).
    assert candidate.kind == "control"
    assert baseline.kind == "control"
    assert candidate.flags == {} and candidate.params == {}


def test_parse_arms_flag_and_params_specs():
    candidate, baseline = runner.parse_arms(
        "cand,base",
        arm_specs=[
            "cand=flag:REVIEW_BEFORE_COMMIT_CONSULT=1,EDIT_MODE=on",
            'base=params:{"temperature": 0.0, "top_p": 1}',
        ],
        roles={"cand": "coder_escalation"},
    )
    assert candidate.kind == "flag"
    assert candidate.flags == {"REVIEW_BEFORE_COMMIT_CONSULT": "1", "EDIT_MODE": "on"}
    assert candidate.role == "coder_escalation"
    assert baseline.kind == "params"
    assert baseline.params == {"temperature": 0.0, "top_p": 1}
    # Transport is always the placement queue, never /chat.
    t = candidate.transport()
    assert t["transport"] == "placement_queue"
    assert t["request_priority"] == "background"
    assert t["workload_class"] == "eval_batch"
    assert t["uses_chat_endpoint"] is False
    assert t["force_role"] == "coder_escalation"


def test_parse_arms_explicit_baseline_arm_first():
    candidate, baseline = runner.parse_arms("role_aware,role_agnostic", baseline_arm="role_aware")
    assert baseline.name == "role_aware"
    assert candidate.name == "role_agnostic"


def test_parse_arms_rejects_wrong_count_and_duplicates():
    with pytest.raises(ValueError):
        runner.parse_arms("only_one")
    with pytest.raises(ValueError):
        runner.parse_arms("a,b,c")
    with pytest.raises(ValueError):
        runner.parse_arms("same,same")


def test_parse_arm_spec_forms():
    assert runner.parse_arm_spec("x=flag:A=1") == ("x", {"A": "1"}, {})
    name, flags, params = runner.parse_arm_spec('y=params:{"k": 2}')
    assert name == "y" and flags == {} and params == {"k": 2}
    assert runner.parse_arm_spec("z=") == ("z", {}, {})
    with pytest.raises(ValueError):
        runner.parse_arm_spec("noeq")


# --------------------------------------------------------------------------- #
# Corpus resolution
# --------------------------------------------------------------------------- #
def _write_manifest(tmp_path: Path) -> Path:
    rows = [
        {"qid": "m1", "prompt": "1+1", "expected": "2", "suite": "math500"},
        {"qid": "m2", "prompt": "2+2", "expected": "4", "suite": "math500"},
        {"qid": "m3", "prompt": "3+3", "expected": "6", "suite": "math500"},
        {"qid": "c1", "prompt": "code a", "expected": "x", "suite": "livecodebench"},
        {"qid": "c2", "prompt": "code b", "expected": "y", "suite": "livecodebench"},
    ]
    p = tmp_path / "corpus.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def test_resolve_corpus_from_manifest_suite_filter_and_cap(tmp_path):
    manifest = _write_manifest(tmp_path)
    res = runner.resolve_corpus(manifest_path=manifest, suites=["math500"], n=2, seed=42)
    assert res.resolved is True
    assert res.n_available == 3  # three math500 rows exist
    assert res.n_selected == 2  # capped to n
    assert {it["suite"] for it in res.items} == {"math500"}
    assert res.dataset_sha256.startswith("sha256:")


def test_resolve_corpus_seed_deterministic_order(tmp_path):
    manifest = _write_manifest(tmp_path)
    a = runner.resolve_corpus(manifest_path=manifest, suites=None, n=5, seed=42)
    b = runner.resolve_corpus(manifest_path=manifest, suites=None, n=5, seed=42)
    c = runner.resolve_corpus(manifest_path=manifest, suites=None, n=5, seed=7)
    ids_a = [it["qid"] for it in a.items]
    ids_b = [it["qid"] for it in b.items]
    assert ids_a == ids_b  # same seed -> identical order
    assert a.dataset_sha256 == b.dataset_sha256
    # A different seed permutes the order (same set), so the hash differs.
    assert set(ids_a) == set(it["qid"] for it in c.items)
    assert a.dataset_sha256 != c.dataset_sha256


def test_resolve_corpus_suite_only_is_unresolved():
    res = runner.resolve_corpus(manifest_path=None, suites=["multifile_edit"], n=100, seed=42)
    assert res.resolved is False
    assert res.items == []
    assert res.suites == ["multifile_edit"]
    assert res.n_requested == 100


# --------------------------------------------------------------------------- #
# Grader dispatch — generic (incl. a FAKE grader) + real patch_verifier
# --------------------------------------------------------------------------- #
def test_generic_graders_exact_and_substring():
    exact = runner.get_grader("exact")
    assert exact.fn("2", {"expected": "2"}) is True
    assert exact.fn("the answer is 2", {"expected": "2"}) is False
    sub = runner.get_grader("substring")
    assert sub.fn("the answer is 2", {"expected": "2"}) is True
    assert sub.fn("nope", {"expected": "2"}) is False


def test_get_grader_unknown_raises():
    with pytest.raises(ValueError):
        runner.get_grader("does_not_exist")


def test_patch_verifier_grader_on_real_diff():
    grader = runner.get_grader("patch_verifier")
    assert grader.needs_base_tree is True
    assert grader.fn(CLEAN_DIFF, {"qid": "p1", "base_tree": BASE_TREE}) is True
    assert grader.fn(NON_APPLYING_DIFF, {"qid": "p2", "base_tree": BASE_TREE}) is False
    with pytest.raises(ValueError):
        grader.fn(CLEAN_DIFF, {"qid": "p3"})  # missing base_tree


def test_build_arm_rows_with_fake_grader():
    """A FAKE generic grader drives build_arm_rows -> rows + outcome vector."""
    fake = runner.Grader(name="fake", fn=lambda pred, item: pred == item["expected"])
    items = [
        {"qid": "q1", "prompt": "p", "expected": "yes", "suite": "s"},
        {"qid": "q2", "prompt": "p", "expected": "no", "suite": "s"},
    ]
    predictions = {"q1": "yes", "q2": "wrong"}
    rows, vector = runner.build_arm_rows(
        runner.ArmConfig(name="candA"), items, predictions, fake, model="M", quant="Q4"
    )
    assert vector["q1"].correct is True
    assert vector["q2"].correct is False
    assert len(rows) == 2
    # Rows are model/quant-stamped, never role-keyed.
    assert rows[0]["model"] == "M" and rows[0]["quant"] == "Q4"
    assert rows[0]["arm"] == "candA"
    assert "role" not in rows[0]
    assert rows[0]["transport"] == "placement_queue"


def test_build_arm_rows_with_real_patch_verifier():
    grader = runner.get_grader("patch_verifier")
    items = [
        {"qid": "d1", "prompt": "edit", "suite": "multifile_edit", "base_tree": BASE_TREE},
        {"qid": "d2", "prompt": "edit", "suite": "multifile_edit", "base_tree": BASE_TREE},
    ]
    predictions = {"d1": CLEAN_DIFF, "d2": NON_APPLYING_DIFF}
    rows, vector = runner.build_arm_rows(
        runner.ArmConfig(name="edit_transaction"), items, predictions, grader,
        model="coder", quant="Q8",
    )
    assert vector["d1"].correct is True
    assert vector["d2"].correct is False


# --------------------------------------------------------------------------- #
# Paired statistics — McNemar + Wilson wiring (concrete expected values)
# --------------------------------------------------------------------------- #
def _vectors_with_table(same_correct, same_wrong, a_correct_b_wrong, a_wrong_b_correct):
    """Build baseline (a) / candidate (b) vectors realizing an exact McNemar table."""
    a: dict[str, object] = {}
    b: dict[str, object] = {}
    i = 0

    def _add(a_ok: bool, b_ok: bool, count: int):
        nonlocal i
        for _ in range(count):
            qid = f"q{i}"
            a[qid] = QuestionOutcome(qid=qid, suite="s", correct=a_ok, trial_id=0)
            b[qid] = QuestionOutcome(qid=qid, suite="s", correct=b_ok, trial_id=1)
            i += 1

    _add(True, True, same_correct)
    _add(False, False, same_wrong)
    _add(True, False, a_correct_b_wrong)
    _add(False, True, a_wrong_b_correct)
    return a, b


def test_compute_paired_result_exact_values():
    baseline_vec, candidate_vec = _vectors_with_table(
        same_correct=4, same_wrong=2, a_correct_b_wrong=1, a_wrong_b_correct=3
    )  # n=10; acc_a=0.5, acc_b=0.7; McNemar b=1,c=3
    result = runner.compute_paired_result(
        baseline_vec,
        candidate_vec,
        baseline_label="baseline",
        candidate_label="candidate",
        dataset_sha256="sha256:deadbeef",
        test_profile="grader=exact;seed=42;sampling=production",
        model="Qwen3.6-35B-A3B",
        quant="Q8_0",
    )
    assert result["shared_qids"] == 10
    assert result["baseline_correct"] == 5
    assert result["candidate_correct"] == 7
    assert result["baseline_accuracy"] == pytest.approx(0.5)
    assert result["candidate_accuracy"] == pytest.approx(0.7)
    assert result["delta_candidate_minus_baseline"] == pytest.approx(0.2)
    # Exact two-sided sign-test over b=1,c=3 discordant pairs: 2*(C4,0+C4,1)/2^4 = 0.625.
    assert result["p_value_two_sided"] == pytest.approx(0.625)
    assert result["a_correct_b_wrong"] == 1
    assert result["a_wrong_b_correct"] == 3
    # Wilson 95% score intervals (z=1.959964): (5/10) and (7/10).
    lo_b, hi_b = result["baseline_wilson95"]
    assert lo_b == pytest.approx(0.2366, abs=1e-3)
    assert hi_b == pytest.approx(0.7634, abs=1e-3)
    lo_c, hi_c = result["candidate_wilson95"]
    assert lo_c == pytest.approx(0.3968, abs=1e-3)
    assert hi_c == pytest.approx(0.8922, abs=1e-3)
    # Model/quant-indexed, NEVER role-indexed.
    assert result["indexed_by"] == "model_quant"
    assert result["model_quant_key"] == "Qwen3.6-35B-A3B/Q8_0"
    assert result["observation_only"] is True


def test_compute_paired_result_end_to_end_from_graded_arms():
    """Grade two arms with the real patch verifier, then pair them."""
    grader = runner.get_grader("patch_verifier")
    items = [
        {"qid": "d1", "suite": "multifile_edit", "base_tree": BASE_TREE},
        {"qid": "d2", "suite": "multifile_edit", "base_tree": BASE_TREE},
    ]
    # baseline gets d1 right, d2 wrong; candidate gets both right -> candidate strictly better.
    _, base_vec = runner.build_arm_rows(
        runner.ArmConfig(name="baseline", is_baseline=True),
        items, {"d1": CLEAN_DIFF, "d2": NON_APPLYING_DIFF}, grader, model="m", quant="q",
    )
    _, cand_vec = runner.build_arm_rows(
        runner.ArmConfig(name="edit_transaction"),
        items, {"d1": CLEAN_DIFF, "d2": CLEAN_DIFF}, grader, model="m", quant="q",
    )
    result = runner.compute_paired_result(
        base_vec, cand_vec,
        baseline_label="baseline", candidate_label="edit_transaction",
        dataset_sha256="sha256:abc", test_profile="grader=patch_verifier;seed=42",
        model="m", quant="q",
    )
    assert result["shared_qids"] == 2
    assert result["baseline_correct"] == 1
    assert result["candidate_correct"] == 2
    assert result["delta_candidate_minus_baseline"] == pytest.approx(0.5)


def test_paired_result_gate_refuses_empty_identity():
    """The dataset_sha256/test_profile gate is wired into compute_paired_result."""
    baseline_vec, candidate_vec = _vectors_with_table(2, 2, 1, 1)
    with pytest.raises(PairedComparisonMismatchError):
        runner.compute_paired_result(
            baseline_vec, candidate_vec,
            baseline_label="a", candidate_label="b",
            dataset_sha256="", test_profile="",  # empty identity -> refusal
            model="m", quant="q",
        )


def test_build_test_profile_excludes_arm_flags():
    """Both arms must share one scoring profile (arm-specific flags excluded)."""
    profile = runner.build_test_profile(
        grader="exact", seed=42, sampling="production", dataset_sha256="sha256:x"
    )
    assert profile == "grader=exact;seed=42;sampling=production;dataset=sha256:x"
    assert "flag" not in profile.lower()


# --------------------------------------------------------------------------- #
# Plan (dry-run) end-to-end via the arg parser — still model-free
# --------------------------------------------------------------------------- #
def test_route_a2_dry_run_plan(tmp_path):
    """The authored ROUTE-A2 command resolves to a patch_verifier dry-run plan."""
    args = runner.build_arg_parser().parse_args(
        [
            "--arms", "edit_transaction,baseline",
            "--verifier", "src/verification/patch_verifier.py",
            "--n", "100", "--suite", "multifile_edit", "--seed", "42",
            "--output", str(tmp_path / "out"),
        ]
    )
    plan = runner.run_paired_ab(args)
    assert plan["mode"] == "dry_run"
    assert plan["inference_ran"] is False
    assert plan["grader"]["name"] == "patch_verifier"
    assert plan["eval_mode"] == "edit_transaction"
    assert plan["arms"]["candidate"]["name"] == "edit_transaction"
    assert plan["arms"]["baseline"]["name"] == "baseline"
    assert plan["transport"]["uses_chat_endpoint"] is False
    assert plan["corpus"]["resolved"] is False  # suite-only, no --manifest


def test_execute_without_env_flag_falls_back_to_dry_run(monkeypatch):
    monkeypatch.delenv(runner.PAIRED_AB_INFERENCE_ENV, raising=False)
    args = runner.build_arg_parser().parse_args(["--arms", "a,b", "--execute"])
    plan = runner.run_paired_ab(args)
    assert plan["mode"] == "dry_run"
    assert any("falling back to dry-run" in n for n in plan["notes"])
