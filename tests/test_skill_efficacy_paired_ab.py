#!/usr/bin/env python3
"""Unit tests for scripts/autopilot/skill_efficacy_paired_ab.py (EV-10a paired A/B).

Coverage is entirely INFERENCE-FREE: arms parsing, corpus resolution (from a
temp JSONL core + tolerance of a missing core), deterministic dev/test_normal
splitting, the placement-queue transport discipline (never /chat), and — the
heart of the runner — the paired-stats wiring (EV-10a efficacy verdict +
paired-McNemar + per-arm Wilson CIs + require_matched_comparison gate) exercised
on SYNTHETIC per-question outcomes. The execution bridge (env-gated,
model-touching) is never called: the flag-OFF path is asserted to return a plan
and NOT run inference.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.autopilot.skill_efficacy_paired_ab import (
    ARM_SKILL_OFF,
    ARM_SKILL_ON,
    PROTOCOL_ID,
    SKILL_EFFICACY_AB_INFERENCE_ENV,
    ArmOutcome,
    arm_outcome_from_question_results,
    build_test_profile,
    compute_paired_efficacy,
    compute_split_paired_efficacy,
    dataset_sha256,
    main,
    parse_arms,
    resolve_corpus,
    resolve_paired_plan,
    run_skill_efficacy_ab,
    split_questions,
)
from scripts.autopilot.paired_stats import PairedComparisonMismatchError


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _write_core(tmp_path: Path, n: int = 12) -> Path:
    """A minimal paired-core JSONL (metadata row + n question rows)."""
    path = tmp_path / "core_fixture.jsonl"
    lines = [json.dumps({"__core_metadata__": True, "core_id": "core_fixture"})]
    suites = ["math", "coder", "web"]
    for i in range(n):
        lines.append(
            json.dumps(
                {
                    "id": f"q{i:03d}",
                    "suite": suites[i % len(suites)],
                    "prompt": f"prompt {i}",
                    "expected": str(i),
                    "scoring_method": "exact_match",
                    "tier": 1,
                }
            )
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def _qr(qids_correct: dict[str, bool], suite_of: dict[str, str]) -> list[dict]:
    """Compact question_results rows: [{qid, suite, correct}]."""
    return [
        {"qid": qid, "suite": suite_of[qid], "correct": correct}
        for qid, correct in qids_correct.items()
    ]


def _arm(
    *,
    label: str,
    skill_enabled: bool,
    split: str,
    ds_sha: str,
    profile: str,
    qids_correct: dict[str, bool],
    suite_of: dict[str, str],
    per_suite_quality: dict[str, float] | None = None,
) -> ArmOutcome:
    return arm_outcome_from_question_results(
        label=label,
        split=split,
        skill_enabled=skill_enabled,
        model="TestModel-7B",
        quant="Q4_K_M",
        dataset_sha256=ds_sha,
        test_profile=profile,
        question_results=_qr(qids_correct, suite_of),
        per_suite_quality=per_suite_quality,
    )


# --------------------------------------------------------------------------- #
# Arms parsing
# --------------------------------------------------------------------------- #
def test_parse_arms_canonical():
    m = parse_arms("with_artifact,no_artifact")
    assert m == {"with_artifact": True, "no_artifact": False}


def test_parse_arms_aliases_and_order_preserved():
    m = parse_arms("baseline,with_skill")
    # exactly one ON + one OFF, insertion order preserved.
    assert list(m) == ["baseline", "with_skill"]
    assert m == {"baseline": False, "with_skill": True}


def test_parse_arms_rejects_two_on():
    with pytest.raises(ValueError):
        parse_arms("with_artifact,with_skill")


def test_parse_arms_rejects_wrong_count():
    with pytest.raises(ValueError):
        parse_arms("with_artifact")
    with pytest.raises(ValueError):
        parse_arms("a,b,c")


def test_parse_arms_rejects_unknown_label():
    with pytest.raises(ValueError):
        parse_arms("frobnicate,no_artifact")


# --------------------------------------------------------------------------- #
# Corpus resolution + splitting
# --------------------------------------------------------------------------- #
def test_resolve_corpus_reads_core_and_skips_metadata(tmp_path):
    path = _write_core(tmp_path, n=12)
    corpus = resolve_corpus(questions_path=str(path))
    assert corpus.exists is True
    assert corpus.n_rows == 12  # metadata row excluded
    assert corpus.error is None
    assert set(corpus.suites) == {"math", "coder", "web"}
    assert corpus.dataset_sha256.startswith("sha256:")
    # qids are the row ids
    assert {q["qid"] for q in corpus.questions} == {f"q{i:03d}" for i in range(12)}


def test_resolve_corpus_suite_filter(tmp_path):
    path = _write_core(tmp_path, n=12)
    corpus = resolve_corpus(questions_path=str(path), suites={"math"})
    assert corpus.suites == ["math"]
    assert all(q["suite"] == "math" for q in corpus.questions)
    assert corpus.n_rows == 4  # 12 rows, every 3rd is math


def test_resolve_corpus_missing_is_tolerant():
    corpus = resolve_corpus(questions_path="/nonexistent/does-not-exist.jsonl")
    assert corpus.exists is False
    assert corpus.n_rows == 0
    assert corpus.error is not None  # recorded, not raised


def test_dataset_sha256_order_independent_and_unique():
    assert dataset_sha256(["b", "a", "c"]) == dataset_sha256(["c", "b", "a", "a"])
    assert dataset_sha256(["a", "b"]) != dataset_sha256(["a", "c"])


def test_split_questions_deterministic_balanced_and_disjoint():
    qs = [{"qid": f"q{i}", "suite": "s"} for i in range(11)]
    a = split_questions(qs, ["dev", "test_normal"], seed=42)
    b = split_questions(qs, ["dev", "test_normal"], seed=42)
    # reproducible
    assert a == b
    dev = {q["qid"] for q in a["dev"]}
    test = {q["qid"] for q in a["test_normal"]}
    # disjoint + total coverage
    assert dev.isdisjoint(test)
    assert dev | test == {f"q{i}" for i in range(11)}
    # balanced to within one
    assert abs(len(dev) - len(test)) <= 1
    # different seed => (very likely) different partition
    c = split_questions(qs, ["dev", "test_normal"], seed=7)
    assert {q["qid"] for q in c["dev"]} != dev or {q["qid"] for q in c["test_normal"]} != test


# --------------------------------------------------------------------------- #
# Plan resolution + transport discipline
# --------------------------------------------------------------------------- #
def _plan_from_core(tmp_path, **over):
    path = _write_core(tmp_path, n=12)
    corpus = resolve_corpus(questions_path=str(path))
    kwargs = dict(
        corpus=corpus,
        arm_map={"with_artifact": True, "no_artifact": False},
        splits=["dev", "test_normal"],
        skill="skill-under-test",
        eval_role="frontdoor",
        model="TestModel-7B",
        quant="Q4_K_M",
        n_per_arm=8,
        seed=42,
        regress_threshold=0.10,
        require_aggregate_gain=True,
        per_suite_negative_delta_guard=True,
    )
    kwargs.update(over)
    return resolve_paired_plan(**kwargs)


def test_resolve_paired_plan_materializes_arm_x_split_jobs(tmp_path):
    plan = _plan_from_core(tmp_path)
    d = plan.to_dict()
    # 2 splits x 2 arms = 4 jobs
    assert d["n_jobs"] == 4
    assert d["split_sizes"] == {"dev": 6, "test_normal": 6}
    assert d["model"] == "TestModel-7B" and d["quant"] == "Q4_K_M"
    assert d["protocol_id"] == PROTOCOL_ID
    # both arms present per split, paired on the SAME dataset hash
    dev_jobs = [j for j in d["jobs"] if j["split"] == "dev"]
    assert {j["arm"] for j in dev_jobs} == {"with_artifact", "no_artifact"}
    assert len({j["dataset_sha256"] for j in dev_jobs}) == 1  # identical questions
    # n is capped to available questions in the split
    assert all(j["n"] == 6 for j in dev_jobs)


def test_plan_transport_is_placement_queue_never_chat(tmp_path):
    plan = _plan_from_core(tmp_path)
    d = plan.to_dict()
    blob = json.dumps({"jobs": d["jobs"], "transport": d["transport"]})
    assert "/chat" not in blob and "/v1/chat" not in blob
    assert d["transport"]["uses_chat_endpoint"] is False
    for job in d["jobs"]:
        assert job["transport"] == "placement_queue"
        assert job["request_priority"] == "background"
        assert job["workload_class"] == "eval_batch"


def test_plan_records_gate_and_inference_env(tmp_path):
    d = _plan_from_core(tmp_path).to_dict()
    assert d["gate_env"] == "AUTOPILOT_SKILL_EFFICACY_GATE"
    assert d["inference_env"] == SKILL_EFFICACY_AB_INFERENCE_ENV
    assert d["inference_ran"] is False


# --------------------------------------------------------------------------- #
# Paired-stats wiring on SYNTHETIC outcomes (the core of the runner)
# --------------------------------------------------------------------------- #
def _matched_arms(qids, suite_of, off_correct, on_correct, split="dev"):
    ds = dataset_sha256(qids)
    profile = build_test_profile(split=split, seed=42, eval_role="frontdoor", n=len(qids))
    off = _arm(
        label=ARM_SKILL_OFF, skill_enabled=False, split=split, ds_sha=ds,
        profile=profile, qids_correct=off_correct, suite_of=suite_of,
        per_suite_quality={"math": 0.50, "coder": 0.60},
    )
    on = _arm(
        label=ARM_SKILL_ON, skill_enabled=True, split=split, ds_sha=ds,
        profile=profile, qids_correct=on_correct, suite_of=suite_of,
        per_suite_quality={"math": 0.60, "coder": 0.66},  # both improve
    )
    return off, on


def test_compute_paired_efficacy_mcnemar_and_wilson():
    qids = ["a", "b", "c", "d"]
    suite_of = {"a": "math", "b": "math", "c": "coder", "d": "coder"}
    # OFF: a,b correct; c,d wrong.  ON: a wrong, b,c,d correct.
    off_c = {"a": True, "b": True, "c": False, "d": False}
    on_c = {"a": False, "b": True, "c": True, "d": True}
    off, on = _matched_arms(qids, suite_of, off_c, on_c)
    res = compute_paired_efficacy(off, on, skill="s")

    mc = res["mcnemar"]
    assert mc["shared_qids"] == 4
    # discordant: c,d flipped OFF-wrong->ON-correct = 2; a flipped OFF-correct->ON-wrong = 1
    assert mc["on_correct_off_wrong"] == 2
    assert mc["off_correct_on_wrong"] == 1
    assert mc["accuracy_off"] == pytest.approx(0.5)
    assert mc["accuracy_on"] == pytest.approx(0.75)
    assert mc["delta_on_minus_off"] == pytest.approx(0.25)

    # per-arm Wilson CIs bracket the point estimate
    off_lo, off_hi = res["wilson_ci"][ARM_SKILL_OFF]
    on_lo, on_hi = res["wilson_ci"][ARM_SKILL_ON]
    assert off_lo < 0.5 < off_hi
    assert on_lo < 0.75 < on_hi

    # efficacy verdict wired from per_suite_quality (both suites improved)
    assert res["efficacy"]["accept"] is True
    assert res["efficacy"]["aggregate_delta"] > 0
    assert res["skill"] == "s"
    assert res["model"] == "TestModel-7B" and res["quant"] == "Q4_K_M"
    assert res["observation_only"] is True


def test_compute_paired_efficacy_negative_delta_guard_rejects():
    # A per-suite regression must reject even with a positive aggregate (SkillsBench).
    qids = ["a", "b"]
    suite_of = {"a": "math", "b": "web"}
    off = _arm(
        label=ARM_SKILL_OFF, skill_enabled=False, split="dev",
        ds_sha=dataset_sha256(qids),
        profile=build_test_profile(split="dev", seed=42, eval_role="frontdoor", n=2),
        qids_correct={"a": False, "b": True}, suite_of=suite_of,
        per_suite_quality={"math": 0.40, "web": 0.60},
    )
    on = _arm(
        label=ARM_SKILL_ON, skill_enabled=True, split="dev",
        ds_sha=dataset_sha256(qids),
        profile=build_test_profile(split="dev", seed=42, eval_role="frontdoor", n=2),
        qids_correct={"a": True, "b": False}, suite_of=suite_of,
        per_suite_quality={"math": 0.95, "web": 0.20},  # web craters -0.40
    )
    res = compute_paired_efficacy(off, on, skill="s")
    assert res["efficacy"]["accept"] is False
    assert res["efficacy"]["regressed_suites"]  # web flagged
    assert res["efficacy"]["regressed_suites"][0][0] == "web"


def test_compute_paired_efficacy_derives_per_suite_quality_when_absent():
    # No explicit per_suite_quality -> derived per-suite pass-rate feeds the verdict.
    qids = ["a", "b", "c", "d"]
    suite_of = {"a": "math", "b": "math", "c": "coder", "d": "coder"}
    ds = dataset_sha256(qids)
    profile = build_test_profile(split="dev", seed=42, eval_role="frontdoor", n=4)
    off = _arm(label=ARM_SKILL_OFF, skill_enabled=False, split="dev", ds_sha=ds,
               profile=profile, qids_correct={"a": False, "b": False, "c": False, "d": True},
               suite_of=suite_of)  # math 0.0, coder 0.5
    on = _arm(label=ARM_SKILL_ON, skill_enabled=True, split="dev", ds_sha=ds,
              profile=profile, qids_correct={"a": True, "b": True, "c": True, "d": True},
              suite_of=suite_of)  # math 1.0, coder 1.0
    res = compute_paired_efficacy(off, on, skill="s")
    assert res["efficacy"]["per_suite_delta"]["math"] == pytest.approx(1.0)
    assert res["efficacy"]["per_suite_delta"]["coder"] == pytest.approx(0.5)
    assert res["efficacy"]["accept"] is True


def test_compute_paired_efficacy_require_matched_rejects_dataset_mismatch():
    # Two arms scored on DIFFERENT question sets must not be paired.
    suite_of = {"a": "math", "b": "math"}
    profile = build_test_profile(split="dev", seed=42, eval_role="frontdoor", n=2)
    off = _arm(label=ARM_SKILL_OFF, skill_enabled=False, split="dev",
               ds_sha=dataset_sha256(["a", "b"]), profile=profile,
               qids_correct={"a": True, "b": False}, suite_of=suite_of,
               per_suite_quality={"math": 0.5})
    on = _arm(label=ARM_SKILL_ON, skill_enabled=True, split="dev",
              ds_sha=dataset_sha256(["a", "x"]), profile=profile,  # different dataset hash
              qids_correct={"a": True, "b": True}, suite_of=suite_of,
              per_suite_quality={"math": 0.6})
    with pytest.raises(PairedComparisonMismatchError):
        compute_paired_efficacy(off, on)


def test_compute_paired_efficacy_require_matched_rejects_profile_mismatch():
    suite_of = {"a": "math", "b": "math"}
    ds = dataset_sha256(["a", "b"])
    off = _arm(label=ARM_SKILL_OFF, skill_enabled=False, split="dev", ds_sha=ds,
               profile=build_test_profile(split="dev", seed=42, eval_role="frontdoor", n=2),
               qids_correct={"a": True, "b": False}, suite_of=suite_of,
               per_suite_quality={"math": 0.5})
    on = _arm(label=ARM_SKILL_ON, skill_enabled=True, split="dev", ds_sha=ds,
              profile=build_test_profile(split="dev", seed=99, eval_role="frontdoor", n=2),  # seed drift
              qids_correct={"a": True, "b": True}, suite_of=suite_of,
              per_suite_quality={"math": 0.6})
    with pytest.raises(PairedComparisonMismatchError):
        compute_paired_efficacy(off, on)


def test_compute_paired_efficacy_rejects_cross_split_pairing():
    suite_of = {"a": "math"}
    ds = dataset_sha256(["a"])
    off = _arm(label=ARM_SKILL_OFF, skill_enabled=False, split="dev", ds_sha=ds,
               profile="p", qids_correct={"a": True}, suite_of=suite_of,
               per_suite_quality={"math": 0.5})
    on = _arm(label=ARM_SKILL_ON, skill_enabled=True, split="test_normal", ds_sha=ds,
              profile="p", qids_correct={"a": True}, suite_of=suite_of,
              per_suite_quality={"math": 0.6})
    with pytest.raises(ValueError):
        compute_paired_efficacy(off, on, require_matched=False)


# --------------------------------------------------------------------------- #
# dev/test_normal split discipline (evaluate_skill_efficacy_split wiring)
# --------------------------------------------------------------------------- #
def test_compute_split_paired_efficacy_requires_both_splits_accept():
    suite_of = {"a": "math", "b": "math"}

    def arms(split, on_quality):
        ds = dataset_sha256(["a", "b"])
        profile = build_test_profile(split=split, seed=42, eval_role="frontdoor", n=2)
        off = _arm(label=ARM_SKILL_OFF, skill_enabled=False, split=split, ds_sha=ds,
                   profile=profile, qids_correct={"a": True, "b": False}, suite_of=suite_of,
                   per_suite_quality={"math": 0.50})
        on = _arm(label=ARM_SKILL_ON, skill_enabled=True, split=split, ds_sha=ds,
                  profile=profile, qids_correct={"a": True, "b": True}, suite_of=suite_of,
                  per_suite_quality={"math": on_quality})
        return off, on

    # dev improves, test_normal REGRESSES -> overall reject (overfit-to-dev guard).
    arms_by_split = {"dev": arms("dev", 0.70), "test_normal": arms("test_normal", 0.30)}
    res = compute_split_paired_efficacy(arms_by_split, skill="s")
    assert res["split_verdict"] is not None
    assert res["split_verdict"]["accept"] is False
    assert "test:" in res["split_verdict"]["reason"]
    # per-split results still computed for both
    assert set(res["per_split"]) == {"dev", "test_normal"}
    assert res["model"] == "TestModel-7B" and res["quant"] == "Q4_K_M"

    # both improve -> accept
    arms_ok = {"dev": arms("dev", 0.60), "test_normal": arms("test_normal", 0.58)}
    res_ok = compute_split_paired_efficacy(arms_ok, skill="s")
    assert res_ok["split_verdict"]["accept"] is True


def test_compute_split_paired_efficacy_single_split_has_no_split_verdict():
    suite_of = {"a": "math"}
    ds = dataset_sha256(["a"])
    profile = build_test_profile(split="dev", seed=42, eval_role="frontdoor", n=1)
    off = _arm(label=ARM_SKILL_OFF, skill_enabled=False, split="dev", ds_sha=ds,
               profile=profile, qids_correct={"a": True}, suite_of=suite_of,
               per_suite_quality={"math": 0.5})
    on = _arm(label=ARM_SKILL_ON, skill_enabled=True, split="dev", ds_sha=ds,
              profile=profile, qids_correct={"a": True}, suite_of=suite_of,
              per_suite_quality={"math": 0.6})
    res = compute_split_paired_efficacy({"dev": (off, on)}, skill="s")
    assert res["split_verdict"] is None
    assert list(res["per_split"]) == ["dev"]


# --------------------------------------------------------------------------- #
# Env-flag gate: OFF => dry-run plan, NO inference
# --------------------------------------------------------------------------- #
def test_run_flag_off_returns_plan_without_inference(tmp_path, monkeypatch):
    monkeypatch.delenv(SKILL_EFFICACY_AB_INFERENCE_ENV, raising=False)
    plan = _plan_from_core(tmp_path)

    def _boom(*a, **k):
        raise AssertionError("execute_paired_ab called with inference flag OFF")

    monkeypatch.setattr(
        "scripts.autopilot.skill_efficacy_paired_ab.execute_paired_ab", _boom
    )
    out = run_skill_efficacy_ab(plan, attempt_run=True)
    assert out["mode"] == "dry_run"
    assert out["inference_ran"] is False
    assert out["kind"] == "skill_efficacy_ab_plan"
    assert SKILL_EFFICACY_AB_INFERENCE_ENV in out["reason"]


def test_run_default_attempt_false_is_plan(tmp_path, monkeypatch):
    monkeypatch.setenv(SKILL_EFFICACY_AB_INFERENCE_ENV, "1")  # even with flag ON...
    plan = _plan_from_core(tmp_path)

    def _boom(*a, **k):
        raise AssertionError("execute_paired_ab called without attempt_run")

    monkeypatch.setattr(
        "scripts.autopilot.skill_efficacy_paired_ab.execute_paired_ab", _boom
    )
    out = run_skill_efficacy_ab(plan, attempt_run=False)  # ...attempt_run gates too
    assert out["mode"] == "dry_run"
    assert out["inference_ran"] is False


def test_run_flag_on_routes_to_execute_bridge(tmp_path, monkeypatch):
    monkeypatch.setenv(SKILL_EFFICACY_AB_INFERENCE_ENV, "1")
    plan = _plan_from_core(tmp_path)
    captured = {}

    def _fake_execute(p, **kwargs):
        captured["plan"] = p
        captured["kwargs"] = kwargs
        return {"kind": "skill_efficacy_ab_paired_result", "split_verdict": {"accept": True}}

    monkeypatch.setattr(
        "scripts.autopilot.skill_efficacy_paired_ab.execute_paired_ab", _fake_execute
    )
    out = run_skill_efficacy_ab(plan, attempt_run=True)
    assert out["mode"] == "execute"
    assert out["inference_ran"] is True
    assert out["result"]["split_verdict"]["accept"] is True
    assert captured["plan"] is plan


# --------------------------------------------------------------------------- #
# CLI __main__ (dry-run, pure)
# --------------------------------------------------------------------------- #
def test_main_dry_run_prints_plan(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv(SKILL_EFFICACY_AB_INFERENCE_ENV, raising=False)
    core = _write_core(tmp_path, n=12)
    code = main([
        "--questions", str(core),
        "--arms", "with_artifact,no_artifact",
        "--splits", "dev,test_normal",
        "--per-suite-negative-delta-guard",
        "--seed", "42",
        "--model", "TestModel-7B",
        "--quant", "Q4_K_M",
        "--n", "8",
    ])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "skill_efficacy_ab_plan"
    assert payload["mode"] == "dry_run"
    assert payload["inference_ran"] is False
    assert payload["transport"]["uses_chat_endpoint"] is False
    assert payload["n_jobs"] == 4
    assert payload["model"] == "TestModel-7B" and payload["quant"] == "Q4_K_M"


def test_main_bad_arms_exits_2(capsys):
    code = main(["--arms", "with_artifact,with_skill"])
    assert code == 2
    assert "error" in json.loads(capsys.readouterr().out)


def test_main_run_without_env_flag_falls_back_to_dry_run(tmp_path, capsys, monkeypatch):
    # --run given but the inference flag is unset -> dry-run fallback, no inference.
    monkeypatch.delenv(SKILL_EFFICACY_AB_INFERENCE_ENV, raising=False)
    core = _write_core(tmp_path, n=9)
    code = main([
        "--questions", str(core),
        "--skill", "some-skill",
        "--model", "M", "--quant", "Q",
        "--run",
    ])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["inference_ran"] is False
    assert payload["mode"] == "dry_run"
