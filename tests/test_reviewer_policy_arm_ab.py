#!/usr/bin/env python3
"""Unit tests for scripts/analysis/reviewer_policy_arm_ab.py (LB-4 sampling-policy A/B).

Coverage is entirely INFERENCE-FREE and asserts CONCRETE expected values on the
non-inference logic:

  * sampling-policy spec parsing + knob coercion/validation (temperature/top_p/
    top_k/max_tokens/verbosity ranges; bad values raise), and arm resolution
    (candidate/baseline, explicit baseline, distinct-name/count guards);
  * reviewer decision extraction + canonicalization;
  * the pure per-policy comparison math on SYNTHETIC outcomes — throughput
    (tokens/sec, mean latency, ratio), decision agreement (raw rate + Cohen's
    kappa), decision distributions, and the gold-scored McNemar/Wilson path
    (reusing the paired_stats seam) — all with exact numbers;
  * the model/quant (never role) indexing of emitted rows/report;
  * the dataset/profile gate; and the env-gated dry-run plan.

The env-gated placement-queue inference bridge is NEVER exercised.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# ── load the runner module by path (robust; no scripts.* package needed) ──────
_MODULE_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts" / "analysis" / "reviewer_policy_arm_ab.py"
)
_SPEC = importlib.util.spec_from_file_location("reviewer_policy_arm_ab", _MODULE_PATH)
runner = importlib.util.module_from_spec(_SPEC)
sys.modules["reviewer_policy_arm_ab"] = runner  # register before exec (dataclasses)
_SPEC.loader.exec_module(runner)

PolicyOutcome = runner.PolicyOutcome
ps = runner._RPAB._load_paired_stats()
PairedComparisonMismatchError = ps.PairedComparisonMismatchError


# --------------------------------------------------------------------------- #
# Sampling-policy spec parsing + knob coercion/validation
# --------------------------------------------------------------------------- #
def test_parse_policy_spec_coerces_and_types_knobs():
    name, knobs = runner.parse_policy_spec(
        "warm=temperature=0.7,top_p=0.95,top_k=40,max_tokens=512,verbosity=verbose"
    )
    assert name == "warm"
    assert knobs["temperature"] == pytest.approx(0.7)  # float
    assert knobs["top_p"] == pytest.approx(0.95)  # float
    assert knobs["top_k"] == 40 and isinstance(knobs["top_k"], int)
    assert knobs["max_tokens"] == 512 and isinstance(knobs["max_tokens"], int)
    assert knobs["verbosity"] == "verbose"  # str preset


def test_parse_policy_spec_empty_body_is_zero_knob_policy():
    assert runner.parse_policy_spec("cold=") == ("cold", {})


def test_parse_policy_spec_rejects_bad_forms_and_ranges():
    with pytest.raises(ValueError):  # no '='
        runner.parse_policy_spec("noeq")
    with pytest.raises(ValueError):  # empty name
        runner.parse_policy_spec("=temperature=0.1")
    with pytest.raises(ValueError):  # knob missing '='
        runner.parse_policy_spec("x=temperature")
    with pytest.raises(ValueError):  # temperature out of [0,2]
        runner.parse_policy_spec("x=temperature=3.0")
    with pytest.raises(ValueError):  # top_p out of (0,1]
        runner.parse_policy_spec("x=top_p=1.5")
    with pytest.raises(ValueError):  # max_tokens must be > 0
        runner.parse_policy_spec("x=max_tokens=0")
    with pytest.raises(ValueError):  # verbosity not a preset
        runner.parse_policy_spec("x=verbosity=loud")
    with pytest.raises(ValueError):  # unknown knob
        runner.parse_policy_spec("x=frequency_penalty=0.1")


def test_parse_policy_arms_defaults_baseline_second_and_pins_shared_role():
    candidate, baseline = runner.parse_policy_arms(
        "warm,cold",
        policy_specs=[
            "warm=temperature=0.7,top_p=0.95",
            "cold=temperature=0.0,top_p=1.0",
        ],
        reviewer_role="architect",
    )
    assert candidate.name == "warm" and candidate.is_baseline is False
    assert baseline.name == "cold" and baseline.is_baseline is True
    # Both arms share ONE reviewer role (they differ only in sampling).
    assert candidate.role == "architect" and baseline.role == "architect"
    # decode_kwargs carries only the decode knobs.
    assert candidate.decode_kwargs() == {"temperature": 0.7, "top_p": 0.95}
    assert baseline.decode_kwargs() == {"temperature": 0.0, "top_p": 1.0}
    # Transport is always the placement queue, never /chat.
    t = candidate.transport()
    assert t["transport"] == "placement_queue"
    assert t["request_priority"] == "background"
    assert t["workload_class"] == "eval_batch"
    assert t["uses_chat_endpoint"] is False
    assert t["force_role"] == "architect"


def test_parse_policy_arms_explicit_baseline_first():
    candidate, baseline = runner.parse_policy_arms(
        "verbose,terse", baseline_arm="verbose"
    )
    assert baseline.name == "verbose"
    assert candidate.name == "terse"


def test_parse_policy_arms_rejects_count_dupes_and_unknown_policy():
    with pytest.raises(ValueError):
        runner.parse_policy_arms("only_one")
    with pytest.raises(ValueError):
        runner.parse_policy_arms("a,b,c")
    with pytest.raises(ValueError):
        runner.parse_policy_arms("same,same")
    with pytest.raises(ValueError):  # --policy names an arm not in --arms
        runner.parse_policy_arms("a,b", policy_specs=["c=temperature=0.1"])


def test_zero_knob_policy_is_valid():
    candidate, baseline = runner.parse_policy_arms("warm,cold", policy_specs=["warm=temperature=0.5"])
    assert candidate.knobs == {"temperature": 0.5}
    assert baseline.knobs == {}  # registry-default policy
    assert baseline.decode_kwargs() == {}


# --------------------------------------------------------------------------- #
# Reviewer decision extraction + canonicalization
# --------------------------------------------------------------------------- #
def test_normalize_decision_maps_aliases_and_defaults():
    assert runner.normalize_decision("approve") == "approve"
    assert runner.normalize_decision("changes") == "request_changes"
    assert runner.normalize_decision("LGTM") == "approve"
    assert runner.normalize_decision("escalate") == "escalate"
    # Unknown / empty -> conservative default (mirrors review_service failure path).
    assert runner.normalize_decision("banana") == "request_changes"
    assert runner.normalize_decision("") == "request_changes"


def test_extract_decision_from_json_and_bare_token():
    assert runner.extract_decision('{"d": "approve", "s": 0.9}') == "approve"
    assert runner.extract_decision('{"decision": "changes"}') == "request_changes"
    assert runner.extract_decision("prefix {\"d\":\"escalate\"} suffix") == "escalate"
    assert runner.extract_decision("approve — looks good") == "approve"
    assert runner.extract_decision("") == "request_changes"  # empty -> default
    assert runner.extract_decision("total gibberish here") == "request_changes"


# --------------------------------------------------------------------------- #
# Pure comparison math (throughput + agreement + kappa)
# --------------------------------------------------------------------------- #
def test_agreement_rate_and_cohen_kappa_exact():
    a = ["approve", "request_changes", "approve", "request_changes", "escalate"]
    b = ["approve", "request_changes", "request_changes", "request_changes", "escalate"]
    assert runner.agreement_rate(a, b) == (4, 0.8)
    # p_o=0.8; p_e=0.4*0.2+0.4*0.6+0.2*0.2=0.36; kappa=(0.8-0.36)/(1-0.36)=0.6875.
    assert runner.cohen_kappa(a, b) == pytest.approx(0.6875)


def test_cohen_kappa_edge_cases():
    # Perfect agreement, both constant -> kappa 1.0 (p_e==1 short-circuit).
    assert runner.cohen_kappa(["approve", "approve"], ["approve", "approve"]) == 1.0
    # Both constant but DIFFERENT constant -> zero observed agreement, p_e==1 -> 0.0.
    assert runner.cohen_kappa(["approve", "approve"], ["reject", "reject"]) == 0.0
    # Perfect non-constant agreement -> kappa 1.0.
    assert runner.cohen_kappa(["approve", "reject"], ["approve", "reject"]) == pytest.approx(1.0)
    # Length mismatch is a hard error.
    with pytest.raises(ValueError):
        runner.cohen_kappa(["approve"], ["approve", "reject"])


def test_policy_throughput_exact():
    outs = [
        PolicyOutcome(qid=f"q{i}", suite="s", decision="approve",
                      tokens_out=100, latency_ms=200.0)
        for i in range(5)
    ]
    tp = runner.policy_throughput(outs)
    assert tp["n"] == 5
    assert tp["total_tokens_out"] == 500
    assert tp["total_latency_ms"] == pytest.approx(1000.0)
    # 500 tokens / 1.0 s = 500 tok/s; mean latency 200 ms.
    assert tp["tokens_per_second"] == pytest.approx(500.0)
    assert tp["mean_latency_ms"] == pytest.approx(200.0)


def test_policy_throughput_zero_latency_guard():
    outs = [PolicyOutcome(qid="q0", suite="s", decision="approve", tokens_out=10, latency_ms=0.0)]
    tp = runner.policy_throughput(outs)
    assert tp["tokens_per_second"] == 0.0  # no divide-by-zero


def test_decision_distribution_counts():
    outs = [
        PolicyOutcome(qid="q0", suite="s", decision="approve", tokens_out=1, latency_ms=1.0),
        PolicyOutcome(qid="q1", suite="s", decision="approve", tokens_out=1, latency_ms=1.0),
        PolicyOutcome(qid="q2", suite="s", decision="request_changes", tokens_out=1, latency_ms=1.0),
    ]
    assert runner.decision_distribution(outs) == {"approve": 2, "request_changes": 1}


def test_build_scoring_profile_excludes_per_policy_sampling():
    """The two arms DIFFER in sampling, so the shared profile must NOT encode it."""
    profile = runner.build_scoring_profile(
        decision_scheme="canonical_v1", seed=42, gold_key="gold_decision",
        dataset_sha256="sha256:x",
    )
    assert profile == "decision_scheme=canonical_v1;seed=42;gold_key=gold_decision;dataset=sha256:x"
    assert "temperature" not in profile
    assert "top_p" not in profile
    assert "verbosity" not in profile


# --------------------------------------------------------------------------- #
# Full paired per-policy comparison on synthetic outcomes (exact values)
# --------------------------------------------------------------------------- #
def _synthetic_pair():
    """Baseline (cold) vs candidate (warm) over 5 shared qids, with gold verdicts."""
    base_dec = ["approve", "request_changes", "approve", "request_changes", "escalate"]
    cand_dec = ["approve", "request_changes", "request_changes", "request_changes", "escalate"]
    gold = ["approve", "request_changes", "approve", "escalate", "escalate"]
    base = [
        PolicyOutcome(qid=f"q{i}", suite="s", decision=base_dec[i],
                      tokens_out=100, latency_ms=200.0, correct=(base_dec[i] == gold[i]))
        for i in range(5)
    ]
    cand = [
        PolicyOutcome(qid=f"q{i}", suite="s", decision=cand_dec[i],
                      tokens_out=40, latency_ms=50.0, correct=(cand_dec[i] == gold[i]))
        for i in range(5)
    ]
    return base, cand


def test_compute_policy_comparison_exact_values():
    base, cand = _synthetic_pair()
    profile = runner.build_scoring_profile(
        decision_scheme="canonical_v1", seed=42, gold_key="gold_decision",
        dataset_sha256="sha256:abc",
    )
    r = runner.compute_policy_comparison(
        base, cand,
        baseline_label="cold", candidate_label="warm",
        dataset_sha256="sha256:abc", test_profile=profile,
        model="gemma-4-26B-A4B-it", quant="Q4_K_M",
    )
    assert r["shared_qids"] == 5

    # Decision agreement: q2 differs (approve vs request_changes) -> 4/5 agree.
    da = r["decision_agreement"]
    assert da["agree"] == 4
    assert da["rate"] == pytest.approx(0.8)
    assert da["cohen_kappa"] == pytest.approx(0.6875)
    assert da["baseline_distribution"] == {"approve": 2, "request_changes": 2, "escalate": 1}
    assert da["candidate_distribution"] == {"approve": 1, "request_changes": 3, "escalate": 1}

    # Throughput: cold 500 tok / 1.0 s = 500 tok/s; warm 200 tok / 0.25 s = 800 tok/s.
    tp = r["throughput"]
    assert tp["baseline"]["tokens_per_second"] == pytest.approx(500.0)
    assert tp["candidate"]["tokens_per_second"] == pytest.approx(800.0)
    assert tp["candidate"]["mean_latency_ms"] == pytest.approx(50.0)
    assert tp["candidate_over_baseline_tps"] == pytest.approx(1.6)

    # Accuracy vs gold: cold 4/5 correct, warm 3/5 (McNemar b=1,c=0 -> p=1.0).
    assert r["has_gold"] is True
    acc = r["accuracy_vs_gold"]
    assert acc["baseline_correct"] == 4
    assert acc["candidate_correct"] == 3
    assert acc["baseline_accuracy"] == pytest.approx(0.8)
    assert acc["candidate_accuracy"] == pytest.approx(0.6)
    assert acc["delta_candidate_minus_baseline"] == pytest.approx(-0.2)
    assert acc["p_value_two_sided"] == pytest.approx(1.0)
    assert acc["a_correct_b_wrong"] == 1
    assert acc["a_wrong_b_correct"] == 0
    # Wilson 95% score intervals for 4/5 and 3/5.
    lo_b, hi_b = acc["baseline_wilson95"]
    assert lo_b == pytest.approx(0.3755, abs=1e-3)
    assert hi_b == pytest.approx(0.9638, abs=1e-3)
    lo_c, hi_c = acc["candidate_wilson95"]
    assert lo_c == pytest.approx(0.2307, abs=1e-3)
    assert hi_c == pytest.approx(0.8824, abs=1e-3)

    # Model/quant-indexed, NEVER role-indexed.
    assert r["indexed_by"] == "model_quant"
    assert r["model_quant_key"] == "gemma-4-26B-A4B-it/Q4_K_M"
    assert "policy" not in r.get("indexed_by", "")
    assert r["observation_only"] is True


def test_compute_policy_comparison_without_gold_skips_accuracy():
    """No gold verdict -> throughput + agreement still emitted, accuracy skipped."""
    base = [PolicyOutcome(qid="q0", suite="s", decision="approve", tokens_out=10, latency_ms=100.0)]
    cand = [PolicyOutcome(qid="q0", suite="s", decision="reject", tokens_out=20, latency_ms=100.0)]
    profile = runner.build_scoring_profile(
        decision_scheme="canonical_v1", seed=1, gold_key="gold_decision",
        dataset_sha256="sha256:z",
    )
    r = runner.compute_policy_comparison(
        base, cand, baseline_label="cold", candidate_label="warm",
        dataset_sha256="sha256:z", test_profile=profile, model="m", quant="q",
    )
    assert r["has_gold"] is False
    assert "accuracy_vs_gold" not in r
    assert r["decision_agreement"]["agree"] == 0  # approve != reject
    assert r["throughput"]["candidate_over_baseline_tps"] == pytest.approx(2.0)


def test_comparison_gate_refuses_empty_identity():
    """The dataset_sha256/test_profile gate is wired into compute_policy_comparison."""
    base, cand = _synthetic_pair()
    with pytest.raises(PairedComparisonMismatchError):
        runner.compute_policy_comparison(
            base, cand, baseline_label="cold", candidate_label="warm",
            dataset_sha256="", test_profile="",  # empty identity -> refusal
            model="m", quant="q",
        )


# --------------------------------------------------------------------------- #
# Per-policy rows are model/quant-stamped, never role-keyed
# --------------------------------------------------------------------------- #
def test_build_policy_rows_model_quant_stamped_not_role_keyed():
    policy = runner.SamplingPolicy(name="warm", role="architect", is_baseline=False)
    outs = [
        PolicyOutcome(qid="q0", suite="s", decision="approve", tokens_out=12,
                      latency_ms=34.0, correct=True),
    ]
    rows = runner.build_policy_rows(policy, outs, model="M", quant="Q4")
    assert len(rows) == 1
    row = rows[0]
    assert row["model"] == "M" and row["quant"] == "Q4"
    assert row["policy"] == "warm"  # policy is metadata...
    assert "role" not in row  # ...role is NOT an index key
    assert row["decision"] == "approve"
    assert row["transport"] == "placement_queue"
    assert row["observation_only"] is True


# --------------------------------------------------------------------------- #
# Dry-run plan via the arg parser — still model-free
# --------------------------------------------------------------------------- #
def test_dry_run_plan_is_model_free_and_placement_queued(tmp_path):
    args = runner.build_arg_parser().parse_args(
        [
            "--arms", "warm,cold",
            "--policy", "warm=temperature=0.7,top_p=0.95,verbosity=verbose,max_tokens=512",
            "--policy", "cold=temperature=0.0,top_p=1.0,verbosity=terse,max_tokens=256",
            "--suite", "reviewer_corpus",
            "--seed", "42",
            "--model", "gemma-4-26B-A4B-it", "--quant", "Q4_K_M",
            "--output", str(tmp_path / "out"),
        ]
    )
    plan = runner.run_policy_ab(args)
    assert plan["mode"] == "dry_run"
    assert plan["inference_ran"] is False
    assert plan["indexed_by"] == "model_quant"
    assert plan["reviewer_role"] == "architect"
    assert plan["policies"]["candidate"]["name"] == "warm"
    assert plan["policies"]["baseline"]["name"] == "cold"
    assert plan["policies"]["candidate"]["decode_kwargs"] == {"temperature": 0.7, "top_p": 0.95}
    assert plan["transport"]["uses_chat_endpoint"] is False
    assert plan["transport"]["workload_class"] == "eval_batch"
    assert plan["corpus"]["resolved"] is False  # suite-only, no --manifest
    # Shared scoring profile must not leak per-policy sampling.
    assert "temperature" not in plan["test_profile"]


def test_dry_run_plan_resolves_manifest_corpus(tmp_path):
    rows = [
        {"qid": "r1", "prompt": "review A", "suite": "reviewer_corpus", "gold_decision": "approve"},
        {"qid": "r2", "prompt": "review B", "suite": "reviewer_corpus", "gold_decision": "reject"},
        {"qid": "r3", "prompt": "review C", "suite": "other", "gold_decision": "approve"},
    ]
    manifest = tmp_path / "corpus.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    args = runner.build_arg_parser().parse_args(
        [
            "--arms", "warm,cold",
            "--manifest", str(manifest),
            "--suite", "reviewer_corpus",
            "--n", "5", "--seed", "42",
            "--output", str(tmp_path / "out"),
        ]
    )
    plan = runner.run_policy_ab(args)
    assert plan["mode"] == "dry_run"
    assert plan["corpus"]["resolved"] is True
    assert plan["corpus"]["n_selected"] == 2  # two reviewer_corpus rows
    assert plan["corpus"]["dataset_sha256"].startswith("sha256:")


def test_execute_without_env_flag_falls_back_to_dry_run(monkeypatch):
    monkeypatch.delenv(runner.REVIEWER_POLICY_AB_INFERENCE_ENV, raising=False)
    args = runner.build_arg_parser().parse_args(["--arms", "warm,cold", "--execute"])
    plan = runner.run_policy_ab(args)
    assert plan["mode"] == "dry_run"
    assert plan["inference_ran"] is False
    assert any("falling back to dry-run" in n for n in plan["notes"])
