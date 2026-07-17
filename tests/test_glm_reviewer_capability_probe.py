#!/usr/bin/env python3
"""Unit tests for scripts/autopilot/glm_reviewer_capability_probe.py (P5 GC-1/2/3).

Coverage is entirely INFERENCE-FREE and single-file (no -n auto). It exercises the
three probe modes' pure logic on SYNTHETIC reviewer outputs — parsing, planning
(task-set -> placement-queue job specs), and deterministic scoring — plus the
model/quant indexing discipline, the RM-3 no-/chat transport, and the triple
execution gate. The one end-to-end run of the execute path uses a STUBBED probe
function (no orchestrator client, no server, no model). Every assertion pins a
concrete expected value; nothing asserts True.
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
    / "scripts" / "autopilot" / "glm_reviewer_capability_probe.py"
)
_SPEC = importlib.util.spec_from_file_location("glm_reviewer_capability_probe", _MODULE_PATH)
probe = importlib.util.module_from_spec(_SPEC)
sys.modules["glm_reviewer_capability_probe"] = probe  # register before exec (dataclasses)
_SPEC.loader.exec_module(probe)


# --------------------------------------------------------------------------- #
# strict_if (GC-1) — typed review_decision emission, GBNF vs free-parse
# --------------------------------------------------------------------------- #
_VALID_APPROVE = '{"decision":"approve","confidence":0.9,"blocking":{"tripwire":false}}'
_VALID_REJECT = '{"decision":"reject","confidence":0.2,"blocking":{"tripwire":true}}'
_PROSE_WRAPPED = (
    'Here is my verdict: {"decision":"request_changes","confidence":0.5,'
    '"blocking":{"tripwire":false}} — done.'
)
_BAD_ENUM = '{"decision":"maybe","confidence":0.5,"blocking":{"tripwire":false}}'
_GARBAGE = "APPROVE"

_STRICT_IF_OUTPUTS = [_VALID_APPROVE, _VALID_REJECT, _PROSE_WRAPPED, _BAD_ENUM, _GARBAGE]


def test_parse_typed_emission_grammar_vs_free_parse():
    # whole-string JSON parses in both lanes.
    assert probe.parse_typed_emission(_VALID_APPROVE, grammar_constrained=True) is not None
    # prose-wrapped: grammar lane rejects, free-parse retry recovers the {...} span.
    assert probe.parse_typed_emission(_PROSE_WRAPPED, grammar_constrained=True) is None
    recovered = probe.parse_typed_emission(_PROSE_WRAPPED, grammar_constrained=False)
    assert recovered is not None and recovered["decision"] == "request_changes"
    # pure garbage never parses.
    assert probe.parse_typed_emission(_GARBAGE, grammar_constrained=False) is None
    assert probe.parse_typed_emission("", grammar_constrained=False) is None


def test_validate_review_decision_required_core():
    ok, errs = probe.validate_review_decision(json.loads(_VALID_APPROVE))
    assert ok is True and errs == []
    # bad enum
    ok, errs = probe.validate_review_decision(json.loads(_BAD_ENUM))
    assert ok is False and any("decision_not_in_enum" in e for e in errs)
    # missing blocking
    ok, errs = probe.validate_review_decision({"decision": "approve", "confidence": 0.5})
    assert ok is False and "missing:blocking" in errs
    # confidence out of range
    ok, errs = probe.validate_review_decision(
        {"decision": "approve", "confidence": 1.5, "blocking": {"tripwire": False}}
    )
    assert ok is False and "confidence_out_of_range" in errs
    # tripwire must be a real bool (not truthy int)
    ok, errs = probe.validate_review_decision(
        {"decision": "approve", "confidence": 0.5, "blocking": {"tripwire": 1}}
    )
    assert ok is False and "missing_or_nonbool:blocking.tripwire" in errs


def test_score_strict_if_grammar_lane_kofm_gate():
    r = probe.score_strict_if(_STRICT_IF_OUTPUTS, grammar_constrained=True, k=2, m=5)
    # grammar lane: only the two whole-string-valid objects count.
    assert r["n_valid"] == 2
    assert r["m"] == 5
    assert r["emission_rate"] == pytest.approx(0.4)
    assert r["passed"] is True          # 2 >= k=2
    # raise the floor above what the lane can clear -> gate fails.
    r_strict = probe.score_strict_if(_STRICT_IF_OUTPUTS, grammar_constrained=True, k=3, m=5)
    assert r_strict["passed"] is False  # 2 < 3


def test_score_strict_if_free_parse_recovers_more():
    r = probe.score_strict_if(_STRICT_IF_OUTPUTS, grammar_constrained=False, k=3, m=5)
    # free-parse retry additionally recovers the prose-wrapped valid decision.
    assert r["n_valid"] == 3
    assert r["emission_rate"] == pytest.approx(0.6)
    assert r["passed"] is True          # 3 >= k=3
    # the bad-enum object parses but is not schema-valid; garbage never parses.
    assert r["n_parsed"] == 4


def test_score_strict_if_m_defaults_to_batch_size():
    r = probe.score_strict_if(_STRICT_IF_OUTPUTS, grammar_constrained=True)
    assert r["m"] == 5                  # defaulted to len(outputs)
    assert r["k"] == probe.DEFAULT_STRICT_IF_K


# --------------------------------------------------------------------------- #
# rubric_authoring (GC-2) — authored rubric vs frontier reference
# --------------------------------------------------------------------------- #
_REFERENCE_RUBRIC = {
    "items": [
        {"id": "R1", "text": "Does the answer address the question?", "axis": "question-alignment", "weight": 3},
        {"id": "R2", "text": "Is every claim grounded in the source?", "axis": "grounding", "weight": 3},
        {"id": "R3", "text": "Are there fabricated facts?", "axis": "integrity", "weight": 2},
        {"id": "R4", "text": "Is the answer complete?", "axis": "completeness", "weight": 1},
    ]
}

_AUTHORED_RUBRIC = {
    "items": [
        {"id": "R1", "text": "Is every claim grounded in the source?", "axis": "grounding", "weight": 3},
        {"id": "R2", "text": "Are any facts fabricated?", "axis": "integrity", "weight": 2},
        {"id": "R3", "text": "The tone is professional.", "axis": "made-up-axis", "weight": 1},
        {"id": "R4", "text": "Invalid weight item.", "axis": "grounding", "weight": 5},  # dropped
        {"id": "bad", "text": "Bad id.", "axis": "grounding", "weight": 1},              # dropped
    ]
}


def test_valid_rubric_items_filters_schema_violations():
    items = probe._valid_rubric_items(_AUTHORED_RUBRIC)
    assert [it["id"] for it in items] == ["R1", "R2", "R3"]  # R4 (weight 5) + 'bad' id dropped


def test_score_rubric_authoring_concrete_axes():
    r = probe.score_rubric_authoring(_AUTHORED_RUBRIC, _REFERENCE_RUBRIC)
    assert r["criteria_count"] == 3
    assert r["reference_criteria_count"] == 4
    assert r["count_ratio"] == pytest.approx(0.75)          # 3/4
    # authored axes {grounding, integrity, made-up-axis} ∩ ref (4 axes) = 2 -> 2/4
    assert r["axis_coverage"] == pytest.approx(0.5)
    assert r["axes_covered"] == ["grounding", "integrity"]
    # 2 of 3 valid items end with '?' (R1,R2 yes; R3 no)
    assert r["grounding_rate"] == pytest.approx(2 / 3)
    assert r["composite"] == pytest.approx((0.75 + 0.5 + 2 / 3) / 3)


def test_score_rubric_authoring_empty_authored_is_zero():
    r = probe.score_rubric_authoring({"items": []}, _REFERENCE_RUBRIC)
    assert r["count_ratio"] == 0.0
    assert r["axis_coverage"] == 0.0
    assert r["grounding_rate"] == 0.0
    assert r["composite"] == 0.0


# --------------------------------------------------------------------------- #
# why_diagnosis (GC-3) — rationale vs gold cause (detect-THAT vs detect-WHY)
# --------------------------------------------------------------------------- #
_WHY_TASKS = [
    {"task_id": "wd-0", "prompt": "diagnose", "gold_cause_aliases": ["off-by-one", "boundary error"]},
    {"task_id": "wd-1", "prompt": "diagnose", "gold_cause_aliases": ["null dereference"]},
    {"task_id": "wd-2", "prompt": "diagnose", "gold_cause_aliases": ["wrong formula"]},
    {"task_id": "wd-3", "prompt": "diagnose", "gold_cause_aliases": ["sign error"]},
]
_WHY_OUTPUTS = [
    "The loop rejects the last element due to an off by one error.",  # that+why
    "This is wrong; it crashes at runtime.",                          # that only
    "The answer looks fine to me.",                                   # neither
    "There is a bug: a sign error flips the result.",                # that+why
]


def test_detect_that_and_cause_matched():
    assert probe.detect_that("This is wrong; it crashes.") is True
    assert probe.detect_that("The answer looks fine to me.") is False
    # normalization lets "off by one" match the "off-by-one" alias.
    assert probe.cause_matched("an off by one error", ["off-by-one"]) is True
    assert probe.cause_matched("the answer is fine", ["off-by-one"]) is False


def test_score_why_diagnosis_that_vs_why_gap():
    scored = [
        probe.score_why_diagnosis_item(out, t["gold_cause_aliases"])
        for t, out in zip(_WHY_TASKS, _WHY_OUTPUTS)
    ]
    agg = probe.score_why_diagnosis(scored)
    assert agg["n"] == 4
    assert agg["n_that_detected"] == 3
    assert agg["n_why_matched"] == 2
    assert agg["that_detection_rate"] == pytest.approx(0.75)
    assert agg["why_match_rate"] == pytest.approx(0.5)
    # the whole point of GC-3: detect-THAT outruns detect-WHY.
    assert agg["that_minus_why_gap"] == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# Task-set parsing / validation (pure)
# --------------------------------------------------------------------------- #
def test_parse_task_set_rejects_bad_tasks_per_probe():
    # missing prompt
    tasks, errs = probe.parse_task_set(
        {"tasks": [{"task_id": "a"}]}, probe="strict_if"
    )
    assert tasks == [] and any("missing_prompt" in e for e in errs)
    # rubric_authoring requires a non-empty reference_rubric
    tasks, errs = probe.parse_task_set(
        {"tasks": [{"task_id": "a", "prompt": "x"}]}, probe="rubric_authoring"
    )
    assert tasks == [] and any("reference_rubric" in e for e in errs)
    # why_diagnosis requires gold_cause_aliases
    tasks, errs = probe.parse_task_set(
        {"tasks": [{"task_id": "a", "prompt": "x"}]}, probe="why_diagnosis"
    )
    assert tasks == [] and any("gold_cause_aliases" in e for e in errs)


def test_parse_task_set_dedups_task_ids():
    tasks, errs = probe.parse_task_set(
        {
            "tasks": [
                {"task_id": "dup", "prompt": "one"},
                {"task_id": "dup", "prompt": "two"},
                {"task_id": "ok", "prompt": "three"},
            ]
        },
        probe="strict_if",
    )
    assert [t["task_id"] for t in tasks] == ["dup", "ok"]
    assert any("duplicate_task_id" in e for e in errs)


# --------------------------------------------------------------------------- #
# Plan resolution -> placement-queue job specs (model/quant indexed, never /chat)
# --------------------------------------------------------------------------- #
def _strict_if_task_set(n: int = 3) -> dict:
    return {
        "probe": "strict_if",
        "task_set_id": "gc1-fixture",
        "tasks": [{"task_id": f"si-{i}", "prompt": f"review candidate {i}"} for i in range(n)],
    }


def test_resolve_capability_probe_builds_model_indexed_jobs():
    resolved = probe.resolve_capability_probe(
        _strict_if_task_set(3), probe="strict_if", grammar_constrained=True
    )
    assert resolved.n_tasks == 3
    assert len(resolved.jobs) == 3
    assert resolved.scoring_config["gate"] == "K_of_M"
    assert resolved.scoring_config["m"] == 3          # defaulted to task count
    assert resolved.scoring_config["k"] == probe.DEFAULT_STRICT_IF_K
    for job in resolved.jobs:
        d = job.to_dict()
        # model/quant indexed — NEVER role-indexed
        assert d["model_key"] == "glm_52_ud_iq2m"
        assert d["quant"] == "UD-IQ2_M"
        assert d["architecture"] == "glm_moe_dsa"
        assert not any("role" in k for k in d.keys())
        assert d["target_binding"]["force_model"] == "glm_52_ud_iq2m"
        # placement-queue transport (RM-3)
        assert d["transport"] == "placement_queue"
        assert d["request_priority"] == "background"
        assert d["workload_class"] == "eval_batch"


def test_resolved_plan_transport_is_never_chat():
    resolved = probe.resolve_capability_probe(_strict_if_task_set(2), probe="strict_if")
    d = resolved.to_dict()
    assert d["transport"]["uses_chat_endpoint"] is False
    # jobs + transport surfaces must never name a /chat endpoint (notes are prose).
    transport_blob = json.dumps({"jobs": d["jobs"], "transport": d["transport"]})
    assert "/chat" not in transport_blob and "/v1/chat" not in transport_blob


def test_resolve_records_parse_errors_without_dropping_plan():
    ts = _strict_if_task_set(2)
    ts["tasks"].append({"task_id": "bad"})  # missing prompt
    resolved = probe.resolve_capability_probe(ts, probe="strict_if")
    assert resolved.n_tasks == 2                      # only the 2 good tasks materialized
    assert any("missing_prompt" in e for e in resolved.parse_errors)


# --------------------------------------------------------------------------- #
# score_probe dispatcher — model/quant indexed summaries (the stub-the-probe seam)
# --------------------------------------------------------------------------- #
def test_score_probe_strict_if_is_model_quant_indexed():
    tasks = _strict_if_task_set(5)["tasks"]
    summary = probe.score_probe(
        "strict_if", tasks, _STRICT_IF_OUTPUTS, grammar_constrained=True, k=2, m=5
    )
    assert summary["model_key"] == "glm_52_ud_iq2m"
    assert summary["quant"] == "UD-IQ2_M"
    assert summary["observation_only"] is True
    assert "role" not in summary                      # never role-indexed
    assert summary["n_valid"] == 2
    assert summary["emission_rate"] == pytest.approx(0.4)


def test_score_probe_rubric_authoring_aggregates():
    tasks = [
        {"task_id": "ra-0", "prompt": "author", "reference_rubric": _REFERENCE_RUBRIC},
    ]
    summary = probe.score_probe("rubric_authoring", tasks, [_AUTHORED_RUBRIC])
    assert summary["model_key"] == "glm_52_ud_iq2m"
    assert summary["mean_count_ratio"] == pytest.approx(0.75)
    assert summary["mean_axis_coverage"] == pytest.approx(0.5)
    assert summary["mean_composite"] == pytest.approx((0.75 + 0.5 + 2 / 3) / 3)


def test_score_probe_why_diagnosis_aggregates():
    summary = probe.score_probe("why_diagnosis", _WHY_TASKS, _WHY_OUTPUTS)
    assert summary["model_key"] == "glm_52_ud_iq2m"
    assert summary["that_detection_rate"] == pytest.approx(0.75)
    assert summary["why_match_rate"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Triple execution gate (dry-run by default; execute needs flag + 2 env gates)
# --------------------------------------------------------------------------- #
def test_execution_gate_requires_all_three(monkeypatch):
    monkeypatch.delenv(probe.INFERENCE_ENV, raising=False)
    monkeypatch.delenv(probe.GLM_ADMISSION_ENV, raising=False)
    assert probe.execution_gate_status(execute_flag=True)["open"] is False
    monkeypatch.setenv(probe.INFERENCE_ENV, "1")
    assert probe.execution_gate_status(execute_flag=True)["open"] is False  # admission still missing
    monkeypatch.setenv(probe.GLM_ADMISSION_ENV, "1")
    assert probe.execution_gate_status(execute_flag=True)["open"] is True
    assert probe.execution_gate_status(execute_flag=False)["open"] is False  # flag still required


def test_run_capability_probe_default_is_dry_run_no_inference(monkeypatch):
    monkeypatch.delenv(probe.INFERENCE_ENV, raising=False)
    monkeypatch.delenv(probe.GLM_ADMISSION_ENV, raising=False)

    def _boom(job, task):  # must NEVER be called on the dry-run path
        raise AssertionError("inference probe invoked during dry-run")

    result = probe.run_capability_probe(
        _strict_if_task_set(3), probe="strict_if", execute=True, probe_fn=_boom
    )
    assert result["mode"] == "dry_run"
    assert result["inference_ran"] is False
    assert result["gate"]["open"] is False
    assert result["n_jobs"] == 3
    assert result["resolved_plan"]["transport"]["uses_chat_endpoint"] is False


def test_run_capability_probe_execute_path_with_stubbed_probe(monkeypatch):
    # Open all three gates, then drive the execute path with a STUB (no inference).
    monkeypatch.setenv(probe.INFERENCE_ENV, "1")
    monkeypatch.setenv(probe.GLM_ADMISSION_ENV, "1")

    outputs_by_id = {
        "si-0": _VALID_APPROVE,
        "si-1": _VALID_REJECT,
        "si-2": _PROSE_WRAPPED,  # invalid under grammar lane
    }
    calls: list[str] = []

    def _stub(job, task):
        calls.append(job.task_id)
        return outputs_by_id[job.task_id]

    result = probe.run_capability_probe(
        _strict_if_task_set(3),
        probe="strict_if",
        execute=True,
        grammar_constrained=True,
        k=2,
        m=3,
        probe_fn=_stub,
    )
    assert result["mode"] == "execute"
    assert result["inference_ran"] is True
    assert sorted(calls) == ["si-0", "si-1", "si-2"]  # stub hit every job, once
    summary = result["result"]
    # grammar lane: 2 valid whole-string objects, prose-wrapped one rejected.
    assert summary["n_valid"] == 2
    assert summary["emission_rate"] == pytest.approx(2 / 3)
    assert summary["passed"] is True                 # 2 >= k=2
    # result stays model/quant indexed, observation-only.
    assert summary["model_key"] == "glm_52_ud_iq2m"
    assert summary["quant"] == "UD-IQ2_M"
    assert summary["observation_only"] is True


# --------------------------------------------------------------------------- #
# CLI main() — default invocation is a pure dry-run, exit 0
# --------------------------------------------------------------------------- #
def test_cli_default_dry_run_exits_zero(capsys, monkeypatch):
    monkeypatch.delenv(probe.INFERENCE_ENV, raising=False)
    monkeypatch.delenv(probe.GLM_ADMISSION_ENV, raising=False)
    rc = probe.main(["--probe", "strict_if"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["mode"] == "dry_run"
    assert out["inference_ran"] is False
    assert out["n_jobs"] == probe.DEFAULT_STRICT_IF_M  # builtin task set is M tasks
    assert out["resolved_plan"]["model"]["model_key"] == "glm_52_ud_iq2m"
