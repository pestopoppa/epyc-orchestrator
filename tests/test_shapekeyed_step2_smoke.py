#!/usr/bin/env python3
"""Fixture test for scripts/autopilot/shapekeyed_step2_smoke.py (ROUTE-A1).

Entirely INFERENCE-FREE. Exercises the two pure surfaces of the shape-keyed
Step-2 smoke driver:

  * PLAN construction — admit-overlap probe classification (region-set overlap =>
    admit/queue) and the vision within-role disjoint re-bench pair enumeration
    (model/quant indexed, J5 priors attached), over a SYNTHETIC region topology.
  * RESULT aggregation — scoring synthetic admit/queue routing outcomes against
    the region-derived expectation, and rolling synthetic co-run ratio samples
    into mean/CV/verdict/clean/ratified rows.

Also asserts placement-queue transport discipline (never /chat) and the
double-gate (no --execute AND/OR no env flag => dry-run, execute bridge never
called). The execution bridge itself is never invoked with real inference here.

Concrete expected values are asserted throughout — no `assert True`.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

# ── load the runner by path (no scripts.* package import needed) ──────────────
_MODULE_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts" / "autopilot" / "shapekeyed_step2_smoke.py"
)
_SPEC = importlib.util.spec_from_file_location("shapekeyed_step2_smoke", _MODULE_PATH)
smoke = importlib.util.module_from_spec(_SPEC)
sys.modules["shapekeyed_step2_smoke"] = smoke  # register before exec (dataclasses)
_SPEC.loader.exec_module(smoke)


# --------------------------------------------------------------------------- #
# Synthetic topology fixtures (hermetic — no live NUMA_CONFIG)
# --------------------------------------------------------------------------- #
def _synthetic_regions() -> dict[tuple[str, int], frozenset[str]]:
    """A minimal region map covering both admit and queue expectations.

    anchor ingest_long_context#0 holds the node0-half {q0,q1}. Against it:
      frontdoor#0 {q0,q1}          -> OVERLAP -> queue
      frontdoor#1 {q2}             -> disjoint -> admit
      vision_escalation#1 {q0}     -> OVERLAP -> queue
      vision_escalation#3 {q2}     -> disjoint -> admit
    """
    return {
        ("ingest_long_context", 0): frozenset({"q0", "q1"}),
        ("frontdoor", 0): frozenset({"q0", "q1"}),
        ("frontdoor", 1): frozenset({"q2"}),
        ("vision_escalation", 1): frozenset({"q0"}),
        ("vision_escalation", 3): frozenset({"q2"}),
    }


def _vision_regions() -> dict[tuple[str, int], frozenset[str]]:
    """The real vision_escalation instance shapes: full={q2,q3}, then q0..q3."""
    return {
        ("vision_escalation", 0): frozenset({"q2", "q3"}),  # full = node1-half
        ("vision_escalation", 1): frozenset({"q0"}),
        ("vision_escalation", 2): frozenset({"q1"}),
        ("vision_escalation", 3): frozenset({"q2"}),
        ("vision_escalation", 4): frozenset({"q3"}),
    }


# --------------------------------------------------------------------------- #
# Region helpers (pure)
# --------------------------------------------------------------------------- #
def test_region_and_pair_labels_never_use_full():
    assert smoke.region_label({"q2", "q3"}) == "q2q3"
    assert smoke.region_label({"q0"}) == "q0"
    # order-independent pair label; a node-half "full" is expressed as its regions.
    assert smoke.pair_label({"q2", "q3"}, {"q0"}) == "q0+q2q3"
    assert smoke.pair_label({"q0"}, {"q1"}) == "q0+q1"
    assert smoke.pair_label({"q1"}, {"q0"}) == "q0+q1"


def test_regions_disjoint():
    assert smoke.regions_disjoint({"q0"}, {"q2", "q3"}) is True
    assert smoke.regions_disjoint({"q2"}, {"q2", "q3"}) is False


# --------------------------------------------------------------------------- #
# Admit-overlap probe construction
# --------------------------------------------------------------------------- #
def test_admit_overlap_probe_classification():
    regions = _synthetic_regions()
    anchor, probes = smoke.build_admit_overlap_probes(
        regions,
        anchor=("ingest_long_context", 0),
        probe_roles=["frontdoor", "vision_escalation"],
    )
    assert anchor.regions == ("q0", "q1")
    # 4 probe-role instances, none is the anchor itself -> 4 probes.
    assert len(probes) == 4

    by_key = {(p.candidate.role, p.candidate.instance_idx): p for p in probes}
    # OVERLAP -> queue
    assert by_key[("frontdoor", 0)].disjoint is False
    assert by_key[("frontdoor", 0)].expected_decision == smoke.DECISION_QUEUE
    assert by_key[("vision_escalation", 1)].expected_decision == "queue"
    # DISJOINT -> admit
    assert by_key[("frontdoor", 1)].disjoint is True
    assert by_key[("frontdoor", 1)].expected_decision == smoke.DECISION_ADMIT
    assert by_key[("vision_escalation", 3)].expected_decision == "admit"

    # exactly 2 admit + 2 queue expectations
    assert sum(1 for p in probes if p.expected_decision == "admit") == 2
    assert sum(1 for p in probes if p.expected_decision == "queue") == 2


def test_admit_overlap_probe_excludes_anchor_and_is_ordered():
    regions = _synthetic_regions()
    _, probes = smoke.build_admit_overlap_probes(
        regions,
        anchor=("ingest_long_context", 0),
        probe_roles=["ingest_long_context", "frontdoor", "vision_escalation"],
    )
    cand_keys = [(p.candidate.role, p.candidate.instance_idx) for p in probes]
    # anchor placement itself must be excluded
    assert ("ingest_long_context", 0) not in cand_keys
    # deterministic ordering by (role, idx)
    assert cand_keys == sorted(cand_keys)


# --------------------------------------------------------------------------- #
# Re-bench pair enumeration (model/quant indexed, disjoint-only, J5 priors)
# --------------------------------------------------------------------------- #
def test_rebench_pairs_disjoint_only_with_priors():
    pairs = smoke.build_rebench_pairs(
        _vision_regions(),
        role="vision_escalation",
        model="Qwen3-VL-30B-A3B-Instruct",
        quant="Q4_K_M",
        target_samples=8,
    )
    # C(5,2)=10 total; the 2 overlapping full+q2 / full+q3 pairs are dropped -> 8.
    assert len(pairs) == 8
    ids = {p.pair_id for p in pairs}
    assert "q0+q2q3" not in {smoke.pair_label({"q2", "q3"}, {"q2"})}  # sanity: label
    # the two overlapping pairs (full{q2,q3}+q2, full+q3) are absent
    assert ids == {
        "q0+q1", "q0+q2", "q0+q3", "q1+q2", "q1+q3", "q2+q3",
        "q0+q2q3", "q1+q2q3",
    }

    by_id = {p.pair_id: p for p in pairs}
    # bench index = model/quant, NEVER role
    assert by_id["q0+q3"].model == "Qwen3-VL-30B-A3B-Instruct"
    assert by_id["q0+q3"].quant == "Q4_K_M"
    assert by_id["q0+q3"].model_quant_key == "Qwen3-VL-30B-A3B-Instruct::Q4_K_M"
    assert by_id["q0+q3"].role == "vision_escalation"  # provenance only
    assert by_id["q0+q3"].target_samples == 8
    # J5 prior attached from the canonical table
    assert by_id["q0+q3"].prior_ratio == pytest.approx(1.266)
    assert by_id["q0+q3"].prior_cv == pytest.approx(0.004)
    assert by_id["q0+q2q3"].prior_ratio == pytest.approx(0.580)  # full+q0
    assert by_id["q1+q3"].prior_cv == pytest.approx(0.420)


def test_rebench_pair_to_dict_indexes_model_quant_not_role():
    pairs = smoke.build_rebench_pairs(_vision_regions())
    d = pairs[0].to_dict()
    # the result dict's primary index fields are model/quant; role is provenance
    assert d["model"] == "Qwen3-VL-30B-A3B-Instruct"
    assert d["quant"] == "Q4_K_M"
    assert d["model_quant_key"] == "Qwen3-VL-30B-A3B-Instruct::Q4_K_M"
    assert d["role_provenance"] == "vision_escalation"
    assert "role" not in d  # role is NOT a top-level index key


# --------------------------------------------------------------------------- #
# Full plan construction
# --------------------------------------------------------------------------- #
def test_build_plan_counts_and_transport():
    regions = {**_synthetic_regions(), **_vision_regions()}
    plan = smoke.build_step2_smoke_plan(
        regions,
        anchor=("ingest_long_context", 0),
        probe_roles=["frontdoor", "vision_escalation"],
        rebench_role="vision_escalation",
    )
    d = plan.to_dict()
    assert d["kind"] == "shapekeyed_step2_smoke_plan"
    assert d["n_rebench_pairs"] == 8
    assert d["floor"] == pytest.approx(0.85)
    assert d["cv_threshold"] == pytest.approx(0.05)
    assert d["target_samples"] == 8
    # transport discipline
    assert d["transport"]["transport"] == "placement_queue"
    assert d["transport"]["request_priority"] == "background"
    assert d["transport"]["workload_class"] == "eval_batch"
    assert d["transport"]["uses_chat_endpoint"] is False
    assert d["n_admit_expected"] + d["n_queue_expected"] == d["n_probes"]


def test_plan_transport_blob_never_names_chat():
    regions = {**_synthetic_regions(), **_vision_regions()}
    plan = smoke.build_step2_smoke_plan(
        regions, probe_roles=["frontdoor", "vision_escalation"]
    )
    d = plan.to_dict()
    # transport-bearing surfaces must never name a /chat endpoint (notes may).
    blob = json.dumps({"probes": d["probes"], "rebench_pairs": d["rebench_pairs"],
                       "transport": d["transport"]})
    assert "/chat" not in blob
    assert "/v1/chat" not in blob
    for probe in d["probes"]:
        assert probe["transport"] == "placement_queue"
        assert probe["workload_class"] == "eval_batch"
    for r in d["rebench_pairs"]:
        assert r["transport"] == "placement_queue"
        assert r["request_priority"] == "background"


# --------------------------------------------------------------------------- #
# Admit-overlap aggregation (synthetic routing outcomes)
# --------------------------------------------------------------------------- #
def _probes_for_agg():
    return smoke.build_admit_overlap_probes(
        _synthetic_regions(),
        anchor=("ingest_long_context", 0),
        probe_roles=["frontdoor", "vision_escalation"],
    )[1]


def test_aggregate_admit_overlap_all_pass():
    probes = _probes_for_agg()
    # feed each probe its EXPECTED decision -> all pass
    observed = {p.probe_id: p.expected_decision for p in probes}
    summary = smoke.aggregate_admit_overlap(probes, observed)
    assert summary["n_probes"] == 4
    assert summary["n_admit_expected"] == 2
    assert summary["n_queue_expected"] == 2
    assert summary["n_evaluated"] == 4
    assert summary["n_pass"] == 4
    assert summary["n_fail"] == 0
    assert summary["all_pass"] is True
    assert summary["observation_only"] is True


def test_aggregate_admit_overlap_detects_mismatch():
    probes = _probes_for_agg()
    observed = {p.probe_id: p.expected_decision for p in probes}
    # flip ONE disjoint probe to a wrong "queue" (a real Step-2 failure)
    disjoint_probe = next(p for p in probes if p.disjoint)
    observed[disjoint_probe.probe_id] = "queue"
    summary = smoke.aggregate_admit_overlap(probes, observed)
    assert summary["n_evaluated"] == 4
    assert summary["n_pass"] == 3
    assert summary["n_fail"] == 1
    assert summary["all_pass"] is False
    bad = next(r for r in summary["rows"] if r["probe_id"] == disjoint_probe.probe_id)
    assert bad["expected"] == "admit"
    assert bad["observed"] == "queue"
    assert bad["pass"] is False


def test_aggregate_admit_overlap_missing_observation_excluded():
    probes = _probes_for_agg()
    # observe only two probes; the other two have no outcome
    partial = {probes[0].probe_id: probes[0].expected_decision,
               probes[1].probe_id: probes[1].expected_decision}
    summary = smoke.aggregate_admit_overlap(probes, partial)
    assert summary["n_evaluated"] == 2
    assert summary["n_pass"] == 2
    # a probe with no observation is pass=None and not counted
    none_rows = [r for r in summary["rows"] if r["pass"] is None]
    assert len(none_rows) == 2
    # all_pass requires at least one evaluation AND no failures
    assert summary["all_pass"] is True


def test_aggregate_admit_overlap_accepts_list_form():
    probes = _probes_for_agg()
    observed_list = [
        {"probe_id": p.probe_id, "decision": p.expected_decision} for p in probes
    ]
    summary = smoke.aggregate_admit_overlap(probes, observed_list)
    assert summary["n_pass"] == 4
    assert summary["all_pass"] is True


# --------------------------------------------------------------------------- #
# Re-bench verdict + summarize (pure computation, exact values)
# --------------------------------------------------------------------------- #
def test_rebench_verdict_thresholds():
    assert smoke.rebench_verdict(1.20, 0.85) == "allow"
    assert smoke.rebench_verdict(1.00, 0.85) == "allow"
    assert smoke.rebench_verdict(0.90, 0.85) == "borderline"
    assert smoke.rebench_verdict(0.85, 0.85) == "borderline"
    assert smoke.rebench_verdict(0.50, 0.85) == "block"
    assert smoke.rebench_verdict(None, 0.85) == "unknown"


def _pair(pair_id="q0+q3", prior_ratio=1.266, prior_cv=0.004, target=8):
    return smoke.RebenchPairSpec(
        pair_id=pair_id, role="vision_escalation",
        model="Qwen3-VL-30B-A3B-Instruct", quant="Q4_K_M",
        instance_a_idx=1, instance_b_idx=4,
        region_a=("q0",), region_b=("q3",),
        target_samples=target, prior_ratio=prior_ratio, prior_cv=prior_cv,
    )


def test_summarize_rebench_clean_allow_ratified():
    # zero-variance super-linear samples -> clean allow, ratified
    row = smoke.summarize_rebench_pair(_pair(), [1.20, 1.20, 1.20], floor=0.85,
                                       cv_threshold=0.05)
    assert row["n_samples"] == 3
    assert row["mean_ratio"] == pytest.approx(1.20)
    assert row["cv"] == pytest.approx(0.0)
    assert row["clean"] is True
    assert row["verdict"] == "allow"
    assert row["ratified_allow"] is True
    assert row["met_sample_target"] is False  # 3 < target 8
    assert row["ratio_delta_vs_prior"] == pytest.approx(1.20 - 1.266)
    # model/quant indexed, role provenance only
    assert row["model_quant_key"] == "Qwen3-VL-30B-A3B-Instruct::Q4_K_M"
    assert row["role_provenance"] == "vision_escalation"
    assert row["transport"] == "placement_queue"
    assert row["observation_only"] is True


def test_summarize_rebench_allow_but_noisy_not_ratified():
    # mean 1.2 (allow) but CV above the 5% gate -> not clean, not ratified
    row = smoke.summarize_rebench_pair(_pair(), [1.0, 1.4], floor=0.85,
                                       cv_threshold=0.05)
    assert row["mean_ratio"] == pytest.approx(1.2)
    # sample stdev of [1.0,1.4] = sqrt(0.08); cv = that / 1.2
    expected_cv = math.sqrt(0.08) / 1.2
    assert row["cv"] == pytest.approx(expected_cv)
    assert row["cv"] > 0.05
    assert row["clean"] is False
    assert row["verdict"] == "allow"
    assert row["ratified_allow"] is False


def test_summarize_rebench_borderline_and_block():
    borderline = smoke.summarize_rebench_pair(_pair(), [0.90, 0.90], floor=0.85,
                                              cv_threshold=0.05)
    assert borderline["mean_ratio"] == pytest.approx(0.90)
    assert borderline["cv"] == pytest.approx(0.0)
    assert borderline["clean"] is True
    assert borderline["verdict"] == "borderline"
    assert borderline["ratified_allow"] is False  # borderline never ratifies allow

    block = smoke.summarize_rebench_pair(_pair(), [0.50, 0.50], floor=0.85)
    assert block["verdict"] == "block"


def test_summarize_rebench_insufficient_samples():
    one = smoke.summarize_rebench_pair(_pair(), [1.30], floor=0.85)
    assert one["n_samples"] == 1
    assert one["mean_ratio"] == pytest.approx(1.30)
    assert one["cv"] is None       # <2 samples -> undefined
    assert one["clean"] is False
    assert one["verdict"] == "allow"
    assert one["ratified_allow"] is False  # can't ratify without a clean CV

    empty = smoke.summarize_rebench_pair(_pair(), [], floor=0.85)
    assert empty["n_samples"] == 0
    assert empty["mean_ratio"] is None
    assert empty["verdict"] == "unknown"
    assert empty["ratio_delta_vs_prior"] is None


def test_aggregate_rebench_over_all_pairs():
    pairs = smoke.build_rebench_pairs(_vision_regions())
    # give every pair the same clean super-linear samples
    samples = {p.pair_id: [1.10, 1.10, 1.10] for p in pairs}
    rows = smoke.aggregate_rebench(pairs, samples, floor=0.85, cv_threshold=0.05)
    assert len(rows) == 8
    assert all(r["verdict"] == "allow" for r in rows)
    assert all(r["ratified_allow"] for r in rows)
    # order preserved (matches pair order)
    assert [r["pair_id"] for r in rows] == [p.pair_id for p in pairs]


# --------------------------------------------------------------------------- #
# Combined smoke aggregation
# --------------------------------------------------------------------------- #
def test_aggregate_smoke_combines_both():
    regions = {**_synthetic_regions(), **_vision_regions()}
    plan = smoke.build_step2_smoke_plan(
        regions, anchor=("ingest_long_context", 0),
        probe_roles=["frontdoor"],  # keep probe set small + deterministic
    )
    observed = {p.probe_id: p.expected_decision for p in plan.probes}
    samples = {p.pair_id: [1.15, 1.15] for p in plan.rebench_pairs}
    report = smoke.aggregate_smoke(
        plan, observed_decisions=observed, rebench_samples=samples
    )
    assert report["kind"] == "shapekeyed_step2_smoke_report"
    assert report["admit_overlap"]["all_pass"] is True
    assert report["smoke_pass"] is True
    assert report["n_rebench_ratified_allow"] == 8
    assert len(report["rebench"]) == 8


# --------------------------------------------------------------------------- #
# Double-gate: dry-run unless BOTH --execute and the env flag are set
# --------------------------------------------------------------------------- #
def _tiny_plan():
    regions = {**_synthetic_regions(), **_vision_regions()}
    return smoke.build_step2_smoke_plan(regions, probe_roles=["frontdoor"])


def test_env_flag_semantics(monkeypatch):
    for val, expected in [("1", True), ("true", True), ("YES", True),
                          ("on", True), ("0", False), ("", False), ("no", False)]:
        monkeypatch.setenv(smoke.SHAPEKEYED_STEP2_INFERENCE_ENV, val)
        assert smoke._env_flag_enabled(smoke.SHAPEKEYED_STEP2_INFERENCE_ENV) is expected


def test_no_execute_flag_is_dry_run_even_with_env(monkeypatch):
    monkeypatch.setenv(smoke.SHAPEKEYED_STEP2_INFERENCE_ENV, "1")

    def _boom(*a, **k):
        raise AssertionError("execute bridge called without --execute")

    monkeypatch.setattr(smoke, "execute_step2_smoke", _boom)
    out = smoke.run_shapekeyed_step2_smoke(_tiny_plan(), execute=False)
    assert out["mode"] == "dry_run"
    assert out["inference_ran"] is False
    assert out["plan"]["kind"] == "shapekeyed_step2_smoke_plan"


def test_execute_without_env_is_dry_run(monkeypatch):
    monkeypatch.delenv(smoke.SHAPEKEYED_STEP2_INFERENCE_ENV, raising=False)

    def _boom(*a, **k):
        raise AssertionError("execute bridge called with env flag OFF")

    monkeypatch.setattr(smoke, "execute_step2_smoke", _boom)
    out = smoke.run_shapekeyed_step2_smoke(_tiny_plan(), execute=True)
    assert out["mode"] == "dry_run"
    assert out["inference_ran"] is False
    assert "AUTOPILOT_SHAPEKEYED_STEP2_SMOKE not set" in out["reason"]


def test_both_gates_route_to_execute_bridge(monkeypatch):
    monkeypatch.setenv(smoke.SHAPEKEYED_STEP2_INFERENCE_ENV, "1")
    captured = {}

    def _fake_execute(plan, **kwargs):
        captured["plan"] = plan
        captured["kwargs"] = kwargs
        return {"kind": "shapekeyed_step2_smoke_report", "smoke_pass": True}

    monkeypatch.setattr(smoke, "execute_step2_smoke", _fake_execute)
    out = smoke.run_shapekeyed_step2_smoke(
        _tiny_plan(), execute=True, output_path=Path("/tmp/does-not-matter.json")
    )
    assert out["mode"] == "execute"
    assert out["inference_ran"] is True
    assert out["report"]["smoke_pass"] is True
    assert isinstance(captured["plan"], smoke.Step2SmokePlan)


# --------------------------------------------------------------------------- #
# CLI __main__ (pure dry-run over live topology)
# --------------------------------------------------------------------------- #
def test_main_dry_run_prints_plan(capsys, monkeypatch):
    monkeypatch.delenv(smoke.SHAPEKEYED_STEP2_INFERENCE_ENV, raising=False)
    code = smoke.main([])  # no --execute -> pure dry-run over live NUMA_CONFIG
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "shapekeyed_step2_smoke_plan"
    assert payload["transport"]["uses_chat_endpoint"] is False
    # live vision_escalation has 8 disjoint within-role pairs
    assert payload["n_rebench_pairs"] == 8
    assert payload["n_probes"] >= 1


def test_main_dry_run_with_json_numa_config(tmp_path, capsys):
    # a synthetic NUMA_CONFIG JSON: instances stored as [cpu_list, port, threads]
    cfg = {
        "ingest_long_context": {"instances": [["0-47", 8085, 96], ["0-23", 8185, 48]]},
        "frontdoor": {"instances": [["0-47", 8070, 96], ["48-71", 8280, 48]]},
        "vision_escalation": {"instances": [
            ["48-95", 8087, 96], ["0-23", 8187, 48], ["24-47", 8287, 48],
            ["48-71", 8387, 48], ["72-95", 8487, 48],
        ]},
    }
    cfg_path = tmp_path / "numa.json"
    cfg_path.write_text(json.dumps(cfg))
    code = smoke.main([
        "--numa-config", str(cfg_path),
        "--probe-roles", "frontdoor,vision_escalation",
    ])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_rebench_pairs"] == 8
    # frontdoor#0 {q0,q1} overlaps anchor; frontdoor#1 {q2} disjoint; etc.
    assert payload["n_admit_expected"] >= 1
    assert payload["n_queue_expected"] >= 1
