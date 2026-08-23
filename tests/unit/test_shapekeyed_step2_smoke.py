"""Unit tests for the ROUTE-A1 shape-keyed Step-2 smoke execution bridge.

Covers the two placement-queue drivers (``_drive_admit_overlap_probes`` and
``_drive_rebench_pairs``) via INJECTED probe/sample callables, the admit-vs-queue
classifier, ``execute_step2_smoke``'s aggregation + artifact write, and the
double-gate dry-run fallback. NOTHING here touches a network, a model, or a live
port, and NO test ever sets ``AUTOPILOT_SHAPEKEYED_STEP2_SMOKE`` with a real
transport — every execute path is driven with synthetic injected callables.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.autopilot import shapekeyed_step2_smoke as sk

# Synthetic (role, idx) -> region-set topology. anchor = ingest#1 {q0,q1}
# (node0-half — the corrected default: quarters do not exist in production;
# idx 0 is the FULL 96t shape, idx 1/2 the 48t halves). Chosen so the default
# probe roles yield a known admit/queue mix and vision_escalation exposes
# several disjoint within-role pairs for re-bench.
SYNTH_REGIONS: dict[tuple[str, int], frozenset[str]] = {
    ("ingest_long_context", 1): frozenset({"q0", "q1"}),          # anchor
    ("frontdoor", 0): frozenset({"q0", "q1"}),                    # overlap -> QUEUE
    ("vision_escalation", 0): frozenset({"q2", "q3"}),            # disjoint -> ADMIT
    ("vision_escalation", 1): frozenset({"q2"}),                  # disjoint -> ADMIT
    ("vision_escalation", 2): frozenset({"q3"}),                  # disjoint -> ADMIT
    ("vision_escalation", 3): frozenset({"q0"}),                  # overlap -> QUEUE
    ("worker_general", 0): frozenset({"q0", "q1", "q2", "q3"}),   # overlap -> QUEUE
}


def _plan() -> sk.Step2SmokePlan:
    # Explicit probe roles including vision_escalation: the SYNTH fixture exists
    # to exercise the classifier machinery with a balanced admit/queue mix. The
    # production DEFAULT_PROBE_ROLES intentionally excludes vision_escalation
    # (GPU-served, no CPU-region instance — 2026-08-23).
    return sk.build_step2_smoke_plan(
        SYNTH_REGIONS,
        topology_hash="test-topo",
        probe_roles=("frontdoor", "ingest_long_context", "vision_escalation", "worker_general"),
    )


def _anchor_held() -> dict[str, list[int]]:
    """Injected anchor-hold scan: the SYNTH anchor (ingest_long_context#1) is held."""
    return {"ingest_long_context": [1]}


# ── admit/queue classifier ────────────────────────────────────────────────────


def test_classify_admit_queue_signal_matrix() -> None:
    c = sk._classify_admit_queue
    # A clean answer with no error => admit.
    assert c({"answer": "OK", "predicted_tps": 40.0}) == sk.DECISION_ADMIT
    # The contention gate's 503 body (parsed) => queue via the "contention" marker.
    assert c({"error": "contention_denied", "detail": "x", "retry_after_s": 5}) == sk.DECISION_QUEUE
    # The same 503 surfaced through call_orchestrator_forced's raise_for_status
    # (empty answer, httpx status string) => queue via the bare-503 rule.
    assert (
        c({"answer": "", "error": "Server error '503 Service Unavailable' for url 'http://localhost:8000/chat'"})
        == sk.DECISION_QUEUE
    )
    # The dispatcher's ContentionDenied reason string => queue.
    assert c({"answer": "", "error": "placement timeout role=frontdoor after 90.0s"}) == sk.DECISION_QUEUE
    # A circuit-open 503 arrives as an [ERROR: ... unavailable] answer + error_code:
    # NOT gate evidence => unscored.
    assert c({"answer": "[ERROR: backend unavailable circuit open]", "error_code": 503,
              "error": "[ERROR: backend unavailable circuit open]"}) is None
    # A generic backend failure => unscored.
    assert c({"answer": "[FAILED: llama backend]", "error_code": 502}) is None
    assert c({"answer": "", "error": "Connection refused"}) is None
    # Pre-classified decision strings pass through; junk => None.
    assert c("admit") == sk.DECISION_ADMIT
    assert c("QUEUE") == sk.DECISION_QUEUE
    assert c("weird") is None
    assert c(None) is None
    assert c(42) is None
    assert c({"answer": "   "}) is None  # empty-after-strip, no error
    assert c({}) is None


# ── admit-overlap driver (injected probe_fn) ──────────────────────────────────


def test_drive_admit_overlap_probes_scores_against_plan() -> None:
    plan = _plan()
    # 6 probes: 3 disjoint (ADMIT) + 3 overlapping (QUEUE).
    assert len(plan.probes) == 6
    assert sum(1 for p in plan.probes if p.expected_decision == sk.DECISION_ADMIT) == 3
    assert sum(1 for p in plan.probes if p.expected_decision == sk.DECISION_QUEUE) == 3

    seen_priority: set[str] = set()
    seen_workload: set[str] = set()

    def fake_probe(spec: sk.AdmitOverlapProbeSpec, *, seed: int):
        # Assert every probe rides the placement queue (background/eval_batch).
        seen_priority.add(spec.request_priority)
        seen_workload.add(spec.workload_class)
        assert spec.transport == sk.PLACEMENT_QUEUE_TRANSPORT
        if spec.expected_decision == sk.DECISION_ADMIT:
            return {"answer": "OK", "tokens_generated": 4, "predicted_tps": 40.0}
        # Simulate the contention gate's fail-closed 503 (as surfaced through
        # call_orchestrator_forced): empty answer + httpx 503 error string.
        return {"answer": "", "error": "Server error '503 Service Unavailable' for url 'http://localhost:8000/chat'"}

    observed = sk._drive_admit_overlap_probes(
        plan, seed=7, probe_fn=fake_probe, anchor_hold_fn=_anchor_held
    )

    assert seen_priority == {sk.PLACEMENT_REQUEST_PRIORITY}   # "background"
    assert seen_workload == {sk.PLACEMENT_WORKLOAD_CLASS}     # "eval_batch"
    # Every probe classified back to its region-derived expectation.
    assert observed == {p.probe_id: p.expected_decision for p in plan.probes}

    summary = sk.aggregate_admit_overlap(plan.probes, observed)
    assert summary["n_evaluated"] == 6
    assert summary["n_pass"] == 6
    assert summary["all_pass"] is True


def test_drive_admit_overlap_probes_omits_unscored_outcomes() -> None:
    plan = _plan()
    # First probe returns a non-gate backend error -> must be OMITTED (unscored).
    dropped = plan.probes[0].probe_id

    def fake_probe(spec: sk.AdmitOverlapProbeSpec, *, seed: int):
        if spec.probe_id == dropped:
            return {"answer": "[ERROR: backend crashed]", "error_code": 500}
        if spec.expected_decision == sk.DECISION_ADMIT:
            return {"answer": "OK"}
        return {"answer": "", "error": "503 Service Unavailable /chat"}

    observed = sk._drive_admit_overlap_probes(
        plan, seed=1, probe_fn=fake_probe, anchor_hold_fn=_anchor_held
    )
    assert dropped not in observed
    assert len(observed) == 5

    summary = sk.aggregate_admit_overlap(plan.probes, observed)
    assert summary["n_evaluated"] == 5  # the dropped probe is not counted
    assert summary["n_pass"] == 5
    # all_pass is over the evaluated set; the omitted probe scores pass=None.
    assert any(row["pass"] is None and row["probe_id"] == dropped for row in summary["rows"])


def test_drive_admit_overlap_probes_timeout_path_classifies_queue() -> None:
    """The 90s/60s background-queue-budget timeout (fail-closed 503) is QUEUE.

    PROBE_TIMEOUT_S=120 maps to a bounded max_queue_wait_ms; an overlapping
    candidate exhausts it and surfaces as the contention handler's 503 through
    call_orchestrator_forced (empty answer + httpx 503 error text). The driver
    must classify that exact timeout shape as "queue", never as unscored.
    """
    plan = _plan()

    def fake_probe(spec: sk.AdmitOverlapProbeSpec, *, seed: int):
        if spec.expected_decision == sk.DECISION_ADMIT:
            return {"answer": "OK", "tokens_generated": 4}
        # The queue-budget-exhausted contention timeout, exactly as surfaced
        # through call_orchestrator_forced's raise_for_status branch.
        return {
            "answer": "",
            "error": (
                "Server error '503 Service Unavailable' for url "
                "'http://localhost:8000/chat'"
            ),
            "failure_reason": "http_status",
        }

    observed = sk._drive_admit_overlap_probes(
        plan, seed=9, probe_fn=fake_probe, anchor_hold_fn=_anchor_held
    )
    assert observed == {p.probe_id: p.expected_decision for p in plan.probes}
    queue_ids = [p.probe_id for p in plan.probes if p.expected_decision == sk.DECISION_QUEUE]
    assert all(observed[qid] == sk.DECISION_QUEUE for qid in queue_ids)


def test_drive_admit_overlap_probes_empty_queue_fails_closed() -> None:
    """An empty probe queue cannot produce the bracket — refuse before probing."""
    plan = _plan()
    plan.probes = []  # simulate a placement queue with no candidates

    def _boom(spec: sk.AdmitOverlapProbeSpec, *, seed: int):  # pragma: no cover
        raise AssertionError("no probe may fire on an empty queue")

    with pytest.raises(RuntimeError, match="signal structurally unobtainable"):
        sk._drive_admit_overlap_probes(plan, seed=3, probe_fn=_boom, anchor_hold_fn=_anchor_held)


def test_drive_admit_overlap_probes_anchor_not_held_fails_closed() -> None:
    """Anchor precondition: the operator must hold the anchor; unverified ⇒ fail closed."""
    plan = _plan()

    def _boom(spec: sk.AdmitOverlapProbeSpec, *, seed: int):  # pragma: no cover
        raise AssertionError("no probe may fire when the anchor is not verifiably held")

    # (a) Empty holder scan (locks disabled / stack down) — cannot verify.
    with pytest.raises(RuntimeError, match="anchor-hold precondition cannot be verified"):
        sk._drive_admit_overlap_probes(
            plan, seed=3, probe_fn=_boom, anchor_hold_fn=lambda: {}
        )
    # (b) Anchor role holds a DIFFERENT instance — not the anchor placement.
    with pytest.raises(RuntimeError, match="anchor-hold precondition cannot be verified"):
        sk._drive_admit_overlap_probes(
            plan, seed=3, probe_fn=_boom,
            anchor_hold_fn=lambda: {"ingest_long_context": [2]},
        )
    # (c) Anchor role absent from the scan entirely.
    with pytest.raises(RuntimeError, match="anchor-hold precondition cannot be verified"):
        sk._drive_admit_overlap_probes(
            plan, seed=3, probe_fn=_boom, anchor_hold_fn=lambda: {"frontdoor": [0]}
        )


def test_drive_admit_overlap_probes_one_sided_signal_fails_closed() -> None:
    """All-queue (or all-admit) expectations are structurally unobtainable.

    Mirrors the handoff's stale-default hazard: with the FULL ingest instance as
    anchor, every candidate overlaps → all probes expect QUEUE → the bracket has
    no disjoint side. The driver must refuse rather than "still report".
    """
    full_anchor_regions = {
        ("ingest_long_context", 1): frozenset({"q0", "q1", "q2", "q3"}),
        ("frontdoor", 0): frozenset({"q0", "q1"}),
        ("vision_escalation", 0): frozenset({"q2", "q3"}),
    }
    plan = sk.build_step2_smoke_plan(full_anchor_regions, topology_hash="test-topo")
    assert len(plan.probes) > 0
    assert all(p.expected_decision == sk.DECISION_QUEUE for p in plan.probes)

    def _boom(spec: sk.AdmitOverlapProbeSpec, *, seed: int):  # pragma: no cover
        raise AssertionError("no probe may fire on a structurally unobtainable plan")

    with pytest.raises(RuntimeError, match="signal structurally unobtainable"):
        sk._drive_admit_overlap_probes(
            plan, seed=3, probe_fn=_boom,
            anchor_hold_fn=lambda: {"ingest_long_context": [0]},
        )


def test_verify_anchor_held_pure_pass_and_fail_paths() -> None:
    plan = _plan()
    # Pure checker: no I/O, no injection seam needed — holder map is the input.
    # The SYNTH anchor is ingest#1 (node0 half — corrected default).
    sk._verify_anchor_held(plan, {"ingest_long_context": [1]})
    sk._verify_anchor_held(plan, {"ingest_long_context": [1, 2]})
    with pytest.raises(RuntimeError, match="anchor-hold precondition cannot be verified"):
        sk._verify_anchor_held(plan, {})
    with pytest.raises(RuntimeError, match="anchor-hold precondition cannot be verified"):
        sk._verify_anchor_held(plan, {"ingest_long_context": [0]})


def test_verify_probe_signal_pure_refuses_one_sided_or_empty() -> None:
    plan = _plan()
    sk._verify_probe_signal(plan)  # 3 admit + 3 queue → valid bracket
    plan.probes = []
    with pytest.raises(RuntimeError, match="signal structurally unobtainable"):
        sk._verify_probe_signal(plan)


# ── re-bench driver (injected sample_fn) ──────────────────────────────────────


def test_drive_rebench_pairs_collects_samples() -> None:
    plan = _plan()
    # 4 disjoint within-role vision pairs.
    assert len(plan.rebench_pairs) == 4
    pair_ids = {p.pair_id for p in plan.rebench_pairs}
    assert pair_ids == {"q0+q2q3", "q2+q3", "q0+q2", "q0+q3"}

    calls: list[tuple[str, int]] = []

    def fake_sample(pair: sk.RebenchPairSpec, *, seed: int, target_samples: int):
        calls.append((pair.pair_id, target_samples))
        assert pair.transport == sk.PLACEMENT_QUEUE_TRANSPORT
        # A clean super-linear allow: mean ~1.20, tiny CV.
        return [1.200 + 0.001 * i for i in range(target_samples)]

    samples = sk._drive_rebench_pairs(plan, seed=3, sample_fn=fake_sample)
    assert set(samples) == pair_ids
    assert all(len(v) == sk.DEFAULT_TARGET_SAMPLES for v in samples.values())
    assert all(ts == sk.DEFAULT_TARGET_SAMPLES for _, ts in calls)

    rows = sk.aggregate_rebench(plan.rebench_pairs, samples, floor=plan.floor,
                                cv_threshold=plan.cv_threshold)
    assert len(rows) == 4
    for row in rows:
        assert row["n_samples"] == sk.DEFAULT_TARGET_SAMPLES
        assert row["met_sample_target"] is True
        assert row["verdict"] == "allow"
        assert row["clean"] is True
        assert row["ratified_allow"] is True
        # Bench index = model/quant, never role.
        assert row["model"] == sk.VISION_DEFAULT_MODEL
        assert row["quant"] == sk.VISION_DEFAULT_QUANT
        assert row["ratio_delta_vs_prior"] is not None


# ── execute_step2_smoke: aggregation + artifact write (injected fns) ───────────


def test_execute_step2_smoke_writes_artifact_with_injected_fns(tmp_path: Path) -> None:
    plan = _plan()
    out = tmp_path / "nested" / "smoke_report.json"

    def fake_probe(spec: sk.AdmitOverlapProbeSpec, *, seed: int):
        if spec.expected_decision == sk.DECISION_ADMIT:
            return {"answer": "OK"}
        return {"answer": "", "error": "contention_denied 503"}

    def fake_sample(pair: sk.RebenchPairSpec, *, seed: int, target_samples: int):
        return [1.15 + 0.0005 * i for i in range(target_samples)]

    report = sk.execute_step2_smoke(
        plan,
        output_path=out,
        seed=42,
        probe_fn=fake_probe,
        sample_fn=fake_sample,
        anchor_hold_fn=_anchor_held,
    )

    assert report["kind"] == "shapekeyed_step2_smoke_report"
    assert report["smoke_pass"] is True
    assert report["admit_overlap"]["all_pass"] is True
    assert report["admit_overlap"]["n_pass"] == 6
    assert report["n_rebench_ratified_allow"] == 4
    assert report["observation_only"] is True

    # Artifact written (nested dirs created) and round-trips to the same report.
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == report


# ── double-gate dry-run: unchanged + bridge never reached ─────────────────────


def test_env_gate_closed_returns_dry_run_and_never_calls_bridge(monkeypatch) -> None:
    monkeypatch.delenv(sk.SHAPEKEYED_STEP2_INFERENCE_ENV, raising=False)
    plan = _plan()

    # execute=False -> dry-run, no inference.
    res = sk.run_shapekeyed_step2_smoke(plan, execute=False)
    assert res["mode"] == "dry_run"
    assert res["inference_ran"] is False
    assert res["n_probes"] == 6
    assert res["n_rebench_pairs"] == 4
    assert res["plan"]["kind"] == "shapekeyed_step2_smoke_plan"

    # execute=True but env gate closed -> STILL dry-run, and the bridge must not be
    # invoked. Poison the bridge to prove the dry-run path never reaches it.
    def _boom(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("execute_step2_smoke must not run when the env gate is closed")

    monkeypatch.setattr(sk, "execute_step2_smoke", _boom)
    res2 = sk.run_shapekeyed_step2_smoke(plan, execute=True)
    assert res2["mode"] == "dry_run"
    assert res2["inference_ran"] is False
    assert sk.SHAPEKEYED_STEP2_INFERENCE_ENV in res2["reason"]
