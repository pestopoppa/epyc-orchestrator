#!/usr/bin/env python3
"""Unit tests for scripts/autopilot/screening_tier_runner.py (B3 / RM-3).

Coverage is entirely INFERENCE-FREE: plan->queue resolution over a synthetic
pool-gen fixture + the real near-miss corpus manifest metadata, dedup/cap/prune/
priority logic, the pure FA/FR/CR scoring helpers, placement-queue transport
assertions (never /chat), and the env-flag gate (flag OFF => dry-run; flag ON =>
routes to a MOCKED execute bridge). The execution bridge itself is never called
with a real EvalTower / orchestrator here.
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
    / "scripts" / "autopilot" / "screening_tier_runner.py"
)
_SPEC = importlib.util.spec_from_file_location("screening_tier_runner", _MODULE_PATH)
runner = importlib.util.module_from_spec(_SPEC)
sys.modules["screening_tier_runner"] = runner  # register before exec (dataclasses)
_SPEC.loader.exec_module(runner)

# The runner puts ORCH_ROOT on sys.path at import; review_policy_trials is now
# importable for the tests that drive the real planner.
rpt = runner._load_review_policy_trials()

REAL_MANIFEST = Path("/mnt/raid0/llm/datasets/nearmiss-corpus-v1/manifest.json")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _pool_gen_output() -> dict:
    """Synthetic reviewer_pool_gen output covering every resolver branch."""
    return {
        "provenance": {"schema_version": "1", "registry_sha256": "deadbeef"},
        "pairings": [
            {
                "pairing_id": "archA__revB__grd",
                "architect": "archA",
                "reviewer": "revB",
                "grader": "grd",
                "cross_family_preferred": True,
                "self_review": False,
                "coresidency": {"fits": True},
                "staged_involved": True,
                "anchor_arm": None,
            },
            {
                "pairing_id": "archA__archA__grd",
                "architect": "archA",
                "reviewer": "archA",
                "grader": "grd",
                "cross_family_preferred": False,
                "self_review": True,
                "coresidency": {"fits": True},
                "staged_involved": False,
                "anchor_arm": "A1",
            },
            {
                "pairing_id": "archA__revC__grd",
                "architect": "archA",
                "reviewer": "revC",
                "grader": "grd",
                "cross_family_preferred": True,
                "self_review": False,
                "coresidency": {"fits": False},  # unfit -> pruned by default
                "staged_involved": False,
                "anchor_arm": None,
            },
        ],
    }


def _manifest() -> dict:
    """Real corpus manifest metadata when present, else a synthetic stand-in."""
    if REAL_MANIFEST.exists():
        return rpt.load_corpus_manifest(REAL_MANIFEST)
    return {
        "corpus_id": "nearmiss-v1",
        "total_rows": 100,
        "content_sha256": "synthetic",
        "schema_version": "nearmiss_corpus_row.v1",
        "counts": {"per_domain": {"code": 40, "thinking": 30}},
        "gate_worthy_multi_oracle": 60,
    }


def _plan(**over) -> dict:
    """A minimal _screening_tier_plan dict (bypasses the real planner)."""
    queue = over.pop("queue", None)
    if queue is None:
        queue = [
            {
                "pairing_id": "archA__revB__grd",
                "architect": "archA",
                "reviewer": "revB",
                "grader": "grd",
                "anchor_arm": None,
                "self_review": False,
                "cross_family": True,
                "n": 12,
                "eval_tier": "T0",
                "corpus_id": "nearmiss-v1",
                "domain": "all",
                "dispatch": "placement_queue",
            }
        ]
    plan = {
        "kind": "screening_tier_plan",
        "corpus_slice": {
            "corpus_id": "nearmiss-v1",
            "domain": "all",
            "n_rows": 11516,
            "content_sha256": "abc123",
        },
        "eval_tier": "T0",
        "per_pairing_n": 12,
        "queue": queue,
        "provenance": {"corpus_content_sha256": "abc123"},
        "notes": [],
    }
    plan.update(over)
    return plan


# --------------------------------------------------------------------------- #
# Plan resolution via the REAL planner (integration on synthetic pool + manifest)
# --------------------------------------------------------------------------- #
def test_resolve_expands_plan_to_jobs_via_real_planner():
    pool = _pool_gen_output()
    manifest = _manifest()
    plan_obj, error = rpt.plan_screening_tier(
        pool, corpus_manifest=manifest, per_pairing_n=8, eval_tier="T0"
    )
    assert error is None and plan_obj is not None

    resolved = runner.resolve_screening_queue(plan_obj.to_dict(), pool)

    # 3 pairings, one is coresidency-unfit -> pruned -> 2 jobs.
    assert len(resolved.jobs) == 2
    assert resolved.n_pruned_unfit == 1
    corpus_id = manifest.get("corpus_id", "nearmiss-v1")
    for job in resolved.jobs:
        assert job.n == 8
        assert job.eval_tier == "T0"
        assert job.corpus_id == corpus_id
        assert job.transport == "placement_queue"
        assert job.request_priority == "background"
        assert job.workload_class == "eval_batch"
        assert job.force_bindings()["force_role"] == job.reviewer


def test_corpus_manifest_metadata_is_real_when_present():
    if not REAL_MANIFEST.exists():
        pytest.skip("real near-miss corpus manifest not on this host")
    manifest = rpt.load_corpus_manifest(REAL_MANIFEST)
    pool = _pool_gen_output()
    plan_obj, error = rpt.plan_screening_tier(pool, corpus_manifest=manifest)
    assert error is None
    resolved = runner.resolve_screening_queue(plan_obj.to_dict(), pool)
    assert resolved.corpus_slice["corpus_id"] == "nearmiss-v1"
    assert resolved.corpus_slice["content_sha256"]  # non-empty real sha
    assert resolved.jobs[0].corpus_content_sha256 == resolved.corpus_slice["content_sha256"]


# --------------------------------------------------------------------------- #
# Transport discipline (RM-3): placement queue, NEVER /chat
# --------------------------------------------------------------------------- #
def test_transport_is_placement_queue_never_chat():
    resolved = runner.resolve_screening_queue(_plan(), _pool_gen_output())
    d = resolved.to_dict()
    # The transport-bearing surfaces (jobs + transport summary) must never name a
    # /chat endpoint; the human-readable `notes` legitimately mention "NEVER /chat"
    # so they are excluded from the substring check.
    transport_blob = json.dumps({"jobs": d["jobs"], "transport": d["transport"]})
    assert "/chat" not in transport_blob
    assert "/v1/chat" not in transport_blob
    assert resolved.transport_summary()["uses_chat_endpoint"] is False
    for job in resolved.jobs:
        jd = job.to_dict()
        assert jd["transport"] == "placement_queue"
        assert jd["request_priority"] == "background"
        assert jd["workload_class"] == "eval_batch"


# --------------------------------------------------------------------------- #
# Dedup / cap / prune / priority
# --------------------------------------------------------------------------- #
def test_dedup_by_pairing_id():
    entry = _plan()["queue"][0]
    plan = _plan(queue=[dict(entry), dict(entry), dict(entry)])
    resolved = runner.resolve_screening_queue(plan, _pool_gen_output())
    assert len(resolved.jobs) == 1
    assert resolved.n_deduped == 2
    assert resolved.pairings_considered == 3


def test_cap_per_pairing_bounds_n():
    resolved = runner.resolve_screening_queue(
        _plan(), _pool_gen_output(), cap_per_pairing=3
    )
    assert all(job.n == 3 for job in resolved.jobs)
    assert resolved.per_pairing_n == 3


def test_cap_per_pairing_never_raises_n():
    # cap larger than plan n must not inflate n.
    resolved = runner.resolve_screening_queue(
        _plan(), _pool_gen_output(), cap_per_pairing=999
    )
    assert all(job.n == 12 for job in resolved.jobs)


def test_prune_unfit_coresidency_toggle():
    pool = _pool_gen_output()
    queue = [
        {"pairing_id": "archA__revB__grd", "reviewer": "revB", "grader": "grd", "n": 5,
         "eval_tier": "T0"},
        {"pairing_id": "archA__revC__grd", "reviewer": "revC", "grader": "grd", "n": 5,
         "eval_tier": "T0"},  # unfit in pool fixture
    ]
    pruned = runner.resolve_screening_queue(_plan(queue=queue), pool, prune_unfit=True)
    assert {j.pairing_id for j in pruned.jobs} == {"archA__revB__grd"}
    assert pruned.n_pruned_unfit == 1

    kept = runner.resolve_screening_queue(_plan(queue=queue), pool, prune_unfit=False)
    assert {j.pairing_id for j in kept.jobs} == {"archA__revB__grd", "archA__revC__grd"}
    assert kept.n_pruned_unfit == 0


def test_unknown_coresidency_is_never_pruned():
    # A plan pairing absent from pool-gen has unknown fit -> must be KEPT.
    queue = [{"pairing_id": "ghost__x__grd", "reviewer": "x", "grader": "grd", "n": 4,
              "eval_tier": "T0"}]
    resolved = runner.resolve_screening_queue(_plan(queue=queue), _pool_gen_output())
    assert len(resolved.jobs) == 1
    assert resolved.jobs[0].coresidency_fits is None


def test_priority_orders_anchor_and_staged_first():
    pool = _pool_gen_output()
    # Same three pairings; make coresidency all-fit so none are pruned.
    for p in pool["pairings"]:
        p["coresidency"] = {"fits": True}
    queue = [
        {"pairing_id": "archA__revC__grd"},   # plain cross-family
        {"pairing_id": "archA__archA__grd"},  # anchor A1 + self-review
        {"pairing_id": "archA__revB__grd"},   # staged, cross-family
    ]
    resolved = runner.resolve_screening_queue(_plan(queue=queue), pool, priority=True)
    order = [j.pairing_id for j in resolved.jobs]
    # anchor first, then staged, then the plain cross-family pairing.
    assert order[0] == "archA__archA__grd"   # anchor_arm A1 wins
    assert order[1] == "archA__revB__grd"    # staged_involved
    assert order[2] == "archA__revC__grd"


def test_max_pairings_truncates_after_priority():
    pool = _pool_gen_output()
    for p in pool["pairings"]:
        p["coresidency"] = {"fits": True}
    queue = [
        {"pairing_id": "archA__revC__grd"},
        {"pairing_id": "archA__archA__grd"},
        {"pairing_id": "archA__revB__grd"},
    ]
    resolved = runner.resolve_screening_queue(
        _plan(queue=queue), pool, max_pairings=2, priority=True
    )
    assert len(resolved.jobs) == 2
    assert resolved.n_truncated == 1
    # highest-priority survive: anchor + staged.
    assert {j.pairing_id for j in resolved.jobs} == {
        "archA__archA__grd", "archA__revB__grd"
    }


# --------------------------------------------------------------------------- #
# Pure scoring helpers (no inference)
# --------------------------------------------------------------------------- #
def test_load_row_ids_dedupes_and_ignores_comments(tmp_path):
    path = tmp_path / "row_ids.txt"
    path.write_text(
        "\n"
        "# comment\n"
        "r1\n"
        "r2  # trailing comment\n"
        "r1\n"
        "\n"
        "r3\n"
    )
    assert runner.load_row_ids(path) == ["r1", "r2", "r3"]
    summary = runner.row_id_filter_summary(path)
    assert summary["row_id_filter_n"] == 3
    assert summary["row_id_filter_sha256"]


def test_attach_row_id_filter_is_idempotent(tmp_path):
    path = tmp_path / "row_ids.txt"
    path.write_text("r1\nr2\n")
    plan = _plan()
    once = runner.attach_row_id_filter(plan, path)
    twice = runner.attach_row_id_filter(once, path)
    assert twice["corpus_slice"]["row_id_filter_n"] == 2
    assert twice["provenance"]["row_id_filter_n"] == 2
    note = "row-id allowlist attached; live execution must filter to this slice."
    assert twice["notes"].count(note) == 1


def test_is_judgeable_row():
    good = {"candidate": "42", "gold_label": "accept", "gold_confidence": "multi_oracle"}
    assert runner.is_judgeable_row(good)
    assert not runner.is_judgeable_row({**good, "candidate": None})
    assert not runner.is_judgeable_row({**good, "gold_label": "None"})
    assert not runner.is_judgeable_row({**good, "gold_confidence": "observation"})
    assert runner.is_judgeable_row({**good, "gold_confidence": "single_oracle"})


def test_gate_from_gold_label():
    assert runner.gate_from_gold_label("accept") == "pass"
    assert runner.gate_from_gold_label("reject") == "fail"
    assert runner.gate_from_gold_label("None") is None
    assert runner.gate_from_gold_label(None) is None


def test_consistency_rate():
    decisions = [
        {"decision": "approve", "gate": "pass"},   # agree
        {"decision": "reject", "gate": "fail"},    # agree
        {"decision": "approve", "gate": "fail"},   # FA (disagree)
        {"decision": "reject", "gate": "pass"},    # FR (disagree)
        {"decision": "approve", "gate": None},     # inconclusive -> excluded
    ]
    assert runner.consistency_rate(decisions) == pytest.approx(0.5)
    assert runner.consistency_rate([]) is None
    assert runner.consistency_rate([{"decision": "approve", "gate": None}]) is None


def test_summarize_pairing_computes_fa_fr_cr():
    job = runner.TrialJobSpec(
        pairing_id="archA__revB__grd", architect="archA", reviewer="revB",
        grader="grd", anchor_arm=None, self_review=False, cross_family=True,
        staged_involved=True, n=4, eval_tier="T0", corpus_id="nearmiss-v1",
        domain="all", corpus_content_sha256="abc", corpus_n_rows=10,
        coresidency_fits=True, priority_rank=0,
    )
    decisions = [
        {"decision": "approve", "gate": "pass", "latency_ms": 10.0},
        {"decision": "reject", "gate": "fail", "latency_ms": 20.0},
        {"decision": "approve", "gate": "fail", "latency_ms": 30.0},  # FA
        {"decision": "reject", "gate": "pass", "latency_ms": 40.0},   # FR
    ]
    result = runner.summarize_pairing(job, decisions)
    assert result["reviewer_fa_rate"] == pytest.approx(0.5)   # 1 FA / 2 gate-fail
    assert result["reviewer_fr_rate"] == pytest.approx(0.5)   # 1 FR / 2 gate-pass
    assert result["consistency_rate"] == pytest.approx(0.5)
    assert result["review_decision_latency_ms"] == pytest.approx(25.0)
    assert result["n_scored"] == 4
    assert result["n_conclusive"] == 4
    assert result["transport"] == "placement_queue"
    assert result["observation_only"] is True


def test_select_rows_for_job_deterministic():
    rows = [{"row_id": f"r{i}"} for i in range(20)]
    a = runner.select_rows_for_job(rows, n=5, seed_key="42:pair")
    b = runner.select_rows_for_job(rows, n=5, seed_key="42:pair")
    assert a == b
    assert len(a) == 5
    # different pairing seed_key -> (very likely) different slice
    c = runner.select_rows_for_job(rows, n=5, seed_key="42:other")
    assert a != c
    assert runner.select_rows_for_job(rows, n=0, seed_key="x") == []
    assert runner.select_rows_for_job([], n=5, seed_key="x") == []


def test_execute_screening_queue_filters_to_row_ids(tmp_path):
    rows_path = tmp_path / "rows.jsonl"
    rows = [
        {
            "row_id": "keep_accept",
            "candidate": "ok",
            "gold_label": "accept",
            "gold_confidence": "multi_oracle",
            "task": "t1",
        },
        {
            "row_id": "drop_reject",
            "candidate": "bad",
            "gold_label": "reject",
            "gold_confidence": "multi_oracle",
            "task": "t2",
        },
        {
            "row_id": "keep_reject",
            "candidate": "bad",
            "gold_label": "reject",
            "gold_confidence": "multi_oracle",
            "task": "t3",
        },
    ]
    rows_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    row_ids_path = tmp_path / "row_ids.txt"
    row_ids_path.write_text("keep_accept\nkeep_reject\n")
    resolved = runner.resolve_screening_queue(
        _plan(per_pairing_n=10),
        _pool_gen_output(),
    )
    seen = []

    def _probe(job, row, tower):
        seen.append(row["row_id"])
        return {
            "decision": "approve" if row["gold_label"] == "accept" else "reject",
            "gate": runner.gate_from_gold_label(row["gold_label"]),
            "latency_ms": 1.0,
            "row_id": row["row_id"],
        }

    results = runner.execute_screening_queue(
        resolved,
        corpus_rows_path=rows_path,
        row_ids_path=row_ids_path,
        tower=object(),
        reviewer_probe=_probe,
    )
    assert sorted(seen) == ["keep_accept", "keep_reject"]
    assert results[0]["n_scored"] == 2
    assert results[0]["reviewer_fa_rate"] == pytest.approx(0.0)
    assert results[0]["reviewer_fr_rate"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Env-flag gate: OFF => dry-run (no inference); ON => routes to execute (mocked)
# --------------------------------------------------------------------------- #
def test_env_flag_off_returns_dry_run(monkeypatch):
    monkeypatch.delenv(runner.SCREENING_TIER_INFERENCE_ENV, raising=False)

    def _boom(*a, **k):  # execution bridge must NOT be called when flag is off
        raise AssertionError("execute_screening_queue called with inference flag OFF")

    monkeypatch.setattr(runner, "execute_screening_queue", _boom)

    out = runner.run_screening_tier(_plan(), _pool_gen_output())
    assert out["mode"] == "dry_run"
    assert out["inference_ran"] is False
    assert out["n_jobs"] == 1
    assert out["resolved_queue"]["kind"] == "resolved_screening_queue"


def test_env_flag_on_routes_to_execute_bridge(monkeypatch):
    monkeypatch.setenv(runner.SCREENING_TIER_INFERENCE_ENV, "1")

    captured = {}

    def _fake_execute(resolved, **kwargs):
        captured["resolved"] = resolved
        captured["kwargs"] = kwargs
        return [{"pairing_id": "archA__revB__grd", "reviewer_fa_rate": 0.0}]

    monkeypatch.setattr(runner, "execute_screening_queue", _fake_execute)

    out = runner.run_screening_tier(
        _plan(), _pool_gen_output(), output_path=Path("/tmp/does-not-matter.jsonl")
    )
    assert out["mode"] == "execute"
    assert out["inference_ran"] is True
    assert out["results"] == [{"pairing_id": "archA__revB__grd", "reviewer_fa_rate": 0.0}]
    assert isinstance(captured["resolved"], runner.ResolvedScreeningQueue)


@pytest.mark.parametrize("val,expected", [
    ("1", True), ("true", True), ("YES", True), ("on", True),
    ("0", False), ("", False), ("no", False),
])
def test_env_flag_semantics(monkeypatch, val, expected):
    monkeypatch.setenv(runner.SCREENING_TIER_INFERENCE_ENV, val)
    assert runner._env_flag_enabled(runner.SCREENING_TIER_INFERENCE_ENV) is expected


# --------------------------------------------------------------------------- #
# CLI __main__ (dry-run, pure)
# --------------------------------------------------------------------------- #
def test_main_dry_run_prints_resolved_queue(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv(runner.SCREENING_TIER_INFERENCE_ENV, raising=False)
    pool_path = tmp_path / "pool.json"
    pool_path.write_text(json.dumps(_pool_gen_output()))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "corpus_id": "nearmiss-v1",
        "total_rows": 500,
        "content_sha256": "cafe",
        "schema_version": "nearmiss_corpus_row.v1",
        "counts": {"per_domain": {"code": 200}},
        "gate_worthy_multi_oracle": 300,
    }))

    code = runner.main([
        "--pool-gen", str(pool_path),
        "--corpus-manifest", str(manifest_path),
        "--per-pairing-n", "6",
    ])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "resolved_screening_queue"
    assert payload["transport"]["uses_chat_endpoint"] is False
    # unfit pairing pruned -> 2 jobs, each n=6.
    assert payload["n_jobs"] == 2
    assert all(j["n"] == 6 for j in payload["jobs"])
    # dry-run writes no results file.
    assert not (tmp_path / "results.jsonl").exists()


def test_main_dry_run_accepts_row_id_filter(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv(runner.SCREENING_TIER_INFERENCE_ENV, raising=False)
    pool_path = tmp_path / "pool.json"
    pool_path.write_text(json.dumps(_pool_gen_output()))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "corpus_id": "nearmiss-v1",
        "total_rows": 500,
        "content_sha256": "cafe",
        "schema_version": "nearmiss_corpus_row.v1",
        "counts": {"per_domain": {"code": 200}},
        "gate_worthy_multi_oracle": 300,
    }))
    row_ids_path = tmp_path / "row_ids.txt"
    row_ids_path.write_text("r1\nr2\n")

    code = runner.main([
        "--pool-gen", str(pool_path),
        "--corpus-manifest", str(manifest_path),
        "--row-ids", str(row_ids_path),
    ])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    corpus_slice = payload["corpus_slice"]
    assert corpus_slice["row_id_filter_n"] == 2
    assert corpus_slice["row_id_filter_path"] == str(row_ids_path)
    assert payload["provenance"]["row_id_filter_sha256"]


def test_main_max_pairings_caps_after_priority(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv(runner.SCREENING_TIER_INFERENCE_ENV, raising=False)
    pool = _pool_gen_output()
    for p in pool["pairings"]:
        p["coresidency"] = {"fits": True}
    # Put a lower-priority plain pairing first; CLI must still keep anchor+staged.
    pool["pairings"] = [
        pool["pairings"][2],  # plain cross-family
        pool["pairings"][1],  # anchor A1
        pool["pairings"][0],  # staged
    ]
    pool_path = tmp_path / "pool.json"
    pool_path.write_text(json.dumps(pool))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "corpus_id": "nearmiss-v1",
        "total_rows": 500,
        "content_sha256": "cafe",
        "schema_version": "nearmiss_corpus_row.v1",
        "counts": {"per_domain": {"code": 200}},
        "gate_worthy_multi_oracle": 300,
    }))

    code = runner.main([
        "--pool-gen", str(pool_path),
        "--corpus-manifest", str(manifest_path),
        "--max-pairings", "2",
    ])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_jobs"] == 2
    assert [j["pairing_id"] for j in payload["jobs"]] == [
        "archA__archA__grd",
        "archA__revB__grd",
    ]


def test_main_errors_on_missing_row_id_filter(tmp_path, capsys):
    pool_path = tmp_path / "pool.json"
    pool_path.write_text(json.dumps(_pool_gen_output()))
    code = runner.main([
        "--pool-gen", str(pool_path),
        "--row-ids", str(tmp_path / "missing.txt"),
    ])
    assert code == 2
    assert "row-id file not found" in capsys.readouterr().out


def test_main_errors_on_empty_pool(tmp_path, capsys):
    pool_path = tmp_path / "empty.json"
    pool_path.write_text(json.dumps({"pairings": []}))
    code = runner.main(["--pool-gen", str(pool_path)])
    assert code == 2
    assert "no pairings" in capsys.readouterr().out
