from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.autopilot import collect_multitier_incumbent_baseline as collector


def _result(*, tier: int = 2, errors: int = 0) -> SimpleNamespace:
    n = collector.EXPECTED_N[tier]
    return SimpleNamespace(
        tier=tier,
        n_questions=n,
        reliability=1.0,
        question_results=[{"qid": f"q{i}", "correct": True} for i in range(n)],
        details={
            "errors": errors,
            "scoring_errors": 0,
            "eval_client_transport_timeout_count": 0,
            "eval_backend_drain_failure_count": 0,
            "eval_orphan_contamination_count": 0,
            "eval_overflow_count": 0,
            "eval_contaminated_by_abandoned_requests": False,
            "eval_execution_instrument_id": collector.EVAL_EXECUTION_INSTRUMENT_ID,
            "eval_scoring_schedule_id": collector.EVAL_SCORING_SCHEDULE_ID,
        },
    )


def test_clean_tier_result_is_admissible() -> None:
    collector._validate_result(_result(), 2)


def test_scorer_or_transport_error_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="errors=1"):
        collector._validate_result(_result(errors=1), 2)


def test_wrong_tier_size_fails_closed() -> None:
    result = _result(tier=3)
    result.n_questions -= 1

    with pytest.raises(RuntimeError, match="n_questions"):
        collector._validate_result(result, 3)


def test_state_readiness_rejects_unresolved_in_flight_trial() -> None:
    readiness = collector._state_collection_readiness(
        {
            "paused": True,
            "in_flight_trial": {"trial_id": 1505, "action": {"type": "deep_eval"}},
        }
    )

    assert readiness == {
        "autopilot_paused": True,
        "in_flight_trial_clear": False,
        "in_flight_trial_id": 1505,
    }


def test_state_readiness_accepts_paused_quiescent_state() -> None:
    assert collector._state_collection_readiness(
        {"paused": True, "in_flight_trial": None}
    ) == {
        "autopilot_paused": True,
        "in_flight_trial_clear": True,
        "in_flight_trial_id": None,
    }


def test_source_guard_covers_learned_routing_instrument() -> None:
    guarded = {
        str(path.relative_to(collector.REPO_ROOT)) for path in collector.SOURCE_PATHS
    }

    assert {
        "orchestration/repl_memory/hybrid_router.py",
        "orchestration/repl_memory/embedder.py",
        "orchestration/repl_memory/episodic_store.py",
        "orchestration/repl_memory/parallel_embedder.py",
        "orchestration/repl_memory/q_scorer.py",
        "src/api/routes/chat_review.py",
        "src/api/__init__.py",
        "src/api/routes/config.py",
        "src/api/routes/chat_pipeline/routing.py",
        "src/api/routes/chat_pipeline/routing_decision.py",
        "src/api/services/memrl.py",
        "src/features.py",
        "src/registry/stack_priors.py",
        "src/roles.py",
        "src/runtime/config_attestation.py",
        "orchestration/derived/stack_priors.yaml",
    } <= guarded


def test_live_config_identity_requires_and_covers_full_worker_roster(monkeypatch) -> None:
    expected = [101, 102, 103]
    monkeypatch.setattr(collector, "_expected_api_worker_pids", lambda: expected)
    monkeypatch.setattr(
        "src.runtime.config_attestation.read_config_attestations",
        lambda pids: {
            pid: {"pid": pid, "flags": {"graph_router": True}, "sources": {}}
            for pid in pids
        },
    )
    monkeypatch.setattr(
        collector,
        "_worker_env",
        lambda pid, names: {name: f"same-{name}" for name in names},
    )

    identity = collector._live_config_identity()

    assert identity["worker_pids"] == expected
    assert identity["expected_worker_pids"] == expected
    assert identity["worker_coverage_complete"] is True


def test_live_config_identity_rejects_missing_worker(monkeypatch) -> None:
    monkeypatch.setattr(collector, "_expected_api_worker_pids", lambda: [101, 102])
    monkeypatch.setattr(
        "src.runtime.config_attestation.read_config_attestations",
        lambda pids: {101: {"pid": 101, "flags": {}, "sources": {}}},
    )
    monkeypatch.setattr(collector, "_worker_env", lambda pid, names: {})

    with pytest.raises(RuntimeError, match=r"missing=\[102\]"):
        collector._live_config_identity()


def test_live_config_identity_rejects_worker_divergence(monkeypatch) -> None:
    monkeypatch.setattr(collector, "_expected_api_worker_pids", lambda: [101, 102])
    monkeypatch.setattr(
        "src.runtime.config_attestation.read_config_attestations",
        lambda pids: {
            101: {"pid": 101, "flags": {"memrl": True}, "sources": {}},
            102: {"pid": 102, "flags": {"memrl": False}, "sources": {}},
        },
    )
    monkeypatch.setattr(collector, "_worker_env", lambda pid, names: {})

    with pytest.raises(RuntimeError, match="workers disagree"):
        collector._live_config_identity()
