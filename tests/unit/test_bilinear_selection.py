"""Model-free controls for opt-in DAR-4 candidate ranking."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from orchestration.repl_memory.bilinear_scorer import (
    MODEL_FEATURE_DIM,
    PROMPT_FEATURE_DIM,
    BilinearScorer,
    ModelFeatures,
)
from orchestration.repl_memory.hybrid_router import HybridRouter
from orchestration.repl_memory.retriever import (
    RetrievalConfig,
    RuleBasedRouter,
    TwoPhaseRetriever,
)


@dataclass
class _Memory:
    action: str
    q_value: float
    embedding: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))
    similarity_score: float = 1.0
    context: dict = field(default_factory=dict)
    update_count: int = 3


def _retriever(memories, **config_overrides):
    store = MagicMock()
    store.retrieve_by_similarity.return_value = memories
    embedder = MagicMock()
    config = RetrievalConfig(
        semantic_k=10,
        min_similarity=0.3,
        min_q_value=0.0,
        q_weight=1.0,
        cost_lambda=0.0,
        top_n=10,
        **config_overrides,
    )
    return TwoPhaseRetriever(store, embedder, config), store


def test_candidate_scorer_changes_ranking_but_not_empirical_confidence():
    memories = [
        _Memory("frontdoor:direct", 0.9, context={"role": "frontdoor"}),
        _Memory("worker_general:direct", 0.1, context={"role": "worker_general"}),
    ]
    retriever, _ = _retriever(memories)
    query = np.array([1.0, 0.0])

    incumbent = retriever._retrieve(query, action_type="routing")
    scored = retriever._retrieve(
        query,
        action_type="routing",
        candidate_value=lambda memory: {
            "frontdoor": 0.1,
            "worker_general": 0.9,
        }[memory.context["role"]],
    )

    assert incumbent[0].memory.action == "frontdoor:direct"
    assert scored[0].memory.action == "worker_general:direct"
    assert scored[0].q_value == pytest.approx(0.1)
    assert scored[0].selection_q_value == pytest.approx(0.9)
    assert scored[0].selection_source == "candidate"
    assert {r.q_confidence for r in scored} == {0.5}


@pytest.mark.parametrize(
    "bad_value",
    [None, float("nan"), float("inf"), -0.1, 1.1, True, np.bool_(False), "0.99"],
)
def test_invalid_or_unsupported_candidate_values_preserve_q_order(bad_value):
    memories = [_Memory("frontdoor", 0.9), _Memory("worker_general", 0.1)]
    retriever, _ = _retriever(memories)
    query = np.array([1.0, 0.0])

    baseline = retriever._retrieve(query)
    controlled = retriever._retrieve(query, candidate_value=lambda _memory: bad_value)

    assert [r.memory.action for r in controlled] == [r.memory.action for r in baseline]
    assert all(r.selection_source == "q" for r in controlled)
    assert [r.selection_q_value for r in controlled] == [r.q_value for r in controlled]


def test_candidate_hook_runs_only_after_similarity_eligibility_and_keeps_filters():
    memories = [
        _Memory("frontdoor", 0.8, embedding=np.array([1.0, 0.0])),
        _Memory("worker_general", 0.8, embedding=np.array([0.0, 1.0])),
    ]
    retriever, store = _retriever(memories)
    scorer_calls = []

    results = retriever._retrieve(
        np.array([1.0, 0.0]),
        action_type="routing",
        candidate_value=lambda memory: scorer_calls.append(memory.action) or 0.99,
    )

    store.retrieve_by_similarity.assert_called_once()
    args, kwargs = store.retrieve_by_similarity.call_args
    np.testing.assert_array_equal(args[0], np.array([1.0, 0.0]))
    assert kwargs == {
        "k": 10,
        "action_type": "routing",
        "min_q_value": 0.0,
    }
    assert scorer_calls == ["frontdoor"]
    assert [r.memory.action for r in results] == ["frontdoor"]


def _trained_scorer():
    scorer = BilinearScorer(
        {
            "frontdoor": ModelFeatures(role="frontdoor", baseline_quality=0.8),
            "worker_general": ModelFeatures(role="worker_general", baseline_tps=20.0),
        }
    )
    prompt = np.zeros(PROMPT_FEATURE_DIM, dtype=np.float32)
    scorer.update(prompt, "frontdoor", reward=1.0)
    return scorer


def test_random_initialized_scorer_is_not_selection_ready():
    scorer = _trained_scorer()
    scorer.updates = 0
    assert not scorer.selection_ready


def test_saved_scorer_requires_versioned_finite_trained_state(tmp_path):
    scorer = _trained_scorer()
    assert scorer.selection_ready
    path = tmp_path / "trained.npz"
    scorer.save(str(path))

    restored = _trained_scorer()
    restored.updates = 0
    assert restored.load(str(path))
    assert restored.selection_ready

    legacy = tmp_path / "legacy.npz"
    np.savez(
        legacy,
        W=np.zeros((MODEL_FEATURE_DIM, PROMPT_FEATURE_DIM), dtype=np.float32),
        b=np.array([0.0]),
        updates=np.array([10]),
    )
    preserved_weights = restored.W.copy()
    preserved_updates = restored.updates
    assert not restored.load(str(legacy))
    assert restored.selection_ready
    np.testing.assert_array_equal(restored.W, preserved_weights)
    assert restored.updates == preserved_updates

    untrained = _trained_scorer()
    untrained.updates = 0
    assert not untrained.load(str(legacy))
    assert not untrained.selection_ready


@pytest.mark.parametrize(
    "metadata",
    [
        {
            "format_version": np.array([1.5]),
            "model_feature_dim": np.array([8]),
            "prompt_feature_dim": np.array([8]),
            "updates": np.array([5]),
        },
        {
            "format_version": np.array([True]),
            "model_feature_dim": np.array([8]),
            "prompt_feature_dim": np.array([8]),
            "updates": np.array([5]),
        },
        {
            "format_version": np.array([1]),
            "model_feature_dim": np.array([8]),
            "prompt_feature_dim": np.array([8]),
            "updates": np.array([2.5]),
        },
    ],
)
def test_loader_rejects_noninteger_metadata_without_mutating_trained_state(tmp_path, metadata):
    scorer = _trained_scorer()
    old_weights = scorer.W.copy()
    old_bias = scorer.b.copy()
    old_updates = scorer.updates
    path = tmp_path / "bad-metadata.npz"
    np.savez(
        path,
        W=np.zeros((MODEL_FEATURE_DIM, PROMPT_FEATURE_DIM), dtype=np.float32),
        b=np.array([0.0]),
        **metadata,
    )

    assert not scorer.load(str(path))
    assert scorer.updates == old_updates
    assert scorer.b == old_bias
    np.testing.assert_array_equal(scorer.W, old_weights)


def test_loader_treats_malformed_archive_as_fallback_without_mutation(tmp_path):
    scorer = _trained_scorer()
    old_weights = scorer.W.copy()
    old_updates = scorer.updates
    path = tmp_path / "truncated.npz"
    path.write_bytes(b"PK\x03\x04truncated")

    assert not scorer.load(str(path))
    assert scorer.updates == old_updates
    np.testing.assert_array_equal(scorer.W, old_weights)


def test_hybrid_router_requires_explicit_optin_and_supported_role():
    scorer = _trained_scorer()
    router = object.__new__(HybridRouter)
    router._bilinear_scorer = scorer
    router.bilinear_selection_enabled = False
    router.retriever = SimpleNamespace(
        _extract_role_from_memory=lambda memory: memory.context.get("role", "")
    )
    task = {"objective": "write a parser", "task_type": "code"}

    assert router._build_bilinear_candidate_value(task) is None
    router.bilinear_selection_enabled = True
    value = router._build_bilinear_candidate_value(task)
    assert value is not None
    assert np.isfinite(value(_Memory("frontdoor", 0.4, context={"role": "frontdoor"})))
    assert value(_Memory("unknown", 0.4, context={"role": "missing_alias"})) is None


def test_hybrid_router_rejects_untrained_injected_scorer():
    scorer = _trained_scorer()
    scorer.updates = 0
    router = object.__new__(HybridRouter)
    router._bilinear_scorer = scorer
    router.bilinear_selection_enabled = True
    router.retriever = SimpleNamespace(
        _extract_role_from_memory=lambda memory: memory.context.get("role", "")
    )

    assert router._build_bilinear_candidate_value({"objective": "task"}) is None


def _route_fixture(*, scorer=None, selection_enabled=False):
    memories = [
        _Memory("frontdoor:direct", 0.53, context={"role": "frontdoor"}),
        _Memory("worker_general:direct", 0.50, context={"role": "worker_general"}),
        _Memory("architect_general:direct", 0.53, context={"role": "architect_general"}),
    ]
    config = RetrievalConfig(
        semantic_k=10,
        min_similarity=0.3,
        min_q_value=0.0,
        q_weight=1.0,
        cost_lambda=0.0,
        top_n=10,
        confidence_min_neighbors=3,
        confidence_threshold=0.5,
        risk_control_enabled=True,
        calibrated_confidence_threshold=0.5,
        risk_gate_min_samples=3,
        risk_gate_rollout_ratio=1.0,
        risk_budget_guardrail_min_events=100,
    )
    store = MagicMock()
    store.retrieve_by_similarity.return_value = memories
    embedder = MagicMock()
    embedder.embed_task_ir.return_value = np.array([1.0, 0.0])
    retriever = TwoPhaseRetriever(store, embedder, config)
    router = HybridRouter(
        retriever,
        RuleBasedRouter(routing_hints=[]),
        bilinear_scorer=scorer,
        bilinear_selection_enabled=selection_enabled,
    )
    router._live_routing_roles = {"frontdoor", "worker_general", "architect_general"}
    return router


def _controlled_trained_scorer(task_ir):
    from orchestration.repl_memory.bilinear_scorer import extract_prompt_features

    scorer = BilinearScorer(
        {
            "frontdoor": ModelFeatures(role="frontdoor", baseline_tps=0.0),
            "worker_general": ModelFeatures(role="worker_general", baseline_tps=40.0),
            "architect_general": ModelFeatures(role="architect_general", baseline_tps=0.0),
        }
    )
    prompt = extract_prompt_features(task_ir)
    scorer.update(prompt, "worker_general", reward=1.0)
    # A controlled, finite fixture makes worker_general rank first. This is a
    # code-path proof only; it carries no learned-quality claim.
    scorer.W[:] = 0.0
    scorer.W[0, 0] = 1.0
    scorer.b = np.float32(0.0)
    assert scorer.selection_ready
    return scorer


def test_hybrid_router_route_selection_is_optin_trained_and_risk_stable():
    task = {
        "task_id": "bilinear-selection-control",
        "task_type": "code",
        "objective": "write parser",
    }
    trained = _controlled_trained_scorer(task)

    incumbent_router = _route_fixture(scorer=trained, selection_enabled=False)
    incumbent_route, incumbent_strategy = incumbent_router.route(task)
    assert (incumbent_route, incumbent_strategy) == (["frontdoor"], "learned")
    assert incumbent_router.last_decision_meta["risk_gate_action"] == "accept"

    selected_router = _route_fixture(scorer=trained, selection_enabled=True)
    selected_route, selected_strategy = selected_router.route(task)
    assert (selected_route, selected_strategy) == (["worker_general"], "learned")
    assert selected_router.last_decision_meta["risk_gate_action"] == "accept"
    assert selected_router.last_decision_meta["q_robust_confidence"] == pytest.approx(0.53)

    untrained = _controlled_trained_scorer(task)
    untrained.updates = 0
    untrained_router = _route_fixture(scorer=untrained, selection_enabled=True)
    assert untrained_router.route(task) == (["frontdoor"], "learned")
