"""Acceptance tests for experimental observed-outcome routing."""

from types import SimpleNamespace

import numpy as np
import pytest

from orchestration.repl_memory.action_outcome_scorer import ActionOutcomeScorer, CandidateScore
from orchestration.repl_memory.hybrid_router import HybridRouter


def test_observed_cells_weights_order_and_missing(monkeypatch):
    from sklearn.ensemble import HistGradientBoostingRegressor

    captured = {}

    def fit(self, x, y, sample_weight):
        captured.update(x=x, y=y, weights=sample_weight)
        return self

    monkeypatch.setattr(HistGradientBoostingRegressor, "fit", fit)
    train = np.zeros((20, 1031))
    train[0, 1029] = np.nan
    outcomes = np.ones((20, 6))
    outcomes[0, 0] = 0
    seen = np.zeros((20, 6), dtype=bool)
    seen[:, 0] = True
    seen[0, 1] = True
    scorer = ActionOutcomeScorer(["a", "b", "c", "d", "e", "f"], 0.1)
    scorer.fit(
        train,
        outcomes,
        seen,
        np.arange(20),
        np.zeros((10, 1031)),
        np.ones((10, 6)),
        np.ones((10, 6), dtype=bool),
        np.arange(20, 30),
    )
    assert captured["x"].shape == (21, 1037)
    assert np.isnan(captured["x"][0, 1029])
    assert captured["x"][0, 1031:].tolist() == [1, 0, 0, 0, 0, 0]
    assert captured["x"][1, 1031:].tolist() == [0, 1, 0, 0, 0, 0]
    assert captured["y"][0] == 0
    assert captured["weights"][:3].tolist() == [0.5, 0.5, 1.0]
    assert scorer.support == {"a": True, "b": False, "c": False, "d": False, "e": False, "f": False}
    monkeypatch.setattr(
        HistGradientBoostingRegressor, "predict", lambda self, x: np.full(len(x), 0.9)
    )
    scores = scorer.score_candidates(train[0], ["a", "b", "unknown"])
    assert scores["a"].rate == 0.9 and scores["a"].supported
    assert not scores["b"].supported and not scores["unknown"].supported


def test_real_fit_disjoint_groups_and_invalid_predictions():
    scorer = ActionOutcomeScorer(["a"], 0.1)
    x = np.zeros((30, 1031))
    seen = np.ones((30, 1), bool)
    scorer.fit(
        x[:20],
        np.ones((20, 1)),
        seen[:20],
        np.arange(20),
        x[20:],
        np.ones((10, 1)),
        seen[20:],
        np.arange(20, 30),
    )
    assert scorer.score_candidates(x[0], ["a"])["a"].supported
    scorer.model = SimpleNamespace(predict=lambda x: np.array([1.01]))
    assert scorer.score_candidates(x[0], ["a"])["a"].reason == "invalid_rate"
    with pytest.raises(ValueError, match="disjoint"):
        scorer.fit(
            x[:20],
            np.ones((20, 1)),
            seen[:20],
            np.arange(20),
            x[20:],
            np.ones((10, 1)),
            seen[20:],
            np.arange(10),
        )


def test_runtime_features_preserve_missing_and_zero():
    router = object.__new__(HybridRouter)
    router.retriever = SimpleNamespace(
        embedder=SimpleNamespace(embed_task_ir=lambda *a, **k: np.ones(1024))
    )
    missing = router._outcome_features({})
    zero = router._outcome_features(
        {"task_type": "general", "context_length": 0, "has_images": False}
    )
    assert np.isnan(missing[1024:]).all()
    assert zero[1024:].tolist() == [0, 0, 0, 0, 1, 0, 0]


def test_menu_cost_preferences_and_refusal(monkeypatch):
    import src.features

    monkeypatch.setattr(src.features, "features", lambda: SimpleNamespace(routing_classifier=True))
    router = object.__new__(HybridRouter)
    router._live_routing_roles = {"frontdoor", "worker_general"}
    router.admissible_roles_provider = lambda task: task["menu"]
    router.action_outcome_scorer = SimpleNamespace(
        quality_floor=0.5,
        score_candidates=lambda x, roles: {
            r: CandidateScore({"frontdoor": 0.9, "worker_general": 0.8}[r], True) for r in roles
        },
    )
    router.retriever = SimpleNamespace(
        embedder=SimpleNamespace(embed_task_ir=lambda *a, **k: np.ones(1024)),
        config=SimpleNamespace(cost_lambda=1.0),
    )

    def result(role, cost, perf, cost_weight):
        return SimpleNamespace(
            memory=SimpleNamespace(action=role),
            normalized_cost=cost,
            expected_cost_s=cost,
            cold_cost_s=1,
            routing_cost_tau=1,
            routing_pref_perf=perf,
            routing_pref_cost=cost_weight,
        )

    results = [result("frontdoor", 0.8, 0.5, 0.5), result("worker_general", 0.1, 0.5, 0.5)]
    assert (
        router._outcome_choice({"menu": ["frontdoor", "worker_general"]}, results)
        == "worker_general"
    )
    assert router._outcome_choice({"menu": ["frontdoor"]}, results) == "frontdoor"
    assert router._outcome_choice({"menu": ["unknown"]}, results) is None
    for r in results:
        r.routing_pref_perf, r.routing_pref_cost = 1, 0
    assert (
        router._outcome_choice(
            {"menu": ["frontdoor", "worker_general"], "routing_preferences": {"perf": 1}}, results
        )
        == "frontdoor"
    )
    monkeypatch.setattr(src.features, "features", lambda: SimpleNamespace(routing_classifier=False))
    assert router._outcome_choice({"menu": ["frontdoor"]}, results) is None


def test_route_flag_off_preserves_incumbent_and_flag_on_respects_menu(monkeypatch):
    from unittest.mock import Mock

    import src.features

    router = object.__new__(HybridRouter)
    router._live_routing_roles = {"frontdoor", "worker_general"}
    router.action_outcome_scorer = SimpleNamespace(
        quality_floor=0.5,
        score_candidates=Mock(return_value={"worker_general": CandidateScore(0.8, True)}),
    )
    router.admissible_roles_provider = lambda task: task["menu"]
    result = SimpleNamespace(
        memory=SimpleNamespace(action="worker_general"),
        normalized_cost=0.1,
        expected_cost_s=0.1,
        cold_cost_s=1,
        routing_cost_tau=1,
        routing_pref_perf=0.5,
        routing_pref_cost=0.5,
    )
    router.retriever = SimpleNamespace(
        embedder=SimpleNamespace(embed_task_ir=lambda *a, **k: np.ones(1024)),
        config=SimpleNamespace(cost_lambda=1, risk_abstain_target_role="frontdoor"),
        retrieve_for_routing=Mock(return_value=[result]),
        evaluate_risk_gate=lambda *a, **k: {},
        should_use_learned=lambda results: False,
        update_last_role=Mock(),
    )
    router.routing_classifier = None
    router.graph_router = None
    router._bilinear_scorer = None
    router.rule_based = SimpleNamespace(route=lambda task: ["frontdoor"])
    router._apply_priors = lambda results, priors: results
    router._record_decision_meta = Mock()
    monkeypatch.setattr(src.features, "features", lambda: SimpleNamespace(routing_classifier=False))
    task = {"menu": ["worker_general"]}
    assert router.route(task) == (["frontdoor"], "rules")
    router.action_outcome_scorer.score_candidates.assert_not_called()
    monkeypatch.setattr(src.features, "features", lambda: SimpleNamespace(routing_classifier=True))
    assert router.route(task) == (["worker_general"], "action_outcome")
    assert router.route({"menu": []}) == (["frontdoor"], "rules")
    router.action_outcome_scorer.score_candidates.side_effect = RuntimeError("unavailable")
    assert router.route(task) == (["frontdoor"], "rules")


@pytest.mark.parametrize("bad", [None, "", "  ", np.nan, np.inf, -np.inf])
def test_missing_or_nonfinite_group_ids_refused(bad):
    scorer = ActionOutcomeScorer(["a"], 0.1)
    x = np.zeros((2, 1031))
    y = np.ones((2, 1))
    mask = np.ones((2, 1), dtype=bool)
    with pytest.raises(ValueError, match="group ids"):
        scorer.fit(x, y, mask, ["valid", bad], x, y, mask, [3, 4])
    with pytest.raises(ValueError, match="group ids"):
        scorer.fit(x, y, mask, [1, 2], x, y, mask, ["valid", bad])


def test_repeated_groups_do_not_inflate_support():
    scorer = ActionOutcomeScorer(["a"], 0.1)
    train_x = np.zeros((20, 1031))
    cal_x = np.zeros((10, 1031))
    scorer.fit(
        train_x,
        np.ones((20, 1)),
        np.ones((20, 1), bool),
        ["same"] * 20,
        cal_x,
        np.ones((10, 1)),
        np.ones((10, 1), bool),
        np.arange(10),
    )
    assert not scorer.support["a"]
    with pytest.raises(ValueError, match="disjoint"):
        scorer.fit(
            train_x,
            np.ones((20, 1)),
            np.ones((20, 1), bool),
            ["same"] * 20,
            cal_x,
            np.ones((10, 1)),
            np.ones((10, 1), bool),
            ["same"] * 10,
        )


@pytest.mark.parametrize(
    "shadow,rate,expected",
    [(False, 0.2, None), (True, 0.2, "frontdoor"), (False, 0.9, "frontdoor")],
)
def test_outcome_frontdoor_respects_verifier_gate(monkeypatch, shadow, rate, expected):
    import src.features

    monkeypatch.setattr(src.features, "features", lambda: SimpleNamespace(routing_classifier=True))
    router = object.__new__(HybridRouter)
    router.action_outcome_scorer = object()
    router.admissible_roles_provider = lambda task: ["frontdoor"]
    router.retriever = SimpleNamespace(
        retrieve_for_routing=lambda task: [],
        evaluate_risk_gate=lambda *args, **kwargs: {},
        update_last_role=lambda role: None,
    )
    router._filter_routable_results = lambda results: results
    router._apply_priors = lambda results, priors: results
    router._outcome_choice = lambda task, results: "frontdoor"
    router._build_classifier_features = lambda task: np.zeros(1031)
    router.frontdoor_verifier = SimpleNamespace(predict=lambda features, action_idx: rate)
    router.frontdoor_verifier_threshold = 0.5
    router.frontdoor_verifier_shadow = shadow
    router.last_decision_meta = {}
    router._record_decision_meta = lambda **kwargs: None
    assert router._try_outcome_route({}, None) == expected
    if expected:
        assert router.last_decision_meta["verifier_verdict"] == (
            "reject" if rate < 0.5 else "accept"
        )
        assert router.last_decision_meta["verifier_shadow"] == shadow
