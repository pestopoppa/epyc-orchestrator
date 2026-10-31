"""Prospective source controls: private requests, database and native progress writer."""

import json
import sys
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "benchmark"))
import seeding_eval as evaluation
import seeding_injection as injection
import seeding_rewards as rewards_module
from seeding_types import ComparativeResult, RoleResult
from orchestration.repl_memory.episodic_store import EpisodicStore
from orchestration.repl_memory.progress_logger import ProgressLogger, ProgressReader
from orchestration.repl_memory.q_scorer import QScorer, ScoringConfig
from orchestration.repl_memory.reward_provenance import native_reward_provenance
from src.api.models.requests import RewardRequest


class Immediate:
    def submit(self, fn, *args):
        f = Future()
        try:
            f.set_result(fn(*args))
        except Exception as e:
            f.set_exception(e)
        return f


@pytest.mark.parametrize("status", [None, True, False, 0, 99, 600, "503", 503.0])
def test_invalid_or_missing_native_http_remains_unknown(status):
    assert native_reward_provenance({"http_status": status})["http_status"] is None


@pytest.mark.parametrize("status", [100, 200, 400, 503, 599])
def test_original_http_status_preserved(status):
    assert native_reward_provenance({"http_status": status})["http_status"] == status


@pytest.mark.parametrize("family", ["three_way", "per_role", "comparative", "escalation"])
@pytest.mark.parametrize("kind", ["correct", "wrong", "native_error", "missing"])
def test_native_writer_serializer_store_and_update_journal(tmp_path, monkeypatch, family, kind):
    # Synthetic source fixture, never an inference observation or historical reconstruction.
    resp = {"answer": "yes" if kind == "correct" else "no", "tokens_generated": 4}
    if kind != "missing":
        resp.update(
            disposition="infra_failed" if kind == "native_error" else "scored",
            infra_reason="http_status" if kind == "native_error" else None,
            http_status=503 if kind == "native_error" else 200,
            source="native-test",
            stage="transport",
            failure_provenance={
                "schema": "epyc.failure_provenance.v1",
                "class": "fixture",
                "stage": "request",
            },
        )
    if kind == "native_error":
        from seeding_orchestrator import _client_transport_timeout_provenance

        resp["failure_provenance"] = _client_transport_timeout_provenance(
            reason="read_timeout",
            exc=TimeoutError("PRIVATE exception text"),
            role="frontdoor",
            workload_class="eval_batch",
            max_queue_wait_ms=1000,
        )
        resp.update(error="deliberately not parsed as status", failure_reason="http_status")
    rr, error_type = evaluation._build_role_result(
        role="frontdoor",
        mode="direct",
        resp=resp,
        elapsed=1.0,
        expected="yes",
        scoring_method="exact_match",
        scoring_config={},
    )
    assert rr.passed is (kind == "correct")
    assert error_type == ("infrastructure" if kind == "native_error" else "none")
    assert rr.reward_provenance["disposition"] == resp.get("disposition")
    assert rr.reward_provenance["http_status"] == resp.get("http_status")
    other = RoleResult(
        role="coder_escalation",
        mode="direct",
        answer="yes",
        passed=True,
        elapsed_seconds=1.0,
        reward_provenance={
            "disposition": "scored",
            "http_status": 201,
            "role": "coder_escalation",
            "mode": "direct",
            "source": "other-test",
        },
    )
    vector = np.random.default_rng(4321).standard_normal(1024).astype(np.float32).tolist()
    logger = ProgressLogger(log_dir=tmp_path / "journal", buffer_size=1)
    store = EpisodicStore(db_path=tmp_path / "store", use_faiss=True)
    scorer = QScorer(
        store=store,
        embedder=Mock(),
        logger=logger,
        reader=None,
        config=ScoringConfig(learning_rate=0.1, temporal_decay_rate=None),
    )
    payloads = []

    class Local:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, *, json, timeout):
            # Actual request model round-trip, including JSON serialization.
            request = RewardRequest.model_validate_json(
                RewardRequest.model_validate(json).model_dump_json()
            )
            payloads.append(request)
            assert scorer.score_external_result(
                task_description=request.task_description,
                action=request.action,
                reward=request.reward,
                context=request.context,
                embedding=vector,
            ) == {
                "memories_created": int(len(payloads) == 1),
                "memories_updated": int(len(payloads) == 2),
            }
            return SimpleNamespace(status_code=200)

    monkeypatch.setattr(injection.httpx, "Client", Local)
    monkeypatch.setattr(injection, "_precompute_embedding", lambda *a, **kw: vector)
    monkeypatch.setattr(injection, "_get_reward_executor", lambda: Immediate())
    context_metadata = {
        "reward_provenance": {
            "SELF:direct": rr.reward_provenance,
            "frontdoor": rr.reward_provenance,
        }
    }
    comp = ComparativeResult(
        suite="fixture",
        question_id="fixture-q",
        prompt="fixture objective",
        expected="yes",
        role_results={"frontdoor:direct": rr},
        rewards={"frontdoor:direct": 1.0},
    )
    # Inject native errors deliberately through writer controls; production filtering is untouched.
    chain = {
        "action": "escalate:frontdoor->coder_escalation",
        "reward": 1.0,
        "from_role": "frontdoor",
        "to_role": "coder_escalation",
        "provenance_sources": [rr.reward_provenance, other.reward_provenance],
    }
    try:
        for value in (1.0, 0.0):
            if family in ("three_way", "per_role"):
                fn = (
                    injection._inject_3way_rewards_http
                    if family == "three_way"
                    else injection._inject_per_role_rewards_http
                )
                action = "SELF:direct" if family == "three_way" else "frontdoor"
                delivery = fn(
                    "fixture objective",
                    "fixture",
                    "fixture-q",
                    {action: value},
                    context_metadata,
                    "http://private",
                    Local(),
                )
                assert delivery["acknowledged"] == 1
            elif family == "comparative":
                comp.rewards = {"frontdoor:direct": value}
                assert rewards_module._inject_rewards_http(comp, "http://private", Local()) == 1
            else:
                chain["reward"] = value
                assert (
                    rewards_module._inject_escalation_chains_http(
                        comp, [chain], "http://private", Local()
                    )
                    == 1
                )
        logger.flush()
        entries = ProgressReader(tmp_path / "journal").read_recent()
        assert [e.event_type.value for e in entries] == ["memory_stored", "q_value_updated"]
        created, updated = [e.data["reward_provenance"] for e in entries]
        assert created == updated
        assert entries[0].data["initial_q"] == 1.0
        assert entries[1].data["old_q"] == 1.0
        assert entries[1].data["new_q"] == 0.9
        assert entries[1].data["reward"] == 0.0
        origin = (
            updated["provenance_sources"][0]
            if family == "escalation"
            else updated["native_reward_provenance"]
        )
        assert origin["evaluation"] == {
            "error_type": error_type,
            "source": "seeding_eval._build_role_result",
        }
        if kind == "native_error":
            failure = origin["failure_provenance"]
            assert {
                k: failure[k]
                for k in ("schema", "class", "phase", "code", "role", "workload_class")
            } == {
                k: resp["failure_provenance"][k]
                for k in ("schema", "class", "phase", "code", "role", "workload_class")
            }
            assert "PRIVATE" not in json.dumps(updated)
        if kind == "missing":
            assert origin["disposition"] is None
            assert origin["infra_reason"] is None
            assert origin["http_status"] is None
        if family == "escalation":
            assert updated["disposition"] is None
            assert [x["http_status"] for x in updated["provenance_sources"]] == [
                resp.get("http_status"),
                201,
            ]
        else:
            assert updated["disposition"] == resp.get("disposition")
            assert updated["http_status"] == resp.get("http_status")
        mem = store.retrieve_by_similarity(np.array(vector, dtype=np.float32), k=1)[0]
        assert mem.q_value == 0.9
        assert mem.outcome == "success"
        assert mem.context["metrics"]["http_status"] == (
            None if family == "escalation" else resp.get("http_status")
        )
    finally:
        store.close()


def test_journal_whitelist_excludes_arbitrary_payloads():
    from orchestration.repl_memory.reward_provenance import reward_journal_provenance

    value = reward_journal_provenance(
        {
            "source": "source",
            "stage": "stage",
            "http_status": True,
            "answer": "PRIVATE",
            "prompt": "PRIVATE",
            "_meta": {"error": "PRIVATE"},
            "native_reward_provenance": {"http_status": 503, "traceback": "PRIVATE"},
            "provenance_sources": [
                {"role": "a", "error": "PRIVATE"},
                {"role": "b", "http_status": 200},
            ],
        },
        "action",
    )
    assert "PRIVATE" not in json.dumps(value)
    assert value["http_status"] is None
    assert value["native_reward_provenance"]["http_status"] == 503


def test_aggregate_sources_are_not_replaced_by_fastest_architect(monkeypatch):
    monkeypatch.setattr(evaluation, "_architect_result_order", lambda roles: list(roles))
    monkeypatch.setattr(evaluation, "_heuristic_architect_role", lambda *a: "architect_general")
    direct = RoleResult(
        role="frontdoor",
        mode="direct",
        answer="no",
        passed=False,
        elapsed_seconds=1,
        reward_provenance={"role": "frontdoor", "mode": "direct", "http_status": 200},
    )
    repl = RoleResult(
        role="frontdoor",
        mode="repl",
        answer="no",
        passed=False,
        elapsed_seconds=1,
        reward_provenance={"role": "frontdoor", "mode": "repl", "http_status": 201},
    )
    first = RoleResult(
        role="architect_general",
        mode="repl",
        answer="yes",
        passed=True,
        elapsed_seconds=9,
        reward_provenance={"role": "architect_general", "mode": "repl", "http_status": 202},
    )
    fastest = RoleResult(
        role="architect_coding",
        mode="repl",
        answer="yes",
        passed=True,
        elapsed_seconds=1,
        reward_provenance={"role": "architect_coding", "mode": "repl", "http_status": 203},
    )
    records = {
        "frontdoor:direct": direct,
        "frontdoor:repl": repl,
        "architect_general:repl": first,
        "architect_coding:repl": fastest,
    }
    arch = {
        rr.role: {"passed": rr.passed, "elapsed_seconds": rr.elapsed_seconds}
        for rr in (first, fastest)
    }
    md = evaluation._compute_3way_metadata(
        records, arch, "fixture", "fixture", False, False, "frontdoor", "direct", "repl", "repl"
    )
    assert md["architect_role"] == "architect_coding"
    assert [x["role"] for x in md["reward_provenance"]["ARCHITECT"]] == [
        "architect_general",
        "architect_coding",
    ]
    from orchestration.repl_memory.reward_provenance import reward_context_provenance

    ctx = reward_context_provenance(md["reward_provenance"]["ARCHITECT"])
    assert ctx["http_status"] is None
    assert [x["http_status"] for x in ctx["provenance_sources"]] == [202, 203]
    monkeypatch.setattr(
        rewards_module,
        "_architect_result_keys",
        lambda r: ["architect_general:repl", "architect_coding:repl"],
    )
    assert rewards_module.compute_3way_rewards(records) == {
        "SELF:direct": 0.0,
        "SELF:repl": 0.0,
        "ARCHITECT": 1.0,
    }
    for rr in (repl, first, fastest):
        rr.delegation_events = [{"type": "fixture"}]
    original = rewards_module.score_delegation_chain(records)
    native = []
    assert (
        rewards_module.score_delegation_chain(records, provenance=native)
        == original
        == {"WORKER": 1.0}
    )
    assert [x["http_status"] for x in native] == [201, 202, 203]


def test_detected_escalation_keeps_both_original_sources(monkeypatch):
    monkeypatch.setattr(rewards_module, "ROLE_COST_TIER", {"frontdoor": 1, "coder_escalation": 2})
    low = RoleResult(
        role="frontdoor",
        mode="direct",
        answer="no",
        passed=False,
        elapsed_seconds=1,
        reward_provenance={"http_status": 200, "disposition": "scored"},
    )
    high = RoleResult(
        role="coder_escalation",
        mode="direct",
        answer="yes",
        passed=True,
        elapsed_seconds=1,
        reward_provenance={"http_status": 201, "disposition": "scored"},
    )
    chains = rewards_module.detect_escalation_chains(
        {"frontdoor:direct": low, "coder_escalation:direct": high}, include_provenance=True
    )
    assert len(chains) == 1
    assert [
        (p["requested_role"], p["requested_mode"], p["http_status"])
        for p in chains[0]["provenance_sources"]
    ] == [("frontdoor", "direct", 200), ("coder_escalation", "direct", 201)]
    assert chains[0]["reward"] == rewards_module.ESCALATION_REWARD


def test_update_retains_previous_store_provenance_and_separate_journal_history(tmp_path):
    logger = ProgressLogger(log_dir=tmp_path / "journal", buffer_size=1)
    store = EpisodicStore(db_path=tmp_path / "store", use_faiss=True)
    scorer = QScorer(
        store=store,
        embedder=Mock(),
        logger=logger,
        reader=None,
        config=ScoringConfig(learning_rate=0.1, temporal_decay_rate=None),
    )
    vector = np.random.default_rng(4321).standard_normal(1024).astype(np.float32).tolist()
    try:
        for source, status, value in [("initial", 200, 1.0), ("later", 503, 0.0)]:
            assert (
                sum(
                    scorer.score_external_result(
                        "objective",
                        "frontdoor",
                        value,
                        context={"source": source, "http_status": status, "question_id": source},
                        embedding=vector,
                    ).values()
                )
                == 1
            )
        rows = ProgressReader(tmp_path / "journal").read_recent()
        assert [e.data["reward_provenance"]["source"] for e in rows] == ["initial", "later"]
        assert [e.data["reward_provenance"]["http_status"] for e in rows] == [200, 503]
        mem = store.retrieve_by_similarity(np.array(vector, dtype=np.float32), k=1)[0]
        assert mem.context["metrics"]["http_status"] == 200
        assert mem.q_value == 0.9
    finally:
        store.close()


def test_aggregate_single_component_does_not_borrow_native_facts():
    from orchestration.repl_memory.reward_provenance import reward_context_provenance

    ctx = reward_context_provenance(
        [{"http_status": 200, "disposition": "scored", "requested_role": "parent"}]
    )
    assert ctx["http_status"] is None
    assert ctx["disposition"] is None
    assert ctx["provenance_sources"][0]["http_status"] == 200
    assert ctx["provenance_sources"][0]["requested_role"] == "parent"


@pytest.mark.parametrize("family", ["three_way", "escalation"])
def test_seventeen_components_are_bounded_before_serializer_and_journal(
    tmp_path, monkeypatch, family
):
    from orchestration.repl_memory.reward_provenance import reward_context_provenance

    # Malformed retained and omitted slots remain explicit; no contributor is borrowed.
    components = [{"http_status": 200, "requested_role": f"role-{i}"} for i in range(18)]
    components[2], components[17] = None, "PRIVATE malformed component"
    vector = np.random.default_rng(4321).standard_normal(1024).astype(np.float32).tolist()
    logger = ProgressLogger(log_dir=tmp_path / "journal", buffer_size=1)
    store = EpisodicStore(db_path=tmp_path / "store", use_faiss=True)
    scorer = QScorer(
        store=store,
        embedder=Mock(),
        logger=logger,
        reader=None,
        config=ScoringConfig(learning_rate=0.1, temporal_decay_rate=None),
    )
    requests = []

    class Local:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def post(self, _url, *, json, timeout):
            request = RewardRequest.model_validate_json(
                RewardRequest.model_validate(json).model_dump_json()
            )
            requests.append(request)
            assert len(request.context["provenance_sources"]) == 16
            assert request.context["http_status"] is None
            assert request.context["disposition"] is None
            assert request.context["provenance_sources_summary"] == {
                "total": 18,
                "retained": 16,
                "omitted": 2,
                "malformed_total": None,
                "malformed_retained": 1,
                "status": "counted_at_writer",
            }
            assert scorer.score_external_result(
                request.task_description,
                request.action,
                request.reward,
                context=request.context,
                embedding=vector,
            ) == {
                "memories_created": int(len(requests) == 1),
                "memories_updated": int(len(requests) == 2),
            }
            return SimpleNamespace(status_code=200)

    monkeypatch.setattr(injection.httpx, "Client", Local)
    monkeypatch.setattr(injection, "_precompute_embedding", lambda *a, **kw: vector)
    monkeypatch.setattr(injection, "_get_reward_executor", lambda: Immediate())
    try:
        for value in (1.0, 0.0):
            if family == "three_way":
                result = injection._inject_3way_rewards_http(
                    "fixture objective",
                    "fixture",
                    "fixture-q",
                    {"ARCHITECT": value},
                    {"reward_provenance": {"ARCHITECT": components}},
                    "http://private",
                    Local(),
                )
                assert result["acknowledged"] == 1
            else:
                chain = {
                    "action": "escalate:a->b",
                    "reward": value,
                    "from_role": "a",
                    "to_role": "b",
                    "provenance_sources": components,
                }
                comp = ComparativeResult(
                    suite="fixture",
                    question_id="fixture-q",
                    prompt="fixture objective",
                    expected="yes",
                )
                assert (
                    rewards_module._inject_escalation_chains_http(
                        comp, [chain], "http://private", Local()
                    )
                    == 1
                )
        rows = ProgressReader(tmp_path / "journal").read_recent()
        assert [r.event_type.value for r in rows] == ["memory_stored", "q_value_updated"]
        for row in rows:
            provenance = row.data["reward_provenance"]
            assert provenance["http_status"] is None
            assert len(provenance["provenance_sources"]) == 16
            assert provenance["provenance_sources"][2]["component_status"] == "unknown_non_mapping"
            assert provenance["provenance_sources_summary"] == {
                "total": 18,
                "retained": 16,
                "omitted": 2,
                "malformed_total": None,
                "malformed_retained": 1,
                "status": "preserved_upstream_counts",
            }
            assert "PRIVATE" not in row.to_json()
        assert rows[1].data["new_q"] == 0.9
        # The journal also bounds a direct caller that did not use an injection builder.
        raw_request = RewardRequest.model_validate_json(
            RewardRequest(
                task_description="raw fixture",
                action="a",
                reward=0,
                context={"provenance_sources": components},
                embedding=vector,
            ).model_dump_json()
        )
        assert scorer.score_external_result(
            raw_request.task_description,
            raw_request.action,
            raw_request.reward,
            context=raw_request.context,
            embedding=raw_request.embedding,
        ) == {"memories_created": 1, "memories_updated": 0}
        direct_row = ProgressReader(tmp_path / "journal").read_recent()[-1]
        direct = direct_row.data["reward_provenance"]
        assert len(direct["provenance_sources"]) == 16
        assert direct["provenance_sources_summary"]["total"] == 18
        assert direct["provenance_sources_summary"]["omitted"] == 2
        assert direct["provenance_sources_summary"]["malformed_total"] is None
        assert "PRIVATE" not in direct_row.to_json()
        unknown = reward_context_provenance([None])
        assert unknown["http_status"] is None
        assert unknown["provenance_sources_summary"]["malformed_total"] == 1
    finally:
        store.close()


@pytest.mark.parametrize("result_kind", ["clean", "timeout", "http_aliases", "conflicting_aliases"])
def test_actual_resilient_emitter_facts_survive_serializer_store_and_journal(tmp_path, result_kind):
    import httpx
    import seeding_orchestrator

    class NativeClient:
        def post(self, _url, *, json, timeout):
            if result_kind == "timeout":
                raise httpx.ReadTimeout("PRIVATE native exception detail")
            body = {"answer": "yes", "tokens_generated": 4}
            if result_kind == "http_aliases":
                body.update(status_code=503, error_code="503")
            elif result_kind == "conflicting_aliases":
                body.update(http_status=503, status_code=400, error_code="503")
            return SimpleNamespace(
                status_code=200, json=lambda: body, raise_for_status=lambda: None
            )

    watcher = SimpleNamespace(
        port_for_role=lambda role: None,
        reference_for_role=lambda role: {},
        invalidate_cache=lambda: None,
        was_restarted_since=lambda refs: {},
    )
    # Normal entry runs actual resilient_post and native _meta attachment. No network client is used.
    response = seeding_orchestrator.call_orchestrator_forced(
        "fixture",
        "frontdoor",
        client=NativeClient(),
        watcher=watcher,
        timeout=60,
        workload_class="fixture",
    )
    assert response["_meta"]["clean"] is (result_kind != "timeout")
    assert response["_meta"]["reason"] == ("read_timeout" if result_kind == "timeout" else "")
    rr, classification = evaluation._build_role_result(
        role="frontdoor",
        mode="direct",
        resp=response,
        elapsed=1,
        expected="yes",
        scoring_method="exact_match",
        scoring_config={},
    )
    assert classification == ("none" if result_kind == "clean" else "infrastructure")
    from orchestration.repl_memory.reward_provenance import reward_context_provenance

    context = {"source": "fixture", **reward_context_provenance(rr.reward_provenance)}
    assert context["http_status"] is None
    transport = context["native_reward_provenance"]["transport"]
    assert transport["meta"] == {
        "clean": response["_meta"]["clean"],
        "reason": response["_meta"]["reason"],
    }
    assert transport["status_fields"] == {
        key: response.get(key) for key in ("http_status", "status_code", "error_code")
    }
    assert transport["conflicting_statuses"] is (result_kind == "conflicting_aliases")
    vector = np.random.default_rng(4321).standard_normal(1024).astype(np.float32).tolist()
    logger = ProgressLogger(log_dir=tmp_path / "journal", buffer_size=1)
    store = EpisodicStore(db_path=tmp_path / "store", use_faiss=True)
    scorer = QScorer(
        store=store,
        embedder=Mock(),
        logger=logger,
        reader=None,
        config=ScoringConfig(learning_rate=0.1, temporal_decay_rate=None),
    )
    try:
        for reward in (1.0, 0.0):
            request = RewardRequest.model_validate_json(
                RewardRequest(
                    task_description="fixture objective",
                    action="frontdoor",
                    reward=reward,
                    context=context,
                    embedding=vector,
                ).model_dump_json()
            )
            scorer.score_external_result(
                request.task_description,
                request.action,
                request.reward,
                context=request.context,
                embedding=request.embedding,
            )
        rows = ProgressReader(tmp_path / "journal").read_recent()
        assert [row.event_type.value for row in rows] == ["memory_stored", "q_value_updated"]
        assert rows[1].data["new_q"] == 0.9
        for row in rows:
            provenance = row.data["reward_provenance"]
            assert provenance["http_status"] is None
            assert provenance["native_reward_provenance"]["transport"] == transport
            assert (
                provenance["native_reward_provenance"]["evaluation"]["error_type"] == classification
            )
            assert "PRIVATE" not in row.to_json()
            assert "detail" not in json.dumps(provenance)
            assert "retry_count" not in json.dumps(provenance)
    finally:
        store.close()


def test_malformed_component_count_summary_remains_explicit_unknown():
    from orchestration.repl_memory.reward_provenance import reward_journal_provenance

    result = reward_journal_provenance(
        {
            "provenance_sources": [None],
            "provenance_sources_summary": {
                "total": 18,
                "retained": 1,
                "omitted": 1,
                "malformed_total": 1,
            },
        },
        "action",
    )
    assert result["provenance_sources_summary"] == {
        "total": None,
        "retained": 1,
        "omitted": None,
        "malformed_total": None,
        "malformed_retained": 1,
        "status": "unknown_upstream_counts",
    }
    assert result["provenance_sources"][0]["component_status"] == "unknown_non_mapping"
    invalid = reward_journal_provenance({"provenance_sources": "PRIVATE invalid list"}, "action")
    assert invalid["provenance_sources"] == []
    assert invalid["provenance_sources_summary"]["status"] == "unknown_component_collection"
    assert "PRIVATE" not in json.dumps(invalid)


def test_upstream_zero_omitted_malformed_count_cannot_claim_false_exact_count():
    from orchestration.repl_memory.reward_provenance import reward_journal_provenance

    result = reward_journal_provenance(
        {
            "provenance_sources": [{}],
            "provenance_sources_summary": {
                "total": 1,
                "retained": 1,
                "omitted": 0,
                "malformed_total": 1,
            },
        },
        "action",
    )
    assert result["provenance_sources_summary"] == {
        "total": None,
        "retained": 1,
        "omitted": None,
        "malformed_total": None,
        "malformed_retained": 0,
        "status": "unknown_upstream_counts",
    }
