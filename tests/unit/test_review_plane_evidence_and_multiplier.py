"""RPK-CONSUMER: request_evidence_round_budget (delegator) and review_token_multiplier (service)."""

from __future__ import annotations

import asyncio
from unittest.mock import Mock

from src.proactive_delegation.delegator import ProactiveDelegator, ReviewPlaneKnobs
from src.proactive_delegation.review_service import ArchitectReviewService
from src.proactive_delegation.types import ArchitectReview, ReviewDecision

RE, AP, RJ = ReviewDecision.REQUEST_EVIDENCE, ReviewDecision.APPROVE, ReviewDecision.REJECT


class Prims:
    def __init__(self):
        self.gen_calls = 0

    def llm_call(self, prompt, role=None, n_tokens=None, **kw):
        self.gen_calls += 1
        return "OUT"


class SeqReview:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.calls: list[dict] = []

    def review(self, spec, subtask, output, quick_mode=False, **kw):
        self.calls.append({"output": output, **kw})
        d = self.decisions[min(len(self.calls) - 1, len(self.decisions) - 1)]
        return ArchitectReview(subtask_id="S0", decision=d, feedback="fb", score=0.5)


def _run(budget, decisions):
    prims = Prims()
    d = ProactiveDelegator(
        Mock(roles={}), prims, review_knobs=ReviewPlaneKnobs(request_evidence_round_budget=budget)
    )
    svc = SeqReview(decisions)
    d.review_service = svc
    task = {"task_id": "t", "objective": "o"}
    step = {"id": "S0", "action": "a", "actor": "worker"}
    res = asyncio.run(d._execute_with_review(task, step))
    return res, svc, prims


def test_budget_zero_unchanged_regenerates_like_request_changes():
    res, svc, prims = _run(0, [RE, AP])
    assert len(svc.calls) == 2 and prims.gen_calls == 2
    assert not any(c.get("force_verdict") for c in svc.calls)
    assert res.success


def test_budget_one_rereviews_same_output_then_forces_verdict():
    res, svc, prims = _run(1, [RE, RE, RJ])
    assert prims.gen_calls == 1  # never regenerated
    assert len(svc.calls) == 3
    assert [bool(c.get("force_verdict")) for c in svc.calls] == [False, False, True]
    assert {c["output"] for c in svc.calls} == {"OUT"}
    assert not res.success and "Rejected" in res.error


def test_budget_one_evidence_round_can_resolve_without_forcing():
    res, svc, prims = _run(1, [RE, AP])
    assert len(svc.calls) == 2 and prims.gen_calls == 1 and res.success


class P:
    def __init__(self):
        self.prompts = []

    def llm_call(self, prompt, role=None, n_tokens=None, **kw):
        self.prompts.append(prompt)
        return '{"d":"approve","s":0.9,"f":"ok"}'


def _svc(monkeypatch, k, m):
    monkeypatch.setenv("ORCHESTRATOR_DELEGATION_REVIEW_MAJORITY_K", str(k))
    if m is None:
        monkeypatch.delenv("ORCHESTRATOR_DELEGATION_REVIEW_TOKEN_MULTIPLIER", raising=False)
    else:
        monkeypatch.setenv("ORCHESTRATOR_DELEGATION_REVIEW_TOKEN_MULTIPLIER", str(m))
    p = P()
    s = ArchitectReviewService(p, trace_sink=lambda e: None)
    s.review({"objective": "o"}, {"id": "s", "action": "a"}, "out")
    return p


def test_k3_m2_runs_two_passes(monkeypatch):
    assert len(_svc(monkeypatch, 3, 2.0).prompts) == 2


def test_env_k3_m3_runs_three_passes(monkeypatch):
    assert len(_svc(monkeypatch, 3, 3.0).prompts) == 3


def test_env_multiplier_clamped_to_5(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_DELEGATION_REVIEW_TOKEN_MULTIPLIER", "9")
    from src.config.models import DelegationConfig

    assert DelegationConfig().review_token_multiplier == 5.0


def test_direct_attrs_k3_m3_runs_three_passes(monkeypatch):
    p = P()
    s = ArchitectReviewService(p, trace_sink=lambda e: None)
    s.review_majority_k, s.review_token_multiplier = 3, 3.0
    s.review({"objective": "o"}, {"id": "s", "action": "a"}, "out")
    assert len(p.prompts) == 3


def test_default_k1_m2_single_pass(monkeypatch):
    monkeypatch.delenv("ORCHESTRATOR_DELEGATION_REVIEW_MAJORITY_K", raising=False)
    assert len(_svc(monkeypatch, 1, None).prompts) == 1


def test_force_verdict_prompt_suffix():
    p = P()
    s = ArchitectReviewService(p, trace_sink=lambda e: None)
    s.review({"objective": "o"}, {"id": "s", "action": "a"}, "out", force_verdict=True)
    assert "MUST return a final decision" in p.prompts[0]
    s.review({"objective": "o"}, {"id": "s", "action": "a"}, "out")
    assert "MUST return" not in p.prompts[1]
