"""RPK-CONSUMER: ``review_majority_k`` is read by ArchitectReviewService.review()."""

from __future__ import annotations

from src.config.models import DelegationConfig
from src.proactive_delegation.review_service import ArchitectReviewService
from src.proactive_delegation.types import ReviewDecision

_APPROVE = '{"d":"approve","s":0.9,"f":"ok"}'
_CHANGES = '{"d":"changes","s":0.4,"f":"fix"}'


class SeqPrimitives:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def llm_call(self, prompt, role=None, n_tokens=None, **kwargs):
        r = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return r


def _run(monkeypatch, k, responses):
    if k is None:
        monkeypatch.delenv("ORCHESTRATOR_DELEGATION_REVIEW_MAJORITY_K", raising=False)
    else:
        monkeypatch.setenv("ORCHESTRATOR_DELEGATION_REVIEW_MAJORITY_K", str(k))
    prims = SeqPrimitives(responses)
    events: list = []
    svc = ArchitectReviewService(prims, trace_sink=events.append)
    svc.review_token_multiplier = 3.0  # lift the LB-2 cap (default 2.0) for this test
    review = svc.review({"objective": "o"}, {"id": "s1", "action": "a"}, "out")
    return svc, prims, review, events


def test_default_is_single_pass(monkeypatch):
    assert DelegationConfig().review_majority_k == 1
    svc, prims, review, events = _run(monkeypatch, None, [_APPROVE])
    assert svc.review_majority_k == 1
    assert prims.calls == 1 and len(events) == 1
    assert review.decision == ReviewDecision.APPROVE


def test_k3_runs_three_passes_and_takes_majority(monkeypatch):
    # first pass approves, the other two request changes -> majority overrides pass 1
    svc, prims, review, events = _run(monkeypatch, 3, [_APPROVE, _CHANGES, _CHANGES])
    assert svc.review_majority_k == 3
    assert prims.calls == 3 and len(events) == 3
    assert review.decision == ReviewDecision.REQUEST_CHANGES


def test_k1_same_inputs_keeps_first_pass(monkeypatch):
    _, prims, review, _ = _run(monkeypatch, 1, [_APPROVE, _CHANGES, _CHANGES])
    assert prims.calls == 1
    assert review.decision == ReviewDecision.APPROVE
