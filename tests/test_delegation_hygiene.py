"""Delegation-mode hygiene + control-plane knob tests (H3 RD-10/RD-11, H-LB LB-5).

Covers:
* RD-10a  — AS-TOOL vs HANDOFF modes (output_extractor seam, input_filter window)
* RD-10b  — complexity-gated per-subtask review + single final-aggregate review
* RD-10c  — sticky decision cache in IterationContext (sequential + parallel wave)
* RD-11/LB-5 — ReviewPlaneKnobs declarations + shadow budget seam + manifest YAML
* RD-9    — plan-reminder call-site contract (cadence knob, absence-tolerant)
* Regression: ALL knob defaults preserve current behavior byte-for-byte.

Zero inference: every completion/review callable is stubbed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from unittest.mock import Mock

from src.proactive_delegation.delegator import (
    DelegationMode,
    ProactiveDelegator,
    ReviewPlaneKnobs,
    _identity_output_extractor,
)
from src.parallel_step_executor import StepExecutor, compute_waves
from src.proactive_delegation.types import (
    ArchitectReview,
    IterationContext,
    ReviewDecision,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "orchestration" / "review_plane_knobs.yaml"


# ── Stubs (no inference) ───────────────────────────────────────────────────


class FakePrimitives:
    """Deterministic llm_call stub that records every call."""

    def __init__(self, response: str = "OUT"):
        self.calls: list[dict] = []
        self.response = response

    def llm_call(self, prompt, role=None, n_tokens=None, **kwargs):
        self.calls.append({"prompt": prompt, "role": role, "n_tokens": n_tokens, "kw": kwargs})
        return self.response


class FakeReviewService:
    """Stub review service. Records calls; decisions configurable per call."""

    def __init__(self, decisions=None, default=ReviewDecision.APPROVE, with_reminder=False):
        self.calls: list[dict] = []
        self.decisions = list(decisions or [])
        self.default = default
        self._with_reminder = with_reminder
        self.reminder_calls: list[int] = []

    def review(self, spec, subtask, output, quick_mode=False):
        self.calls.append({"subtask": subtask, "output": output})
        decision = self.decisions.pop(0) if self.decisions else self.default
        return ArchitectReview(
            subtask_id=subtask.get("id", "?"),
            decision=decision,
            feedback=f"fb-{len(self.calls)}",
            approved_output=output if decision == ReviewDecision.APPROVE else None,
        )

    # RD-9 plan-reminder helper (only present when with_reminder=True).
    def __getattr__(self, name):
        if name == "build_plan_reminder" and object.__getattribute__(self, "_with_reminder"):
            def _helper(objective, plan_steps, step_index):
                self.reminder_calls.append(step_index)
                return f"REMINDER@{step_index}"

            return _helper
        raise AttributeError(name)


def _make_delegator(review_service, review_knobs=None, output_extractor=None):
    registry = Mock(roles={})
    primitives = FakePrimitives()
    delegator = ProactiveDelegator(
        registry,
        primitives,
        review_knobs=review_knobs,
        output_extractor=output_extractor,
    )
    delegator.review_service = review_service
    return delegator, primitives


def _plan(*actions):
    return {
        "task_id": "t1",
        "objective": "do the thing",
        "plan": {"steps": [{"id": f"S{i}", "action": a, "actor": "worker"} for i, a in enumerate(actions)]},
    }


# ── RD-11 knob defaults ────────────────────────────────────────────────────


class TestReviewPlaneKnobDefaults:
    def test_defaults_are_behavior_preserving(self):
        k = ReviewPlaneKnobs()
        assert k.per_subtask_review_enabled is True
        assert k.review_trigger_complexity_threshold == 0
        assert k.input_filter_window == 3
        assert k.decision_cache_enabled is False
        assert k.reminder_cadence == 0
        assert k.review_majority_k == 1
        assert k.request_evidence_round_budget == 0
        assert k.review_token_multiplier == 2.0
        assert k.budget_shadow_logging_enabled is False

    def test_lb2_per_decision_budget_defaults(self):
        k = ReviewPlaneKnobs()
        assert k.plan_review_token_budget == 350
        assert k.candidate_review_token_budget == 300
        assert k.rubric_authoring_token_budget == 800
        assert k.rubric_grading_token_budget == 180

    def test_from_config_returns_defaults_when_unwired(self):
        # Delegation config carries no review_plane fields yet → defaults.
        assert ReviewPlaneKnobs.from_config() == ReviewPlaneKnobs()

    def test_delegation_mode_names(self):
        assert DelegationMode.AS_TOOL.value == "as_tool"
        assert DelegationMode.HANDOFF.value == "handoff"


# ── Regression: default behavior byte-identical ────────────────────────────


class TestDefaultBehaviorUnchanged:
    async def test_review_every_subtask_no_aggregate_no_cache(self):
        review = FakeReviewService(default=ReviewDecision.APPROVE)
        delegator, primitives = _make_delegator(review)

        result = await delegator.delegate(_plan("implement feature A", "implement feature B"))

        # One specialist call + one review per subtask; NO final-aggregate review.
        assert len(primitives.calls) == 2
        assert len(review.calls) == 2
        assert all(c["subtask"]["id"] in {"S0", "S1"} for c in review.calls)
        assert result.all_approved is True
        assert result.total_iterations == 2
        assert [r.output for r in result.subtask_results] == ["OUT", "OUT"]
        # Sticky cache stays inert on the shared iteration context.
        assert delegator.iteration_context.decision_cache_enabled is False
        assert delegator.iteration_context.decision_cache == {}

    async def test_default_extractor_is_identity(self):
        assert _identity_output_extractor("abc") == "abc"
        review = FakeReviewService()
        delegator, _ = _make_delegator(review)
        result = await delegator.delegate(_plan("do A"))
        assert result.subtask_results[0].output == "OUT"


# ── RD-10a: AS-TOOL output_extractor + HANDOFF input_filter ────────────────


class TestAsToolAndHandoff:
    async def test_output_extractor_applied(self):
        review = FakeReviewService()
        # extractor uppercases the raw as-tool output.
        delegator, primitives = _make_delegator(review, output_extractor=lambda s: s.upper())
        primitives.response = "out"
        result = await delegator.delegate(_plan("do A"))
        assert result.subtask_results[0].output == "OUT"

    async def test_input_filter_window_bounds_feedback_slice(self):
        # Three REQUEST_CHANGES iterations, window=2 → 3rd prompt carries 2 bullets.
        review = FakeReviewService(
            decisions=[
                ReviewDecision.REQUEST_CHANGES,
                ReviewDecision.REQUEST_CHANGES,
                ReviewDecision.APPROVE,
            ]
        )
        knobs = ReviewPlaneKnobs(input_filter_window=2)
        delegator, primitives = _make_delegator(review, review_knobs=knobs)
        delegator.iteration_context = IterationContext(max_iterations=5, max_total_iterations=10)
        delegator.iteration_context.decision_cache_enabled = False

        await delegator.delegate(_plan("do A"))
        # 3rd specialist call includes the previous-feedback block with 2 bullets.
        third_prompt = primitives.calls[2]["prompt"]
        assert "## Previous Feedback" in third_prompt
        assert third_prompt.count("- fb-") == 2

    async def test_input_filter_window_default_three(self):
        review = FakeReviewService(
            decisions=[ReviewDecision.REQUEST_CHANGES] * 4 + [ReviewDecision.APPROVE]
        )
        delegator, primitives = _make_delegator(review)
        delegator.iteration_context = IterationContext(max_iterations=6, max_total_iterations=20)
        await delegator.delegate(_plan("do A"))
        # By the 4th call, feedback_history has 3 entries → [-3:] → 3 bullets.
        assert primitives.calls[3]["prompt"].count("- fb-") == 3


# ── RD-10b: complexity-gated review + final-aggregate ──────────────────────


class TestComplexityGatedReview:
    async def test_high_threshold_skips_per_subtask_runs_aggregate(self):
        # Trivial actions + COMPLEX(3) threshold → per-subtask review skipped,
        # a single final-aggregate review runs.
        review = FakeReviewService(default=ReviewDecision.APPROVE)
        knobs = ReviewPlaneKnobs(review_trigger_complexity_threshold=3)
        delegator, primitives = _make_delegator(review, review_knobs=knobs)

        result = await delegator.delegate(_plan("say hello", "say hi there"))

        assert len(primitives.calls) == 2  # each subtask still executed once
        assert len(review.calls) == 1  # only the aggregate review
        assert review.calls[0]["subtask"]["id"] == "__final_aggregate__"
        assert result.all_approved is True

    async def test_per_subtask_disabled_runs_aggregate(self):
        review = FakeReviewService(default=ReviewDecision.APPROVE)
        knobs = ReviewPlaneKnobs(per_subtask_review_enabled=False)
        delegator, primitives = _make_delegator(review, review_knobs=knobs)

        result = await delegator.delegate(_plan("implement A", "implement B"))

        assert len(review.calls) == 1
        assert review.calls[0]["subtask"]["id"] == "__final_aggregate__"
        assert result.all_approved is True

    async def test_aggregate_reject_flips_all_approved(self):
        review = FakeReviewService(default=ReviewDecision.REJECT_TO_EMPTY)
        knobs = ReviewPlaneKnobs(per_subtask_review_enabled=False)
        delegator, _ = _make_delegator(review, review_knobs=knobs)
        result = await delegator.delegate(_plan("implement A"))
        assert result.all_approved is False


# ── RD-10c: sticky decision cache ──────────────────────────────────────────


class TestStickyDecisionCache:
    async def test_cache_skips_rereview_sequential(self):
        review = FakeReviewService(default=ReviewDecision.APPROVE)
        knobs = ReviewPlaneKnobs(decision_cache_enabled=True)
        delegator, _ = _make_delegator(review, review_knobs=knobs)
        assert delegator.iteration_context.decision_cache_enabled is True

        # Two structurally-identical subtasks (same action → same signature).
        result = await delegator.delegate(_plan("implement widget", "implement widget"))
        assert len(review.calls) == 1  # 2nd subtask served from cache
        assert result.all_approved is True

    async def test_reject_never_cached(self):
        review = FakeReviewService(default=ReviewDecision.REJECT)
        knobs = ReviewPlaneKnobs(decision_cache_enabled=True)
        delegator, _ = _make_delegator(review, review_knobs=knobs)
        await delegator.delegate(_plan("implement widget"))
        assert delegator.iteration_context.decision_cache == {}

    async def test_cache_skips_rereview_parallel_wave(self):
        review = FakeReviewService(default=ReviewDecision.APPROVE)
        ctx = IterationContext(max_iterations=3, max_total_iterations=10)
        ctx.decision_cache_enabled = True
        executor = StepExecutor(
            primitives=FakePrimitives(),
            review_service=review,
            iteration_context=ctx,
            # Force sequential (no burst gather) so the first step's APPROVE is
            # cached before the second step consults the cache — deterministic.
            burst_worker_roles=frozenset(),
        )
        task_ir = {"objective": "obj"}
        steps = [
            {"id": "S1", "action": "same task", "actor": "worker"},
            {"id": "S2", "action": "same task", "actor": "worker"},
        ]
        waves = compute_waves(steps)
        await executor.execute_plan(task_ir, waves, {"worker": "worker_general"})
        assert len(review.calls) == 1  # second wave step hits cache


# ── IterationContext extensions (unit) ─────────────────────────────────────


class TestIterationContextExtensions:
    def test_signature_stable_and_action_sensitive(self):
        ctx = IterationContext(max_iterations=3, max_total_iterations=10)
        task = {"objective": "obj"}
        s1 = ctx.subtask_signature(task, {"action": "build X"}, "candidate output")
        s2 = ctx.subtask_signature(task, {"action": "build X"}, "candidate output")
        s3 = ctx.subtask_signature(task, {"action": "build Y"}, "candidate output")
        assert s1 == s2
        assert s1 != s3

    def test_cached_decision_off_by_default(self):
        ctx = IterationContext(max_iterations=3, max_total_iterations=10)
        sig = ctx.subtask_signature({}, {"action": "a"}, "o")
        ctx.remember_decision(sig, ReviewDecision.APPROVE)
        assert ctx.decision_cache == {}  # disabled → no write
        assert ctx.cached_decision(sig) is None

    def test_cache_stores_only_approve(self):
        ctx = IterationContext(max_iterations=3, max_total_iterations=10)
        ctx.decision_cache_enabled = True
        sig_ok = ctx.subtask_signature({}, {"action": "a"}, "o")
        sig_bad = ctx.subtask_signature({}, {"action": "b"}, "o")
        ctx.remember_decision(sig_ok, ReviewDecision.APPROVE)
        ctx.remember_decision(sig_bad, ReviewDecision.REJECT)
        assert ctx.cached_decision(sig_ok) == ReviewDecision.APPROVE
        assert ctx.cached_decision(sig_bad) is None

    def test_check_token_budget_no_breach(self):
        ctx = IterationContext(max_iterations=3, max_total_iterations=10)
        assert ctx.check_token_budget("plan_review", tokens_used=100, token_budget=350) is None
        assert ctx.budget_violations == []

    def test_check_token_budget_records_breach_never_blocks(self):
        ctx = IterationContext(max_iterations=3, max_total_iterations=10)
        rec = ctx.check_token_budget(
            "candidate_review",
            tokens_used=500,
            token_budget=300,
            latency_ms=9000,
            latency_budget_ms=6000,
        )
        assert rec is not None
        assert "tokens" in rec["breaches"] and "latency_ms" in rec["breaches"]
        assert ctx.budget_violations == [rec]


# ── RD-9: plan-reminder call-site contract ─────────────────────────────────


class TestPlanReminderCallSite:
    async def test_reminder_fires_on_cadence(self):
        review = FakeReviewService(with_reminder=True)
        knobs = ReviewPlaneKnobs(reminder_cadence=2)
        delegator, primitives = _make_delegator(review, review_knobs=knobs)
        await delegator.delegate(_plan("a", "b", "c"))
        # step_index 2 hits cadence → 3rd prompt carries the reminder.
        assert review.reminder_calls == [2]
        assert primitives.calls[2]["prompt"].startswith("REMINDER@2")
        assert not primitives.calls[0]["prompt"].startswith("REMINDER")

    async def test_default_cadence_no_reminder(self):
        review = FakeReviewService(with_reminder=True)
        delegator, primitives = _make_delegator(review)  # cadence default 0
        await delegator.delegate(_plan("a", "b", "c"))
        assert review.reminder_calls == []
        assert not any(c["prompt"].startswith("REMINDER") for c in primitives.calls)

    async def test_helper_absence_tolerated(self):
        review = FakeReviewService(with_reminder=False)  # no build_plan_reminder
        knobs = ReviewPlaneKnobs(reminder_cadence=2)
        delegator, primitives = _make_delegator(review, review_knobs=knobs)
        # Must not raise despite the missing helper.
        result = await delegator.delegate(_plan("a", "b", "c"))
        assert len(primitives.calls) == 3
        assert result.all_approved is True


# ── Manifest YAML ──────────────────────────────────────────────────────────


class TestReviewPlaneManifest:
    def _load(self):
        return yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_manifest_loads_with_merge_point(self):
        m = self._load()
        assert m["version"] == 1
        assert m["tuning_surface_class"] == 1
        assert "merge_point" in m
        assert "config_applicator" in m["merge_point"]["consumer"]

    def test_all_required_knobs_declared(self):
        m = self._load()
        names = {k["name"].split(".")[-1] for k in m["knobs"]}
        required = {
            "review_trigger_complexity_threshold",
            "max_review_iterations",
            "max_total_review_iterations",
            "reminder_cadence",
            "per_subtask_review_enabled",
            "review_majority_k",
            "request_evidence_round_budget",
            "review_token_multiplier",
            "input_filter_window",
            "decision_cache_enabled",
        }
        assert required <= names

    def test_each_knob_has_governance_fields(self):
        m = self._load()
        for knob in m["knobs"]:
            assert "default" in knob, knob["name"]
            assert knob["dtype"] in {"int", "float", "bool"}, knob["name"]
            assert knob["restart_cost"] == "none", knob["name"]
            assert "provenance" in knob, knob["name"]

    def test_lb2_per_decision_budgets(self):
        m = self._load()
        b = m["per_decision_budgets"]
        assert b["plan_review"]["token_budget"] == 350
        assert b["candidate_review"]["token_budget"] == 300
        assert b["rubric_authoring"]["token_budget"] == 800
        assert b["rubric_authoring"]["amortized"] is True
        assert b["rubric_grading"]["token_budget"] == 180
