"""Architect review and aggregation services.

Decision-plane core (H3: RD-1/3/5/6/8/9 + H1 TM-3). The heavyweight additions to
``ArchitectReviewService`` implement an explicit, typed, evidence-linked review
decision surface in SHADOW mode: it emits/records but never enforces. Enforcement
is gated by ``review_decision_enforce`` (blocked on H-LB LB-6) and is NOT wired
here. Model calls stay on the existing ``primitives.llm_call`` seam; every unit is
exercised with stub/fake completion callables (zero real inference).
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import replace
from typing import Any, Callable, TYPE_CHECKING

from src.config import get_config
from src.config import _registry_timeout
from src.proactive_delegation.review_grammar import parse_review_decision
from src.proactive_delegation.types import (
    ArchitectReview,
    PlanReviewResult,
    ReviewDecision,
    SubtaskResult,
)

# CP1: the RD-3 mechanical verifier-precedence + reject-admissibility constants now
# live in the deterministic reducer (policy_reducer), which SUBSUMES this precedence.
# review_service emits recommendations + findings and DELEGATES the precedence
# decision to the reducer; the constants are re-exported here for back-compat
# (existing importers and tests read them off this module).
from src.proactive_delegation.policy_reducer import (
    FA_CANDIDATE,
    FR_CANDIDATE,  # noqa: F401 — re-exported for back-compat (tests/consumers import it off this module)
    OBJECTIVE_EVIDENCE_KINDS,
    conclusive_verdict as _reducer_conclusive_verdict,
    fail_certificates as _reducer_fail_certificates,
    verifier_precedence_recommendation,
)

if TYPE_CHECKING:
    from src.llm_primitives import LLMPrimitives

logger = logging.getLogger(__name__)

# Default review timeout from registry
_REVIEW_TIMEOUT = float(_registry_timeout("external", "review_service", 15))


# ── Decision-plane trace categories + constants ───────────────────────────────
# These strings match ``src.trace.store.EventCategory`` values (kept as literals so
# the service does not hard-depend on the trace package at import time; the emit
# helper imports the store lazily and best-effort).
CAT_REVIEW_DECISION = "review_decision"
CAT_REVIEW_ESCALATION = "review_escalation"
CAT_PLAN_REMINDER = "plan_reminder"

# RD-3 verifier-precedence disagreement categories (FA_CANDIDATE/FR_CANDIDATE) and
# the RD-8 objective-evidence kinds (OBJECTIVE_EVIDENCE_KINDS) are defined in
# policy_reducer and imported above. A conclusive objective verdict overrides the
# reviewer; the disagreement is logged as a FA/FR *candidate* for the calibration
# ledger (H4). An evidence-free reject is inadmissible as enforcement (overcorrection
# runs 10:1–440:1). The DECISION about which precedence branch applies is the
# reducer's (verifier_precedence_recommendation); this service only composes the
# resulting ArchitectReview + shadow trace emission.

# RD-5 env gate mirroring safety_gate.SAFETY_GATE_WARN_ONLY (default ON): a would-be
# BLOCKING decision is downgraded + logged rather than enforced.
REVIEW_WARN_ONLY_ENV = "REVIEW_DECISION_WARN_ONLY"

# RD-7 Trinity tri-role axis: a review-decision turn IS a Verifier turn. Tag every
# emitted review-plane trace event's detail with the Trinity ``assigned_role`` so the
# tri-role shadow telemetry (scripts/analysis/trinity_shadow_telemetry.py, which scans
# the telemetry detail/``data`` for ``assigned_role``) can correlate review dispatches
# with Verifier-role semantics. This axis is ORTHOGONAL to the model role carried in
# ``Event.role`` (e.g. ``architect_general``); it mirrors the convention in
# routing_decision.routing_meta()'s ``assigned_role`` field. Value == TrinityRole.VERIFIER.value.
REVIEW_ASSIGNED_ROLE = "verifier"


def _estimate_tokens(text: str | None) -> int:
    """Cheap local token estimate (~4 chars/token) for decisions with no usage data."""
    if not text:
        return 0
    return max(1, round(len(text) / 4))


class AggregationService:
    """Service for combining outputs from multiple specialists.

    Strategies:
    - concatenate: Simple concatenation with headers
    - merge_code: Merge code outputs intelligently
    - structured: Combine into structured JSON
    """

    def aggregate(
        self,
        results: list[SubtaskResult],
        strategy: str = "concatenate",
    ) -> str:
        """Aggregate multiple specialist outputs into final result."""
        if not results:
            return ""

        if strategy == "concatenate":
            return self._aggregate_concatenate(results)
        elif strategy == "merge_code":
            return self._aggregate_merge_code(results)
        elif strategy == "structured":
            return self._aggregate_structured(results)
        else:
            logger.warning(f"Unknown aggregation strategy '{strategy}', using concatenate")
            return self._aggregate_concatenate(results)

    def _aggregate_concatenate(self, results: list[SubtaskResult]) -> str:
        """Simple concatenation with section headers."""
        sections = []
        for result in results:
            if (result.success or getattr(result, 'partial', False)) and result.output:
                header = f"## {result.subtask_id} ({result.role})"
                sections.append(f"{header}\n\n{result.output}")
        return "\n\n---\n\n".join(sections)

    def _aggregate_merge_code(self, results: list[SubtaskResult]) -> str:
        """Merge code outputs, handling imports and dependencies."""
        imports = set()
        code_blocks = []

        for result in results:
            if (not result.success and not getattr(result, 'partial', False)) or not result.output:
                continue

            lines = result.output.split("\n")
            current_block = []

            for line in lines:
                # Extract imports
                if line.startswith("import ") or line.startswith("from "):
                    imports.add(line)
                else:
                    current_block.append(line)

            if current_block:
                code_blocks.append(
                    f"# From {result.subtask_id} ({result.role})\n" + "\n".join(current_block)
                )

        # Combine imports at top, then code blocks
        output_parts = []
        if imports:
            output_parts.append("\n".join(sorted(imports)))
        if code_blocks:
            output_parts.append("\n\n".join(code_blocks))

        return "\n\n".join(output_parts)

    def _aggregate_structured(self, results: list[SubtaskResult]) -> str:
        """Combine into structured JSON output."""
        structured = {
            "results": [
                {
                    "subtask_id": r.subtask_id,
                    "role": r.role,
                    "success": r.success,
                    "output": r.output,
                    "error": r.error,
                }
                for r in results
            ],
            "summary": {
                "total": len(results),
                "successful": sum(1 for r in results if r.success),
                "failed": sum(1 for r in results if not r.success),
            },
        }
        return json.dumps(structured, indent=2)


class ArchitectReviewService:
    """Service for architect to review specialist outputs.

    The architect evaluates outputs against the original spec and provides
    feedback for iteration or approval.

    IMPORTANT: Prompts are designed for minimal token output from expensive
    architect models (Qwen3-235B, Qwen3-Coder-480B).

    RD-6 framing-neutrality audit (legacy prompts below): ``REVIEW_PROMPT_TEMPLATE``
    and ``QUICK_REVIEW_PROMPT`` were audited for the two measured attack surfaces —
    (a) "assume competent / expert authored this" priming and (b) explain-then-fix as
    the PRIMARY path (which doubles false-rejects, intake-836). Both legacy prompts
    are already free of (a) and put the verdict field FIRST while demanding
    "JSON only (no explanation)", so they do not fall into (b). They are therefore
    left byte-identical (the legacy ``review()``/``review_plan()`` paths must not
    change). The framing-neutral, sanitized-package, pointwise reviewer for the new
    decision plane is ``FRAMING_NEUTRAL_REVIEW_PROMPT`` + ``review_candidate()``, and
    the plan rubric (phase-coverage/order/executor-alignment, NOT prose) is
    ``PLAN_REVIEW_RUBRIC_PROMPT`` + ``review_plan_rubric()``.
    """

    # Concise review prompt - minimize architect output tokens
    REVIEW_PROMPT_TEMPLATE = """Review specialist output. Be BRIEF.

Objective: {objective}
Subtask: {action}
Output (truncated):
{output}

Reply JSON only (no explanation):
{{"d":"approve|changes|escalate|reject","s":0.0-1.0,"f":"<10 words","c":["fix1"]}}

d=decision, s=score, f=feedback, c=changes (optional, max 3 items)"""

    # Even more compact for simple approve/reject decisions
    QUICK_REVIEW_PROMPT = """Review: {action}
Output: {output_preview}
Reply: {{"d":"approve|changes","s":0.0-1.0,"f":"<5 words"}}"""

    # RD-6: framing-neutral, pointwise, single-candidate reviewer over the SANITIZED
    # CandidatePackage view. Verdict FIRST; fixes only AFTER the verdict, phrased as
    # checkable artifacts (not prose advice); no competence priming; no comparison to
    # alternatives. Output shape matches review_decision_response_schema so
    # parse_review_decision() (RA-9) validates it.
    FRAMING_NEUTRAL_REVIEW_PROMPT = """Adjudicate ONE candidate against its objective.

Objective: {objective}
Acceptance checks:
{acceptance_checks}
Candidate output:
{outputs}

Rules:
- Judge only THIS candidate against the acceptance checks. Do not compare to alternatives.
- Decide FIRST. Do not explain or reason before the decision.
- Any fix belongs AFTER the verdict, phrased as a checkable test — not prose advice.

Reply JSON only:
{{"decision":"approve|request_changes|request_evidence|reject|reject_to_empty|escalate","confidence":0.0-1.0,"blocking":{{"tripwire":false}},"advisory":{{"score":0.0-1.0,"feedback":"<=12 words"}}}}"""

    # RD-9: plan rubric — checks phase-coverage / order / executor-alignment ONLY.
    # Explicitly NOT prose quality; penalizes over-specification as much as gaps.
    PLAN_REVIEW_RUBRIC_PROMPT = """Adjudicate a PLAN (not prose). Decide FIRST.

Objective: {objective}
Type: {task_type}
Plan steps:
{steps}

Judge ONLY these three axes — ignore wording/prose quality, do not reward verbosity:
- phase_coverage: does the plan cover every phase the objective needs (no missing step)?
- order: is every step ordered after the steps it depends on?
- executor_alignment: is each step assigned to an actor able to execute it?
Penalize over-specification (needless steps) as much as gaps.

Reply JSON only:
{{"decision":"approve|request_changes|reject_to_empty|escalate","confidence":0.0-1.0,"phase_coverage":true,"order":true,"executor_alignment":true,"advisory":{{"score":0.0-1.0,"feedback":"<=12 words"}}}}"""

    # TaskIR generation prompt (for future use when frontdoor queries architect)
    TASKIR_GENERATION_PROMPT = """Break down task into subtasks. Be MINIMAL.

Task: {objective}

Reply JSON only:
{{"steps":[{{"id":"S1","actor":"coder|worker|math","action":"<10 words","out":["file.py"]}}]}}

Rules:
- Max 5 steps
- actor: coder (code), worker (docs/tests), math (proofs)
- action: imperative, <10 words
- out: expected output files"""

    def __init__(
        self,
        primitives: "LLMPrimitives",
        architect_role: str | None = None,
        *,
        reviewer_role: "str | None" = None,
        warn_only: bool | None = None,
        trace_sink: "Callable[[Any], None] | None" = None,
        trace_db_path: "str | None" = None,
    ):
        """Initialize the review service.

        Args:
            primitives: LLM primitives seam (all model calls go through it).
            architect_role: Explicit role override. When ``None`` (default), RD-1's
                config-level reviewer binding resolves it (``resolve_reviewer_role``);
                the default binding is ``architect_general`` so behavior is unchanged.
            reviewer_role: Optional binding override forwarded to the resolver.
            warn_only: RD-5 shadow-downgrade of would-be blocking decisions. ``None``
                reads ``REVIEW_DECISION_WARN_ONLY`` (default ON), mirroring safety_gate.
            trace_sink: TM-3 test seam — a callable receiving each ``Event`` instead of
                writing through to the store. ``None`` → best-effort write via emit.py.
            trace_db_path: Optional trace DB path override (tests point at a temp DB).
        """
        self.primitives = primitives
        cfg = get_config()
        deleg_cfg = cfg.delegation
        # RD-1: config-level reviewer role binding (default → architect_general).
        if architect_role is None:
            from src.roles import resolve_reviewer_role

            architect_role = str(resolve_reviewer_role(override=reviewer_role, config=cfg))
        self.architect_role = architect_role
        self.max_review_tokens = deleg_cfg.max_review_tokens
        self.max_taskir_tokens = deleg_cfg.max_taskir_tokens
        self.max_plan_review_tokens = deleg_cfg.max_plan_review_tokens
        # RD-5: warn-only shadow downgrade (env-gated; default ON, mirrors safety_gate).
        if warn_only is None:
            warn_only = os.environ.get(REVIEW_WARN_ONLY_ENV, "1").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
        self.warn_only = warn_only
        # TM-3: trace emission sink (injectable for tests; default → src/trace/emit.py).
        self._trace_sink = trace_sink
        self._trace_db_path = trace_db_path

    # ── TM-3 / RD-* trace emission (best-effort, never observable) ────────────

    def _emit_review_event(
        self,
        *,
        category: str,
        summary: str,
        detail: Any,
        status: str | None = None,
        role: str | None = None,
        session_id: Any | None = None,
        trial_id: int | None = None,
    ) -> None:
        """Write-through a review-plane trace event. NEVER raises, NEVER alters the
        caller's return value (TM-3 always-on emission). When a ``trace_sink`` is
        injected it receives the ``Event``; otherwise the event is committed via
        ``src/trace/emit.py`` (write-through, content-addressed, idempotent).
        """
        try:
            from src.trace.store import Event, EventSource, detail_to_json

            # RD-7: tag the emitted detail with the Trinity Verifier-role axis (mirror of
            # routing_meta()'s ``assigned_role``). Non-mutating shallow merge — the
            # caller's dict is untouched and the service's returns stay byte-identical.
            if isinstance(detail, dict) and "assigned_role" not in detail:
                detail = {**detail, "assigned_role": REVIEW_ASSIGNED_ROLE}

            ev = Event(
                ts_utc="",  # emit() stamps 'now' when empty
                source=EventSource.REVIEW_PLANE,
                source_path="",  # emit() assigns a content-addressed synthetic path
                source_line=None,
                session_id=str(session_id) if session_id is not None else None,
                trial_id=trial_id if isinstance(trial_id, int) else None,
                role=role or self.architect_role,
                category=category,
                status=status,
                summary=summary,
                detail_json=detail_to_json(detail),
            )
            if self._trace_sink is not None:
                self._trace_sink(ev)
                return
            from src.trace.emit import emit

            if self._trace_db_path is not None:
                emit(ev, db_path=self._trace_db_path)
            else:
                emit(ev)
        except Exception as exc:  # pragma: no cover - emission is best-effort
            logger.debug("Review trace emission skipped: %s", exc)

    @staticmethod
    def _response_tokens(response: Any) -> dict[str, int]:
        """Per-decision token accounting (TM-3 / RD-12).

        Prefers real counts off a response object (``tokens_generated`` on
        ``LLMResult``, a ``usage`` mapping) and falls back to a length-based estimate
        for plain-string completions (the ``llm_call`` seam returns ``str``).
        """
        text = response if isinstance(response, str) else str(getattr(response, "text", response) or "")
        tokens_out = _estimate_tokens(text)
        tg = getattr(response, "tokens_generated", None)
        if isinstance(tg, int) and tg > 0:
            tokens_out = tg
        usage = getattr(response, "usage", None)
        if isinstance(usage, dict):
            for key in ("completion_tokens", "output_tokens", "tokens_out"):
                val = usage.get(key)
                if isinstance(val, int) and val > 0:
                    tokens_out = val
                    break
        return {"tokens_out": int(tokens_out), "chars_out": len(text)}

    def review(
        self,
        spec: dict[str, Any],
        subtask: dict[str, Any],
        output: str,
        quick_mode: bool = False,
        *,
        session_id: Any | None = None,
        trial_id: int | None = None,
    ) -> ArchitectReview:
        """Have architect review a specialist's output.

        TM-3: every invocation emits a REVIEW_DECISION trace event (with per-decision
        latency_ms + token counts) regardless of downstream acting. The RETURNED
        ``ArchitectReview`` is byte-identical to the pre-decision-plane behavior.
        """
        subtask_id = subtask.get("id", "unknown")
        action = subtask.get("action", "")

        # Truncate output aggressively to save input tokens
        output_truncated = output[:500] + "..." if len(output) > 500 else output

        # Build concise review prompt - only include objective, not full spec
        objective = spec.get("objective", "")[:200]

        if quick_mode:
            prompt = self.QUICK_REVIEW_PROMPT.format(
                action=action[:50],
                output_preview=output[:200],
            )
        else:
            prompt = self.REVIEW_PROMPT_TEMPLATE.format(
                objective=objective,
                action=action,
                output=output_truncated,
            )

        start = time.perf_counter()
        parse_ok = True
        tokens: dict[str, int] = {"tokens_out": 0}
        try:
            # Call architect with strict token limit
            response = self.primitives.llm_call(
                prompt,
                role=self.architect_role,
                n_tokens=self.max_review_tokens,
            )

            # Parse abbreviated JSON response
            review_data = self._parse_review_response(response)

            # Map abbreviated keys to full names
            decision_str = review_data.get("d", review_data.get("decision", "changes"))
            # Normalize decision string
            if decision_str == "changes":
                decision_str = "request_changes"
            decision = ReviewDecision(decision_str)

            review = ArchitectReview(
                subtask_id=subtask_id,
                decision=decision,
                feedback=review_data.get("f", review_data.get("feedback", "")),
                score=float(review_data.get("s", review_data.get("score", 0.5))),
                suggested_changes=review_data.get("c", review_data.get("suggested_changes", [])),
                approved_output=output if decision == ReviewDecision.APPROVE else None,
            )
            tokens = self._response_tokens(response)

        except Exception as e:
            logger.warning(f"Architect review failed: {e}", exc_info=True)
            parse_ok = False
            # Default to request_changes on failure
            review = ArchitectReview(
                subtask_id=subtask_id,
                decision=ReviewDecision.REQUEST_CHANGES,
                feedback=f"Review failed: {e}",
                score=0.3,
            )

        # TM-3: always-on shadow emission (side effect only; return is unchanged).
        latency_ms = (time.perf_counter() - start) * 1000.0
        self._emit_review_event(
            category=CAT_REVIEW_DECISION,
            summary=f"review {subtask_id}: {review.decision.value}",
            status=review.decision.value,
            detail={
                "mode": "review",
                "subtask_id": subtask_id,
                "decision": review.decision.value,
                "score": review.score,
                "confidence": review.confidence,
                "tripwire": review.tripwire,
                "quick_mode": quick_mode,
                "parse_ok": parse_ok,
                "latency_ms": latency_ms,
                "tokens": tokens,
            },
            session_id=session_id,
            trial_id=trial_id,
        )
        return review

    def review_plan(
        self,
        objective: str,
        task_type: str,
        plan_steps: list[dict[str, Any]],
        timeout_seconds: float = _REVIEW_TIMEOUT,
        *,
        session_id: Any | None = None,
        trial_id: int | None = None,
    ) -> PlanReviewResult | None:
        """Have architect review a plan before specialist execution.

        Returns PlanReviewResult on success, None on timeout/error.
        Non-blocking: never prevents request completion.

        TM-3: every invocation emits a REVIEW_DECISION trace event (latency_ms +
        token counts) regardless of whether ``plan_review`` acts — this is the
        DECOUPLED shadow-emission path (RD-5): the trace flows even when the
        ``plan_review`` feature (which requires ``memrl``) is off. The RETURNED value
        is byte-identical to the pre-decision-plane behavior.
        """
        from src.prompt_builders import build_plan_review_prompt

        prompt = build_plan_review_prompt(objective, task_type, plan_steps)

        start = time.perf_counter()
        try:
            response = self.primitives.llm_call(
                prompt,
                role=self.architect_role,
                n_tokens=self.max_plan_review_tokens,
            )

            # Parse abbreviated JSON response
            review_data = self._parse_review_response(response)

            decision = review_data.get("d", review_data.get("decision", "ok"))
            # Normalize valid decisions
            valid_decisions = {"ok", "reorder", "drop", "add", "reroute"}
            if decision not in valid_decisions:
                decision = "ok"

            result = PlanReviewResult(
                decision=decision,
                score=float(review_data.get("s", review_data.get("score", 0.5))),
                feedback=review_data.get("f", review_data.get("feedback", "")),
                patches=review_data.get("p", review_data.get("patches", [])),
                raw_response=response[:200],
            )
            latency_ms = (time.perf_counter() - start) * 1000.0
            self._emit_review_event(
                category=CAT_REVIEW_DECISION,
                summary=f"plan_review: {result.decision}",
                status=result.decision,
                detail={
                    "mode": "plan_review",
                    "decision": result.decision,
                    "score": result.score,
                    "patches": len(result.patches),
                    "parse_ok": True,
                    "latency_ms": latency_ms,
                    "tokens": self._response_tokens(response),
                },
                session_id=session_id,
                trial_id=trial_id,
            )
            return result

        except Exception as e:
            logger.warning(f"Plan review failed: {e}", exc_info=True)
            latency_ms = (time.perf_counter() - start) * 1000.0
            self._emit_review_event(
                category=CAT_REVIEW_DECISION,
                summary="plan_review: error",
                status="error",
                detail={
                    "mode": "plan_review",
                    "error": str(e),
                    "parse_ok": False,
                    "latency_ms": latency_ms,
                    "tokens": {"tokens_out": 0},
                },
                session_id=session_id,
                trial_id=trial_id,
            )
            return None  # Non-blocking -- proceed without review

    def generate_taskir(self, objective: str) -> dict[str, Any]:
        """Have architect generate minimal TaskIR for an objective."""
        prompt = self.TASKIR_GENERATION_PROMPT.format(
            objective=objective[:300],  # Truncate long objectives
        )

        try:
            response = self.primitives.llm_call(
                prompt,
                role=self.architect_role,
                n_tokens=self.max_taskir_tokens,
            )

            taskir_data = self._parse_review_response(response)

            # Normalize abbreviated format to full TaskIR
            steps = taskir_data.get("steps", [])
            normalized_steps = []
            for i, step in enumerate(steps):
                normalized_steps.append(
                    {
                        "id": step.get("id", f"S{i + 1}"),
                        "actor": step.get("actor", "worker"),
                        "action": step.get("action", ""),
                        "outputs": step.get("out", step.get("outputs", [])),
                        "inputs": step.get("in", step.get("inputs", [])),
                    }
                )

            return {
                "task_id": f"arch-{uuid.uuid4().hex[:8]}",
                "objective": objective,
                "plan": {"steps": normalized_steps},
            }

        except Exception as e:
            logger.warning(f"TaskIR generation failed: {e}", exc_info=True)
            # Return single-step fallback
            return {
                "task_id": f"arch-{uuid.uuid4().hex[:8]}",
                "objective": objective,
                "plan": {
                    "steps": [
                        {
                            "id": "S1",
                            "actor": "coder",
                            "action": objective[:50],
                            "outputs": ["output.txt"],
                        }
                    ]
                },
            }

    def _parse_review_response(self, response: str) -> dict[str, Any]:
        """Parse JSON from architect response."""
        # Try to extract JSON from response
        response = response.strip()

        # Handle markdown code blocks
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end > start:
                response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                response = response[start:end].strip()

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to find JSON object in response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(response[start:end])
                except json.JSONDecodeError:
                    pass

            logger.warning(f"Could not parse review response: {response[:200]}")
            return {"decision": "request_changes", "feedback": "Parse error"}

    # ══════════════════════════════════════════════════════════════════════════
    # Decision plane (SHADOW-ONLY): RD-3 / RD-5 / RD-6 / RD-8 / RD-9.
    #
    # These methods EMIT and RECORD but never enforce. Enforcement is gated by the
    # ``review_decision_enforce`` flag (blocked on H-LB LB-6) and is intentionally
    # NOT wired here. They stay on the ``primitives.llm_call`` seam and are fully
    # exercisable with stub completion callables.
    # ══════════════════════════════════════════════════════════════════════════

    # ── rendering helpers ────────────────────────────────────────────────────

    @staticmethod
    def _render_outputs(outputs: list[dict[str, Any]] | None) -> str:
        lines = []
        for o in (outputs or [])[:8]:
            ref = str(o.get("ref", ""))[:400]
            lines.append(f"- [{o.get('type', 'artifact')}] {ref}")
        return "\n".join(lines) or "(none)"

    @staticmethod
    def _render_checks(checks: list[dict[str, Any]] | None) -> str:
        lines = []
        for c in (checks or [])[:8]:
            lines.append(f"- {c.get('id', '?')}: {str(c.get('statement', ''))[:120]}")
        return "\n".join(lines) or "(none)"

    @staticmethod
    def _render_plan_steps(plan_steps: list[dict[str, Any]] | None) -> str:
        lines = []
        for s in (plan_steps or [])[:8]:
            sid = s.get("id", "S?")
            actor = s.get("actor", s.get("role", "worker"))
            action = str(s.get("action", ""))[:50]
            deps = s.get("deps", s.get("inputs", []))
            dep_str = f" ({','.join(str(d) for d in deps[:3])})" if deps else ""
            lines.append(f"{sid}:{actor}:{action}{dep_str}")
        return "\n".join(lines) or "(no steps)"

    # ── RD-6: framing-neutral, pointwise, single-candidate review ─────────────

    def review_candidate(
        self,
        sanitized_view: dict[str, Any],
        *,
        subtask_id: str = "candidate",
        session_id: Any | None = None,
        trial_id: int | None = None,
    ) -> ArchitectReview:
        """Pointwise review of a PRE-SANITIZED CandidatePackage view (RD-6).

        ``sanitized_view`` MUST be the ``candidate_package.schema.json``
        ``sanitized_view`` projection (author self-assessment / confidence assertions
        / quality labels already stripped at assembly time). This method consumes
        ONLY that projection; if it detects framing-leaking keys it logs and ignores
        them (defense-in-depth — sanitization is an assembly-time contract).

        Emits a REVIEW_DECISION trace event with latency + tokens. A parse failure is
        recorded and withheld as REQUEST_EVIDENCE — never a reject (admissibility).
        """
        for banned in ("author_self_assessment", "author_confidence_assertion", "quality_labels"):
            if isinstance(sanitized_view, dict) and banned in sanitized_view:
                logger.warning(
                    "review_candidate received an UNSANITIZED package (found %r); "
                    "ignoring framing-leaking field",
                    banned,
                )

        objective = str((sanitized_view or {}).get("objective", ""))[:400]
        outputs = self._render_outputs((sanitized_view or {}).get("outputs"))
        checks = self._render_checks((sanitized_view or {}).get("acceptance_checks"))
        prompt = self.FRAMING_NEUTRAL_REVIEW_PROMPT.format(
            objective=objective, outputs=outputs, acceptance_checks=checks
        )

        start = time.perf_counter()
        response: Any = ""
        try:
            response = self.primitives.llm_call(
                prompt, role=self.architect_role, n_tokens=self.max_review_tokens
            )
        except Exception as exc:
            logger.warning("review_candidate model call failed: %s", exc)
            response = ""

        text = response if isinstance(response, str) else str(getattr(response, "text", "") or "")
        obj, failure = parse_review_decision(text)
        if obj is None:
            # Admissibility: a parse failure must NOT become a reject — withhold.
            review = ArchitectReview(
                subtask_id=subtask_id,
                decision=ReviewDecision.REQUEST_EVIDENCE,
                feedback=f"parse_failure:{failure.reason.value if failure else 'unknown'}",
                score=0.0,
                confidence=0.0,
            )
            parse_ok = False
        else:
            review = self._architect_review_from_decision(obj, subtask_id)
            parse_ok = True

        latency_ms = (time.perf_counter() - start) * 1000.0
        self._emit_review_event(
            category=CAT_REVIEW_DECISION,
            summary=f"review_candidate {review.subtask_id}: {review.decision.value}",
            status=review.decision.value,
            detail={
                "mode": "review_candidate",
                "framing_neutral": True,
                "pointwise": True,
                "candidate_ref": (sanitized_view or {}).get("task_ref"),
                "decision": review.decision.value,
                "confidence": review.confidence,
                "score": review.score,
                "tripwire": review.tripwire,
                "parse_ok": parse_ok,
                "parse_failure": failure.to_dict() if failure else None,
                "latency_ms": latency_ms,
                "tokens": self._response_tokens(response),
            },
            session_id=session_id,
            trial_id=trial_id,
        )
        return review

    @staticmethod
    def _architect_review_from_decision(
        obj: dict[str, Any], subtask_id: str
    ) -> ArchitectReview:
        """Map a validated review_decision object → ArchitectReview (RD-6)."""
        blocking = obj.get("blocking") or {}
        advisory = obj.get("advisory") or {}
        return ArchitectReview(
            subtask_id=obj.get("subtask_id", subtask_id),
            decision=ReviewDecision(obj["decision"]),
            feedback=str(advisory.get("feedback", "")),
            score=float(advisory.get("score", 0.0) or 0.0),
            suggested_changes=[
                i["summary"] for i in blocking.get("blocking_issues", []) if "summary" in i
            ],
            approved_output=None,
            confidence=float(obj.get("confidence", 0.0) or 0.0),
            tripwire=bool(blocking.get("tripwire", False)),
            evidence=obj.get("evidence", []) or [],
            verifier_requests=obj.get("verifier_requests", []) or [],
        )

    # ── RD-3: verifier precedence (mechanical, three-valued) ──────────────────

    @staticmethod
    def _conclusive_verdict(report: dict[str, Any]) -> str:
        """Aggregate a VerificationReport-shaped dict → pass|fail|inconclusive.

        Precedence over the reviewer applies ONLY to conclusive (pass/fail) verdicts;
        ``inconclusive`` hands control back to the reviewer. Uses ``summary.
        conclusive_verdict`` when present, else derives from required checks
        (any required inconclusive → inconclusive; else any required fail → fail;
        else ≥1 required pass and none fail/inconclusive → pass; else inconclusive).

        CP1: delegates to ``policy_reducer.conclusive_verdict`` — the reducer subsumes
        this precedence mechanic. Behavior is byte-identical (same algorithm).
        """
        return _reducer_conclusive_verdict(report)

    @staticmethod
    def _fail_certificates(report: dict[str, Any]) -> list[dict[str, Any]]:
        """Collect failing checks' certificates — the request_evidence payload.

        CP1: delegates to ``policy_reducer.fail_certificates`` (subsumed).
        """
        return _reducer_fail_certificates(report)

    def apply_verifier_precedence(
        self,
        review: ArchitectReview,
        report: dict[str, Any] | None,
        *,
        session_id: Any | None = None,
        trial_id: int | None = None,
    ) -> ArchitectReview:
        """Mechanical verifier-precedence over a reviewer decision (RD-3).

        Rules (conclusive objective verdicts override reviewer claims):
          1. reviewer-approve + conclusive-gate-FAIL → emit ``fa_candidate``; override
             to REQUEST_EVIDENCE with the failing certificate(s) attached.
          2. reviewer-reject + conclusive-gate-PASS → emit ``fr_candidate``; downgrade
             to REQUEST_EVIDENCE (never keep a reject when objective PASSED — rule 3).
          3. objective-PASS + advisory-low reject-class → REQUEST_EVIDENCE (folded into
             rule 2: objective PASS never yields a reject).
          4. ``inconclusive`` → defer to the reviewer (return unchanged).

        Emits a disagreement trace event on override. Returns the (possibly adjusted)
        review; whether it is ACTED on is the shadow/enforce split (RD-5), not here.

        CP1: the precedence DECISION (which disagreement class, what adjusted verdict)
        is now the reducer's — ``policy_reducer.verifier_precedence_recommendation``.
        This method only composes the resulting ``ArchitectReview`` + shadow trace so
        the return value and emitted event stay byte-identical to the RD-3 behavior.
        """
        verdict = self._conclusive_verdict(report or {})
        adjusted_decision, trace_cat = verifier_precedence_recommendation(
            review.decision.value, verdict
        )
        if trace_cat is None:  # inconclusive verdict, or reviewer/verifier agreement
            return review

        fail_certs = self._fail_certificates(report or {})
        adjusted: ArchitectReview

        if trace_cat == FA_CANDIDATE:  # rule 1: reviewer-approve + conclusive-fail
            evidence = list(review.evidence or []) + [
                {"kind": "gate_result", "ref": fc.get("check_id"), "summary": "conclusive FAIL certificate"}
                for fc in fail_certs
            ]
            adjusted = replace(
                review,
                decision=ReviewDecision(adjusted_decision),
                approved_output=None,
                evidence=evidence,
            )
        else:  # FR_CANDIDATE — rules 2 + 3: reviewer-reject + conclusive-pass
            adjusted = replace(
                review, decision=ReviewDecision(adjusted_decision), tripwire=False
            )

        self._emit_review_event(
            category=trace_cat,
            summary=f"verifier precedence {trace_cat}: "
            f"{review.decision.value}->{adjusted.decision.value}",
            status=verdict,
            detail={
                "kind": trace_cat,
                "conclusive_verdict": verdict,
                "original_decision": review.decision.value,
                "adjusted_decision": adjusted.decision.value,
                "report_id": (report or {}).get("report_id"),
                "certificates": fail_certs,
            },
            session_id=session_id,
            trial_id=trial_id,
        )
        return adjusted

    # ── RD-5: warn-only shadow downgrade ──────────────────────────────────────

    def apply_warn_only(
        self,
        review: ArchitectReview,
        *,
        session_id: Any | None = None,
        trial_id: int | None = None,
    ) -> ArchitectReview:
        """Env-gated shadow downgrade of a would-be BLOCKING decision (RD-5).

        Mirrors ``safety_gate.warn_only``: when active, a blocking verdict (REJECT /
        REJECT_TO_EMPTY, or ``tripwire=True``) is downgraded to advisory
        REQUEST_CHANGES and LOGGED — never enforced. When ``warn_only`` is off the
        review is returned unchanged (enforcement remains separately gated by
        ``review_decision_enforce``, which is blocked on H-LB LB-6).
        """
        if not self.warn_only:
            return review
        is_blocking = review.tripwire or review.decision in (
            ReviewDecision.REJECT,
            ReviewDecision.REJECT_TO_EMPTY,
        )
        if not is_blocking:
            return review
        downgraded = replace(review, decision=ReviewDecision.REQUEST_CHANGES, tripwire=False)
        self._emit_review_event(
            category=CAT_REVIEW_DECISION,
            summary=f"warn_only downgrade {review.subtask_id}",
            status="warn_only",
            detail={
                "mode": "warn_only_downgrade",
                "warn_only_active": True,
                "original_decision": review.decision.value,
                "downgraded_decision": downgraded.decision.value,
                "original_tripwire": review.tripwire,
            },
            session_id=session_id,
            trial_id=trial_id,
        )
        return downgraded

    # ── RD-8: reject-admissibility + escalate stub ────────────────────────────

    @staticmethod
    def check_reject_admissibility(review: ArchitectReview) -> bool:
        """True if a REJECT/REJECT_TO_EMPTY carries ≥1 objective-evidence item.

        Non-reject decisions are trivially admissible. Objective-evidence kinds are
        gate_result / test_result / scorer_result (RD-8).
        """
        if review.decision not in (ReviewDecision.REJECT, ReviewDecision.REJECT_TO_EMPTY):
            return True
        return any(
            (e or {}).get("kind") in OBJECTIVE_EVIDENCE_KINDS for e in (review.evidence or [])
        )

    def mark_reject_admissibility(
        self,
        review: ArchitectReview,
        *,
        session_id: Any | None = None,
        trial_id: int | None = None,
    ) -> dict[str, Any]:
        """Return a decision artifact flagging ``unverified_rejection`` (RD-8).

        A reject with no objective evidence is marked ``unverified_rejection=True`` in
        the emitted artifact + a trace event and (in shadow mode) recorded but never
        acted on. The returned dict is a superset of ``ArchitectReview.to_dict()`` so
        the ArchitectReview object itself stays schema-compatible.
        """
        admissible = self.check_reject_admissibility(review)
        unverified = (
            review.decision in (ReviewDecision.REJECT, ReviewDecision.REJECT_TO_EMPTY)
            and not admissible
        )
        artifact = review.to_dict()
        artifact["unverified_rejection"] = unverified
        if unverified:
            self._emit_review_event(
                category=CAT_REVIEW_DECISION,
                summary=f"unverified_rejection {review.subtask_id}",
                status="unverified_rejection",
                detail={
                    "mode": "reject_admissibility",
                    "decision": review.decision.value,
                    "unverified_rejection": True,
                    "reason": "reject without objective evidence (gate/test/scorer)",
                    "evidence_kinds": [
                        (e or {}).get("kind") for e in (review.evidence or [])
                    ],
                },
                session_id=session_id,
                trial_id=trial_id,
            )
        return artifact

    def escalate(
        self,
        review: ArchitectReview,
        *,
        reason: str = "",
        session_id: Any | None = None,
        trial_id: int | None = None,
    ) -> ArchitectReview:
        """RD-8 escalate stub: route to the escalation pipeline surface.

        Emits a REVIEW_ESCALATION trace event and returns the decision unchanged.
        No UI (HS-4-gated). This is the server-side stub surface the escalation
        pipeline consumes.
        """
        self._emit_review_event(
            category=CAT_REVIEW_ESCALATION,
            summary=f"escalate {review.subtask_id}",
            status="escalate",
            detail={
                "subtask_id": review.subtask_id,
                "decision": review.decision.value,
                "confidence": review.confidence,
                "reason": reason or review.feedback,
            },
            session_id=session_id,
            trial_id=trial_id,
        )
        return review

    # ── RD-3/5/8 shadow orchestration entry point ─────────────────────────────

    def shadow_decide(
        self,
        review: ArchitectReview,
        *,
        verification_report: dict[str, Any] | None = None,
        session_id: Any | None = None,
        trial_id: int | None = None,
    ) -> dict[str, Any]:
        """Run the full SHADOW decision pipeline over a review (RD-3/5/8).

        Order: verifier precedence → warn-only downgrade → escalate-stub (if escalate)
        → reject-admissibility marking. Emits trace only; never enforces or mutates
        external state. Returns a decision artifact dict (superset of
        ``ArchitectReview.to_dict()``). This is the single entry point a caller in
        shadow mode (gated by ``review_decision_shadow``) invokes.
        """
        r = review
        if verification_report is not None:
            r = self.apply_verifier_precedence(
                r, verification_report, session_id=session_id, trial_id=trial_id
            )
        r = self.apply_warn_only(r, session_id=session_id, trial_id=trial_id)
        if r.decision == ReviewDecision.ESCALATE:
            self.escalate(r, session_id=session_id, trial_id=trial_id)
        artifact = self.mark_reject_admissibility(r, session_id=session_id, trial_id=trial_id)
        artifact["shadow"] = True
        return artifact

    # ── RD-9: plan-review specifics ───────────────────────────────────────────

    def review_plan_rubric(
        self,
        objective: str,
        task_type: str,
        plan_steps: list[dict[str, Any]],
        *,
        session_id: Any | None = None,
        trial_id: int | None = None,
    ) -> dict[str, Any]:
        """Structured plan review keyed to phase-coverage/order/executor-alignment (RD-9).

        Returns a structured dict (NOT ``PlanReviewResult``) so it is additive to the
        legacy ``review_plan()`` path. Prose quality is deliberately ignored;
        over-specification is penalized like gaps. Shadow-only; emits a trace event.
        """
        steps = self._render_plan_steps(plan_steps)
        prompt = self.PLAN_REVIEW_RUBRIC_PROMPT.format(
            objective=str(objective)[:200], task_type=task_type, steps=steps
        )

        start = time.perf_counter()
        response: Any = ""
        try:
            response = self.primitives.llm_call(
                prompt, role=self.architect_role, n_tokens=self.max_plan_review_tokens
            )
        except Exception as exc:
            logger.warning("review_plan_rubric model call failed: %s", exc)
            response = ""

        data = self._parse_review_response(response) if response else {}
        decision = data.get("decision", data.get("d", "approve"))
        valid = {"approve", "request_changes", "reject_to_empty", "escalate"}
        if decision not in valid:
            decision = "approve"
        advisory = data.get("advisory") or {}
        result = {
            "decision": decision,
            "confidence": float(data.get("confidence", 0.5) or 0.5),
            "phase_coverage": bool(data.get("phase_coverage", True)),
            "order": bool(data.get("order", True)),
            "executor_alignment": bool(data.get("executor_alignment", True)),
            "score": float(advisory.get("score", data.get("s", 0.5)) or 0.5),
            "feedback": str(advisory.get("feedback", data.get("f", ""))),
        }
        latency_ms = (time.perf_counter() - start) * 1000.0
        self._emit_review_event(
            category=CAT_REVIEW_DECISION,
            summary=f"plan_rubric: {result['decision']}",
            status=result["decision"],
            detail={
                "mode": "plan_rubric",
                **result,
                "latency_ms": latency_ms,
                "tokens": self._response_tokens(response),
            },
            session_id=session_id,
            trial_id=trial_id,
        )
        return result

    @staticmethod
    def plan_review_reject_to_empty_fallback(decision: "ReviewDecision | str") -> bool:
        """RD-9: True when a plan-review verdict means 'discard the plan → no-plan default'.

        A ``reject_to_empty`` plan is worse than none: the caller, on True, drops the
        plan and proceeds with its default no-plan workflow. WIRE-POINT (documented for
        the delegator owner; delegator.py is NOT edited here): the plan-execution loop
        calls this on the plan verdict and, on True, replaces the plan with the no-plan
        default rather than iterating on a bad plan.
        """
        d = decision.value if isinstance(decision, ReviewDecision) else str(decision)
        return d == "reject_to_empty"

    def build_plan_reminder(
        self,
        approved_plan: list[dict[str, Any]] | None,
        *,
        cadence_n: int = 5,
        step_index: int = 0,
        emit: bool = False,
        session_id: Any | None = None,
        trial_id: int | None = None,
    ) -> str | None:
        """RD-9 cheap plan-reminder re-injection knob (PREFERRED over re-review).

        Given an APPROVED plan and a cadence N, return a compact reminder message to
        re-inject every N steps/tool-calls (fires when ``step_index > 0`` and
        ``step_index % cadence_n == 0``), else ``None``.

        WIRE-POINT (documented for the delegator owner; delegator.py NOT edited here):
        the execution loop calls this each step and, on a non-``None`` return, prepends
        the reminder to the next specialist prompt. Set ``emit=True`` to also record a
        PLAN_REMINDER trace event (needed for plan-compliance metrics, intake-835).
        """
        if (
            not approved_plan
            or cadence_n <= 0
            or step_index <= 0
            or step_index % cadence_n != 0
        ):
            return None
        steps = self._render_plan_steps(approved_plan)
        message = (
            "Reminder — you are executing this approved plan; stay on it:\n" + steps
        )
        if emit:
            self._emit_review_event(
                category=CAT_PLAN_REMINDER,
                summary=f"plan reminder @step {step_index}",
                status="reminder",
                detail={
                    "step_index": step_index,
                    "cadence_n": cadence_n,
                    "n_steps": len(approved_plan),
                },
                session_id=session_id,
                trial_id=trial_id,
            )
        return message

    @staticmethod
    def iteration_bound_by_compliance_trend(
        compliance_trend: list[float] | None,
        *,
        drift_threshold: float = 0.5,
        collapse_threshold: float = 0.2,
    ) -> str:
        """RD-9 policy HOOK (intended policy documented; minimal implementation).

        INTENDED POLICY: iteration bounds are keyed to the plan-compliance trend —
        ``drift`` → cheap reminder re-injection; ``collapse`` → re-plan. This hook maps
        a per-step compliance-score sequence (each in [0,1]) to one of
        ``'ok' | 'reminder' | 'replan'``. It is the surface the delegator/autopilot
        policy will call; the numeric thresholds are RD-11 tuning-surface candidates
        (declared to ``config_applicator.apply_params``, not owned here).
        """
        if not compliance_trend:
            return "ok"
        latest = float(compliance_trend[-1])
        if latest <= collapse_threshold:
            return "replan"
        if latest <= drift_threshold:
            return "reminder"
        return "ok"
