"""TE-1 (UFH-13) / HS-4 P4 subset — /v1 escalation parity with /chat.

Flag ``v1_escalation`` (default OFF) plus the per-request key ``x_escalation``
(``auto`` | ``off`` | ``architect_general`` | ``force_architect_general``).
Escalation is OPT-IN per request: with the key absent nothing here runs and the
route is byte-identical whether the flag is on or off (golden-pinned both ways).
``auto`` uses /chat's quality trigger and default target; ``architect_general``
uses that same trigger and pins the target. ``force_architect_general`` requests
one explicit direct-stage consultant re-answer without the quality detector.
All values still require the flag and the existing eligible frontdoor path.

The force mode does not create a reviewer/critique path. It reuses the existing
consultant call contract after a completed direct-stage answer; the default REPL
stage has no post-answer hook after RI-18c.

WHICH ESCALATION, AND WHY THESE TRIGGERS
========================================
This module keeps /chat's existing quality-trigger policy for ``auto`` and
``architect_general``. The explicit force value is the one caller-requested
exception and reuses the same bounded re-answer helper; it adds no shared
review policy or typed reviewer-plane invocation.

* ``direct`` stage — client tool mode (the mode OpenCode uses) and
  ``x_disable_repl``. A client-mode backend call is one direct completion, so
  it gets /chat's direct-stage chain (``chat_pipeline/direct_stage.py``):

  1. ``quality_escalation`` — ``chat_pipeline.stages._quality_escalate``: when
     the ``generation_monitor`` flag is on and the quality detector flags the
     answer, the SAME prompt is re-answered by ``coder_escalation`` (``auto``;
     today an alias on architect_general's :8083 process, and it stays on the 27B
     after the role swap) or by the pinned consultant (``architect_general``),
     and that answer replaces it.

  The direct stage's review gate (``chat_review._should_review`` -> architect verdict
  -> ``worker_general`` revision, triggers ``review_gate`` / ``review_gate_revision``)
  was removed by RI-18c, applying RI-18's pre-registered DROP verdict: reviewing
  every answer was net-harmful (-38.3 per 100) and the production 0.6 Q gate never
  fired. /chat lost the same hook at every site, so parity is preserved.

* ``repl`` stage — the default REPL bridge. /chat's REPL stage ran only that
  review gate after the graph (``chat_pipeline/repl_executor.py``); with it removed
  the stage runs no hook, here as in /chat. The stage is still classified so a
  receipt records which /chat stage the answer mapped to.

NOT reproduced, and why (so the semantics stay identical rather than invented):

* failure-driven graph escalation (``FrontdoorNode`` -> ``CoderEscalationNode``
  -> ``ArchitectNode``: EARLY_ABORT, retries exhausted, promoted nudges) needs
  the graph's failure accounting over REPL execution errors. The /v1 REPL
  bridge is not the graph (it never feeds an error back), and in client tool
  mode tools run in the CLIENT, so the orchestrator never observes a tool
  failure. Porting /v1 onto the graph would change the no-escalation arm too.
* model-requested ``escalate()``: a REPL tool, absent in client mode, and
  ignored by ``FrontdoorNode`` in /chat's graph path (only ``CoderNode`` reads
  ``_escalation_requested``).
* ``chat_delegation`` is architect -> specialist delegation, not escalation.

/chat skips every hook when ``force_role`` is set; the /v1 analogue is a role
override (``x_force_role`` / ``x_force_model`` / ``x_orchestrator_role``), so an
eval pin (UFH-13 arm A0) never escalates.

TELEMETRY
=========
Every escalation call is recorded as a step with its trigger, from/to role, the
server URL the role resolves to, and llama-server's own timings for that call
(``prompt_ms`` + ``gen_ms``, the deltas of ``LLMPrimitives``' per-request
accumulators, which sum ``timings.prompt_ms`` / ``timings.predicted_ms``). A
step whose URL is ``architect_general``'s server is a consultant step, and
``consultant_device_seconds`` sums exactly those. While a step's call runs, the
primitives' trace keys carry ``escalation_trigger`` / ``escalation_from_role``
/ ``escalation_to_role``, so the call's own tap ``timings`` event is tagged
too. The per-request receipt goes to the tap as a ``v1_escalation`` event and
to ``x_orchestrator_metadata.escalation`` (with ``x_show_routing``).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from src.roles import Role

log = logging.getLogger(__name__)

CONSULTANT_ROLE = str(Role.ARCHITECT_GENERAL)
QUALITY_ESCALATION_ROLE = str(Role.CODER_ESCALATION)

TRIGGER_QUALITY = "quality_escalation"
TRIGGER_FORCED = "caller_forced"
FORCE_MODE = "force_architect_general"

STAGE_DIRECT = "direct"
STAGE_REPL = "repl"

TAP_EVENT = "v1_escalation"

# Explicit target pin and force-mode mapping for the consultant role.
PINNED_TARGETS = frozenset({CONSULTANT_ROLE})


def _role_name(role: Any) -> str:
    return role.value if isinstance(role, Role) else str(role)


def _number(owner: Any, name: str) -> float:
    value = getattr(owner, name, 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _counters(primitives: Any) -> dict[str, float]:
    return {
        "calls": _number(primitives, "total_calls"),
        "prompt_ms": _number(primitives, "total_prompt_eval_ms"),
        "gen_ms": _number(primitives, "total_generation_ms"),
        "tokens": _number(primitives, "total_tokens_generated"),
        "prompt_tokens": _number(primitives, "total_prompt_tokens_reported"),
    }


def _server_url(primitives: Any, role: str) -> str | None:
    urls = getattr(primitives, "server_urls", None)
    if not isinstance(urls, dict):
        return None
    url = urls.get(role)
    return url if isinstance(url, str) and url else None


def _server_ports(url: str | None) -> list[int]:
    if not url:
        return []
    from src.llm_primitives.backend import _url_str_ports

    return _url_str_ports(url)


def _model_id(primitives: Any, role: str) -> str | None:
    """The registry's model name for ``role`` (best effort; None when unknown)."""
    registry = getattr(primitives, "registry", None)
    getter = getattr(registry, "get_role", None)
    if not callable(getter):
        return None
    try:
        name = getattr(getattr(getter(role), "model", None), "name", None)
    except Exception:
        return None
    return name if isinstance(name, str) and name else None


@dataclass
class V1EscalationPlan:
    """What one /v1 request may do, decided once before any model call."""

    requested: str | None
    enabled: bool
    disabled_reason: str | None
    from_role: str
    # The consultant role pinned by the request, or None for /chat's target.
    # The force enum is fixed to CONSULTANT_ROLE (architect_general).
    target_role: str | None = None
    base_trace_keys: dict[str, Any] = field(default_factory=dict)
    steps: list[dict[str, Any]] = field(default_factory=list)
    final_answer_role: str = ""
    consultant_url: str | None = None
    error: str | None = None
    # Independent default-off immutable writer; legacy receipt fields stay unchanged.
    strict_capture: dict[str, Any] | None = field(default=None, repr=False)

    @property
    def fired(self) -> bool:
        return any(step["calls"] > 0 for step in self.steps)

    @property
    def consultant_role(self) -> str:
        return self.target_role or CONSULTANT_ROLE

    @property
    def quality_escalation_role(self) -> str:
        return self.target_role or QUALITY_ESCALATION_ROLE

    @property
    def force(self) -> bool:
        return self.requested == FORCE_MODE

    @property
    def consultant_device_seconds(self) -> float:
        return round(sum(step["device_seconds"] for step in self.steps if step["consultant"]), 6)

    def receipt(self) -> dict[str, Any]:
        """The escalation receipt (metadata block and tap event body)."""
        return {
            "requested": self.requested,
            "enabled": self.enabled,
            "disabled_reason": self.disabled_reason,
            "fired": self.fired,
            "target": "pinned" if self.target_role else "chat_default",
            "target_role": self.target_role,
            "from_role": self.from_role,
            "final_answer_role": self.final_answer_role or self.from_role,
            "consultant_role": self.consultant_role,
            "consultant_url": self.consultant_url,
            "consultant_ports": _server_ports(self.consultant_url),
            "consultant_device_seconds": self.consultant_device_seconds,
            "steps": [dict(step) for step in self.steps],
            "error": self.error,
        }

    def usage_delta(self) -> tuple[int, int]:
        """(prompt_tokens, completion_tokens) the escalation calls added."""
        return (
            sum(step["prompt_tokens"] for step in self.steps),
            sum(step["tokens"] for step in self.steps),
        )


def plan_v1_escalation(
    *,
    flag_on: bool,
    requested: str | None,
    role: Any,
    role_override: bool,
    image_input: bool,
) -> V1EscalationPlan | None:
    """Decide eligibility. ``None`` = no key sent: touch nothing, flag on or off.

    Opt-in: explicit ``auto`` / ``architect_general`` uses the quality trigger;
    ``force_architect_general`` requests the one force re-answer. All require the
    flag on and the same role/image eligibility checks. An explicit ``off`` gets
    a disabled receipt. Flag off + key sent is recorded as disabled (``flag_off``).
    """
    if requested is None:
        return None
    from_role = _role_name(role)
    reason: str | None = None
    if not flag_on:
        reason = "flag_off"
    elif requested == "off":
        reason = "x_escalation_off"
    elif role_override:
        reason = "role_override"
    elif from_role != str(Role.FRONTDOOR):
        reason = "not_frontdoor"
    elif image_input:
        reason = "image_input"
    target_role = requested if requested in PINNED_TARGETS else None
    if requested == FORCE_MODE:
        target_role = CONSULTANT_ROLE
    plan = V1EscalationPlan(
        requested=requested,
        enabled=reason is None,
        disabled_reason=reason,
        from_role=from_role,
        target_role=target_role,
        final_answer_role=from_role,
    )
    try:
        from src.runtime.hg5_request_event import begin_capture

        plan.strict_capture = begin_capture()
        if plan.strict_capture is not None:
            plan.strict_capture["feature_enabled"] = flag_on
    except Exception as exc:
        log.debug("HG5 strict event capture unavailable: %s", type(exc).__name__)
    return plan


@contextmanager
def _tagged_trace(
    plan: V1EscalationPlan, primitives: Any, trigger: str, from_role: str, to_role: str
) -> Iterator[None]:
    """Tag the escalation call's tap section; restore the request's own keys after."""
    setter = getattr(primitives, "set_request_trace_keys", None)
    if not callable(setter):
        yield
        return
    setter(
        {
            **plan.base_trace_keys,
            "escalation_trigger": trigger,
            "escalation_from_role": from_role,
            "escalation_to_role": to_role,
        }
    )
    try:
        yield
    finally:
        setter(dict(plan.base_trace_keys))


def _record_step(
    plan: V1EscalationPlan,
    primitives: Any,
    *,
    trigger: str,
    from_role: str,
    to_role: str,
    before: dict[str, float],
    outcome: str,
) -> dict[str, Any] | None:
    if plan.strict_capture is not None:
        try:
            from src.runtime.hg5_request_event import record_step

            record_step(plan.strict_capture, primitives, trigger=trigger,
                        initial_role=from_role, target_role=to_role, outcome=outcome)
        except Exception as exc:
            log.debug("HG5 strict step emitter failed: %s", type(exc).__name__)
    after = _counters(primitives)
    calls = int(after["calls"] - before["calls"])
    if calls <= 0:
        return None  # the hook decided not to call anything: no escalation
    prompt_ms = max(0.0, after["prompt_ms"] - before["prompt_ms"])
    gen_ms = max(0.0, after["gen_ms"] - before["gen_ms"])
    server_url = _server_url(primitives, to_role)
    last_meta = {}
    getter = getattr(primitives, "get_last_inference_meta", None)
    if callable(getter):
        meta = getter()
        last_meta = meta if isinstance(meta, dict) else {}
    step = {
        "trigger": trigger,
        "from_role": from_role,
        "to_role": to_role,
        "server_url": server_url,
        "ports": _server_ports(server_url),
        "model_id": _model_id(primitives, to_role),
        "consultant": bool(server_url) and server_url == plan.consultant_url,
        "calls": calls,
        "prompt_ms": round(prompt_ms, 3),
        "gen_ms": round(gen_ms, 3),
        "device_seconds": round((prompt_ms + gen_ms) / 1000.0, 6),
        "tokens": int(max(0.0, after["tokens"] - before["tokens"])),
        "prompt_tokens": int(max(0.0, after["prompt_tokens"] - before["prompt_tokens"])),
        "completion_reason": str(last_meta.get("completion_reason") or "") or None,
        "outcome": outcome,
    }
    plan.steps.append(step)
    return step


def _finish_strict_capture(plan: V1EscalationPlan | None, *, request_id: str,
                           primitives: Any, failure_type: str | None = None,
                           failure_status: str | None = None) -> None:
    if plan is None or plan.strict_capture is None:
        return
    capture = plan.strict_capture
    try:
        from src.runtime.hg5_request_event import finish_capture, snapshot

        finish_capture(capture, request_id=request_id, plan=plan,
                       counters=snapshot(primitives), failure_type=failure_type,
                       failure_status=failure_status)
    except Exception as exc:
        log.debug("HG5 strict native event write failed: %s", type(exc).__name__)
    finally:
        # One request owns one attempt; a failure is not retried from a later renderer.
        plan.strict_capture = None


def escalate_answer(
    plan: V1EscalationPlan | None,
    *,
    stage: str,
    answer: str,
    question: str,
    direct_prompt: str,
    primitives: Any,
    state: Any,
    task_id: str | None = None,
    chat_id: str | None = None,
) -> str:
    """Run /chat's post-answer escalation hooks on a /v1 answer; return the answer.

    ``stage`` is ``direct`` (client tool mode, ``x_disable_repl``) or ``repl``
    (a FINAL answer of the REPL bridge). A no-op unless ``plan.enabled``.
    """
    if plan is None:
        return answer
    if plan.strict_capture is not None:
        try:
            plan.strict_capture["stage"] = stage if stage in {STAGE_DIRECT, STAGE_REPL} else None
        except Exception as exc:
            log.debug("HG5 strict stage capture failed: %s", type(exc).__name__)
    request_id = chat_id or task_id or ""
    failure_type = None
    failure_status = None
    try:
        if not plan.enabled or not answer:
            return answer
        return _escalate_answer(
            plan,
            stage=stage,
            answer=answer,
            question=question,
            direct_prompt=direct_prompt,
            primitives=primitives,
            state=state,
            task_id=request_id,
        )
    except Exception as exc:
        # /chat's hooks never block an answer (each helper already swallows its
        # own backend failure); anything else is recorded in the receipt, not hidden.
        log.warning("v1 escalation failed (%s); serving the unescalated answer", type(exc).__name__)
        plan.error = f"{type(exc).__name__}: {exc}"
        plan.final_answer_role = plan.from_role
        failure_type = type(exc).__name__
        failure_status = "request_failed"
        return answer
    finally:
        if plan.strict_capture is not None:
            failure_type = failure_type or plan.strict_capture.get("failure_type")
            failure_status = failure_status or plan.strict_capture.get("failure_status")
        _finish_strict_capture(plan, request_id=request_id, primitives=primitives,
                               failure_type=failure_type, failure_status=failure_status)


def _escalate_answer(
    plan: V1EscalationPlan,
    *,
    stage: str,
    answer: str,
    question: str,
    direct_prompt: str,
    primitives: Any,
    state: Any,
    task_id: str,
) -> str:
    from src.api.routes.chat_pipeline import stages as chat_stages

    plan.consultant_url = _server_url(primitives, plan.consultant_role)
    role = plan.final_answer_role or plan.from_role

    if stage == STAGE_DIRECT:
        quality_role = plan.quality_escalation_role
        before = _counters(primitives)
        trigger = TRIGGER_FORCED if plan.force else TRIGGER_QUALITY
        force_kwargs = {"force": True} if plan.force else {}
        try:
            with _tagged_trace(plan, primitives, trigger, role, quality_role):
                if plan.strict_capture is not None:
                    try:
                        from src.runtime.hg5_request_event import snapshot

                        plan.strict_capture["step_before"] = snapshot(primitives)
                    except Exception as exc:
                        log.debug("HG5 strict before-snapshot failed: %s", type(exc).__name__)
                        plan.strict_capture["step_before"] = {key: None for key in ("calls", "prompt_tokens", "completion_tokens", "prompt_ms", "generation_ms")}
                new_answer, new_role = chat_stages._quality_escalate(
                    answer,
                    direct_prompt,
                    primitives,
                    role,
                    allow_escalation=True,
                    escalation_role=Role(quality_role),
                    **force_kwargs,
                )
        except Exception as exc:
            if not plan.force:
                if plan.strict_capture is not None:
                    try:
                        from src.runtime.hg5_request_event import record_step

                        record_step(plan.strict_capture, primitives, trigger=trigger,
                                    initial_role=role, target_role=quality_role, outcome="failed")
                    except Exception as emitter_exc:
                        log.debug("HG5 strict failure-step emitter failed: %s", type(emitter_exc).__name__)
                raise
            failed_step = _record_step(
                plan,
                primitives,
                trigger=trigger,
                from_role=role,
                to_role=quality_role,
                before=before,
                outcome="failed",
            )
            if failed_step is not None:
                failed_step["completion_reason"] = None
            plan.error = f"{type(exc).__name__[:80]}: forced consultant call failed"
            plan.final_answer_role = role
            if plan.strict_capture is not None:
                plan.strict_capture["failure_type"] = type(exc).__name__[:80]
                plan.strict_capture["failure_status"] = "request_failed"
            return answer
        adopted = _role_name(new_role) != role
        _record_step(
            plan,
            primitives,
            trigger=trigger,
            from_role=role,
            to_role=quality_role,
            before=before,
            outcome="adopted" if adopted else "not_adopted",
        )
        if adopted:
            answer, role = new_answer, _role_name(new_role)

    plan.final_answer_role = role
    return answer


def record_escalation(
    plan: V1EscalationPlan | None,
    *,
    chat_id: str,
    request_keys: dict[str, Any],
    primitives: Any,
) -> dict[str, Any] | None:
    """Emit the receipt to the tap; return it for the response metadata."""
    if plan is None:
        return None
    if plan.consultant_url is None and primitives is not None:
        plan.consultant_url = _server_url(primitives, plan.consultant_role)
    if plan.strict_capture is not None:
        _finish_strict_capture(plan, request_id=chat_id, primitives=primitives)
    receipt = plan.receipt()
    if primitives is not None:
        # Whole-request llama-server time (frontdoor + every escalation call),
        # for the experiment's "frontdoor device-seconds" beside the consultant's.
        receipt["request_device_seconds"] = round(
            (
                _number(primitives, "total_prompt_eval_ms")
                + _number(primitives, "total_generation_ms")
            )
            / 1000.0,
            6,
        )
    try:
        from src.runtime.inference_tap import emit_request_event

        emit_request_event(TAP_EVENT, chat_id=chat_id, request_keys=dict(request_keys), **receipt)
    except Exception as exc:  # the tap must never affect the response
        log.debug("v1_escalation tap event failed: %s", exc)
    return receipt
