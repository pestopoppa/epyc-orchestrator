"""First-feature evaluator through the orchestrator ``/chat`` with the AP-54 fence (HIL-6h). Build-only.

:class:`ChatArmEvaluator` implements :class:`~src.harness_loop.interfaces.Evaluator`. It is
feature-agnostic on purpose: the pilot feature is chosen in HIL-6i, so the three feature-specific
pieces are injected callables, not a hardcoded suite:

* ``request_builder(arm, arm_text, item) -> dict``: the ``/chat`` body for one item under one arm
  (for example a prompt arm materialized under ``x_orchestrator_prompt_root``);
* ``scorer(item, response) -> ItemScore``: whether the item passed (``None`` = not scorable) and
  whether the treatment fired on it (``None`` = not observable);
* ``content_for(arm) -> str``: the arm's bytes, verified against ``arm.content_sha256`` (items 6
  and 14) before any request is built.

Every request goes through ``substrate.fenced_eval_request`` (an explicit disarm is refused) and
every echo through ``substrate.require_fence_active``; one non-active echo refuses the whole
result. The returned :class:`ArmResult` is filled completely: metric, value, unit, direction,
scored/attempted, per-item outcomes, the solver seat's tokens, ``treatment_fired``, the fence
state, the infra fingerprint (passed through) and the verification-signal mode.

``/chat`` echoes the backend decode count (``tokens_generated``) but no prompt-token count, so the
solver seat's prefill columns are ``None`` ("not captured"), never estimated from characters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from src.harness_loop import substrate
from src.harness_loop.interfaces import ArmResult, ArmSpec, SplitRef, TokenCount, sha256_text
from src.harness_loop.pilot import VERIFICATION_ARMS
from src.harness_loop.transport import JsonTransport, TransportError

DIRECTIONS = frozenset({"higher_better", "lower_better"})


class EvaluationRefused(ValueError):
    """The evaluator refused to produce a result (pinning, fence, or missing item)."""


@dataclass(frozen=True)
class ItemScore:
    passed: bool | None
    fired: bool | None


@dataclass
class EvalDiagnostics:
    """Per-call diagnostics kept beside the result (not part of ``ArmResult``)."""

    transport_errors: dict[str, str] = field(default_factory=dict)
    fence_denied_total: int = 0
    fence_enforcement: set[str] = field(default_factory=set)
    fired_unobservable: list[str] = field(default_factory=list)


def _decode_tokens(response: Mapping[str, Any]) -> int | None:
    value = response.get("tokens_generated")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def pass_share(outcomes: Mapping[str, bool | None]) -> float | None:
    """Default metric value: passed / scored over the scorable items (``None`` when none)."""
    scored = [v for v in outcomes.values() if v is not None]
    return (sum(1 for v in scored if v) / len(scored)) if scored else None


class ChatArmEvaluator:
    seat = "solver"

    def __init__(
        self,
        *,
        transport: JsonTransport,
        items: Mapping[str, Mapping[str, Any]],
        request_builder: Callable[[ArmSpec, str, Mapping[str, Any]], Mapping[str, Any]],
        scorer: Callable[[Mapping[str, Any], Mapping[str, Any]], ItemScore],
        content_for: Callable[[ArmSpec], str],
        metric: str,
        unit: str,
        direction: str,
        verification_signal: str = "with",
        infra_fingerprint_fn: Callable[[], Mapping[str, Any] | None] | None = None,
        workload_class: str = "eval_batch",
        aggregate: Callable[[Mapping[str, bool | None]], float | None] | None = None,
    ) -> None:
        if direction not in DIRECTIONS:
            raise ValueError(f"direction {direction!r} not in {sorted(DIRECTIONS)}")
        if verification_signal not in VERIFICATION_ARMS:
            raise ValueError(f"verification_signal {verification_signal!r} not in "
                             f"{sorted(VERIFICATION_ARMS)}")
        if not (metric and unit):
            raise ValueError("metric and unit are declared before arm generation")
        self._transport = transport
        self._items = items
        self._build = request_builder
        self._score = scorer
        self._content_for = content_for
        self.metric = metric
        self.unit = unit
        self.direction = direction
        self.verification_signal = verification_signal
        self._infra = infra_fingerprint_fn
        self.workload_class = workload_class
        self._aggregate = aggregate or pass_share
        self.last_diagnostics: EvalDiagnostics | None = None

    def _arm_text(self, arm: ArmSpec) -> str:
        text = self._content_for(arm)
        if not isinstance(text, str) or sha256_text(text) != arm.content_sha256:
            raise EvaluationRefused(f"items 6/14: arm {arm.arm_id} bytes do not match its pinned "
                                    f"content_sha256")
        return text

    def build_request(self, arm: ArmSpec, arm_text: str, qid: str) -> dict[str, Any]:
        item = self._items.get(qid)
        if item is None:
            raise EvaluationRefused(f"split item {qid!r} has no eval item")
        body = dict(self._build(arm, arm_text, item))
        body.setdefault("workload_class", self.workload_class)
        body.setdefault("request_id", f"hil-{arm.arm_id}-{qid}"[:128])
        return substrate.fenced_eval_request(body)

    def evaluate(self, arm: ArmSpec, split: SplitRef) -> ArmResult:
        arm_text = self._arm_text(arm)
        diag = EvalDiagnostics()
        outcomes: dict[str, bool | None] = {}
        fired = 0
        fired_known = True
        decode_total: int | None = 0
        for qid in sorted(split.qids):
            body = self.build_request(arm, arm_text, qid)
            try:
                response = self._transport(body)
            except TransportError as exc:
                diag.transport_errors[qid] = str(exc)[:300]
                outcomes[qid] = None
                continue
            if not isinstance(response, Mapping):
                raise EvaluationRefused(f"item {qid!r}: response is not an object")
            echo = response.get(substrate.EVAL_FENCE_FIELD)
            try:
                substrate.require_fence_active(echo if isinstance(echo, Mapping) else None)
            except ValueError as exc:
                self.last_diagnostics = diag
                raise EvaluationRefused(f"item {qid!r}: {exc}") from exc
            diag.fence_denied_total += int((echo or {}).get("denied_count") or 0)
            if (echo or {}).get("enforcement"):
                diag.fence_enforcement.add(str(echo["enforcement"]))
            score = self._score(self._items[qid], response)
            outcomes[qid] = score.passed
            if score.fired is None:
                fired_known = False
                diag.fired_unobservable.append(qid)
            elif score.fired:
                fired += 1
            n = _decode_tokens(response)
            decode_total = None if (n is None or decode_total is None) else decode_total + n
        scored = sum(1 for v in outcomes.values() if v is not None)
        responded = len(outcomes) - len(diag.transport_errors)
        self.last_diagnostics = diag
        return ArmResult(
            arm_id=arm.arm_id,
            split_id=split.split_id,
            metric=self.metric,
            value=self._aggregate(outcomes),
            unit=self.unit,
            direction=self.direction,
            scored=scored,
            attempted=len(outcomes),
            item_outcomes=outcomes,
            tokens=(TokenCount(uncached_prefill=None, cached_prefill=None,
                               decode=decode_total if responded else None, seat=self.seat),),
            treatment_fired=fired if (fired_known and responded) else None,
            fence_state="active" if responded else "unobserved",
            infra_fingerprint=self._infra() if self._infra is not None else None,
            verification_signal=self.verification_signal,
        )


__all__ = ["ChatArmEvaluator", "EvalDiagnostics", "EvaluationRefused", "ItemScore", "pass_share"]
