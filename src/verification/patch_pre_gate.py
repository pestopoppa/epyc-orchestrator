"""EV-12 — patch pre-gate SIGNAL (execution-free, zero inference cost).

Research-evaluation index **RE-2 / EV-12** ("Dockerless execution-free patch
verdicts at zero inference cost"). This module is a thin, deterministic *policy
layer* over :func:`src.verification.patch_verifier.verify_patch`. It turns that
static verdict (``pass | fail | inconclusive``) into a single reusable
**pre-gate signal** that a ``coder_escalation`` dispatch — or an eval-tower
verifier — *could* consult to decide whether spending inference on a candidate
patch is worthwhile.

What this module is (and is NOT)
--------------------------------
  * It runs **no inference**, starts **no server**, and — because it only calls
    ``verify_patch`` — **never executes the patched program**. The verdict is a
    static read (git-apply-check / hunk-context / AST+``py_compile`` /
    import-resolution / ruff).
  * It is **pure and deterministic**: same patch + same base tree ⇒ same signal.
  * It does **NOT** wire itself into the live dispatch / serving path. Producing
    the signal is decoupled from acting on it; the actual coder-escalation
    wiring is a serving-path change and is deliberately **deferred**. A caller
    imports :func:`evaluate_patch_pre_gate`, gets a :class:`PreGateSignal`, and
    decides what to do with ``should_escalate`` itself.

Verdict → escalation policy
---------------------------
The static verdict is cheap; the coder-escalation inference is expensive. A
static ``PASS`` only means the patch *applies and compiles* — it does **not**
prove correctness (nothing was executed, no tests ran). So:

===============  =================  ==========================================
static verdict   should_escalate    rationale
===============  =================  ==========================================
``FAIL``         **False**          The patch is provably non-viable — it will
                                    not apply or will not compile — and carries
                                    a machine-checkable ``certificate`` saying
                                    why. Rejecting it costs zero inference;
                                    escalating an expensive coder to "fix" a
                                    patch we can already prove is broken is
                                    wasted inference.
``PASS``         **True**           A viable candidate: it applies and compiles.
                                    Proceed and spend the coder-escalation /
                                    deeper-verification inference.
``INCONCLUSIVE`` **True**           The cheap gate could not rule the patch out
                                    (base tree unresolved, empty patch, only
                                    advisory signals, …). Err toward spending
                                    inference rather than dropping a possibly
                                    good candidate on an inconclusive read.
===============  =================  ==========================================

Only a **conclusive** ``FAIL`` suppresses escalation. This mirrors the verifier
precedence rule in ``verify_patch``: an inconclusive *required* check never
yields a conclusive fail, so it never suppresses escalation here either.

Public API::

    from src.verification.patch_pre_gate import evaluate_patch_pre_gate
    signal = evaluate_patch_pre_gate(patch, base_ref_or_tree)  # -> PreGateSignal
    if signal.should_escalate:
        ...  # (serving-path) spend the coder-escalation inference — DEFERRED
    signal.to_dict()  # {verdict, certificate_type, reason, should_escalate}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.verification.patch_verifier import (
    FAIL,
    INCONCLUSIVE,
    PASS,
    BaseTree,
    VerdictResult,
    verify_patch,
)

__all__ = [
    "PreGateSignal",
    "evaluate_patch_pre_gate",
    "should_escalate_patch",
    "ESCALATE_ON_VERDICT",
]

# ── policy table: static verdict -> spend the coder-escalation inference? ──
#
# Single source of truth for the escalation policy. FAIL is the only verdict
# that suppresses escalation (a conclusive, certificate-backed non-viability
# proof); PASS and INCONCLUSIVE both proceed. Kept as a mapping so the policy is
# inspectable/testable rather than buried in branches.
ESCALATE_ON_VERDICT: dict[str, bool] = {
    PASS: True,
    INCONCLUSIVE: True,
    FAIL: False,
}

_MAX_DETAIL = 240


def _truncate(text: str, limit: int = _MAX_DETAIL) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass
class PreGateSignal:
    """A single execution-free pre-gate verdict.

    Attributes:
      verdict: aggregate static verdict from ``verify_patch``
        (``pass | fail | inconclusive``).
      certificate_type: on a ``FAIL``, the ``certificate.type`` of the first
        failing required check (e.g. ``"diff"`` for an apply/context failure,
        ``"stack_trace"`` for a ``py_compile`` syntax failure). ``None`` for
        ``PASS`` / ``INCONCLUSIVE``.
      reason: human-readable, one-line explanation of the verdict + policy call.
      should_escalate: whether the caller SHOULD spend coder-escalation
        inference on this candidate (see the module policy table).
      report_id / verdict_summary: provenance passthrough from the underlying
        ``VerdictResult`` (not part of the canonical 4-key signal dict).
    """

    verdict: str
    certificate_type: Optional[str]
    reason: str
    should_escalate: bool
    report_id: Optional[str] = None
    verdict_summary: Optional[dict] = None

    def to_dict(self) -> dict:
        """The canonical structured signal: exactly the four contract keys."""
        return {
            "verdict": self.verdict,
            "certificate_type": self.certificate_type,
            "reason": self.reason,
            "should_escalate": self.should_escalate,
        }


def _fail_signal_fields(result: VerdictResult) -> tuple[Optional[str], str]:
    """(certificate_type, reason) for a conclusive-FAIL verdict."""
    failing = result.failing_check
    if failing is None:  # defensive: aggregate FAIL always has a failing check
        return None, (
            "static verdict FAIL — candidate is provably non-viable; "
            "skipping coder-escalation inference"
        )
    cert = failing.certificate
    cert_type = cert.type if cert is not None else None

    detail = ""
    if cert is not None:
        payload = cert.payload
        if isinstance(payload, dict):
            first = payload.get("first")
            if isinstance(first, dict):
                detail = (
                    f"{first.get('kind', 'context')} mismatch in "
                    f"{first.get('file')}:{first.get('line')}"
                )
            else:
                detail = _truncate(payload)
        else:
            detail = _truncate(payload)

    reason = f"static verdict FAIL at check '{failing.check_id}'"
    if detail:
        reason += f": {detail}"
    reason += " — candidate is provably non-viable; skipping coder-escalation inference"
    return cert_type, reason


def _inconclusive_reason(result: VerdictResult) -> str:
    reason = next(
        (
            c.inconclusive_reason
            for c in result.checks
            if c.required and c.outcome == INCONCLUSIVE and c.inconclusive_reason
        ),
        None,
    )
    base = "static verdict INCONCLUSIVE"
    if reason:
        base += f": {_truncate(reason)}"
    return base + " — cannot statically rule the patch out; proceeding to coder-escalation inference"


def evaluate_patch_pre_gate(
    patch: str,
    base_ref_or_tree: BaseTree,
    *,
    run_lint: bool = True,
    use_git: bool = True,
    strip: int = 1,
    candidate_ref: Optional[str] = None,
) -> PreGateSignal:
    """Produce the execution-free pre-gate signal for ``patch``.

    Delegates entirely to :func:`verify_patch` (no verification logic is
    reimplemented here) and applies the module escalation policy. NEVER executes
    the patched program; runs no inference and no server.

    Args:
      patch: unified diff text.
      base_ref_or_tree: mapping ``{relpath: source_or_None}`` OR a work-tree dir
        — passed straight through to ``verify_patch``.
      run_lint / use_git / strip / candidate_ref: passthrough to ``verify_patch``.

    Returns:
      PreGateSignal with ``verdict``, ``certificate_type``, ``reason`` and
      ``should_escalate`` (plus provenance).
    """
    result = verify_patch(
        patch,
        base_ref_or_tree,
        run_lint=run_lint,
        use_git=use_git,
        strip=strip,
        candidate_ref=candidate_ref,
    )
    verdict = result.verdict
    should_escalate = ESCALATE_ON_VERDICT[verdict]

    if verdict == FAIL:
        cert_type, reason = _fail_signal_fields(result)
    elif verdict == PASS:
        cert_type = None
        reason = (
            "static verdict PASS: patch applies and compiles cleanly "
            "(execution-free) — viable candidate; proceeding to "
            "coder-escalation inference"
        )
    else:  # INCONCLUSIVE
        cert_type = None
        reason = _inconclusive_reason(result)

    return PreGateSignal(
        verdict=verdict,
        certificate_type=cert_type,
        reason=reason,
        should_escalate=should_escalate,
        report_id=result.report_id,
        verdict_summary=result.summary(),
    )


def should_escalate_patch(
    patch: str,
    base_ref_or_tree: BaseTree,
    **kwargs,
) -> bool:
    """Convenience boolean: ``evaluate_patch_pre_gate(...).should_escalate``."""
    return evaluate_patch_pre_gate(patch, base_ref_or_tree, **kwargs).should_escalate
