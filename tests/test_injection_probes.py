"""Corpus-driven injection-probe runner (control-plane spec §13.1 + §20.5).

Loads every probe in ``tests/fixtures/injection_probes/`` and asserts, per probe,
whether the LANDED CandidatePackage sanitizer neutralizes the attack:

  * ``expected == "neutralized"`` → a landed control defeats the attack; the runner
    asserts the neutralization property holds NOW (LIVE).
  * ``expected == "gap"`` → the landed sanitizer does NOT neutralize it; the runner
    asserts the CURRENT (unneutralized) behavior, and an ``xfail(strict=True)``
    records the spec's desired-but-unmet property (closed by a follow-up).

``_neutralizes`` is the single dispatcher mapping each threat class to the landed
defense it must clear. NO inference (reviewer prompt captured from a stub).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.fixtures.injection_probes import (
    BANNED_FRAMING_FIELDS,
    load_probes,
    sanitize_candidate_package,
)

PROBES = load_probes()
NEUTRALIZED = [p for p in PROBES if p["expected"] == "neutralized"]
GAPS = [p for p in PROBES if p["expected"] == "gap"]


def _pid(p):
    return p["id"]


class _StubPrimitives:
    def __init__(self):
        self.calls: list[str] = []

    def llm_call(self, prompt, role=None, n_tokens=None, **kw):
        self.calls.append(prompt)
        return '{"decision":"approve","confidence":0.5,"blocking":{"tripwire":false}}'


def _reviewer_prompt_for(sanitized_view: dict) -> str:
    from src.proactive_delegation.review_service import ArchitectReviewService

    stub = _StubPrimitives()
    ArchitectReviewService(stub, trace_sink=lambda ev: None).review_candidate(sanitized_view)
    return stub.calls[0]


def _ledger_resolves(evidence_id: str | None, tmp_path: Path) -> bool:
    from src.trace.review_ledger import iter_review_ledger_rows
    from src.trace.store import ensure_schema

    conn = ensure_schema(tmp_path / "probe_events.sqlite")
    try:
        rows = list(iter_review_ledger_rows(conn))
    finally:
        conn.close()
    resolvable = {r.get("decision_id") for r in rows} | {r.get("candidate_id") for r in rows}
    return evidence_id in resolvable


def _neutralizes(probe: dict, tmp_path: Path) -> bool:
    """Whether the landed sanitizer neutralizes this probe's attack (per its threat class)."""
    cat = probe["category"]
    full = probe["full_package"]

    if cat == "framing_manipulation":
        sv = sanitize_candidate_package(full)["sanitized_view"]
        present_framing = set(BANNED_FRAMING_FIELDS) & set(full)
        stripped = all(b not in sv for b in BANNED_FRAMING_FIELDS)
        recorded = present_framing <= set(sv["sanitization"]["removed_fields"])
        return stripped and recorded

    if cat in ("authority_laundering", "fabricated_evidence_reference"):
        # Neutralized iff the claimed evidence id resolves to nothing in the ledger.
        return not _ledger_resolves(probe.get("claimed_evidence_id"), tmp_path)

    if cat == "embedded_tool_instruction":
        sv = sanitize_candidate_package(full)["sanitized_view"]
        stub = _StubPrimitives()
        from src.proactive_delegation.review_service import ArchitectReviewService

        ArchitectReviewService(stub, trace_sink=lambda ev: None).review_candidate(sv)
        return len(stub.calls) == 1  # exactly one llm_call; no tool execution surface

    if cat == "position_bias":
        import inspect

        from src.proactive_delegation.review_service import ArchitectReviewService

        sig = inspect.signature(ArchitectReviewService.review_candidate)
        pos = [
            p
            for p in sig.parameters.values()
            if p.name != "self" and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        return len(pos) == 1  # pointwise — no second candidate slot

    if cat == "direct_prompt_injection":
        sv = sanitize_candidate_package(full)["sanitized_view"]
        return probe["payload"] not in _reviewer_prompt_for(sv)

    if cat == "context_flooding":
        sv = sanitize_candidate_package(full)["sanitized_view"]
        return probe["payload"] in _reviewer_prompt_for(sv)  # neutralized iff preserved

    if cat == "secret_inclusion":
        outs = sanitize_candidate_package(full)["sanitized_view"]["outputs"]
        allow = tuple(probe.get("allowlist", []))
        refs = [str(o.get("ref", "")) for o in outs]
        return all(any(r.startswith(a) for a in allow) for r in refs)

    raise AssertionError(f"unhandled probe category {cat!r} (probe {probe['id']})")


def test_corpus_is_nonempty_and_covers_both_outcomes():
    assert len(PROBES) >= 8
    assert NEUTRALIZED and GAPS  # the corpus must exercise both a defended and a gapped case


@pytest.mark.parametrize("probe", NEUTRALIZED, ids=[_pid(p) for p in NEUTRALIZED])
def test_neutralized_probe_is_defeated(probe, tmp_path):
    assert _neutralizes(probe, tmp_path) is True, (
        f"{probe['id']}: landed defense ({probe['landed_defense']}) should neutralize this"
    )


@pytest.mark.parametrize("probe", GAPS, ids=[_pid(p) for p in GAPS])
def test_gap_probe_current_behavior(probe, tmp_path):
    """LIVE: the landed sanitizer does NOT neutralize this attack (documented gap)."""
    assert _neutralizes(probe, tmp_path) is False, (
        f"{probe['id']}: unexpectedly neutralized — the gap may have been closed; update the corpus"
    )


@pytest.mark.parametrize(
    "probe",
    GAPS,
    ids=[f"{_pid(p)}-desired" for p in GAPS],
)
def test_gap_probe_desired_property(probe, tmp_path, request):
    """xfail: the spec's desired property is unmet until the follow-up lands."""
    request.node.add_marker(
        pytest.mark.xfail(strict=True, reason=f"GAP (follow-up): {probe['desired_property']}")
    )
    assert _neutralizes(probe, tmp_path) is True
