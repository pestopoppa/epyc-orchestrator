"""Narrow facade over the AutoPilot substrate the HIL sibling loop shares (HIL-3 / HIL-6 option B).

One implementation, two callers. Every function here delegates to the module AutoPilot itself
uses; nothing is re-implemented. What the facade ADDS is only the sibling's boundary:

* **Journal** (``scripts/autopilot/experiment_journal.py``): :func:`open_loop_journal` demands an
  explicit directory and refuses AutoPilot's live journal directory (``ExperimentJournal()``'s
  default) or any directory that already holds an AutoPilot journal. Storage, append-only,
  flock + fsync and torn-tail repair are the journal's own.
* **AP-53 rejected-mutation ledger** (``scripts/autopilot/rejected_mutation_ledger.py``): the same
  record builder and durable append, written beside the LOOP journal, never AutoPilot's. The
  proposer sees the ledger through the score-free render (seam S3,
  :func:`render_rejections_for_proposer`).
* **MHS guards** (``src/autopilot_core/mutation_guards.py``, extraction seams S1 and S2):
  import-safe, no PromptForge proposer is imported or instantiated to call them.
  :func:`leakage_reason` applies the same MHS-3 checks PromptForge applies to a prompt mutation
  (eval-instance and eval-content leakage over the ADDED text, then suite special-casing), with
  the same fail-closed contract: an unavailable eval-id vocabulary refuses every text.
* **AP-55 infra fingerprint** (``src/autopilot_core/infra_fingerprint.py``): reused directly;
  identity is passed as data.
* **AP-54 eval fence** (``src/repl_environment/knowledge_fence.py``, request field
  ``eval_fence``): the sibling arms the existing request seam on BOTH arms and creates no second
  fence. :func:`fenced_eval_request` and :func:`require_fence_active` are the two checks.
* **Safety/promotion gate** (seam S4): :func:`autopilot_gate_verdict` runs AutoPilot's own
  verdict legs (``safety_gate.verdict_from_evidence``: the same checks, order and messages as
  ``SafetyGate.check``) against the sibling's OWN incumbent arm as the reference. It never
  calls ``SafetyGate.check``, which mutates gate state, and never reads AutoPilot's baseline,
  MAD window, host state or live archive. The sibling consumes a verdict; it can never ratchet
  AutoPilot's baseline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.autopilot_core import infra_fingerprint as _infra
from src.autopilot_core import mutation_guards as _guards
from src.autopilot_core.mutation_guards import (
    MutationEffect,
    classify_prompt_effect,
    mutation_effect_risk,
    mutation_risk_gate_reason,
    prompt_interference_lint,
    rank_mutations_by_risk,
    screen_static_safety,
)

__all__ = [
    "AUTOPILOT_JOURNAL_STEM",
    "EVAL_FENCE_FIELD",
    "LOOP_JOURNAL_MARKER",
    "LiveSubstrateRefused",
    "MutationEffect",
    "SeamNotExtracted",
    "append_rejection",
    "autopilot_gate_verdict",
    "classify_prompt_effect",
    "compare_infra_fingerprints",
    "fenced_eval_request",
    "leakage_reason",
    "load_rejections",
    "mutation_effect_risk",
    "mutation_risk_gate_reason",
    "open_loop_journal",
    "prompt_interference_lint",
    "rank_mutations_by_risk",
    "regime_digest",
    "render_rejections_for_proposer",
    "render_trace_context",
    "require_fence_active",
    "screen_static_safety",
]

#: File stem AutoPilot's ``ExperimentJournal`` writes (``autopilot_journal*.jsonl`` / ``.tsv``).
AUTOPILOT_JOURNAL_STEM = "autopilot_journal"
#: Marker the sibling writes into a directory it owns; a directory with AutoPilot journal shards
#: and no marker is AutoPilot's, never the loop's.
LOOP_JOURNAL_MARKER = ".harness_loop_journal"
#: Request field of the AP-54 fence (``src/api/models/requests.py`` ``ChatRequest.eval_fence``).
EVAL_FENCE_FIELD = "eval_fence"

# Shared-clone and symlinked spellings of AutoPilot's live journal directory. The worktree's own
# ``DEFAULT_JOURNAL_DIR`` is added at call time (it differs per checkout).
_KNOWN_LIVE_JOURNAL_DIRS = (
    Path("/mnt/raid0/llm/epyc-orchestrator/orchestration"),
    Path("/workspace/repos/epyc-orchestrator/orchestration"),
)


class LiveSubstrateRefused(ValueError):
    """The sibling was pointed at AutoPilot's live journal/ledger, or at no path at all."""


class SeamNotExtracted(RuntimeError):
    """A substrate seam the sibling needs is planned but not extracted yet (fail closed)."""


# ── journal ────────────────────────────────────────────────────────────────


def _autopilot_default_journal_dir() -> Path:
    from scripts.autopilot.experiment_journal import DEFAULT_JOURNAL_DIR

    return Path(DEFAULT_JOURNAL_DIR)


def _live_journal_dirs() -> set[Path]:
    dirs = {_autopilot_default_journal_dir()} | set(_KNOWN_LIVE_JOURNAL_DIRS)
    out: set[Path] = set()
    for d in dirs:
        try:
            out.add(d.resolve())
        except OSError:
            out.add(d)
    return out


def _check_loop_dir(journal_dir: Path | str | None) -> Path:
    if journal_dir is None or str(journal_dir).strip() == "":
        raise LiveSubstrateRefused(
            "the sibling loop needs an explicit journal directory; ExperimentJournal()'s default "
            "is AutoPilot's live journal"
        )
    resolved = Path(journal_dir).resolve()
    if resolved in _live_journal_dirs():
        raise LiveSubstrateRefused(f"{resolved} is AutoPilot's live journal directory")
    if resolved.is_dir() and not (resolved / LOOP_JOURNAL_MARKER).exists():
        if any(resolved.glob(f"{AUTOPILOT_JOURNAL_STEM}*")):
            raise LiveSubstrateRefused(
                f"{resolved} holds an AutoPilot journal and no {LOOP_JOURNAL_MARKER} marker"
            )
    return resolved


def open_loop_journal(journal_dir: Path | str, *, segment_snapshots: bool = False) -> Any:
    """Open (create) the LOOP's own ``ExperimentJournal``; never AutoPilot's live one.

    The loop writes typed ledger events (``ExperimentJournal.append_ledger_event``) rather than
    AutoPilot ``JournalEntry`` trial rows, so it emits nothing into the shared trace store and
    never closes AutoPilot-shaped segments. ``segment_snapshots`` defaults off for the same reason.
    """
    resolved = _check_loop_dir(journal_dir)
    resolved.mkdir(parents=True, exist_ok=True)
    marker = resolved / LOOP_JOURNAL_MARKER
    if not marker.exists():
        marker.write_text("harness-improvement sibling loop journal (HIL-6 option B)\n")
    from scripts.autopilot.experiment_journal import ExperimentJournal

    return ExperimentJournal(journal_dir=resolved, segment_snapshots=segment_snapshots)


# ── AP-53 rejected-mutation ledger ───────────────────────────────────────────


def append_rejection(journal_dir: Path | str, **record_fields: Any) -> bool:
    """Build an AP-53 record with AutoPilot's own builder and append it beside the LOOP journal."""
    from scripts.autopilot import rejected_mutation_ledger as ledger

    resolved = _check_loop_dir(journal_dir)
    return ledger.append_record(resolved, ledger.build_record(**record_fields))


def load_rejections(journal_dir: Path | str) -> list[dict[str, Any]]:
    from scripts.autopilot import rejected_mutation_ledger as ledger

    return ledger.load_records(_check_loop_dir(journal_dir))


def render_rejections_for_proposer(journal_dir: Path | str, target: str, *, n: int = 3) -> str:
    """The loop's AP-53 rejections of ``target`` as the proposer may see them (seam S3).

    AutoPilot's own renderer over the LOOP directory, score-free (HIL-2 item 7): no per-suite
    deltas and no gate detail; the gate name, repeat count and diff remain.
    """
    from scripts.autopilot import rejected_mutation_ledger as ledger

    return ledger.render_for_prompt(_check_loop_dir(journal_dir), target, n=n, include_scores=False)


# ── MHS-3 leakage (seam S2) ─────────────────────────────────────────────────


def leakage_reason(
    text: str,
    *,
    original: str = "",
    trace_context: str = "",
    vocabulary: Any = None,
    harness: Any = None,
) -> str | None:
    """MHS-3 verdict for a candidate: a refusal reason, or None when clean.

    ``text`` is the candidate's full mutated text. The checks and their order are PromptForge's
    for a prompt mutation (``PromptForge._leakage_reason`` then ``suite_special_casing_reason``):
    eval-instance ids and eval content are scored over the text the candidate ADDS to
    ``original``; ``trace_context`` is what the proposer was shown. ``vocabulary`` defaults to the
    data-sourced eval-id vocabulary and ``harness`` to this checkout's harness corpus. Fail
    closed: an unavailable vocabulary (or a crashed content check) refuses every text.
    """
    vocab = vocabulary if vocabulary is not None else _guards.load_eval_id_vocabulary()
    if harness is not None:
        known = harness
    elif vocab.available:
        known = _guards.harness_content()
    else:  # refused before the corpus is consulted; skip hashing the checkout
        known = None
    reason = _guards.eval_leakage_reason(
        _guards._added_text(original, text),
        vocab,
        original=original,
        trace_context=trace_context,
        harness=known,
    )
    if reason is not None:
        return reason
    return _guards.suite_special_casing_reason(original, text, vocab, harness=known)


def render_trace_context(traces: Any) -> str:
    """The view's traces in the contrastive-trace shape the MHS-3 trace-bank checks parse.

    One ``[i] trace=<id> role=<role>`` entry per trace with its excerpt and event names, under
    the ``## Contrastive Execution Traces`` heading (``mutation_guards._trace_blocks``). Without
    a recognised heading the trace-bank n-gram and source-identity checks would see no blocks.
    """
    entries = []
    for i, t in enumerate(traces, start=1):
        body = "\n".join(x for x in (t.excerpt, " ".join(t.events)) if x)
        entries.append(f"[{i}] trace={t.trace_id} role={t.role}\n{body}")
    if not entries:
        return ""
    return "## Contrastive Execution Traces\n" + "\n".join(entries) + "\n"


# ── AP-55 infra fingerprint ───────────────────────────────────────────────────

compare_infra_fingerprints = _infra.compare_infra_fingerprints
regime_digest = _infra.regime_digest
NON_COMPARABLE = _infra.NON_COMPARABLE
COMPARABLE = _infra.COMPARABLE
UNVERIFIED = _infra.UNVERIFIED


# ── AP-54 eval fence ─────────────────────────────────────────────────────────


def fenced_eval_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``payload`` with the AP-54 fence armed. An explicit disarm is refused, not overridden."""
    out = dict(payload)
    if EVAL_FENCE_FIELD in out and out[EVAL_FENCE_FIELD] is not True:
        raise ValueError(
            f"HIL-2 item 7: loop eval requests arm the fence on both arms; got "
            f"{EVAL_FENCE_FIELD}={out[EVAL_FENCE_FIELD]!r}"
        )
    out[EVAL_FENCE_FIELD] = True
    return out


def require_fence_active(response_fence: Mapping[str, Any] | None) -> None:
    """Refuse a result whose response does not echo an ACTIVE fence (``eval_fence.state``)."""
    state = (response_fence or {}).get("state")
    if state != "active":
        raise ValueError(f"HIL-2 item 7: eval fence state {state!r}, expected 'active'")


# ── safety/promotion gate (seam S4) ───────────────────────────────────────────


def autopilot_gate_verdict(
    candidate: Any,
    incumbent: Any,
    *,
    mad_window: Any = (),
    reliability_floor: float | None = None,
) -> Any:
    """AutoPilot's SafetyGate verdict for ``candidate`` against the sibling's incumbent arm.

    Both arguments are EvalResult-shaped (``tier``, ``quality``, ``speed``, ``reliability``,
    ``per_suite_quality``, optional ``per_suite_counts`` / ``routing_distribution`` /
    ``quality_measured``). The reference is the INCUMBENT (quality, per-suite scores and counts,
    speed), never an AutoPilot tier baseline. ``mad_window`` is the loop's own same-tier quality
    history (empty: no noise filter). No host read: throttle demotion never applies. Pure: it
    returns a ``SafetyVerdict`` and changes nothing.
    """
    from scripts.autopilot import safety_gate as sg

    evidence = sg.GateEvidence(
        baseline_quality=incumbent.quality,
        baseline_per_suite=dict(incumbent.per_suite_quality or {}),
        baseline_per_suite_counts=dict(getattr(incumbent, "per_suite_counts", None) or {}),
        frontdoor_speed=incumbent.speed,
        proxy_per_suite=dict(incumbent.per_suite_quality or {}),
        mad_window=tuple(mad_window),
        reliability_floor=reliability_floor,
    )
    return sg.verdict_from_evidence(candidate, evidence)
