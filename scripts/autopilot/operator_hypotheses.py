#!/usr/bin/env python3
"""Operator hypothesis channel — steering AutoPilot's planner without authority.

WHY THIS EXISTS
---------------
AutoPilot's planner already carries hypotheses with an explicit falsifier
("one-line predicted outcome whose absence invalidates this hypothesis"), keeps
them still-open until resolved, and re-surfaces the open set into each planning
round (``autopilot.py`` "Hypotheses Under Test" / "Still-open hypotheses").
There was no channel in the other direction: the "Operator Outbox" is
autopilot -> operator only. Operator steering therefore arrived out-of-band — a
prompt edit, a config change, a verbal instruction — carrying no falsifier and
leaving no resolution record, so it could never be refuted.

This module is that inbound channel, and nothing more. It is a *proposal
source*, never an authority.

THE FALSIFIER IS MANDATORY
--------------------------
A hypothesis without a falsifier is refused at load. That is the whole point:
the falsifier is what makes an operator statement a resolvable claim rather
than a standing instruction. Refusing it here is cheaper than discovering six
weeks later that "the operator said so" was never checkable.

THE GRADE IS WHAT MAKES THIS SAFE
---------------------------------
An operator hypothesis is a PRIOR, never evidence, and it must not be able to
raise its own standing by virtue of who wrote it. Rather than inventing a
parallel grade, this module conforms to the provenance vocabulary AutoPilot
already uses for operator-supplied input — the StrategyStore operator seeds
written by ``seed_operator_strategies.py``:

    seeded_by: operator          # authorship, not authority
    evidence_trial_ids: []       # a prior has cited NO measured trial
    confidence: low|medium|high  # the operator's self-declared confidence

``evidence_trial_ids`` is the load-bearing field. AutoPilot distinguishes a
measured fact from a suggestion by whether trials back it, so a statement here
is REFUSED if it carries ``evidence_trial_ids``: evidence ids belong to a
resolution, which is a record of what the loop measured, not to the claim.
That is the concrete mechanism that stops a hunch being laundered into a
measured fact.

Correspondingly, the channel is subject to the existing failure blacklist:
``blacklist_conflicts()`` runs an operator hypothesis' proposed action through
``state_store.check_blacklist`` so the critic can say "this repeats a recorded
negative". Being the operator's idea is not new evidence.

TWO FILES, ONE TRUTH EACH
-------------------------
``orchestration/operator_hypotheses.yaml``
    Operator-authored statements. Hand-edited, comment-friendly, NEVER machine
    written — so no tool run can eat the operator's prose.

``orchestration/operator_hypothesis_resolutions.jsonl``
    Append-only resolution ledger, machine-written (mirrors the operator-outbox
    JSONL convention: append + flush + fsync). ``confirmed`` / ``refuted`` /
    ``inconclusive``, each with the evidence that resolved it. A refutation is
    exactly as recordable as a confirmation — that is the main reason to have
    this channel rather than steering out-of-band.

FAILURE IS EXPLICIT
-------------------
A malformed store RAISES ``OperatorHypothesisError``. It never degrades to an
empty list, because an empty list reads as "the operator has no hypotheses" and
that is a lie the planner would act on. An absent file is different, and does
mean exactly that.

CLI
---
    python scripts/autopilot/operator_hypotheses.py validate
    python scripts/autopilot/operator_hypotheses.py list [--open-only]
    python scripts/autopilot/operator_hypotheses.py resolve <id> \
        --status refuted --trial 1234 --trial 1235 --note "..."
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_HYPOTHESES_PATH = ORCH_ROOT / "orchestration" / "operator_hypotheses.yaml"
DEFAULT_RESOLUTIONS_PATH = (
    ORCH_ROOT / "orchestration" / "operator_hypothesis_resolutions.jsonl"
)

TOP_LEVEL_KEY = "operator_hypotheses"

#: Same vocabulary as ``seed_operator_strategies.VALID_CONFIDENCE``. This is the
#: operator's self-declared confidence, NOT an evidence grade.
VALID_CONFIDENCE = ("low", "medium", "high")
DEFAULT_CONFIDENCE = "medium"

RESOLUTION_STATUSES = ("confirmed", "refuted", "inconclusive")
#: Statuses that assert an outcome, and therefore owe at least one evidence id.
EVIDENCE_REQUIRING_STATUSES = ("confirmed", "refuted")

REQUIRED_FIELDS = ("id", "hypothesis", "falsifier", "stated_by", "stated")
OPTIONAL_FIELDS = ("confidence", "proposed_action", "notes")
KNOWN_FIELDS = frozenset(REQUIRED_FIELDS + OPTIONAL_FIELDS)

#: Fields refused on a statement because they would grade a prior as evidence.
EVIDENCE_FIELDS_REFUSED_ON_STATEMENT = ("evidence_trial_ids", "evidence_event_ids")

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*[a-z0-9]$")

#: Rendered into the planner prompt so the model sees the provenance, not just
#: the claim. Mirrors the shape ``seed_operator_strategies.py`` writes.
PLANNER_PROVENANCE = "seeded_by=operator | evidence_trial_ids=[]"

DEFAULT_RENDER_LIMIT = 8


class OperatorHypothesisError(RuntimeError):
    """Raised when the operator hypothesis store cannot be trusted as read.

    Deliberately fatal rather than fail-open. A parse error, a missing
    falsifier, a resolution for an unknown hypothesis — any of these mean the
    channel's contents are not what the file says, and the planner must not be
    handed a silently truncated set that reads as "the operator has nothing to
    say".
    """


@dataclass(frozen=True)
class Resolution:
    """One recorded outcome for an operator hypothesis."""

    hypothesis_id: str
    status: str
    resolved_at: str
    evidence_trial_ids: tuple[int, ...] = ()
    evidence_event_ids: tuple[str, ...] = ()
    note: str = ""
    recorded_by: str = "autopilot"

    def as_row(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "status": self.status,
            "resolved_at": self.resolved_at,
            "evidence_trial_ids": list(self.evidence_trial_ids),
            "evidence_event_ids": list(self.evidence_event_ids),
            "note": self.note,
            "recorded_by": self.recorded_by,
        }

    def evidence_text(self) -> str:
        parts: list[str] = []
        if self.evidence_trial_ids:
            parts.append("trials " + ",".join(f"#{t}" for t in self.evidence_trial_ids))
        if self.evidence_event_ids:
            parts.append("events " + ",".join(self.evidence_event_ids))
        return "; ".join(parts) or "(none)"


@dataclass(frozen=True)
class OperatorHypothesis:
    """One operator-stated, falsifiable claim awaiting resolution."""

    id: str
    hypothesis: str
    falsifier: str
    stated_by: str
    stated: str
    confidence: str = DEFAULT_CONFIDENCE
    proposed_action: Mapping[str, Any] | None = None
    notes: str = ""
    resolution: Resolution | None = None

    @property
    def is_open(self) -> bool:
        return self.resolution is None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "hypothesis": self.hypothesis,
            "falsifier": self.falsifier,
            "stated_by": self.stated_by,
            "stated": self.stated,
            "confidence": self.confidence,
            "notes": self.notes,
            # Provenance, restated explicitly: this is a prior with no measured
            # trial behind it. See module docstring.
            "seeded_by": "operator",
            "evidence_trial_ids": [],
        }
        if self.proposed_action is not None:
            out["proposed_action"] = dict(self.proposed_action)
        if self.resolution is not None:
            out["resolution"] = self.resolution.as_row()
        return out


# ── Loading / validation ─────────────────────────────────────────


def _require_text(raw: Mapping[str, Any], key: str, where: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise OperatorHypothesisError(
            f"{where}: '{key}' is required and must be a non-empty string"
        )
    return value.strip()


def _hypothesis_from_raw(index: int, raw: Any, path: Path) -> OperatorHypothesis:
    where = f"{path} entry {index}"
    if not isinstance(raw, Mapping):
        raise OperatorHypothesisError(f"{where}: entry must be a mapping, got {type(raw).__name__}")

    unknown = sorted(set(raw) - KNOWN_FIELDS)
    for refused in EVIDENCE_FIELDS_REFUSED_ON_STATEMENT:
        if refused in raw:
            raise OperatorHypothesisError(
                f"{where}: '{refused}' is refused on a statement. An operator "
                f"hypothesis is a PRIOR, not evidence — evidence ids belong to a "
                f"resolution recorded in {DEFAULT_RESOLUTIONS_PATH.name}, which "
                f"says what the loop measured. Stating them here would grade the "
                f"claim by its authorship."
            )
    if "resolution" in raw:
        raise OperatorHypothesisError(
            f"{where}: 'resolution' is refused here. Resolutions live in the "
            f"append-only ledger {DEFAULT_RESOLUTIONS_PATH.name} so there is one "
            f"source of truth for what was measured; record one with "
            f"`operator_hypotheses.py resolve`."
        )
    if unknown:
        raise OperatorHypothesisError(
            f"{where}: unknown field(s) {unknown}; allowed: {sorted(KNOWN_FIELDS)}"
        )

    hypothesis_id = _require_text(raw, "id", where)
    if not SLUG_RE.match(hypothesis_id):
        raise OperatorHypothesisError(
            f"{where}: id {hypothesis_id!r} must be a lowercase slug "
            f"(a-z, 0-9, '-', '_', '.')"
        )

    hypothesis = _require_text(raw, "hypothesis", where)

    falsifier_raw = raw.get("falsifier")
    if not isinstance(falsifier_raw, str) or not falsifier_raw.strip():
        raise OperatorHypothesisError(
            f"{where} ({hypothesis_id}): 'falsifier' is MANDATORY and must be a "
            f"non-empty one-line predicted outcome whose absence invalidates the "
            f"hypothesis. Without it this is a standing instruction, not a "
            f"resolvable claim, and the loop could never refute it."
        )

    stated_by = _require_text(raw, "stated_by", where)
    stated = _require_text(raw, "stated", where)

    confidence = raw.get("confidence", DEFAULT_CONFIDENCE)
    if not isinstance(confidence, str) or confidence.strip().lower() not in VALID_CONFIDENCE:
        raise OperatorHypothesisError(
            f"{where} ({hypothesis_id}): 'confidence' must be one of "
            f"{list(VALID_CONFIDENCE)}, got {confidence!r}"
        )

    proposed_action = raw.get("proposed_action")
    if proposed_action is not None:
        if not isinstance(proposed_action, Mapping):
            raise OperatorHypothesisError(
                f"{where} ({hypothesis_id}): 'proposed_action' must be a mapping "
                f"shaped like an AutoPilot action so it can be checked against "
                f"the failure blacklist"
            )
        proposed_action = dict(proposed_action)

    notes = raw.get("notes", "")
    if notes is None:
        notes = ""
    if not isinstance(notes, str):
        raise OperatorHypothesisError(f"{where} ({hypothesis_id}): 'notes' must be a string")

    return OperatorHypothesis(
        id=hypothesis_id,
        hypothesis=hypothesis,
        falsifier=falsifier_raw.strip(),
        stated_by=stated_by,
        stated=stated,
        confidence=confidence.strip().lower(),
        proposed_action=proposed_action,
        notes=notes.strip(),
    )


def _load_statements(path: Path) -> list[OperatorHypothesis]:
    """Parse the operator-authored YAML. Absent file => genuinely none."""
    if not path.exists():
        return []
    try:
        text = path.read_text()
    except OSError as exc:
        raise OperatorHypothesisError(f"cannot read operator hypothesis store {path}: {exc}") from exc
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise OperatorHypothesisError(
            f"operator hypothesis store {path} is not valid YAML ({exc}). "
            f"Refusing to read it as an empty set."
        ) from exc

    if loaded is None:
        return []
    if not isinstance(loaded, Mapping):
        raise OperatorHypothesisError(
            f"operator hypothesis store {path} must be a mapping with a "
            f"'{TOP_LEVEL_KEY}' list, got {type(loaded).__name__}"
        )
    if TOP_LEVEL_KEY not in loaded:
        raise OperatorHypothesisError(
            f"operator hypothesis store {path} has no '{TOP_LEVEL_KEY}' key"
        )
    raw_entries = loaded.get(TOP_LEVEL_KEY)
    if raw_entries is None:
        return []
    if not isinstance(raw_entries, list):
        raise OperatorHypothesisError(
            f"operator hypothesis store {path}: '{TOP_LEVEL_KEY}' must be a list, "
            f"got {type(raw_entries).__name__}"
        )

    statements: list[OperatorHypothesis] = []
    seen: dict[str, int] = {}
    for index, raw in enumerate(raw_entries, start=1):
        statement = _hypothesis_from_raw(index, raw, path)
        if statement.id in seen:
            raise OperatorHypothesisError(
                f"{path}: duplicate hypothesis id {statement.id!r} "
                f"(entries {seen[statement.id]} and {index})"
            )
        seen[statement.id] = index
        statements.append(statement)
    return statements


def _resolution_from_row(index: int, row: Any, path: Path) -> Resolution:
    where = f"{path} line {index}"
    if not isinstance(row, Mapping):
        raise OperatorHypothesisError(f"{where}: resolution row must be a JSON object")
    hypothesis_id = _require_text(row, "hypothesis_id", where)
    status = row.get("status")
    if not isinstance(status, str) or status.strip().lower() not in RESOLUTION_STATUSES:
        raise OperatorHypothesisError(
            f"{where} ({hypothesis_id}): 'status' must be one of "
            f"{list(RESOLUTION_STATUSES)}, got {status!r}"
        )
    status = status.strip().lower()
    resolved_at = _require_text(row, "resolved_at", where)

    trial_ids_raw = row.get("evidence_trial_ids", []) or []
    if not isinstance(trial_ids_raw, Sequence) or isinstance(trial_ids_raw, (str, bytes)):
        raise OperatorHypothesisError(
            f"{where} ({hypothesis_id}): 'evidence_trial_ids' must be a list of ints"
        )
    trial_ids: list[int] = []
    for item in trial_ids_raw:
        if isinstance(item, bool) or not isinstance(item, int):
            raise OperatorHypothesisError(
                f"{where} ({hypothesis_id}): evidence trial id {item!r} is not an int"
            )
        trial_ids.append(item)

    event_ids_raw = row.get("evidence_event_ids", []) or []
    if not isinstance(event_ids_raw, Sequence) or isinstance(event_ids_raw, (str, bytes)):
        raise OperatorHypothesisError(
            f"{where} ({hypothesis_id}): 'evidence_event_ids' must be a list of strings"
        )
    event_ids = [str(item) for item in event_ids_raw if str(item).strip()]

    note = row.get("note", "") or ""
    if not isinstance(note, str):
        raise OperatorHypothesisError(f"{where} ({hypothesis_id}): 'note' must be a string")

    _assert_evidence_sufficient(hypothesis_id, status, trial_ids, event_ids, note)

    return Resolution(
        hypothesis_id=hypothesis_id,
        status=status,
        resolved_at=resolved_at,
        evidence_trial_ids=tuple(trial_ids),
        evidence_event_ids=tuple(event_ids),
        note=note.strip(),
        recorded_by=str(row.get("recorded_by", "autopilot") or "autopilot"),
    )


def _assert_evidence_sufficient(
    hypothesis_id: str,
    status: str,
    trial_ids: Sequence[int],
    event_ids: Sequence[str],
    note: str,
) -> None:
    """No queue-jumping: nothing is marked resolved without evidence."""
    if status in EVIDENCE_REQUIRING_STATUSES and not (trial_ids or event_ids):
        raise OperatorHypothesisError(
            f"{hypothesis_id}: a '{status}' resolution requires at least one "
            f"evidence id (trial id or journal event id). Marking an operator "
            f"hypothesis resolved on assertion alone is exactly the "
            f"queue-jumping this channel must not allow."
        )
    if status == "inconclusive" and not (trial_ids or event_ids or note.strip()):
        raise OperatorHypothesisError(
            f"{hypothesis_id}: an 'inconclusive' resolution requires evidence ids "
            f"or a note saying what was run and why it did not decide"
        )


def _load_resolutions(path: Path) -> dict[str, Resolution]:
    """Parse the append-only resolution ledger. Later rows supersede earlier."""
    if not path.exists():
        return {}
    resolutions: dict[str, Resolution] = {}
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise OperatorHypothesisError(
            f"cannot read operator hypothesis resolution ledger {path}: {exc}"
        ) from exc
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise OperatorHypothesisError(
                f"{path} line {index} is not valid JSON ({exc}). Refusing to skip "
                f"it: a dropped row would silently reopen a resolved hypothesis."
            ) from exc
        resolution = _resolution_from_row(index, row, path)
        resolutions[resolution.hypothesis_id] = resolution
    return resolutions


def load_operator_hypotheses(
    path: Path = DEFAULT_HYPOTHESES_PATH,
    resolutions_path: Path = DEFAULT_RESOLUTIONS_PATH,
) -> list[OperatorHypothesis]:
    """Load statements joined to their recorded resolutions.

    Raises:
        OperatorHypothesisError: the store or ledger is malformed, a statement
            lacks its mandatory falsifier, or the ledger resolves a hypothesis
            that does not exist (stale ledger row = drift, same reasoning as an
            expired accepted-gap waiver).
    """
    statements = _load_statements(path)
    resolutions = _load_resolutions(resolutions_path)

    known = {statement.id for statement in statements}
    orphans = sorted(set(resolutions) - known)
    if orphans:
        raise OperatorHypothesisError(
            f"{resolutions_path}: resolution(s) for unknown hypothesis id(s) "
            f"{orphans}. Either the statement was deleted from {path.name} while "
            f"its resolution stands, or the id is misspelled; both make the "
            f"open set wrong."
        )

    return [
        OperatorHypothesis(
            id=statement.id,
            hypothesis=statement.hypothesis,
            falsifier=statement.falsifier,
            stated_by=statement.stated_by,
            stated=statement.stated,
            confidence=statement.confidence,
            proposed_action=statement.proposed_action,
            notes=statement.notes,
            resolution=resolutions.get(statement.id),
        )
        for statement in statements
    ]


def still_open(
    hypotheses: Iterable[OperatorHypothesis] | None = None,
    *,
    path: Path = DEFAULT_HYPOTHESES_PATH,
    resolutions_path: Path = DEFAULT_RESOLUTIONS_PATH,
) -> list[OperatorHypothesis]:
    """Operator hypotheses that carry a falsifier and are not yet resolved."""
    if hypotheses is None:
        hypotheses = load_operator_hypotheses(path, resolutions_path)
    return [item for item in hypotheses if item.is_open]


# ── Resolution recording ─────────────────────────────────────────


def record_resolution(
    hypothesis_id: str,
    status: str,
    *,
    evidence_trial_ids: Sequence[int] = (),
    evidence_event_ids: Sequence[str] = (),
    note: str = "",
    recorded_by: str = "autopilot",
    resolved_at: str | None = None,
    path: Path = DEFAULT_HYPOTHESES_PATH,
    resolutions_path: Path = DEFAULT_RESOLUTIONS_PATH,
) -> Resolution:
    """Append one resolution to the ledger. Refutation is a first-class outcome.

    Raises:
        OperatorHypothesisError: unknown id, invalid status, insufficient
            evidence, or the hypothesis is already resolved (a correction is a
            deliberate hand-append to the ledger, not an accidental overwrite).
    """
    status_normalized = (status or "").strip().lower()
    if status_normalized not in RESOLUTION_STATUSES:
        raise OperatorHypothesisError(
            f"status must be one of {list(RESOLUTION_STATUSES)}, got {status!r}"
        )

    trial_ids = [int(t) for t in evidence_trial_ids]
    event_ids = [str(e) for e in evidence_event_ids if str(e).strip()]
    _assert_evidence_sufficient(hypothesis_id, status_normalized, trial_ids, event_ids, note)

    existing = load_operator_hypotheses(path, resolutions_path)
    by_id = {item.id: item for item in existing}
    if hypothesis_id not in by_id:
        raise OperatorHypothesisError(
            f"no operator hypothesis with id {hypothesis_id!r} in {path}"
        )
    already = by_id[hypothesis_id].resolution
    if already is not None:
        raise OperatorHypothesisError(
            f"{hypothesis_id} is already resolved as '{already.status}' at "
            f"{already.resolved_at}; refusing to overwrite the record"
        )

    resolution = Resolution(
        hypothesis_id=hypothesis_id,
        status=status_normalized,
        resolved_at=resolved_at or datetime.now(timezone.utc).isoformat(),
        evidence_trial_ids=tuple(trial_ids),
        evidence_event_ids=tuple(event_ids),
        note=(note or "").strip(),
        recorded_by=recorded_by,
    )

    resolutions_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(resolution.as_row(), sort_keys=True, default=str) + "\n"
    with open(resolutions_path, "a") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())
    return resolution


# ── Negative history / blacklist ─────────────────────────────────


def _check_blacklist_fn():
    """Import the live blacklist checker; never reimplement its matching."""
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    from state_store import check_blacklist  # noqa: PLC0415 — deferred by design

    return check_blacklist


def blacklist_conflicts(
    hypotheses: Iterable[OperatorHypothesis],
    blacklist: list[dict[str, Any]],
) -> dict[str, str]:
    """Map hypothesis id -> blacklist reason for proposals repeating a negative.

    Authorship is not new evidence. An operator hypothesis whose proposed action
    matches a recorded failure gets flagged so the critic can say so, exactly as
    it would for a planner-authored draft.
    """
    check_blacklist = _check_blacklist_fn()
    conflicts: dict[str, str] = {}
    for item in hypotheses:
        if not item.proposed_action:
            continue
        reason = check_blacklist(dict(item.proposed_action), blacklist or [])
        if reason:
            conflicts[item.id] = reason
    return conflicts


# ── Planner rendering ────────────────────────────────────────────


def render_planner_block(
    hypotheses: Iterable[OperatorHypothesis] | None = None,
    *,
    blacklist: list[dict[str, Any]] | None = None,
    limit: int = DEFAULT_RENDER_LIMIT,
    path: Path = DEFAULT_HYPOTHESES_PATH,
    resolutions_path: Path = DEFAULT_RESOLUTIONS_PATH,
) -> str:
    """Render the still-open operator set for the planner prompt.

    Wording deliberately mirrors the existing Operator Outbox block: planner
    context only, NOT an action gate.
    """
    if limit <= 0:
        return "  (disabled by AUTOPILOT_OPERATOR_HYPOTHESIS_RENDER_CAP)"
    if hypotheses is None:
        hypotheses = load_operator_hypotheses(path, resolutions_path)
    hypotheses = list(hypotheses)
    open_items = still_open(hypotheses)
    if not open_items:
        return "  (none)"

    conflicts: dict[str, str] = {}
    check_note = ""
    if blacklist:
        try:
            conflicts = blacklist_conflicts(open_items, blacklist)
        except Exception as exc:  # noqa: BLE001 — rendering must not abort a trial
            check_note = f"  (negative-history check unavailable: {exc})"

    lines = [
        "  Open operator hypotheses (planner context only; NOT an action gate). "
        "Each is a PRIOR, not a measurement:",
    ]
    if check_note:
        lines.append(check_note)
    for item in open_items[:limit]:
        lines.append(
            f"  - [{item.id} | {PLANNER_PROVENANCE} | confidence={item.confidence}] "
            f"{item.hypothesis[:240]}"
        )
        lines.append(f"      falsifier: {item.falsifier[:240]}")
        if item.proposed_action:
            action_text = json.dumps(dict(item.proposed_action), sort_keys=True, default=str)
            lines.append(f"      proposed action: {action_text[:180]}")
        if item.id in conflicts:
            lines.append(
                f"      NEGATIVE HISTORY: repeats a recorded failure — "
                f"{conflicts[item.id][:200]}. Authorship is not new evidence; do "
                f"not re-propose without a reason the prior failure no longer applies."
            )
    if len(open_items) > limit:
        lines.append(f"  ... and {len(open_items) - limit} more still open")
    lines.append(
        "  These carry NO measured evidence (evidence_trial_ids=[]). Rank them with "
        "everything else, prefer the cheapest action that could resolve one, and note "
        "in your rationale which one you are testing. They do not bypass the critic, "
        "the blacklist, or any gate, and an operator hypothesis the loop refutes is "
        "REFUTED — record it."
    )
    return "\n".join(lines)


# ── Call-site helpers (keep the autopilot.py diff to one line each) ──


#: Rendered when the store exists but cannot be parsed. NOT "(none)". Three
#: states, not two: a rendered open set, an empty set, and an unreadable store —
#: collapsing the third into the second is the lie this channel must not tell.
PLANNER_BLOCK_UNREADABLE_TEMPLATE = (
    "  !! OPERATOR HYPOTHESIS CHANNEL UNREADABLE: {error}\n"
    "  This does NOT mean the operator has no hypotheses. The store could not be\n"
    "  parsed, so the open set is UNKNOWN this turn. Do not read the absence of\n"
    "  operator priors as evidence about them; prefer an action that is sound under\n"
    "  either reading and say in your reasoning that the channel failed to load."
)


def build_planner_block(
    blacklist: list[dict[str, Any]] | None = None,
    *,
    limit: int = DEFAULT_RENDER_LIMIT,
    path: Path = DEFAULT_HYPOTHESES_PATH,
    resolutions_path: Path = DEFAULT_RESOLUTIONS_PATH,
) -> str:
    """Planner-prompt text for the open operator set; never raises.

    The store contract still raises — this wrapper exists so a broken store
    surfaces to the planner as an explicit alarm rather than aborting the trial
    OR masquerading as an empty set.
    """
    try:
        return render_planner_block(
            blacklist=blacklist, limit=limit, path=path, resolutions_path=resolutions_path
        )
    except OperatorHypothesisError as exc:
        log.error("Operator hypothesis store unreadable: %s", exc)
        return PLANNER_BLOCK_UNREADABLE_TEMPLATE.format(error=exc)
    except Exception as exc:  # noqa: BLE001 — prompt assembly must not abort a trial
        log.error("Operator hypothesis block assembly failed: %s", exc)
        return PLANNER_BLOCK_UNREADABLE_TEMPLATE.format(error=exc)


def record_resolution_from_rationale(
    rationale: Mapping[str, Any] | None,
    trial_id: int,
    *,
    path: Path = DEFAULT_HYPOTHESES_PATH,
    resolutions_path: Path = DEFAULT_RESOLUTIONS_PATH,
) -> Resolution | None:
    """Record a resolution the planner claimed in its rationale sidecar.

    The planner may name ONE operator hypothesis it believes this trial
    resolved, e.g.::

        {"operator_hypothesis": {"id": "...", "status": "refuted",
                                 "note": "what the outcome showed"}}

    The evidence is the trial that just ran, supplied HERE — the planner never
    supplies trial ids, so it cannot cite a trial it did not run, and it cannot
    mark anything resolved without a real trial behind it.

    A malformed or unknown claim is REJECTED (returns ``None`` and logs). That
    is not fail-open on the store: an untrusted planner claim is not a store
    defect, and refusing it leaves the hypothesis correctly still-open.
    """
    if not isinstance(rationale, Mapping):
        return None
    claim = rationale.get("operator_hypothesis")
    if not isinstance(claim, Mapping):
        return None
    hypothesis_id = str(claim.get("id", "") or "").strip()
    status = str(claim.get("status", "") or "").strip().lower()
    if not hypothesis_id or status not in RESOLUTION_STATUSES:
        log.warning(
            "Ignoring malformed operator-hypothesis resolution claim from trial %s: %r",
            trial_id,
            claim,
        )
        return None
    note = str(claim.get("note", "") or "").strip()
    try:
        return record_resolution(
            hypothesis_id,
            status,
            evidence_trial_ids=[trial_id],
            note=note,
            recorded_by="autopilot_planner",
            path=path,
            resolutions_path=resolutions_path,
        )
    except OperatorHypothesisError as exc:
        log.warning(
            "Rejected operator-hypothesis resolution claim %r from trial %s: %s",
            hypothesis_id,
            trial_id,
            exc,
        )
        return None


# ── CLI ──────────────────────────────────────────────────────────


def _cmd_validate(args: argparse.Namespace) -> int:
    items = load_operator_hypotheses(args.path, args.resolutions_path)
    open_count = len(still_open(items))
    print(
        f"OK: {len(items)} operator hypothes{'is' if len(items) == 1 else 'es'} "
        f"loaded from {args.path} ({open_count} still open)"
    )
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    items = load_operator_hypotheses(args.path, args.resolutions_path)
    if args.open_only:
        items = still_open(items)
    if args.json:
        print(json.dumps([item.as_dict() for item in items], indent=2, default=str))
        return 0
    if not items:
        print("(none)")
        return 0
    for item in items:
        state = "OPEN" if item.is_open else item.resolution.status.upper()  # type: ignore[union-attr]
        print(f"[{state}] {item.id} ({item.stated_by}, {item.stated}, confidence={item.confidence})")
        print(f"    hypothesis: {item.hypothesis}")
        print(f"    falsifier:  {item.falsifier}")
        if item.resolution is not None:
            print(
                f"    resolved:   {item.resolution.status} at "
                f"{item.resolution.resolved_at} — evidence {item.resolution.evidence_text()}"
            )
    return 0


def _cmd_resolve(args: argparse.Namespace) -> int:
    resolution = record_resolution(
        args.hypothesis_id,
        args.status,
        evidence_trial_ids=args.trial,
        evidence_event_ids=args.event,
        note=args.note or "",
        recorded_by=args.recorded_by,
        path=args.path,
        resolutions_path=args.resolutions_path,
    )
    print(
        f"recorded {resolution.status} for {resolution.hypothesis_id} "
        f"(evidence: {resolution.evidence_text()})"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", type=Path, default=DEFAULT_HYPOTHESES_PATH)
    parser.add_argument("--resolutions-path", type=Path, default=DEFAULT_RESOLUTIONS_PATH)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("validate", help="parse the store; non-zero exit on any defect")

    lister = sub.add_parser("list", help="list hypotheses and their state")
    lister.add_argument("--open-only", action="store_true")
    lister.add_argument("--json", action="store_true")

    resolver = sub.add_parser("resolve", help="record a resolution with its evidence")
    resolver.add_argument("hypothesis_id")
    resolver.add_argument("--status", required=True, choices=list(RESOLUTION_STATUSES))
    resolver.add_argument("--trial", type=int, action="append", default=[])
    resolver.add_argument("--event", action="append", default=[])
    resolver.add_argument("--note", default="")
    resolver.add_argument("--recorded-by", default="operator")

    args = parser.parse_args(argv)
    handlers = {"validate": _cmd_validate, "list": _cmd_list, "resolve": _cmd_resolve}
    try:
        return handlers[args.command](args)
    except OperatorHypothesisError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
