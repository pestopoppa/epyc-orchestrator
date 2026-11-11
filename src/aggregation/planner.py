"""Typed planner skeleton (AGG-0.3): one decision per document question.

Paths: ``whole_doc`` | ``repl_nav`` | ``retrieval`` | ``sql``. Corpus size C and
questions-so-far Q are deterministic facts; the size rules below choose the path
whenever the question type cannot change it. Inside the ambiguous band (C > 60k,
where an aggregation question would switch to ``sql``) the plan asks one typed
decision (Choice over paths + Score "needs exhaustive aggregation"). Phase 0
ships that band as a STUB: unless a ``decide_fn`` is injected it returns the
deterministic size default and logs a ``decision_receipt.v1`` saying so. No
model is ever called from here; a caller that wants the typed call passes
``decide_fn`` (e.g. a closure over ``run_typed_decisions_native``).

Two-threshold rule (JEV-6): aggregation score >= ``hi`` -> ``sql``; <= ``lo`` ->
size default; in between -> the model's Choice.

The current token-count rule (``routing_decision.py``) remains the fallback and
is not touched; nothing in the request path calls this module yet.

``sql(+backfill)``: :func:`question_fields` names, deterministically and without a
model, the canonical fields a question needs; :func:`backfill_plan` compares them with
the canonical catalog and lists, per field, the documents with no row for the asked
period. When any is missing the ``sql`` path is a ``sql+backfill`` path: ``plan(...,
backfill=...)`` puts that plan in the ``decision_receipt.v1`` (``receipt["backfill"]``)
so the trigger is on the record. Building the missing L4 units stays the caller's
(``LazyFieldBackfill``) job; this module never calls a model.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Mapping, Sequence

from src.aggregation import require_enabled
from src.typed_decisions.types import Question, QuestionKind

log = logging.getLogger(__name__)

PATHS = ("whole_doc", "repl_nav", "retrieval", "sql")
SMALL_C = 60_000
ONE_DOC_C = 200_000
LARGE_C = 1_000_000
AGG_LEVELS = (0, 1, 2, 3)
DEFAULT_LO, DEFAULT_HI = 1, 3

LAYERS = {
    "whole_doc": ("L0",),
    "repl_nav": ("L0", "L1"),
    "retrieval": ("L0", "L1", "L2"),
    "sql": ("L0", "L1", "L3", "L4"),  # L4 lazy: backfill only for fields L3 lacks
}

PLANNER_QUESTIONS = (
    Question(
        id="path", kind=QuestionKind.CHOICE,
        text="Which answer path fits this question over the stated corpus?",
        options=PATHS,
        option_descriptions=(
            "read the whole document(s) in one long-context call",
            "navigate sections of one document with the REPL",
            "lexical+dense pointer search, then read the hits",
            "query the fact table; backfill missing fields lazily",
        ),
    ),
    Question(
        id="needs_aggregation", kind=QuestionKind.SCORE,
        text="Does answering require exhaustively aggregating >= ~10 facts across the corpus? "
             "0 = single lookup, 3 = certainly exhaustive aggregation.",
        levels=AGG_LEVELS,
    ),
)


@dataclass(frozen=True)
class PlannerFacts:
    corpus_tokens: int
    n_docs: int
    questions_so_far: int = 1
    available_layers: tuple[str, ...] = ()
    sql_fields_present: bool = False
    scope_doc: str | None = None  # set by :func:`scoped_facts` when the question targets ONE document


@dataclass
class PlanDecision:
    path: str
    layers: tuple[str, ...]
    llm_called: bool
    decided_by: str  # "size_rule" | "deterministic_default_stub" | "typed_threshold" | "typed_choice"
    receipt: dict = field(default_factory=dict)
    backfill: dict = field(default_factory=dict)  # the ``backfill_plan`` when the ``sql`` path must backfill


DecideFn = Callable[[str, tuple[Question, ...]], Mapping[str, object]]


def size_default(f: PlannerFacts) -> str:
    """Path from deterministic size rules for a lookup/navigation question."""
    c, q = f.corpus_tokens, f.questions_so_far
    if c <= SMALL_C:
        return "whole_doc"
    if c <= ONE_DOC_C:
        return "whole_doc" if q <= 1 else "repl_nav"
    if c < LARGE_C:
        return "retrieval"
    return "sql" if f.sql_fields_present else "retrieval"


def in_ambiguous_band(f: PlannerFacts) -> bool:
    """Question type can change the path only above the small-corpus threshold."""
    return f.corpus_tokens > SMALL_C and size_default(f) != "sql"


# ------------------------------------------------------------------ question scope
# A question that names exactly ONE corpus entity and carries no corpus-wide cue is a
# single-document question: the size rules apply to THAT document, not to the corpus it sits
# in. Without this, every question over a >= 1M-token corpus took the ``sql`` path, including
# single-fact lookups that no fact table row can answer (AGG Phase 0 first run: 10 of 10
# single-fact questions routed to ``sql``, 0 rows). Deterministic, no model call.
_AGG_CUES = re.compile(
    r"\b(companies|corpus|combined|across|collectively|all|each|every|which company|"
    r"how many|in total|total of)\b", re.I)
_LEGAL = frozenset({"inc", "corp", "corporation", "company", "co", "plc", "ltd", "llc", "the"})


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _entity_key(name: str) -> str:
    return "".join(t for t in _tokens(name) if t not in _LEGAL)


def mentions_entity(question: str, entity: str) -> bool:
    """True iff the entity's compact name equals 1-5 consecutive question tokens joined
    ('Coca Cola' and 'Coca-Cola' both match 'cocacola'; 'aes' never matches 'diseases')."""
    key = _entity_key(entity)
    qt = _tokens(question)
    if not key:
        return False
    for i in range(len(qt)):
        acc = ""
        for j in range(i, min(i + 5, len(qt))):
            acc += qt[j]
            if acc == key:
                return True
            if not key.startswith(acc):
                break
    return False


def resolve_scope(question: str, docs: Sequence[Mapping[str, object]]
                  ) -> tuple[Mapping[str, object] | None, str]:
    """``(doc, reason)``: the one document the question is about, else ``(None, reason)``.

    ``docs`` carry ``doc_id``, ``entity``, ``n_tokens_est``. Reasons: ``single_entity``,
    ``aggregation_cue``, ``no_entity``, ``multi_entity``.
    """
    if _AGG_CUES.search(question):
        return None, "aggregation_cue"
    hit = [d for d in docs if d.get("entity") and mentions_entity(question, str(d["entity"]))]
    if len(hit) == 1:
        return hit[0], "single_entity"
    return None, "multi_entity" if hit else "no_entity"


def scoped_facts(question: str, docs: Sequence[Mapping[str, object]], *, questions_so_far: int = 1,
                 available_layers: tuple[str, ...] = (), sql_fields_present: bool = False
                 ) -> tuple[PlannerFacts, str]:
    """PlannerFacts for ``question`` over ``docs``: the target document's size when scoped to
    one document (never ``sql``: ``sql_fields_present`` is cleared), else the corpus' size."""
    doc, reason = resolve_scope(question, docs)
    if doc is not None:
        return PlannerFacts(corpus_tokens=int(doc.get("n_tokens_est") or 0), n_docs=1,
                            questions_so_far=questions_so_far, available_layers=available_layers,
                            sql_fields_present=False, scope_doc=str(doc["doc_id"])), reason
    return PlannerFacts(corpus_tokens=sum(int(d.get("n_tokens_est") or 0) for d in docs),
                        n_docs=len(docs), questions_so_far=questions_so_far,
                        available_layers=available_layers,
                        sql_fields_present=sql_fields_present), reason


# ------------------------------------------------------------------ backfill trigger
# Which canonical fields does a question need? A phrase table, not a model: every label of
# ``facts_det.FIELD_ALIASES`` (longest first) plus the field names themselves, matched as whole-word
# sequences on alphanumeric tokens ('long-term debt' and 'long term debt' are one phrase).
_YEAR_Q = re.compile(r"(?<!\d)((?:19[89]|20[0-4])\d)(?!\d)")


def _phrase(text: str) -> str:
    return " ".join(_tokens(text))


def _field_phrases() -> list[tuple[str, str]]:
    from src.aggregation.facts_det import FIELD_ALIASES  # noqa: PLC0415 - keeps the planner import-light

    table = {_phrase(label): fld for label, fld in FIELD_ALIASES}
    for fld in {f for _l, f in FIELD_ALIASES}:
        table.setdefault(_phrase(fld.replace("_", " ")), fld)
    return sorted(table.items(), key=lambda kv: (-len(kv[0]), kv[0]))


def question_fields(question: str) -> list[str]:
    """Canonical fields named by ``question``, in order of first mention; a longer phrase hides the shorter ones
    inside it ('net income attributable to' is ``net_income_attributable`` only, not also ``net_income``)."""
    text = f" {_phrase(question)} "
    hits: list[tuple[int, str]] = []
    for phrase, fld in _field_phrases():
        at = text.find(f" {phrase} ")
        if at < 0:
            continue
        hits.append((at, fld))
        text = text[:at] + " " * (len(phrase) + 2) + text[at + len(phrase) + 2:]
    out: list[str] = []
    for _at, fld in sorted(hits):
        if fld not in out:
            out.append(fld)
    return out


def question_period(question: str) -> str | None:
    """The fiscal year a question asks about ('FY2022', '2022'), or None."""
    m = _YEAR_Q.search(question)
    return m.group(1) if m else None


def backfill_plan(question: str, canonical: Sequence[Mapping[str, object]], docs: Sequence[Mapping[str, object]],
                  *, period: str | None = None, fields: Sequence[str] | None = None) -> dict:
    """The lazy-backfill trigger: for every field the question needs, the documents with no canonical row for
    ``period`` (all periods when None). ``canonical`` rows carry ``doc_id``, ``field``, ``period`` (as
    ``CorpusStore.canonical_facts`` or ``select_canonical`` return them); ``docs`` carry ``doc_id``.
    Pure and deterministic: no store write, no model."""
    needed = list(fields) if fields is not None else question_fields(question)
    if period is None:
        period = question_period(question)
    doc_ids = sorted(str(d["doc_id"]) for d in docs)
    per: dict[str, dict] = {}
    for fld in needed:
        have = {str(r["doc_id"]) for r in canonical
                if r.get("field") == fld and (period is None or str(r.get("period")) == str(period))}
        missing = [d for d in doc_ids if d not in have]
        per[fld] = {"n_present": len(doc_ids) - len(missing), "n_missing": len(missing), "missing_docs": missing}
    to_fill = [f for f in needed if per[f]["n_missing"]]
    return {"schema": "backfill_plan.v1", "trigger": bool(to_fill), "needed_fields": needed, "fields": to_fill,
            "period": period, "n_docs": len(doc_ids),
            "missing_docs": sorted({d for f in to_fill for d in per[f]["missing_docs"]}), "per_field": per}


def _state(question: str, f: PlannerFacts) -> str:
    return json.dumps({"question": question, "facts": asdict(f)}, sort_keys=True)


def plan(
    question: str,
    facts: PlannerFacts,
    *,
    decide_fn: DecideFn | None = None,
    lo: int = DEFAULT_LO,
    hi: int = DEFAULT_HI,
    backfill: Mapping | None = None,
) -> PlanDecision:
    require_enabled()
    t0 = time.monotonic()
    default = size_default(facts)
    state = _state(question, facts)
    receipt = {
        "schema": "decision_receipt.v1",
        "surface": "agg_planner",
        "state_sha256": hashlib.sha256(state.encode()).hexdigest(),
        "facts": asdict(facts),
        "size_default": default,
        "thresholds": {"lo": lo, "hi": hi},
    }
    if not in_ambiguous_band(facts):
        dec = PlanDecision(default, LAYERS[default], False, "size_rule")
    elif decide_fn is None:
        dec = PlanDecision(default, LAYERS[default], False, "deterministic_default_stub")
        receipt["stub"] = True
    else:
        answers = decide_fn(state, PLANNER_QUESTIONS)
        score = int(answers.get("needs_aggregation", 0))  # type: ignore[arg-type]
        choice = str(answers.get("path", default))
        receipt["answers"] = {"path": choice, "needs_aggregation": score}
        if score >= hi:
            dec = PlanDecision("sql", LAYERS["sql"], True, "typed_threshold")
        elif score <= lo:
            dec = PlanDecision(default, LAYERS[default], True, "typed_threshold")
        else:
            path = choice if choice in PATHS else default
            dec = PlanDecision(path, LAYERS[path], True, "typed_choice")
    if backfill is not None:
        # Only the ``sql`` path reads L4; on any other path the trigger is recorded as not applicable.
        if dec.path == "sql" and backfill.get("trigger"):
            dec.backfill = dict(backfill)
            receipt["backfill"] = dec.backfill
            receipt["path_label"] = "sql+backfill"
        else:
            receipt["backfill"] = {**dict(backfill), "trigger": False,
                                   "reason": "path " + dec.path + " does not read L4"
                                   if dec.path != "sql" else "no needed field is missing"}
    receipt.update(path=dec.path, layers=list(dec.layers), llm_called=dec.llm_called,
                   decided_by=dec.decided_by, elapsed_ms=(time.monotonic() - t0) * 1000)
    dec.receipt = receipt
    log.info("agg_planner receipt %s", json.dumps(receipt, sort_keys=True))
    return dec
