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
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Mapping

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


@dataclass
class PlanDecision:
    path: str
    layers: tuple[str, ...]
    llm_called: bool
    decided_by: str  # "size_rule" | "deterministic_default_stub" | "typed_threshold" | "typed_choice"
    receipt: dict = field(default_factory=dict)


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


def _state(question: str, f: PlannerFacts) -> str:
    return json.dumps({"question": question, "facts": asdict(f)}, sort_keys=True)


def plan(
    question: str,
    facts: PlannerFacts,
    *,
    decide_fn: DecideFn | None = None,
    lo: int = DEFAULT_LO,
    hi: int = DEFAULT_HI,
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
    receipt.update(path=dec.path, layers=list(dec.layers), llm_called=dec.llm_called,
                   decided_by=dec.decided_by, elapsed_ms=(time.monotonic() - t0) * 1000)
    dec.receipt = receipt
    log.info("agg_planner receipt %s", json.dumps(receipt, sort_keys=True))
    return dec
