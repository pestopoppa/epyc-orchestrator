#!/usr/bin/env python3
"""RTE-PFX-1 (1): static first-divergence check for REPL root-LM prompts, zero inference.

Renders the prompt for turn N and N+1 of one session through the real
``PromptBuilder.build_root_lm_prompt`` (and the ``src/graph/helpers.py`` tail the
graph path appends) under both section orders (``prefix_stable_order`` off / on) and a
set of turn-to-turn transitions, then reports the first-divergence character offset,
the section it falls in, a cause class, and the prefix-reusable share of the N+1
prompt.  Token figures are chars/4 ESTIMATES (no tokenizer, no model).  The root-LM
prompt is role-independent (same builder, same system/tools/rules for every role), so
rows are keyed by path, not role; the role-dependent additions are tail-only riders.

Cause classes: volatile_field (per-turn counter or per-call field), key_order
(same content, different order), tool_block_edit, history_rewrite (earlier text
changed or removed), state_preview_change.  A divergence inside the system/tools/
rules block is flagged ``PREFIX_BLOCK_DIVERGENCE``.

Usage: rte_pfx1_first_divergence.py [--json OUT] [--turn-counter-tail]   (run with the repo venv)

``--turn-counter-tail`` (RTE-PFX-3) renders with flag ``repl_turn_counter_tail`` on: the
``Turn N`` counter moves from the state-block head to the builder prompt's last line.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.prompt_builders import builder as B  # noqa: E402
from src.prompt_builders.types import PromptConfig, PromptStyle  # noqa: E402

TASK = "Refactor the parser module to use dataclasses and add unit tests."
STATE_A = (
    "context: str (4210 chars)\nartifacts: ['plan', 'files']\n"
    "  artifacts['plan']: read parser.py, then edit\n  artifacts['files']: ['parser.py', 'lexer.py']\n"
    "  n (int) = 3\n  names (list) = ['a', 'b']"
)
STATE_B = STATE_A.replace("n (int) = 3", "n (int) = 4")  # a changed variable preview
STATE_REORDERED = (
    "context: str (4210 chars)\nartifacts: ['files', 'plan']\n"
    "  artifacts['files']: ['parser.py', 'lexer.py']\n  artifacts['plan']: read parser.py, then edit\n"
    "  n (int) = 3\n  names (list) = ['a', 'b']"
)
STATE_GROWN = STATE_A + "\n  result (str) = 'ok'"
OUT1 = "parser.py: 120 lines\nclass Parser: ..."
OUT2 = "tests passed: 4"
WORKSPACE_BLOCK = "[Workspace]\nroot=/tmp/ws files=2"
SESSION_N = "[Session log]\nturn 1: read"
SESSION_N1 = "[Session log]\nturn 1: read\nturn 2: edit"
TURN_COUNTER_TAIL = False  # RTE-PFX-3 flag, set by --turn-counter-tail


def render(stable: bool, *, state: str, turn: int, last_output: str = "", last_error: str = "",
           routing: str = "", tools_variant: str | None = None, graph_tail: bool = False,
           session: str = "") -> str:
    B._get_features = lambda: SimpleNamespace(prefix_stable_order=stable, deferred_tool_results=False,
                                              repl_turn_counter_tail=TURN_COUNTER_TAIL)
    builder = B.PromptBuilder(PromptConfig(style=PromptStyle.MINIMAL))
    if tools_variant is not None:
        builder.config.tools_file = None
        builder._resolve_tools = lambda: tools_variant  # type: ignore[method-assign]
    text = builder.build_root_lm_prompt(
        state=state, original_prompt=TASK, last_output=last_output, last_error=last_error,
        turn=turn, routing_context=routing)
    if graph_tail:  # what src/graph/helpers.py appends after the builder
        text += "\n\n" + WORKSPACE_BLOCK
        if session:
            text += "\n\n" + session
    return text


def section_of(prompt: str, offset: int, stable: bool) -> str:
    """Name the builder section containing ``offset`` using the real section spans."""
    from src.prompt_builders.types import RootLMPrompt  # noqa: F401  (spans need a structured prompt)
    marks = [("## Available Tools", "tools"), ("## Rules", "rules"), ("## Task", "task"),
             ("## Your Code", "instruction"), ("## Current State", "state"),
             ("## Last", "context"), ("## Routing Intelligence", "context"),
             ("[Workspace]", "graph_tail"), ("[Session log]", "graph_tail")]
    found = [(prompt.find(m), name) for m, name in marks if prompt.find(m) >= 0]
    if TURN_COUNTER_TAIL:  # RTE-PFX-3: the builder's last line is the counter
        end = prompt.find("\n\n[Workspace]")
        tail = prompt.rfind("\nTurn ", 0, end if end >= 0 else len(prompt))
        if tail >= 0:
            found.append((tail + 1, "turn_tail"))
    found.sort()
    cur = "system"
    for pos, name in found:
        if pos <= offset:
            cur = name
    return cur


def classify(a: str, b: str, off: int, sec: str, label: str) -> str:
    if label in ("tool_block_edit", "key_order", "history_rewrite", "state_preview_change"):
        return label
    return "volatile_field" if sec in ("state", "context", "graph_tail", "task", "instruction",
                                       "turn_tail") else label


SCENARIOS = [
    # (name, N-kwargs, N+1-kwargs, declared cause)
    ("turn counter only", dict(state=STATE_A, turn=1, last_output=OUT1),
     dict(state=STATE_A, turn=2, last_output=OUT1), "volatile_field"),
    ("new last output", dict(state=STATE_A, turn=1, last_output=OUT1),
     dict(state=STATE_A, turn=2, last_output=OUT2), "volatile_field"),
    ("variable preview changed", dict(state=STATE_A, turn=1, last_output=OUT1),
     dict(state=STATE_B, turn=2, last_output=OUT1), "state_preview_change"),
    ("state key order changed", dict(state=STATE_A, turn=1, last_output=OUT1),
     dict(state=STATE_REORDERED, turn=2, last_output=OUT1), "key_order"),
    ("state grew by one variable", dict(state=STATE_A, turn=1, last_output=OUT1),
     dict(state=STATE_GROWN, turn=2, last_output=OUT1), "volatile_field"),
    ("routing context dropped after turn 0", dict(state=STATE_A, turn=0, routing="Role: frontdoor (Tier 1)"),
     dict(state=STATE_A, turn=1, last_output=OUT1), "history_rewrite"),
    ("tool block edited (negative control)", dict(state=STATE_A, turn=1, last_output=OUT1),
     dict(state=STATE_A, turn=2, last_output=OUT1, tools_variant="edited tools block"), "tool_block_edit"),
    ("graph tail: session log grows", dict(state=STATE_A, turn=1, last_output=OUT1, graph_tail=True, session=SESSION_N),
     dict(state=STATE_A, turn=2, last_output=OUT1, graph_tail=True, session=SESSION_N1), "volatile_field"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--turn-counter-tail", action="store_true")
    a = ap.parse_args()
    global TURN_COUNTER_TAIL
    TURN_COUNTER_TAIL = a.turn_counter_tail
    rows = []
    for stable in (False, True):
        for name, kn, kn1, declared in SCENARIOS:
            pn = render(stable, **kn)
            pn1 = render(stable, **kn1)
            off = next((i for i, (x, y) in enumerate(zip(pn, pn1)) if x != y), min(len(pn), len(pn1)))
            sec = section_of(pn1, off, stable)
            blockflag = sec in ("system", "tools", "rules") and off < len(pn1)
            rows.append({
                "prefix_stable_order": stable, "repl_turn_counter_tail": TURN_COUNTER_TAIL,
                "scenario": name,
                "first_divergence_char": off, "first_divergence_tok_est": off // 4,
                "section": sec, "cause_class": classify(pn, pn1, off, sec, declared),
                "prefix_share_of_n1": round(off / len(pn1), 4), "n1_chars": len(pn1),
                "PREFIX_BLOCK_DIVERGENCE": blockflag,
            })
    hdr = f"{'order':<8}{'scenario':<40}{'div@char':>9}{'~tok':>6}{'section':>12}{'cause':>22}{'prefix%':>9}  flag"
    print(hdr)
    for r in rows:
        print(f"{'stable' if r['prefix_stable_order'] else 'legacy':<8}{r['scenario']:<40}"
              f"{r['first_divergence_char']:>9}{r['first_divergence_tok_est']:>6}{r['section']:>12}"
              f"{r['cause_class']:>22}{100 * r['prefix_share_of_n1']:>8.1f}%  "
              f"{'PREFIX_BLOCK_DIVERGENCE' if r['PREFIX_BLOCK_DIVERGENCE'] else ''}")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
