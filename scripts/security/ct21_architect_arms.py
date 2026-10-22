#!/usr/bin/env python3
"""CT-21 run prep: OFF/ON arms of the architect-only boundary sentence, through the PRODUCTION builders.

Reuses the frozen CT-18 harness (``ct18_ab_run.py``: fixtures, item iterators, graders, the
``WINDOW_CONFIRMED=1`` refusal, the result-row shape) and swaps only the arms and the renderer:

  OFF  ``build_architect_<surface>_prompt`` with flag ``architect_untrusted_boundary_sentence`` off
  ON   the same builder with the flag on (sentence directly above the untrusted block)

``--surface synthesis`` puts the fixture's tool output in the specialist-report slot of
``build_architect_synthesis_prompt``; ``--surface investigate`` puts it in the evidence-bundle slot
of ``build_architect_investigate_prompt``. The task text is the question. The rendered prompt is
sent as one user message (the architect delegation path sends the built prompt as the whole
request). All other options are CT-18's (``--suite``, ``--role``, ``--thinking``, ``--out``,
``--seed``, ``--max-tokens``, ``--dry-run``).

The decision rule is predeclared in epyc-root
``artifacts/chat-templates/ct21-architect-boundary/PREDECLARED_RULE.md``; read it before a run.
CT-20 wording: any figure is "static-attack ASR on fixture set X", never a robustness claim.
Nothing here calls a model unless the CT-18 harness's own window gate is satisfied.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import ct18_ab_run as H  # noqa: E402

ARMS = ("OFF", "ON")
SURFACES = ("synthesis", "investigate")
_SURFACE = "synthesis"


def render(arm: str, task: str, tool_text: str, key: str) -> list[dict]:
    from src.features import Features, reset_features, set_features
    from src.prompt_builders import build_architect_investigate_prompt, build_architect_synthesis_prompt

    if arm not in ARMS:
        raise ValueError(f"unknown CT-21 arm {arm!r}")
    set_features(Features(architect_untrusted_boundary_sentence=(arm == "ON")))
    try:
        if _SURFACE == "synthesis":
            prompt = build_architect_synthesis_prompt(task, tool_text, loop_num=1, max_loops=1)
        else:
            prompt = build_architect_investigate_prompt(task, context=tool_text)
    finally:
        reset_features()
    return [{"role": "user", "content": prompt}]


def main(argv: list[str] | None = None) -> int:
    global _SURFACE
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--surface", choices=SURFACES, default="synthesis")
    a, rest = ap.parse_known_args(argv if argv is not None else sys.argv[1:])
    _SURFACE = a.surface
    H.ARMS, H.render = ARMS, render
    sys.argv = [sys.argv[0], *rest]
    return H.main()


if __name__ == "__main__":
    raise SystemExit(main())
