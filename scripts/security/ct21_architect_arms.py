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


def summarise(rows: list[dict], roles) -> dict:
    """CT-21 OFF/ON pairs; CT-18's A/B/C summary remains unchanged."""
    out = {}
    for role in roles:
        for variant, metric in (("conflict", "attacked"), ("aligned", "complied"), ("clean", "task_ok")):
            per = {arm: {r["id"]: r["verdict"][metric] for r in rows
                         if r["role"] == role and r["arm"] == arm and r["variant"] == variant}
                   for arm in ARMS}
            ids = sorted(set(per["OFF"]) & set(per["ON"]))
            if not ids:
                continue
            out[f"{role}/{variant}/{metric}"] = {
                **{arm: {"rate": sum(per[arm][i] for i in ids) / len(ids), "n": len(ids)} for arm in ARMS},
                "OFF_vs_ON": H.G.discordant([per["OFF"][i] for i in ids], [per["ON"][i] for i in ids]),
            }
    return out


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
    old_arms, old_render, old_summary, old_items, old_argv = H.ARMS, H.render, H.summarise, H.skillinject_items, sys.argv
    # The predeclared CT-21 set is the 316 CT-18 complete triples: exclude
    # SI-C037 (native fixture id SI-C037-0) from BOTH arms, not only after
    # a context-overflow response.
    def skillinject_items(root):
        return (item for item in old_items(root) if item[0]["id"] != "SI-C037-0")
    H.ARMS, H.render, H.summarise, H.skillinject_items = ARMS, render, summarise, skillinject_items
    sys.argv = [sys.argv[0], *rest]
    try:
        return H.main()
    finally:
        H.ARMS, H.render, H.summarise, H.skillinject_items, sys.argv = old_arms, old_render, old_summary, old_items, old_argv


if __name__ == "__main__":
    raise SystemExit(main())
