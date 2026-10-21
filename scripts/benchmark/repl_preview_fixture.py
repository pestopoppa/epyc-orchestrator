#!/usr/bin/env python3
"""D-RI-PREVIEW-1: bounded-preview grammar comparator + blind-generalisation fixture.

Two renderings of a REPL variable for the root prompt:
  current  : name (type) = repr[:80]            (state.py get_state(); no true length)
  bounded  : name = type(len=N, first k: [...], last k: [...])   (true length stated)

Fixture: a 50-item batch where the visible half hides a rule. The first 25 items are all
tag "S" (sale, +qty). Items 25..49 mix in tag "R" (return, -qty) and tag "S". Task: net quantity
per tag over the WHOLE batch. A preview showing only the first half invites two failure modes
the scorer separates mechanically from the final answer:
  retyping_loss  : answer equals the net over only the items the model could see/copy
  blind_rule_fit : answer equals the rule fitted on the visible half applied to everything
                   (every item treated as a sale, qty summed)
  correct        : answer equals the full-batch net
The grammar's fit to training data is the authors' assertion; this fixture is how we test it on
our served worker roles (the run needs its own window). Deterministic: no randomness.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

N_ITEMS = 50


def build_batch() -> list[dict]:
    batch = []
    for i in range(N_ITEMS):
        if i < 25:
            tag, qty = "S", 10 + i
        else:
            tag = "R" if i % 3 == 0 else "S"
            qty = 10 + i
        batch.append({"id": f"ord-{i:03d}", "tag": tag, "qty": qty})
    return batch


def net(batch: list[dict]) -> dict:
    out: dict = {}
    for it in batch:
        sign = -1 if it["tag"] == "R" else 1
        out["net"] = out.get("net", 0) + sign * it["qty"]
    return out


def render_current(name: str, value) -> str:
    preview = repr(value)
    if len(preview) > 80:
        preview = preview[:80] + "..."
    return f"  {name} ({type(value).__name__}) = {preview}"


def render_bounded(name: str, value, k: int = 3) -> str:
    n = len(value)
    t = type(value).__name__
    return (f"  {name} = {t}(len={n}, first {k}: {list(value[:k])!r}, "
            f"last {k}: {list(value[-k:])!r})")


def render_half(name: str, value) -> str:
    """The test case itself: only the first half of the batch is shown, length not stated."""
    h = len(value) // 2
    return f"  {name} ({type(value).__name__}) = {list(value[:h])!r}"


def build_fixture(k: int = 3) -> dict:
    batch = build_batch()
    visible_current = []   # what the 80-char repr actually shows
    shown = render_current("batch", batch)
    for it in batch:
        if repr(it)[:40] in shown:
            visible_current.append(it)
    visible_bounded = batch[:k] + batch[-k:]
    visible_half = batch[:N_ITEMS // 2]
    full = net(batch)["net"]
    return {
        "schema": "epyc.repl_preview_fixture.v1",
        "task": "Variable `batch` is a list of order dicts (id, tag, qty). tag 'S' is a sale "
                "(adds qty), tag 'R' is a return (subtracts qty). Using `batch`, set FINAL to "
                "the net quantity over the WHOLE batch as an integer.",
        "batch": batch,
        "renderings": {"current": render_current("batch", batch),
                       f"bounded_k{k}": render_bounded("batch", batch, k),
                       "half_first25": render_half("batch", batch)},
        "expected": {"correct": full,
                     "retyping_loss_current": net(visible_current)["net"],
                     "retyping_loss_bounded": net(visible_bounded)["net"],
                     "retyping_loss_half": net(visible_half)["net"],
                     "blind_rule_fit": sum(it["qty"] for it in batch)},
        "visible_counts": {"current": len(visible_current), "bounded": len(visible_bounded), "half": len(visible_half),
                           "total": N_ITEMS},
        "scoring": "classify(answer, fixture, rendering)",
    }


def classify(answer, fx: dict, rendering: str) -> str:
    e = fx["expected"]
    if answer == e["correct"]:
        return "correct"
    if answer == e["blind_rule_fit"]:
        return "blind_rule_fit"
    key = {"current": "retyping_loss_current", "half_first25": "retyping_loss_half"}.get(
        rendering, "retyping_loss_bounded")
    if answer == e[key]:
        return "retyping_loss"
    return "other"


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/fixtures/repl_preview/batch50.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build_fixture(), indent=1) + "\n")
    print(out)
