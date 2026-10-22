#!/usr/bin/env python3
"""HIL-LB1a: label the JSON-shaped inference-tap replies as grammar/schema-constrained or not.

The streaming repetition guard (``src/llm_primitives/inference.py``) is OFF for requests carrying
``json_schema`` / ``grammar``. HIL-LB1 found the guard replaying as firing on 26 of 649 JSON-shaped
recorded replies, 16 of them looking like live cuts, but could not say whether those requests were
constrained (and so guard-exempt in production) because the tap never recorded the request grammar.

This script labels each JSON-shaped reply from the best evidence available, kept in separate classes:

* ``recorded``: the tap ``start`` event carries ``request_constraints`` (written by the tap hook
  ``_request_constraints``, HIL-LB1a). A list: ``[]`` is recorded-unconstrained, otherwise the flags.
* ``inferred_call_site``: the event predates the hook, but the prompt text opens with a template that
  only ever reaches the model through a call site that passes ``json_schema`` (SIGNATURES, each cited
  to source). This is an inference from code, NOT a recovered flag, and is reported as such.
* ``unrecoverable``: neither. Counted, never guessed.

Counts only (recorded replies are private); no reply text and no request ids are written.
Zero inference, read-only.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import hil_lb1_loop_score as H  # noqa: E402

#: prompt-opening template -> (label, source evidence that the call site passes a schema).
#: Only call sites verified in source are listed; anything else stays unrecoverable.
SIGNATURES = {
    "Answer every question in the catalogue": (
        "typed_decisions.runner", "src/typed_decisions/runner.py llm_call(..., json_schema=schema)"),
    "Review plan. Reply JSON ONLY": (
        "review.plan", "src/proactive_delegation/review_service.py llm_call_no_think(..., "
        "json_schema=REVIEW_PLAN_ABBREV_SCHEMA)"),
    "Convert the reply given as the user message into exactly one JSON": (
        "structured_output.repair", "src/structured_output/repair.py CompleteFn -> llm_call(..., "
        "json_schema=dict(schema)) / response_format"),
}
SIGNATURE_WINDOW = 400


def signature_of(prompt: str) -> str | None:
    head = (prompt or "")[:SIGNATURE_WINDOW]
    for sig in SIGNATURES:
        if sig in head:
            return sig
    return None


def label_request(start: dict | None) -> tuple[str, str]:
    """(class, detail) for one request's ``start`` event."""
    if not start:
        return "unrecoverable", "no start event"
    rc = start.get("request_constraints")
    if isinstance(rc, list):
        return "recorded", "constrained:" + ",".join(rc) if rc else "unconstrained"
    sig = signature_of(start.get("prompt", ""))
    if sig:
        return "inferred_call_site", SIGNATURES[sig][0]
    return "unrecoverable", "pre-hook event, no known constrained call-site template"


def start_events(paths: list[Path], wanted: set[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in paths:
        with open(p, errors="replace") as fh:
            for line in fh:
                if '"start"' not in line:
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                rid = d.get("request_id")
                if d.get("event") == "start" and rid in wanted and rid not in out:
                    out[rid] = d
    return out


def label_tap(paths: list[Path]) -> dict:
    present = [p for p in paths if p.exists()]
    if not present:
        return {"status": "absent"}
    shaped: dict[str, list[str]] = {}
    for rid, chunks in H.tap_replies(present):
        text = "".join(chunks)
        if text.strip() and H._json_shaped(text):
            shaped[rid] = chunks
    starts = start_events(present, set(shaped))
    total, by_class, by_detail = Counter(), Counter(), Counter()
    guard: dict[str, Counter] = {"fired": Counter(), "fired_last_chunk(cut live)": Counter(),
                                 "fired_reply_continued": Counter()}
    for rid, chunks in shaped.items():
        cls, detail = label_request(starts.get(rid))
        total["json_shaped"] += 1
        by_class[cls] += 1
        by_detail[f"{cls}:{detail}"] += 1
        fire = H._current_guard_fire(chunks) if len(chunks) > 1 else None
        if fire is None and len(chunks) == 1:
            fire = H._current_guard_fire(H.LD.regex_tokenize(chunks[0]))
        if fire is not None:
            guard["fired"][cls] += 1
            if len(chunks) > 1:
                key = "fired_last_chunk(cut live)" if fire == len(chunks) else "fired_reply_continued"
                guard[key][cls] += 1
    return {
        "files": [p.name for p in present], "json_shaped": total["json_shaped"],
        "with_start_event": len(starts),
        "label_class": dict(by_class), "label_detail": dict(by_detail),
        "guard_fire_by_label_class": {k: dict(v) for k, v in guard.items()},
        "note": "inferred_call_site is code-derived (SIGNATURES), not a recovered request flag; "
                "unrecoverable replies stay unlabelled",
        "signatures": {k: v[1] for k, v in SIGNATURES.items()},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tap", action="append", required=True, help="inference_tap_events.jsonl (repeatable)")
    ap.add_argument("--out", help="write the counts JSON here (default: stdout only)")
    a = ap.parse_args(argv)
    res = label_tap([Path(p) for p in a.tap])
    txt = json.dumps(res, indent=1, sort_keys=True) + "\n"
    if a.out:
        Path(a.out).write_text(txt)
    print(txt, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
