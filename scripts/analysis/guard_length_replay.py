#!/usr/bin/env python3
"""TD-30f: replay known-coherent text through the runtime repetition guards, by length.

ZERO INFERENCE. The question (RECTIFY 6f): does any runtime guard's trigger drift with
output length on text that is coherent by construction? For each corpus document and
each word-prefix length, run the guards unchanged and report, per guard x length bin:
documents, flagged, false-positive rate, and the minimum trigram unique ratio seen.

Guards (current source, unmodified):
  quality_detector  src.classifiers.quality_detector.detect_output_quality_issue
                    (all of its branches; only the trigram branch is length-sensitive,
                    reported separately as ``quality_detector_trigram``)
  anomaly_loop      src.pipeline_monitor.anomaly.detect_repetition_loop
  streaming         src.llm_primitives.inference._detect_streaming_repetition
                    (tail-4000-char exact 3-line repeat; length enters only through the tail)

Corpora are author-supplied paths: coherence here means "written by a person or agent as
prose", NOT an independently labelled coherence set, so a flag is a candidate false
positive, and a zero is a bound on these corpora only. Stdlib + the three guards.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_LENGTHS = (20, 50, 100, 200, 500, 1000, 2000, 5000)


def trigram_unique_ratio(words: list[str]) -> float | None:
    tri = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
    return len(set(tri)) / len(tri) if tri else None


def replay(docs: dict[str, str], guards: dict, lengths=DEFAULT_LENGTHS) -> dict:
    """docs: name -> text. guards: name -> callable(text) -> truthy if flagged."""
    out: dict = {"lengths": list(lengths), "guards": {}}
    for gname, fn in guards.items():
        per_len = {}
        for n in lengths:
            seen = flagged = 0
            min_ratio = None
            flagged_docs = []
            for name, text in docs.items():
                words = text.split()
                if len(words) < n:
                    continue  # only full-length prefixes: no mixing of lengths in a bin
                prefix = " ".join(words[:n])
                seen += 1
                ratio = trigram_unique_ratio(prefix.split())
                if ratio is not None:
                    min_ratio = ratio if min_ratio is None else min(min_ratio, ratio)
                if fn(prefix):
                    flagged += 1
                    flagged_docs.append(name)
            per_len[str(n)] = {
                "docs": seen, "flagged": flagged,
                "false_positive_rate": (flagged / seen) if seen else None,
                "min_trigram_unique_ratio": min_ratio,
                "flagged_docs": flagged_docs[:10],
            }
        out["guards"][gname] = per_len
    return out


def load_docs(patterns: list[str]) -> dict[str, str]:
    docs = {}
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            try:
                docs[f] = Path(f).read_text(errors="ignore")
            except OSError:
                continue
    return docs


def real_guards() -> dict:
    from src.classifiers.quality_detector import detect_output_quality_issue
    from src.llm_primitives.inference import _detect_streaming_repetition
    from src.pipeline_monitor.anomaly import detect_repetition_loop

    def trigram_branch(text: str):
        issue = detect_output_quality_issue(text)
        return issue if issue and issue.startswith("high_repetition") else None

    return {
        "quality_detector": detect_output_quality_issue,
        "quality_detector_trigram": trigram_branch,
        "anomaly_loop": detect_repetition_loop,
        "streaming": _detect_streaming_repetition,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", action="append", required=True, help="corpus glob (repeatable)")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()
    docs = load_docs(args.glob)
    if not docs:
        print("no documents matched")
        return 1
    report = {"n_docs": len(docs), "globs": args.glob, **replay(docs, real_guards())}
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(text + "\n", encoding="utf-8")
    for g, rows in report["guards"].items():
        print(g, {n: f"{r['flagged']}/{r['docs']}" for n, r in rows.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
