#!/usr/bin/env python3
"""B7 scorer golden-delta harness.

Runs EVERY row of ``golden_corpus.jsonl`` through BOTH scorers:

* PRE  = ``debug_scorer_pre.py`` (extracted from commit 2a41c0bc — post
  hard-fail A1, PRE the 07a20a7c + 8f24679a semantics package).
* POST = the live ``scripts/benchmark/debug_scorer.py`` at HEAD.

Both are loaded via importlib under distinct module names so their
module-private state (each defines its own ``ScoringUnavailableError``) never
collides. Outcomes are captured as ``True`` / ``False`` / ``"ERROR:<ExcType>"``.

Writes:
* ``results.jsonl`` — one row per case: {case_id, scoring_method, provenance,
  old, new, changed, pin_fast}.
* ``REPORT.md`` — the operator-facing golden-delta document.

Deterministic except for llm_judge rows, which attempt a TCP connect to
127.0.0.1:1 (always refused, sub-millisecond) — both scorers see the same
unreachable-judge condition, so the delta is stable.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
CORPUS = HERE / "golden_corpus.jsonl"
RESULTS = HERE / "results.jsonl"
REPORT = HERE / "REPORT.md"

PRE_PATH = HERE / "debug_scorer_pre.py"
POST_PATH = REPO_ROOT / "scripts" / "benchmark" / "debug_scorer.py"

PRE_COMMIT = "2a41c0bc"
POST_COMMIT = "fe1da2f1 (HEAD, spec-dec-mtp-refresh-2026-06-22)"
PACKAGE_COMMITS = "07a20a7c + 8f24679a"

# code_execution spawns python3 subprocesses; llm_judge does a (refused) TCP
# connect. Both are excluded from the fast pin subset so the pin test stays
# well under its runtime budget on pure in-process scoring.
_SUBPROCESS_OR_NET = {"code_execution", "llm_judge"}

# Which audit finding explains each provenance tag (for the changed-row table).
FINDING_RATIONALE = {
    "SCORE-03": "colon/quote fallback now confined to the final-answer region (not CoT)",
    "SCORE-04": "entry_point + string expected without cases now ERRORs (no zero-arg synth)",
    "SCORE-05": "entry_point_cases oracle is a post-package capability",
    "SCORE-06": "substring now boundary-aware (digit/word units, not raw containment)",
    "SCORE-16": "exact_match now extracts \\boxed{...} incl. nested braces",
    "SCORE-21": "vacuous `assert True` no longer counts as an executable oracle",
    "SCORE-23": "non-string expected coerced to str instead of raising",
    "SCORE-24": "F1 multiset overlap + single-capture-group enforcement",
    "MC": "multiple_choice resolves textual/overlapping choice labels",
    "B7": "llm_judge substring fast-path is now boundary-aware",
}


def _load(path: Path, mod_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _outcome(fn, row: dict) -> Any:
    try:
        return bool(
            fn(
                answer=row["answer"],
                expected=row["expected"],
                scoring_method=row["scoring_method"],
                scoring_config=row.get("scoring_config"),
            )
        )
    except Exception as exc:  # noqa: BLE001 — we classify by type name
        return f"ERROR:{type(exc).__name__}"


def _direction(old: Any, new: Any) -> str | None:
    if old == new:
        return None
    old_err = isinstance(old, str)
    new_err = isinstance(new, str)
    if new_err and not old_err:
        return "X->ERROR"
    if old_err and not new_err:
        return "ERROR->X"
    if old_err and new_err:
        return "ERROR->ERROR"  # both error, different ExcType
    if old is False and new is True:
        return "False->True"
    if old is True and new is False:
        return "True->False"
    return "other"


def _finding_for(provenance: str) -> str:
    # provenance is "sentinel:<id>/<variant>" or "audit:<FINDING>/<variant>"
    tag = provenance.split(":", 1)[1].split("/", 1)[0]
    if tag.startswith("SCORE-"):
        return tag
    if provenance.startswith("audit:MC"):
        return "MC"
    if provenance.startswith("audit:B7"):
        return "B7"
    return tag  # sentinel id


def _rationale_for(provenance: str) -> str:
    finding = _finding_for(provenance)
    if finding in FINDING_RATIONALE:
        return f"{finding}: {FINDING_RATIONALE[finding]}"
    # sentinel-derived row: infer the finding from the variant suffix
    variant = provenance.split("/", 1)[1] if "/" in provenance else ""
    if "boxed" in variant:
        return f"SCORE-16: {FINDING_RATIONALE['SCORE-16']}"
    if "wrongfinal" in variant or "score03" in variant:
        return f"SCORE-03: {FINDING_RATIONALE['SCORE-03']}"
    if "nearmiss" in variant:
        return f"SCORE-06: {FINDING_RATIONALE['SCORE-06']}"
    return "semantics package (see per-method summary)"


def main() -> None:
    pre = _load(PRE_PATH, "debug_scorer_pre")
    post = _load(POST_PATH, "debug_scorer_post")

    rows = [json.loads(line) for line in CORPUS.read_text().splitlines() if line.strip()]

    results: list[dict] = []
    for row in rows:
        old = _outcome(pre.score_answer, row)
        new = _outcome(post.score_answer, row)
        results.append(
            {
                "case_id": row["case_id"],
                "scoring_method": row["scoring_method"],
                "provenance": row["provenance"],
                "old": old,
                "new": new,
                "changed": old != new,
                "pin_fast": row["scoring_method"] not in _SUBPROCESS_OR_NET,
            }
        )

    with RESULTS.open("w") as fh:
        for r in results:
            fh.write(json.dumps(r, sort_keys=True) + "\n")

    _write_report(rows, results)
    n_changed = sum(1 for r in results if r["changed"])
    print(f"scored {len(results)} rows; {n_changed} changed; wrote {RESULTS} and {REPORT}")


def _fmt(v: Any) -> str:
    return {True: "True", False: "False"}.get(v, str(v))


def _write_report(rows: list[dict], results: list[dict]) -> None:
    by_id = {r["case_id"]: r for r in rows}
    n = len(results)
    changed = [r for r in results if r["changed"]]
    n_changed = len(changed)

    # by scoring_method
    per_method = defaultdict(lambda: {"total": 0, "changed": 0})
    for r in results:
        per_method[r["scoring_method"]]["total"] += 1
        if r["changed"]:
            per_method[r["scoring_method"]]["changed"] += 1

    # by direction
    dir_counts = Counter(_direction(r["old"], r["new"]) for r in changed)

    # direction x method
    dir_method = defaultdict(Counter)
    for r in changed:
        dir_method[r["scoring_method"]][_direction(r["old"], r["new"])] += 1

    # sentinel scoring changes
    sentinel_changed = [
        r for r in changed if r["provenance"].startswith("sentinel:")
    ]
    sentinel_ids_changed = sorted(
        {r["provenance"].split(":", 1)[1].split("/", 1)[0] for r in sentinel_changed}
    )

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines: list[str] = []
    lines.append("# B7 Scorer Semantics — Golden Before/After Delta")
    lines.append("")
    lines.append(f"_Generated: {ts} · op-bundle ESC-6 option B (operator-granted 2026-07-21)_")
    lines.append("")
    lines.append("## What this is")
    lines.append("")
    lines.append(
        "The before/after behavioral delta the operator is owed for the "
        "**scorer-semantics package** — commits "
        f"`{PACKAGE_COMMITS}`'s changes to `scripts/benchmark/debug_scorer.py`. "
        "A golden input corpus is scored by BOTH the pre-package and the "
        "post-package scorer; every changed outcome is enumerated and tied to "
        "the audit finding that explains it."
    )
    lines.append("")
    lines.append(f"- **PRE scorer** = `debug_scorer_pre.py`, extracted from `{PRE_COMMIT}` "
                 "(post hard-fail A1, pre the semantics package).")
    lines.append(f"- **POST scorer** = live `scripts/benchmark/debug_scorer.py` at "
                 f"`{POST_COMMIT}`.")
    lines.append(f"- **Corpus**: `golden_corpus.jsonl` — {n} rows "
                 f"({sum(1 for r in results if r['provenance'].startswith('sentinel:'))} "
                 "derived from every live sentinel, "
                 f"{sum(1 for r in results if r['provenance'].startswith('audit:'))} "
                 "synthetic per-finding).")
    lines.append(f"- **Outcomes**: `{n_changed}` of `{n}` rows changed outcome under the package.")
    lines.append("")

    lines.append("## Summary — changed rows by scoring method")
    lines.append("")
    lines.append("| scoring_method | rows | changed | unchanged |")
    lines.append("|---|---:|---:|---:|")
    for m in sorted(per_method):
        d = per_method[m]
        lines.append(f"| {m} | {d['total']} | {d['changed']} | {d['total'] - d['changed']} |")
    lines.append(f"| **TOTAL** | **{n}** | **{n_changed}** | **{n - n_changed}** |")
    lines.append("")

    lines.append("## Summary — changed rows by direction")
    lines.append("")
    lines.append("| direction | count | meaning |")
    lines.append("|---|---:|---|")
    dir_meaning = {
        "False->True": "PRE scored wrong-form as miss / POST now credits it (boxed, multiset, textual MC)",
        "True->False": "PRE credited a non-answer / POST rejects it (boundary, final-region)",
        "X->ERROR": "POST refuses to score (ScoringUnavailableError / ValueError) where PRE returned a bool",
        "ERROR->X": "PRE raised where POST now returns a bool (non-string expected coerced)",
        "ERROR->ERROR": "both raise, but the ExcType changed (guarded ValueError vs incidental Attribute/Type error)",
    }
    for direction, count in sorted(dir_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {direction} | {count} | {dir_meaning.get(direction, '')} |")
    lines.append("")

    lines.append("## Direction × scoring method")
    lines.append("")
    all_dirs = sorted({d for c in dir_method.values() for d in c})
    header = "| scoring_method | " + " | ".join(all_dirs) + " |"
    lines.append(header)
    lines.append("|---" * (len(all_dirs) + 1) + "|")
    for m in sorted(dir_method):
        cells = " | ".join(str(dir_method[m].get(d, 0)) for d in all_dirs)
        lines.append(f"| {m} | {cells} |")
    lines.append("")

    lines.append("## Live sentinel scoring changes")
    lines.append("")
    lines.append(
        f"**{len(sentinel_changed)}** sentinel-derived rows changed, spanning "
        f"**{len(sentinel_ids_changed)}** distinct live sentinels (out of 39 in "
        "`scripts/autopilot/sentinel_questions.yaml`, the file eval_tower.py loads "
        "via `SENTINEL_PATH`)."
    )
    lines.append("")
    if sentinel_ids_changed:
        lines.append("Sentinels with at least one changed synthetic-answer row:")
        lines.append("")
        for sid in sentinel_ids_changed:
            hit = [r for r in sentinel_changed
                   if r["provenance"].split(":", 1)[1].split("/", 1)[0] == sid]
            variants = ", ".join(sorted(
                r["provenance"].split("/", 1)[1] for r in hit))
            lines.append(f"- `{sid}` ({hit[0]['scoring_method']}): {variants}")
        lines.append("")
    lines.append(
        "> These are changes on **synthetic probe answers** constructed to "
        "exercise the changed behaviors, not on real model transcripts. They "
        "show *which* sentinels' scoring is sensitive to the package (e.g. a "
        "`\\boxed{}`-only answer to a numeric sentinel, or an embedded-digit "
        "near-miss). The sentinels' own gold values are unchanged."
    )
    lines.append("")

    lines.append("## Every changed row")
    lines.append("")
    lines.append("| case_id | method | PRE | POST | dir | audit finding |")
    lines.append("|---|---|---|---|---|---|")
    for r in sorted(changed, key=lambda x: (x["scoring_method"], x["case_id"])):
        lines.append(
            f"| `{r['case_id']}` | {r['scoring_method']} | {_fmt(r['old'])} | "
            f"{_fmt(r['new'])} | {_direction(r['old'], r['new'])} | "
            f"{_rationale_for(r['provenance'])} |"
        )
    lines.append("")

    lines.append("## Era note")
    lines.append("")
    lines.append(
        "This delta is part of the **E7 eval-instrument boundary** (the "
        "measurement trust boundary: MEASUREMENT.md, eval tower, scoring, safety "
        "gates, era registry). The scorer-semantics package tightens how the "
        "deterministic eval-tower scorer credits/refuses answers; per the "
        "instrument constitution these are human-amendment-only changes, and "
        "this golden file is the ratified before/after record. "
        "**op-bundle ESC-6 option B is satisfied by this document.**"
    )
    lines.append("")
    lines.append("## Reproduce / pin")
    lines.append("")
    lines.append("```")
    lines.append("# regenerate corpus (deterministic)")
    lines.append("python orchestration/reports/b7_scorer_golden_delta_20260721/build_corpus.py")
    lines.append("# re-run the delta")
    lines.append("python orchestration/reports/b7_scorer_golden_delta_20260721/run_delta.py")
    lines.append("# pin: current scorer must still reproduce results.jsonl 'new' column")
    lines.append("python -m pytest tests/unit/test_b7_golden_corpus_pin.py -q")
    lines.append("```")
    lines.append("")
    lines.append(
        "`results.jsonl` records the ratified POST (`new`) outcome per case. "
        "`tests/unit/test_b7_golden_corpus_pin.py` re-runs the live scorer over "
        "the corpus and asserts it still produces exactly those outcomes — so a "
        "future scorer change must update this golden file **deliberately**."
    )
    lines.append("")

    REPORT.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
