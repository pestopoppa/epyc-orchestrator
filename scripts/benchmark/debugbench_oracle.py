#!/usr/bin/env python3
"""Build a debugbench oracle from the buggy->solution DIFF, and prove it discriminates.

WHY THIS MODULE EXISTS
======================

`artifacts/audit/debugbench-oracle-vacuity-20260812.md` (epyc-root) proved the
suite's oracle measured nothing. Every row's ``expected`` was a byte-exact
100-character PREFIX of the upstream reference solution, scored with
``substring``; those first 100 characters are class/constructor boilerplate that
is ALREADY PRESENT in the buggy code the model is handed, so a model that
changed nothing and echoed its input scored a PASS — on 4 of 4 pool rows and on
76.1% of the upstream corpus.

The 1,414 ``python3`` rows were broken in the opposite direction and nobody
noticed: the adapter set ``scoring_method="code_execution"`` with no
``test_code`` and no ``entry_point``, a configuration ``_score_code_execution``
answers with an unconditional ``False``. Half a suite that always passes, a
third that never can.

Widening the prefix is not a fix. The defect is containment in the input, not
length — a 500-character prefix is contained too. What separates a fix from a
no-op is not what the answer REPRODUCES but what it CHANGES, and the upstream
data does carry that: ``buggy_code`` and ``solution`` differ by a small,
localised patch (median 2 changed lines each way, measured over all 4,253 rows).

THE ORACLE
==========

For each row, two line sets, compared whitespace-free so formatting is not the
thing under test:

* ``required_lines`` — lines the solution has that the buggy code does not.
* ``forbidden_lines`` — lines the buggy code has that the solution does not,
  i.e. the broken statements the fix must remove.

Scored by ``debug_scorer``'s ``code_patch`` programmatic verifier: every
required line present, no forbidden line present, within one fenced block.

SELF-VALIDATION IS PART OF THE BUILD, NOT A REPORT ABOUT IT
===========================================================

``build_validated_oracle`` runs the candidate oracle through the REAL scorer
against three echo answers (the prompt, the buggy code, the buggy code fenced)
and four correct answers (the solution raw/fenced/reformatted/narrated) and
returns ``None`` unless all three fail and all four pass. A row that cannot
prove both directions is not emitted. That makes the two rates this remediation
is judged on structural rather than aspirational: on emitted rows the echo-pass
rate is 0 by construction and the reference-pass rate is 1 by construction, and
the number that carries real information is COVERAGE — how much of the corpus
survives: 3,676 of 4,253 rows (86.43%) at the default cap. 557 rows have no small
localised patch to test against, and 20 are refused by the gate itself (upstream's
``illegal comment`` subtype injects the bug by commenting a correct line OUT, so
the required text is still present in the buggy input and echo would pass). Run
``--report`` to reproduce every number in this docstring.

WHAT THIS ORACLE DOES NOT DO
============================

It tests "did you produce the reference patch", not "did you fix the bug" — a
different but correct repair fails it. That is a bounded risk here (the prompt
tells the model to fix only the bug and keep the structure) and it is why
``MAX_REQUIRED_LINES`` exists: rows whose reference patch ADDS more than a few
lines are demanding reproduction of a specific implementation, so they are
dropped rather than scored. Nothing short of executable tests removes this
limitation, and upstream ships none (fields: bug_explanation, buggy_code,
category, constraints, examples, language, level, question, release_time, slug,
solution, solution_explanation, subtype — no tests).

Usage::

    python3 scripts/benchmark/debugbench_oracle.py --report /tmp/out.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: Upstream snapshot cached on this host. Read-only; never rebuilt from here.
DEFAULT_EVAL_JSON = Path(
    "/mnt/raid0/llm/hf-home/hub/datasets--Rtian--DebugBench/snapshots/"
    "f474dcd2ad9276dfb48f96670f830da694870447/eval.json"
)

#: A line shorter than this (whitespace removed) is not evidence of anything: a
#: bare ``}`` or ``);`` appears and disappears in every patch, and pinning one
#: would make the oracle a brace-counter. Measured over the corpus, raising the
#: floor from 6 to 12 costs 4pp of coverage and buys nothing.
MIN_LINE_KEY_LEN = 6

#: Rows whose reference patch adds more than this many distinct lines are
#: dropped, not scored. Beyond a handful of lines the oracle stops asking "did
#: you fix the bug" and starts asking "did you write THIS implementation" — the
#: over-strict failure mode, which is as useless as the vacuous one. Coverage by
#: cap, measured: 2 -> 65.6%, 3 -> 79.1%, 4 -> 86.2%, 6 -> 91.4% of rows have a
#: buildable patch before validation.
MAX_REQUIRED_LINES = 4

#: Comment lines are excluded from both sides. Requiring a model to reproduce
#: the reference's ``// normal case so will go for top and left only`` measures
#: prose, and forbidding a comment the buggy version carried punishes a model
#: for leaving a comment alone.
_COMMENT = re.compile(r"^\s*(?://|#|/\*|\*/|\*(?!\w))")

_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")

_SCORER_MODULE_KEY = "epyc_debugbench_oracle_debug_scorer"


def _scorer() -> Any:
    """Load the sibling ``debug_scorer.py`` by path, under a private key.

    Same reason ``seeding_scoring`` does it: the research repo ships a diverged
    copy of this filename, and a bare import binds whichever won the sys.path
    race. The builder validates against the scorer that will actually score, so
    scorer identity cannot be left to import order.
    """
    cached = sys.modules.get(_SCORER_MODULE_KEY)
    if cached is not None:
        return cached
    path = Path(__file__).resolve().parent / "debug_scorer.py"
    spec = importlib.util.spec_from_file_location(_SCORER_MODULE_KEY, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load debug_scorer from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_SCORER_MODULE_KEY] = module
    spec.loader.exec_module(module)
    return module


def code_key(text: str) -> str:
    """Whitespace-free identity of a code line. Mirrors the scorer's ``_code_key``."""
    return re.sub(r"\s+", "", text)


def _is_evidence(raw_line: str) -> bool:
    if _COMMENT.match(raw_line):
        return False
    key = code_key(raw_line)
    return len(key) >= MIN_LINE_KEY_LEN and bool(_IDENTIFIER.search(key))


def patch_sides(buggy_code: str, solution: str) -> tuple[list[str], list[str]]:
    """Return ``(required_lines, forbidden_lines)`` for one buggy/solution pair.

    Membership is whole-line and whitespace-free, computed against the WHOLE
    other file rather than against aligned diff hunks on purpose: the question
    "does the answer still contain this broken statement" is a question about
    the file, not about where difflib chose to anchor.
    """
    buggy_lines = buggy_code.splitlines()
    solution_lines = solution.splitlines()
    buggy_keys = {code_key(line) for line in buggy_lines}
    solution_keys = {code_key(line) for line in solution_lines}

    def _side(lines: list[str], other_keys: set[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for line in lines:
            key = code_key(line)
            if key in other_keys or key in seen or not _is_evidence(line):
                continue
            seen.add(key)
            out.append(line.strip())
        return out

    required = _side(solution_lines, buggy_keys)
    forbidden = _side(buggy_lines, solution_keys)
    return required, forbidden


def build_oracle(
    buggy_code: str,
    solution: str,
    *,
    max_required: int = MAX_REQUIRED_LINES,
) -> dict[str, Any] | None:
    """Candidate ``scoring_config`` for one row, or ``None`` if not buildable."""
    required, forbidden = patch_sides(buggy_code, solution)
    if not required and not forbidden:
        return None
    if len(required) > max_required:
        return None
    return {
        "verifier": "code_patch",
        "required_lines": required,
        "forbidden_lines": forbidden,
    }


def _fence(code: str, language: str = "") -> str:
    return f"```{language}\n{code}\n```"


def _reformat(code: str) -> str:
    """A cosmetically different but semantically identical rendering.

    Doubles indentation and pads binary operators. A correct answer that happens
    to be formatted by a different hand must still pass, or the oracle is a
    style checker.
    """
    out = []
    for line in code.splitlines():
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        spaced = re.sub(r"(?<=[\w\)\]])([+\-*/<>=]=?)(?=[\w\(\[])", r" \1 ", stripped)
        out.append(" " * (indent * 2) + spaced)
    return "\n".join(out)


def echo_answers(*, prompt: str, buggy_code: str, language: str = "") -> dict[str, str]:
    """Answers that make NO change. Every one of these must score False."""
    return {
        "echo_buggy_code": buggy_code,
        "echo_buggy_fenced": _fence(buggy_code, language),
        "echo_whole_prompt": prompt,
    }


def correct_answers(*, solution: str, language: str = "") -> dict[str, str]:
    """Answers that ARE the known-correct fix. Every one must score True."""
    return {
        "reference_raw": solution,
        "reference_fenced": _fence(solution, language),
        "reference_reformatted": _fence(_reformat(solution), language),
        "reference_narrated": (
            "The bug is on the highlighted line. Here is the corrected code:\n\n"
            + _fence(solution, language)
            + "\n\nThe fix changes only that statement."
        ),
    }


def validate_oracle(
    oracle: dict[str, Any],
    *,
    prompt: str,
    buggy_code: str,
    solution: str,
    language: str = "",
) -> dict[str, Any]:
    """Score the echo answers and the correct answers through the REAL scorer."""
    score_answer = _scorer().score_answer
    echoes = {
        name: score_answer(text, "", "programmatic", oracle)
        for name, text in echo_answers(
            prompt=prompt, buggy_code=buggy_code, language=language
        ).items()
    }
    corrects = {
        name: score_answer(text, "", "programmatic", oracle)
        for name, text in correct_answers(solution=solution, language=language).items()
    }
    return {
        "echo": echoes,
        "correct": corrects,
        "any_echo_passes": any(echoes.values()),
        "all_correct_pass": all(corrects.values()),
        "discriminates": (not any(echoes.values())) and all(corrects.values()),
    }


def build_validated_oracle(
    *,
    prompt: str,
    buggy_code: str,
    solution: str,
    language: str = "",
    max_required: int = MAX_REQUIRED_LINES,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Build an oracle and emit it ONLY if it fails every echo and passes every fix.

    Returns ``(oracle_or_None, diagnostics)``. This is the whole point of the
    module: the discrimination check is a build gate, not a post-hoc report, so a
    row that cannot prove it discriminates never reaches a pool.
    """
    oracle = build_oracle(buggy_code, solution, max_required=max_required)
    if oracle is None:
        return None, {"reason": "no_buildable_patch"}
    result = validate_oracle(
        oracle, prompt=prompt, buggy_code=buggy_code, solution=solution, language=language
    )
    if not result["discriminates"]:
        return None, {"reason": "failed_validation", **result}
    return oracle, {"reason": "ok", **result}


# ── corpus report ────────────────────────────────────────────────────────


def _old_oracle_verdicts(row: dict[str, Any], prompt: str) -> dict[str, Any]:
    """What the SHIPPED oracle does on this row, measured through the scorer."""
    score_answer = _scorer().score_answer
    solution = row.get("solution", "")
    buggy = row.get("buggy_code", "")
    language = {"python3": "python"}.get(row.get("language", ""), row.get("language", ""))
    expected = solution[:100] if solution else "def "
    if language == "python":
        method, config = "code_execution", {"language": "python", "timeout": 30}
    else:
        method, config = "substring", {"case_sensitive": True}
    try:
        echo = score_answer(buggy, expected, method, config)
    except Exception:
        echo = False
    try:
        ref = score_answer(solution, expected, method, config)
    except Exception:
        ref = False
    try:
        prompt_echo = score_answer(prompt, expected, method, config)
    except Exception:
        prompt_echo = False
    return {"echo_passes": echo or prompt_echo, "reference_passes": ref}


def live_pool_report(pool_path: Path, upstream: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure the oracle the LIVE pool actually carries, row by row.

    Not the same instrument as the adapter code: on 2026-08-04 the pool's python
    rows were retargeted ``code_execution`` -> ``substring`` in the pool file
    itself (eval_tower.py:112), so what has been scored since then is not what a
    rebuild would produce. This function reads the shipped rows and answers the
    only two questions that matter about them — does echoing the prompt pass, and
    does the known-correct reference solution pass.
    """
    from collections import Counter

    score_answer = _scorer().score_answer
    by_id: dict[str, list[dict[str, Any]]] = {}
    for row in upstream:
        lang = {"python3": "python"}.get(row["language"], row["language"])
        by_id.setdefault(f"debugbench_{row['slug']}_{lang}", []).append(row)

    counts: Counter = Counter()
    per_lang: dict[str, Counter] = {}
    with pool_path.open() as handle:
        for line in handle:
            if '"debugbench' not in line:
                continue
            row = json.loads(line)
            if row.get("suite") != "debugbench":
                continue
            lang = (row.get("metadata") or {}).get("language", "?")
            counter = per_lang.setdefault(lang, Counter())
            counts["rows"] += 1
            counter["rows"] += 1
            method = row.get("scoring_method", "")
            config = row.get("scoring_config") or {}
            expected = row.get("expected", "")
            counts[f"method_{method}"] += 1
            echo = score_answer(row.get("prompt", ""), expected, method, config)
            counts["echo_passes"] += int(echo)
            counter["echo_passes"] += int(echo)
            for upstream_row in by_id.get(row.get("id", ""), []):
                if upstream_row["solution"][:100] == expected:
                    reference = score_answer(
                        upstream_row["solution"], expected, method, config)
                    counts["reference_matched"] += 1
                    counts["reference_passes"] += int(reference)
                    counter["reference_matched"] += 1
                    counter["reference_passes"] += int(reference)
                    break

    n = max(1, counts["rows"])
    matched = max(1, counts["reference_matched"])
    return {
        "path": str(pool_path),
        "rows": counts["rows"],
        "scoring_methods": {
            k[len("method_"):]: v for k, v in counts.items() if k.startswith("method_")
        },
        "echo_the_prompt_pass_rate_pct": round(100.0 * counts["echo_passes"] / n, 2),
        "reference_solution_matched": counts["reference_matched"],
        "reference_solution_pass_rate_pct": round(
            100.0 * counts["reference_passes"] / matched, 2),
        "by_language": {
            lang: {
                "rows": c["rows"],
                "echo_the_prompt_pass_rate_pct": round(
                    100.0 * c["echo_passes"] / max(1, c["rows"]), 2),
                "reference_solution_pass_rate_pct": round(
                    100.0 * c["reference_passes"] / max(1, c["reference_matched"]), 2),
            }
            for lang, c in sorted(per_lang.items())
        },
    }


def corpus_report(
    rows: list[dict[str, Any]],
    *,
    max_required: int = MAX_REQUIRED_LINES,
    include_old: bool = True,
    prompt_builder: Any = None,
) -> dict[str, Any]:
    """Measure old and new oracle over every upstream row."""
    from collections import Counter

    per_lang: dict[str, Counter] = {}
    totals = Counter()
    required_sizes: list[int] = []
    forbidden_sizes: list[int] = []
    for row in rows:
        buggy = row.get("buggy_code", "")
        solution = row.get("solution", "")
        language = row.get("language", "")
        # Same shape the adapter emits (fenced buggy code + instruction). Only the
        # code matters for the echo test; the surrounding question text cannot make
        # an oracle satisfiable that the buggy code does not already satisfy.
        prompt = (
            prompt_builder(row)
            if prompt_builder
            else f"## Buggy Code\n```{language}\n{buggy}\n```\n\nFind and fix the bug(s)."
        )
        counter = per_lang.setdefault(language, Counter())
        totals["rows"] += 1
        counter["rows"] += 1

        required, forbidden = patch_sides(buggy, solution)
        required_sizes.append(len(required))
        forbidden_sizes.append(len(forbidden))

        oracle, diag = build_validated_oracle(
            prompt=prompt,
            buggy_code=buggy,
            solution=solution,
            language=language,
            max_required=max_required,
        )
        if oracle is None:
            totals[f"dropped_{diag['reason']}"] += 1
            counter[f"dropped_{diag['reason']}"] += 1
            if diag["reason"] == "failed_validation":
                if diag.get("any_echo_passes"):
                    totals["dropped_echo_would_pass"] += 1
                if not diag.get("all_correct_pass"):
                    totals["dropped_reference_would_fail"] += 1
        else:
            totals["emitted"] += 1
            counter["emitted"] += 1

        if include_old:
            old = _old_oracle_verdicts(row, prompt)
            totals["old_echo_passes"] += int(old["echo_passes"])
            totals["old_reference_passes"] += int(old["reference_passes"])
            counter["old_echo_passes"] += int(old["echo_passes"])
            counter["old_reference_passes"] += int(old["reference_passes"])

    n = max(1, totals["rows"])
    return {
        "rows": totals["rows"],
        "max_required_lines": max_required,
        "new_oracle": {
            "emitted": totals["emitted"],
            "coverage_pct": round(100.0 * totals["emitted"] / n, 2),
            "dropped_no_buildable_patch": totals["dropped_no_buildable_patch"],
            "dropped_failed_validation": totals["dropped_failed_validation"],
            "dropped_because_echo_would_pass": totals["dropped_echo_would_pass"],
            "dropped_because_reference_would_fail": totals["dropped_reference_would_fail"],
            "emitted_echo_pass_rate_pct": 0.0,
            "emitted_reference_pass_rate_pct": 100.0,
            # The honest denominator: of the rows that HAVE a buildable patch,
            # what fraction would still have been vacuous (or unsatisfiable) had
            # the gate not been there? This is the number that says whether the
            # scheme itself is sound or whether the gate is carrying it.
            "ungated_echo_pass_pct": round(
                100.0
                * totals["dropped_echo_would_pass"]
                / max(1, totals["rows"] - totals["dropped_no_buildable_patch"]),
                2,
            ),
            "ungated_reference_fail_pct": round(
                100.0
                * totals["dropped_reference_would_fail"]
                / max(1, totals["rows"] - totals["dropped_no_buildable_patch"]),
                2,
            ),
        },
        "old_oracle": {
            "echo_pass_rate_pct": round(100.0 * totals["old_echo_passes"] / n, 2),
            "reference_pass_rate_pct": round(100.0 * totals["old_reference_passes"] / n, 2),
        },
        "patch_size": {
            "required_median": sorted(required_sizes)[len(required_sizes) // 2],
            "required_mean": round(sum(required_sizes) / n, 2),
            "required_max": max(required_sizes),
            "forbidden_median": sorted(forbidden_sizes)[len(forbidden_sizes) // 2],
            "forbidden_mean": round(sum(forbidden_sizes) / n, 2),
            "forbidden_max": max(forbidden_sizes),
        },
        "by_language": {
            lang: {
                "rows": c["rows"],
                "emitted": c["emitted"],
                "coverage_pct": round(100.0 * c["emitted"] / max(1, c["rows"]), 2),
                "old_echo_pass_rate_pct": round(
                    100.0 * c["old_echo_passes"] / max(1, c["rows"]), 2
                ),
                "old_reference_pass_rate_pct": round(
                    100.0 * c["old_reference_passes"] / max(1, c["rows"]), 2
                ),
            }
            for lang, c in sorted(per_lang.items())
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-json", type=Path, default=DEFAULT_EVAL_JSON)
    parser.add_argument("--report", type=Path, help="write the JSON report here")
    parser.add_argument("--max-required", type=int, default=MAX_REQUIRED_LINES)
    parser.add_argument("--limit", type=int, default=0, help="first N rows only (debug)")
    parser.add_argument(
        "--no-old", action="store_true", help="skip the shipped-oracle comparison"
    )
    parser.add_argument(
        "--pool",
        type=Path,
        help="also measure the LIVE question pool's shipped debugbench rows (read-only)",
    )
    args = parser.parse_args(argv)

    rows = json.loads(args.eval_json.read_text())
    if args.limit:
        rows = rows[: args.limit]
    report = corpus_report(
        rows, max_required=args.max_required, include_old=not args.no_old
    )
    if args.pool:
        report["live_pool_as_shipped"] = live_pool_report(args.pool, rows)
    # A JSON blob with no provenance is not evidence — it is a number someone
    # remembers producing. Stamp what made it and from what.
    report["provenance"] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": str(Path(__file__).resolve()),
        "argv": sys.argv[1:],
        "upstream_snapshot": str(args.eval_json),
        "remediates": "epyc-root artifacts/audit/debugbench-oracle-vacuity-20260812.md",
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
