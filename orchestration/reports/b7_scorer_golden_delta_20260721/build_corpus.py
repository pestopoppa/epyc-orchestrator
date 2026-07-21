#!/usr/bin/env python3
"""Build the B7 scorer golden-delta input corpus.

Emits ``golden_corpus.jsonl`` in this directory. Each row:

    {case_id, scoring_method, answer, expected, scoring_config, provenance}

Two provenance families:

* ``sentinel:<id>/<variant>`` — derived from EVERY live sentinel in
  ``scripts/autopilot/sentinel_questions.yaml`` (the exact file eval_tower.py
  loads via ``SENTINEL_PATH``). Each sentinel gets 2-3 synthetic model answers:
  a correct-form answer, a near-miss the PRE-package scorer accepted wrongly,
  and a chain-of-thought answer whose value appears mid-reasoning.
* ``audit:<FINDING>/<variant>`` — synthetic cases pinned to a specific audit
  finding (SCORE-03/04/05/06/16/21/23/24, mc-textual, llm_judge fast-path).

The corpus is deterministic: no randomness, no network, no clock. Re-running
regenerates byte-identical output (module dict order is insertion order).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
# repo root = .../epyc-orchestrator ; this file lives at
# orchestration/reports/b7_scorer_golden_delta_20260721/build_corpus.py
REPO_ROOT = HERE.parents[2]
SENTINEL_PATH = REPO_ROOT / "scripts" / "autopilot" / "sentinel_questions.yaml"
OUT_PATH = HERE / "golden_corpus.jsonl"

_SEP = re.compile(r"[,_ ]")


def _digits_only(s: str) -> str:
    return _SEP.sub("", s)


def _is_pure_number(s: str) -> bool:
    d = _digits_only(s)
    return bool(d) and d.isdigit()


def _is_glueable_word(s: str) -> bool:
    """A token/phrase whose last char is alnum so a glued suffix breaks the
    word boundary the post-package substring scorer enforces."""
    return bool(s) and bool(re.fullmatch(r"[A-Za-z0-9_ ]+", s)) and s[-1].isalnum()


def _substring_rows(sid: str, expected: str, cfg: dict) -> list[dict]:
    rows: list[dict] = []
    # (a) correct-form — expected present as a clean text unit
    rows.append(
        dict(
            case_id=f"{sid}__substr_correct",
            scoring_method="substring",
            answer=f"After working through it, the answer is {expected}.",
            expected=expected,
            scoring_config=cfg,
            provenance=f"sentinel:{sid}/correct",
        )
    )
    # (b) near-miss the PRE scorer accepted wrongly (SCORE-06 boundary):
    #     glue the expected value into a longer number/word so plain-substring
    #     matched but boundary-aware matching rejects it. Only meaningful when
    #     the expected value ends in an alnum char (pure number or glueable word);
    #     symbolic/bracketed expecteds get no boundary applied, so skip them.
    near = None
    if _is_pure_number(expected):
        near = f"The intermediate computation yielded {expected}0 exactly."
    elif _is_glueable_word(expected):
        near = f"The relevant token here is {expected}s inside a larger word."
    if near is not None:
        rows.append(
            dict(
                case_id=f"{sid}__substr_nearmiss",
                scoring_method="substring",
                answer=near,
                expected=expected,
                scoring_config=cfg,
                provenance=f"sentinel:{sid}/nearmiss_embedded",
            )
        )
    # (c) CoT — value appears mid-reasoning as a clean unit (stability: both pass)
    rows.append(
        dict(
            case_id=f"{sid}__substr_cot",
            scoring_method="substring",
            answer=(
                "Let me reason step by step.\n"
                f"Working through the problem, I find {expected} along the way.\n"
                f"Therefore the final answer is {expected}."
            ),
            expected=expected,
            scoring_config=cfg,
            provenance=f"sentinel:{sid}/cot_midreasoning",
        )
    )
    return rows


def _exact_match_rows(sid: str, expected: str, cfg: dict) -> list[dict]:
    wrong = "999" if expected.strip() != "999" else "111"
    return [
        dict(
            case_id=f"{sid}__em_correct",
            scoring_method="exact_match",
            answer=f"Reasoning omitted for brevity. <answer>{expected}</answer>",
            expected=expected,
            scoring_config=cfg,
            provenance=f"sentinel:{sid}/correct",
        ),
        # SCORE-16: value only inside \boxed{} with no <answer> tag.
        dict(
            case_id=f"{sid}__em_boxed",
            scoring_method="exact_match",
            answer=f"After the derivation, the result is \\boxed{{{expected}}}.",
            expected=expected,
            scoring_config=cfg,
            provenance=f"sentinel:{sid}/boxed_score16",
        ),
        # SCORE-03: expected value sits in a mid-reasoning colon line, but the
        # FINAL answer line is a different (wrong) value. Pre scorer harvested the
        # expected value out of the reasoning region; post scorer confines the
        # colon/quote fallback to the final-answer region.
        dict(
            case_id=f"{sid}__em_cot_wrongfinal",
            scoring_method="exact_match",
            answer=f"Candidate: {expected}\nFinal answer: {wrong}",
            expected=expected,
            scoring_config=cfg,
            provenance=f"sentinel:{sid}/cot_wrongfinal_score03",
        ),
    ]


def _f1_rows(sid: str, expected: str, cfg: dict) -> list[dict]:
    return [
        dict(
            case_id=f"{sid}__f1_correct",
            scoring_method="f1",
            answer=f"Based on the context. <answer>{expected}</answer>",
            expected=expected,
            scoring_config=cfg,
            provenance=f"sentinel:{sid}/correct",
        ),
        dict(
            case_id=f"{sid}__f1_wrong",
            scoring_method="f1",
            answer="<answer>completely unrelated placeholder tokens</answer>",
            expected=expected,
            scoring_config=cfg,
            provenance=f"sentinel:{sid}/wrong",
        ),
        dict(
            case_id=f"{sid}__f1_cot",
            scoring_method="f1",
            answer=(
                "Let me narrow it down.\n"
                f"After considering the evidence, <answer>{expected}</answer>"
            ),
            expected=expected,
            scoring_config=cfg,
            provenance=f"sentinel:{sid}/cot_midreasoning",
        ),
    ]


def _mc_rows(sid: str, expected: str, cfg: dict) -> list[dict]:
    exp = expected.strip().upper()
    wrong = "A" if exp != "A" else "B"
    return [
        dict(
            case_id=f"{sid}__mc_correct",
            scoring_method="multiple_choice",
            answer=f"After analysis, the answer is {exp}.",
            expected=expected,
            scoring_config=cfg,
            provenance=f"sentinel:{sid}/correct",
        ),
        dict(
            case_id=f"{sid}__mc_wrong",
            scoring_method="multiple_choice",
            answer=f"After analysis, the answer is {wrong}.",
            expected=expected,
            scoring_config=cfg,
            provenance=f"sentinel:{sid}/wrong",
        ),
        dict(
            case_id=f"{sid}__mc_cot",
            scoring_method="multiple_choice",
            answer=(
                f"Option {wrong} looks tempting at first glance, but on closer "
                f"reading the correct choice is {exp}.\n{exp}"
            ),
            expected=expected,
            scoring_config=cfg,
            provenance=f"sentinel:{sid}/cot_midreasoning",
        ),
    ]


def _sentinel_rows() -> list[dict]:
    sentinels = yaml.safe_load(SENTINEL_PATH.read_text())
    rows: list[dict] = []
    for q in sentinels:
        sid = q["id"]
        method = q["scoring_method"]
        expected = str(q["expected"])
        cfg = q.get("scoring_config") or {}
        if method == "substring":
            rows.extend(_substring_rows(sid, expected, cfg))
        elif method == "exact_match":
            rows.extend(_exact_match_rows(sid, expected, cfg))
        elif method == "f1":
            rows.extend(_f1_rows(sid, expected, cfg))
        elif method == "multiple_choice":
            rows.extend(_mc_rows(sid, expected, cfg))
        else:  # pragma: no cover - defensive
            raise ValueError(f"unhandled sentinel scoring_method {method!r}")
    return rows


def _audit_rows() -> list[dict]:
    R: list[dict] = []

    # ── SCORE-06: substring boundary awareness ───────────────────────────
    R += [
        dict(case_id="audit_score06_630", scoring_method="substring",
             answer="The total came out to 630 items in the end.",
             expected="63", scoring_config={},
             provenance="audit:SCORE-06/digit_embed_630"),
        dict(case_id="audit_score06_2630", scoring_method="substring",
             answer="The total came out to 2,630 items in the end.",
             expected="63", scoring_config={},
             provenance="audit:SCORE-06/digit_embed_2630"),
        dict(case_id="audit_score06_concat", scoring_method="substring",
             answer="We concatenate these strings together.",
             expected="cat", scoring_config={},
             provenance="audit:SCORE-06/word_embed_concatenate"),
        dict(case_id="audit_score06_blackcat_ok", scoring_method="substring",
             answer="the black cat slept on the mat",
             expected="cat", scoring_config={},
             provenance="audit:SCORE-06/word_boundary_ok"),
        dict(case_id="audit_score06_grouped_ok", scoring_method="substring",
             answer="The factorial evaluates to 479,001,600 exactly.",
             expected="479001600", scoring_config={},
             provenance="audit:SCORE-06/digit_separator_ok"),
    ]

    # ── SCORE-16: boxed extraction (incl. nested braces) in exact_match ──
    R += [
        dict(case_id="audit_score16_nested_frac", scoring_method="exact_match",
             answer=r"Work omitted. Therefore \boxed{\frac{1}{2}}.",
             expected=r"\frac{1}{2}", scoring_config={},
             provenance="audit:SCORE-16/nested_boxed"),
        dict(case_id="audit_score16_simple", scoring_method="exact_match",
             answer=r"So the count is \boxed{42} in total.",
             expected="42", scoring_config={},
             provenance="audit:SCORE-16/simple_boxed"),
    ]

    # ── SCORE-03: colon/quote fallback confined to final-answer region ───
    R += [
        dict(case_id="audit_score03_quote_final", scoring_method="exact_match",
             answer='Earlier evidence mentions "Paris".\nFinal answer: "London"',
             expected="London", scoring_config={},
             provenance="audit:SCORE-03/quote_final_region_correct"),
        dict(case_id="audit_score03_quote_cotwrong", scoring_method="exact_match",
             answer='Earlier evidence mentions "Paris".\nFinal answer: "London"',
             expected="Paris", scoring_config={},
             provenance="audit:SCORE-03/quote_cot_value_rejected"),
        dict(case_id="audit_score03_colon_cotwrong", scoring_method="exact_match",
             answer="Consider: London\nResult: Paris",
             expected="London", scoring_config={},
             provenance="audit:SCORE-03/colon_cot_value_rejected"),
    ]

    # ── SCORE-23: non-string expected coerced to str (not an ERROR) ──────
    R += [
        dict(case_id="audit_score23_int_exact", scoring_method="exact_match",
             answer="1,234", expected=1234, scoring_config={},
             provenance="audit:SCORE-23/int_expected_exact_match"),
        dict(case_id="audit_score23_none_substr", scoring_method="substring",
             answer="anything at all here", expected=None, scoring_config={},
             provenance="audit:SCORE-23/none_expected_substring"),
        dict(case_id="audit_score23_list_f1", scoring_method="f1",
             answer="<answer>1 2</answer>", expected=[1, 2], scoring_config={},
             provenance="audit:SCORE-23/list_expected_f1"),
    ]

    # ── SCORE-24: multiset F1 + single-capture-group enforcement ─────────
    R += [
        dict(case_id="audit_score24_multiset_a", scoring_method="f1",
             answer="<answer>red red blue</answer>", expected="red red green",
             scoring_config={"threshold": 0.6},
             provenance="audit:SCORE-24/multiset_recovers_repeat"),
        dict(case_id="audit_score24_multiset_b", scoring_method="f1",
             answer="<answer>apple apple apple pear</answer>",
             expected="apple apple apple plum",
             scoring_config={"threshold": 0.7},
             provenance="audit:SCORE-24/multiset_recovers_repeat2"),
        dict(case_id="audit_score24_multigroup_f1", scoring_method="f1",
             answer="xy", expected="x",
             scoring_config={"extract_pattern": r"(x)(y)"},
             provenance="audit:SCORE-24/multigroup_pattern_f1"),
        dict(case_id="audit_score24_multigroup_em", scoring_method="exact_match",
             answer="xy", expected="x",
             scoring_config={"extract_pattern": r"(x)(y)"},
             provenance="audit:SCORE-24/multigroup_pattern_exact_match"),
        dict(case_id="audit_score24_zerogroup_em", scoring_method="exact_match",
             answer="xy", expected="x",
             scoring_config={"extract_pattern": r"xy"},
             provenance="audit:SCORE-24/zerogroup_pattern_exact_match"),
    ]

    # ── multiple_choice: textual labels, overlapping choices, "(B)" label ─
    R += [
        dict(case_id="audit_mc_textual_label", scoring_method="multiple_choice",
             answer="I choose the black cat.",
             expected="black cat",
             scoring_config={"choices": ["cat", "black cat"]},
             provenance="audit:MC/textual_label"),
        dict(case_id="audit_mc_overlap_cat", scoring_method="multiple_choice",
             answer="just a cat here, nothing else",
             expected="cat",
             scoring_config={"choices": ["cat", "black cat"]},
             provenance="audit:MC/overlapping_choice_shorter"),
        dict(case_id="audit_mc_overlap_blackcat", scoring_method="multiple_choice",
             answer="the black cat is the one",
             expected="black cat",
             scoring_config={"choices": ["cat", "black cat"]},
             provenance="audit:MC/overlapping_choice_longer"),
        dict(case_id="audit_mc_none_of_above", scoring_method="multiple_choice",
             answer="The answer is None of the above.",
             expected="None of the above",
             scoring_config={"choices": ["None", "None of the above"]},
             provenance="audit:MC/none_vs_none_of_the_above"),
        dict(case_id="audit_mc_paren_label", scoring_method="multiple_choice",
             answer="After analysis, the answer is B.",
             expected="(B)", scoring_config={},
             provenance="audit:MC/paren_wrapped_expected"),
    ]

    # ── llm_judge: boundary-aware fast path (judge on port 1 => unreachable) ─
    judge_cfg = {"judge_port": 1, "judge_host": "127.0.0.1", "timeout": 2}
    R += [
        dict(case_id="audit_judge_fastpath_concat", scoring_method="llm_judge",
             answer="the word concatenate appears here", expected="cat",
             scoring_config=dict(judge_cfg),
             provenance="audit:B7/llm_judge_fastpath_boundary"),
        dict(case_id="audit_judge_fastpath_unit_ok", scoring_method="llm_judge",
             answer="the black cat sat down", expected="cat",
             scoring_config=dict(judge_cfg),
             provenance="audit:B7/llm_judge_fastpath_unit_ok"),
        dict(case_id="audit_judge_absent_both_error", scoring_method="llm_judge",
             answer="the model said something entirely different",
             expected="mg/2", scoring_config=dict(judge_cfg),
             provenance="audit:B7/llm_judge_absent_unreachable"),
    ]

    # ── SCORE-04/05 + SCORE-21: code_execution oracle semantics ──────────
    #    (subprocess rows — excluded from the fast pin subset)
    R += [
        # SCORE-04/05: entry_point + string expected, NO entry_point_cases.
        # Pre scorer synthesised `assert f() == 5` (a zero-arg assertion) and
        # ran it; post scorer refuses (ScoringUnavailableError).
        dict(case_id="audit_score04_entrypoint_noargs", scoring_method="code_execution",
             answer="```python\ndef f():\n    return 5\n```",
             expected="5",
             scoring_config={"entry_point": "f", "timeout": 10},
             provenance="audit:SCORE-04/entrypoint_zero_arg_synth"),
        # SCORE-05: proper entry_point_cases oracle (post-only capability).
        dict(case_id="audit_score05_entrypoint_cases_pass", scoring_method="code_execution",
             answer="```python\ndef add(a, b):\n    return a + b\n```",
             expected="",
             scoring_config={"entry_point": "add",
                             "entry_point_cases": [{"args": [2, 3], "expected": 5}],
                             "timeout": 10},
             provenance="audit:SCORE-05/entrypoint_cases_pass"),
        dict(case_id="audit_score05_entrypoint_cases_fail", scoring_method="code_execution",
             answer="```python\ndef add(a, b):\n    return a + b\n```",
             expected="",
             scoring_config={"entry_point": "add",
                             "entry_point_cases": [{"args": [2, 3], "expected": 6}],
                             "timeout": 10},
             provenance="audit:SCORE-05/entrypoint_cases_fail"),
        # SCORE-21: vacuous `assert True` oracle no longer counts as executable.
        dict(case_id="audit_score21_vacuous_assert", scoring_method="code_execution",
             answer="```python\ndef foo():\n    return 1\n```",
             expected="",
             scoring_config={"test_code": "assert True", "timeout": 10},
             provenance="audit:SCORE-21/vacuous_assert_true"),
        dict(case_id="audit_score21_real_assert_ok", scoring_method="code_execution",
             answer="```python\ndef foo():\n    return 1\n```",
             expected="",
             scoring_config={"test_code": "assert foo() == 1", "timeout": 10},
             provenance="audit:SCORE-21/real_assert_still_runs"),
    ]

    return R


def main() -> None:
    rows = _sentinel_rows() + _audit_rows()
    # stable, unique case_ids
    seen: set[str] = set()
    for r in rows:
        if r["case_id"] in seen:
            raise ValueError(f"duplicate case_id {r['case_id']!r}")
        seen.add(r["case_id"])
    with OUT_PATH.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=True, sort_keys=True) + "\n")
    print(f"wrote {len(rows)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
