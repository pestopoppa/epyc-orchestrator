#!/usr/bin/env python3
"""Deterministic scorer for debug benchmark suite.

Scores model outputs against ground-truth answers using methods from
public benchmarks (exact_match, multiple_choice, code_execution,
programmatic, substring). No heuristics, no Claude-as-Judge needed.

Usage:
    from scripts.benchmark.debug_scorer import score_answer

    result = score_answer(
        answer="The answer is 42",
        expected="42",
        scoring_method="exact_match",
        scoring_config={"extract_pattern": r"#### (\\d+)"},
    )
    print(result)  # True/False
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any


class ScoringUnavailableError(RuntimeError):
    """Raised when a requested scorer cannot run.

    Signals a scorer-infrastructure defect — a missing dependency, an
    unreachable judge, or an unparseable GOLD/expected answer — as distinct
    from the model merely producing a wrong answer. Callers MUST surface this
    as an eval ERROR (an item that could not be scored) and NEVER fold it into
    a ``False`` (wrong-answer) result. Silently swapping in a different scorer
    (e.g. exact_match / substring) on such a failure is exactly how threaded
    ``math_verify`` parses were mis-scored en masse; this exception makes the
    failure loud instead of catastrophic-and-quiet.
    """


def score_answer(
    answer: str,
    expected: Any,
    scoring_method: str,
    scoring_config: dict[str, Any] | None = None,
) -> bool:
    """Score a model answer against expected ground truth.

    Args:
        answer: The model's raw output.
        expected: The expected correct answer.
        scoring_method: One of: exact_match, multiple_choice,
            code_execution, programmatic, substring.
        scoring_config: Method-specific configuration.

    Returns:
        True if the answer is correct, False otherwise.
    """
    if not answer or not answer.strip():
        return False

    # Strip <think>...</think> blocks before scoring (architect models produce these)
    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
    if not answer:
        return False

    expected = "" if expected is None else str(expected)
    config = scoring_config or {}

    scorers = {
        "exact_match": _score_exact_match,
        "multiple_choice": _score_multiple_choice,
        "code_execution": _score_code_execution,
        "programmatic": _score_programmatic,
        "substring": _score_substring,
        "f1": _score_f1,
        "llm_judge": _score_llm_judge,
        "math_verify": _score_math_verify,
    }

    scorer = scorers.get(scoring_method)
    if scorer is None:
        raise ValueError(f"Unknown scoring method: {scoring_method}")

    return scorer(answer, expected, config)


def _score_exact_match(
    answer: str, expected: str, config: dict[str, Any]
) -> bool:
    """Extract answer via regex, compare to expected.

    Used for: GSM8K, MATH — where the answer is a number or expression.

    Config:
        extract_pattern: Regex with one capture group to extract the answer.
            Default: ``<answer>...</answer>`` tag extraction.
            Legacy fallback: ``#### (\\S+)`` (GSM8K standard).
        normalize: If True, strip whitespace and lowercase both sides.
    """
    pattern = config.get("extract_pattern", r"<answer>(.*?)</answer>")
    normalize = config.get("normalize", True)

    # Try to extract via pattern first
    extracted = _extract_answer(answer, pattern)
    if extracted is None:
        # Legacy fallback: try #### pattern for backward compatibility
        extracted = _extract_answer(answer, r"####[ \t]*\n?(\S+)")
    if extracted is None:
        boxed = _extract_boxed_answer(answer)
        if boxed is not None:
            extracted = boxed
    if extracted is None:
        # Last resort: try to find the expected value anywhere in the last line
        last_line = answer.strip().split("\n")[-1]
        extracted = last_line.strip()

    if normalize:
        extracted = extracted.strip().lower().rstrip(".")
        expected_norm = expected.strip().lower().rstrip(".")
    else:
        expected_norm = expected.strip()

    # Numeric comparison for numbers (including word forms like "three" vs "3")
    _NUMBER_WORDS = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
        "nineteen": 19, "twenty": 20,
    }
    def _to_number(s: str) -> float | None:
        try:
            return float(s.replace(",", ""))
        except (ValueError, TypeError):
            return _NUMBER_WORDS.get(s.lower()) if isinstance(s, str) else None

    ext_num = _to_number(extracted)
    exp_num = _to_number(expected_norm)
    if ext_num is not None and exp_num is not None:
        return abs(ext_num - exp_num) < 1e-6

    if extracted == expected_norm:
        return True

    # Fallback: vision models wrap OCR results in prose like
    #   'The text in the image is "iRaeenlc".' or 'The image contains the text: iRaeenlc'
    # Try extracting quoted text or text after colon from the full answer.
    if normalize:
        answer_lower = _final_answer_region(answer).lower()
        # Check quoted: "answer" or 'answer'
        for q in ('"', "'", "\u201c"):
            q_end = "\u201d" if q == "\u201c" else q
            idx = answer_lower.find(q)
            if idx >= 0:
                end = answer_lower.find(q_end, idx + 1)
                if end > idx:
                    candidate = answer_lower[idx + 1:end].strip().rstrip(".")
                    if candidate == expected_norm:
                        return True
        # Check after colon on last meaningful line
        for line in reversed(_final_answer_region(answer).split("\n")):
            if ":" in line:
                candidate = line.split(":", 1)[1].strip().lower().rstrip(".")
                if candidate == expected_norm:
                    return True

    return False


def _score_multiple_choice(
    answer: str, expected: str, config: dict[str, Any]
) -> bool:
    """Parse A/B/C/D or configured choice text from output.

    Used for: ARC-Challenge, MMLU, HellaSwag.

    Config:
        choices: Optional list of choice texts.
    """
    choices = config.get("choices")
    if not isinstance(choices, list):
        choices = []

    expected_letter = _expected_choice_letter(expected, choices)
    expected_index = _expected_choice_index(expected, choices)
    if expected_letter is None and expected_index is None:
        return False

    parsed_letter = _extract_multiple_choice_letter(answer)
    if parsed_letter is not None and expected_letter is not None:
        return parsed_letter == expected_letter

    parsed_index = _extract_multiple_choice_text_index(answer, choices)
    if parsed_index is not None and expected_index is not None:
        return parsed_index == expected_index

    return False


def _expected_choice_letter(expected: str, choices: list[Any]) -> str | None:
    expected_match = re.fullmatch(
        r"\s*[\(\[\{]?\s*([A-H])\s*[\)\]\}]?\s*\.?\s*",
        expected,
        re.IGNORECASE,
    )
    if expected_match:
        return expected_match.group(1).upper()

    idx = _expected_choice_index(expected, choices)
    if idx is not None and idx < 8:
        return chr(ord("A") + idx)
    return None


def _expected_choice_index(expected: str, choices: list[Any]) -> int | None:
    if not choices:
        return None
    expected_norm = _normalize_choice_text(expected)
    for idx, choice in enumerate(choices):
        if _normalize_choice_text(str(choice)) == expected_norm:
            return idx
    return None


def _normalize_choice_text(text: str) -> str:
    text = re.sub(r'<think>.*?</think>', '', str(text), flags=re.DOTALL)
    text = re.sub(r"[*_`~]+", "", text)
    text = text.strip().lower()
    text = text.strip("\"'“”‘’()[]{}")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_multiple_choice_letter(answer: str) -> str | None:
    # Strategy 1: Explicit "Answer: X" — take LAST match (verbose models repeat)
    # Negative lookahead prevents "option is correct" matching as letter "C"
    explicit_pat = r"(?:answer|choice|option)\s*(?:is|:)\s*\(?([A-H])\)?(?![a-zA-Z])"
    explicit_matches = re.findall(explicit_pat, answer, re.IGNORECASE)
    if explicit_matches:
        return explicit_matches[-1].upper()

    # Strategy 2: Letter on its own line near the end of output
    last_line_pat = r"^\s*\(?([A-H])\)?\s*$"
    line_matches = re.findall(last_line_pat, answer, re.MULTILINE)
    if line_matches:
        return line_matches[-1].upper()

    # Strategy 3: Letter at very start of output (before any prose)
    match = re.match(r"\s*\(?([A-H])\)?\s*[.:\-\n]", answer)
    if match:
        return match.group(1).upper()

    # Strategy 4: Bold letter — take LAST match
    bold_matches = re.findall(r"\*\*([A-H])\*\*", answer)
    if bold_matches:
        return bold_matches[-1].upper()

    # Strategy 5: Last standalone letter A-H in the text (not first!)
    standalone = re.findall(r"\b([A-H])\b", answer)
    if standalone:
        return standalone[-1].upper()

    return None


def _extract_multiple_choice_text_index(answer: str, choices: list[Any]) -> int | None:
    if not choices:
        return None

    answer_norm = _normalize_choice_text(answer)
    if not answer_norm:
        return None

    matches: list[tuple[int, int, int]] = []
    for idx, choice in enumerate(choices):
        choice_norm = _normalize_choice_text(str(choice))
        if not choice_norm:
            continue
        pattern = rf"(?<!\w){re.escape(choice_norm)}(?!\w)"
        found = list(re.finditer(pattern, answer_norm))
        if found:
            match = found[-1]
            matches.append((match.end(), len(choice_norm), idx))

    if not matches:
        return None
    return max(matches)[2]


def _score_stdin_program(
    code: str, test_code: str, preamble: str, timeout: int
) -> bool:
    """Run a stdin/stdout program against TEST_CASES.

    For competitive programming (USACO, etc.) where solutions read from stdin
    and write to stdout.  Each test case is (input_str, expected_output_str).
    The program passes if ALL test cases produce the expected output.

    Strategy: write solution to a temp file, then run it once per test case
    with stdin piped in.  Compare stdout to expected output.
    """
    # Parse TEST_CASES from the test_code string
    try:
        ns: dict = {}
        exec(test_code, ns)
        cases = ns.get("TEST_CASES", [])
    except Exception:
        return False

    if not cases:
        return False

    full_code = preamble + code

    try:
        sol_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False,
            dir="/mnt/raid0/llm/tmp",
        )
        sol_file.write(full_code)
        sol_file.flush()
        sol_file.close()

        for inp, expected_out in cases:
            try:
                result = subprocess.run(
                    ["python3", sol_file.name],
                    input=inp,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd="/mnt/raid0/llm/tmp",
                )
            except subprocess.TimeoutExpired:
                Path(sol_file.name).unlink(missing_ok=True)
                return False

            if result.returncode != 0:
                Path(sol_file.name).unlink(missing_ok=True)
                return False

            got = result.stdout.strip()
            want = expected_out.strip()
            if got != want:
                Path(sol_file.name).unlink(missing_ok=True)
                return False

        Path(sol_file.name).unlink(missing_ok=True)
        return True
    except OSError as exc:
        raise ScoringUnavailableError(
            "code_execution could not create or execute its temporary stdin harness"
        ) from exc


def _has_executable_assertion(test_code: str) -> bool:
    for line in test_code.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("assert ") or stripped.startswith("assert("):
            expr = (
                stripped[6:].strip()
                if stripped.startswith("assert ")
                else stripped[7:].strip()
            )
            expr = expr.split(",", 1)[0].strip().strip("()")
            if expr == "True":
                continue
            return True
    return False


def _has_unittest_case(test_code: str) -> bool:
    return "unittest.TestCase" in test_code or "(TestCase)" in test_code


def _score_code_execution(
    answer: str, expected: str, config: dict[str, Any]
) -> bool:
    """Extract code from model output, run against test cases.

    Used for: HumanEval, MBPP.

    Config:
        test_code: Test code to append after the model's function.
        language: Programming language (default: "python").
        timeout: Execution timeout in seconds (default: 10).
        entry_point: Function name to test (for HumanEval).
    """
    language = config.get("language", "python")
    timeout = config.get("timeout", 10)
    test_code = config.get("test_code", "")
    entry_point = config.get("entry_point", "")
    entry_point_cases = config.get("entry_point_cases")

    if language != "python":
        # Only Python execution supported currently
        return False

    # Extract code block from model output
    code = _extract_code_block(answer, language)
    if not code:
        return False

    # Prepend common imports so extracted code with type annotations
    # (e.g. List[int], Optional[str]) doesn't crash on NameError.
    _TYPING_PREAMBLE = (
        "from typing import List, Optional, Tuple, Dict, Set, Any\n"
        "from collections import defaultdict, deque, Counter\n"
        "import math, heapq, bisect, itertools, functools\n\n"
    )

    # Detect stdin-based competitive programming solutions (USACO etc.)
    # These use input() to read from stdin, so we must feed test cases via stdin.
    _uses_stdin = "input()" in code or "sys.stdin" in code
    _has_test_cases = test_code.strip().startswith("TEST_CASES")

    if _has_test_cases:
        if not _uses_stdin:
            return False
        return _score_stdin_program(code, test_code, _TYPING_PREAMBLE, timeout)

    has_test_oracle = (
        _has_executable_assertion(test_code) or _has_unittest_case(test_code)
    )
    has_entrypoint_oracle = bool(
        entry_point and isinstance(entry_point_cases, list) and entry_point_cases
    )
    if test_code and not has_test_oracle:
        return False
    if not test_code and not has_entrypoint_oracle:
        if entry_point and expected:
            raise ScoringUnavailableError(
                "code_execution entry_point oracle requires executable "
                "entry_point_cases or test_code; refusing to synthesize a "
                "zero-argument assertion from expected text"
            )
        return False

    # Build full test script
    full_code = _TYPING_PREAMBLE + code
    if test_code:
        full_code += "\n\n" + test_code
        if _has_unittest_case(test_code) and "unittest.main" not in test_code:
            full_code += "\n\nif __name__ == '__main__':\n    unittest.main()\n"
    elif entry_point:
        if not _is_safe_entry_point(entry_point):
            raise ScoringUnavailableError(
                f"code_execution entry_point {entry_point!r} is not a safe "
                "Python identifier path"
            )
        cases_literal = repr(entry_point_cases)
        full_code += (
            "\n\n"
            f"_EPYC_ENTRY_POINT_CASES = {cases_literal}\n"
            "for _case in _EPYC_ENTRY_POINT_CASES:\n"
            "    if isinstance(_case, dict):\n"
            "        _args = _case.get('args', [])\n"
            "        _kwargs = _case.get('kwargs', {})\n"
            "        _expected = _case.get('expected')\n"
            "    else:\n"
            "        _args, _expected = _case\n"
            "        _kwargs = {}\n"
            f"    assert {entry_point}(*_args, **_kwargs) == _expected\n"
        )

    # Execute in sandboxed subprocess
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False,
            dir="/mnt/raid0/llm/tmp",
        ) as f:
            f.write(full_code)
            f.flush()
            result = subprocess.run(
                ["python3", f.name],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd="/mnt/raid0/llm/tmp",
            )
            Path(f.name).unlink(missing_ok=True)
            return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except OSError as exc:
        raise ScoringUnavailableError(
            "code_execution could not create or execute its temporary harness"
        ) from exc


def _score_programmatic(
    answer: str, expected: str, config: dict[str, Any]
) -> bool:
    """Run IFEval-style programmatic verifiers.

    Used for: IFEval — checks format constraints.

    Config:
        verifier: Name of verifier to run. Options:

            YAML prompt verifiers:
            - word_count_min/max/range: word count checks (threshold, min_val, max_val)
            - contains_keyword / no_keyword: keyword presence (keyword)
            - starts_with / ends_with: text prefix/suffix (text)
            - json_valid / all_uppercase / all_lowercase: format checks
            - bullet_list / numbered_list: list format checks
            - paragraph_count / sentence_count_min: structure checks (threshold)
            - comma_separated / title_case: format checks

            IFEval adapter verifiers (from dataset_adapters.py):
            - no_comma: answer contains no commas
            - has_title: first line is short + title-cased
            - placeholder_count: count of [placeholder] patterns (count)
            - bullet_count: minimum bullet points (count)
            - contains_keywords: all keywords present (keywords list)
            - no_forbidden_words: none of forbidden words present (forbidden list)
            - language: language check (always passes — no langdetect)
            - non_empty: answer is non-empty
            - highlighted_sections: contains **bold** or ## headings
            - word_count: word count with relation (count, relation)
            - sentence_count: sentence count with relation (count, relation)

        threshold: Numeric threshold for count-based verifiers.
        count: Alias for threshold (used by IFEval adapter).
        relation: "at_least" | "at_most" | "exactly" (IFEval word/sentence count).
        keyword: Keyword for contains/no_keyword verifiers.
        keywords: Keyword list for contains_keywords verifier.
        forbidden: Forbidden word list for no_forbidden_words verifier.
        text: Text for starts_with/ends_with verifiers.
        min_val / max_val: Range for range-based verifiers.
    """
    verifier = config.get("verifier", "")
    threshold = config.get("threshold", 0)
    keyword = config.get("keyword", "")
    keywords = config.get("keywords", [])
    forbidden = config.get("forbidden", [])
    text = config.get("text", "")
    min_val = config.get("min_val", 0)
    max_val = config.get("max_val", 0)
    # IFEval adapter uses "count" and "relation" instead of threshold/min_val/max_val
    count = config.get("count") or threshold or 0
    relation = config.get("relation", "at_least")

    answer_stripped = answer.strip()
    words = answer_stripped.split()
    wc = len(words)
    lines = answer_stripped.split("\n")

    def _word_count_by_relation() -> bool:
        """Handle word_count/sentence_count with 'relation' from IFEval adapter."""
        if relation == "at_least":
            return wc >= count
        elif relation == "at_most":
            return wc <= count
        elif relation == "exactly":
            return wc == count
        return wc >= count  # default: at_least

    def _sentence_count_by_relation() -> bool:
        sc = len(re.findall(r"[.!?]+", answer_stripped))
        if relation == "at_least":
            return sc >= count
        elif relation == "at_most":
            return sc <= count
        elif relation == "exactly":
            return sc == count
        return sc >= count

    verifiers = {
        # Original verifiers (YAML prompts use these names)
        "word_count_min": lambda: wc >= (count or threshold),
        "word_count_max": lambda: wc <= (count or threshold),
        "word_count_range": lambda: min_val <= wc <= max_val,
        "contains_keyword": lambda: keyword.lower() in answer_stripped.lower(),
        "no_keyword": lambda: keyword.lower() not in answer_stripped.lower(),
        "starts_with": lambda: answer_stripped.lower().startswith(text.lower()),
        "ends_with": lambda: answer_stripped.rstrip(".!?").lower().endswith(text.lower()),
        "json_valid": lambda: _is_valid_json(answer_stripped),
        "all_uppercase": lambda: answer_stripped == answer_stripped.upper(),
        "all_lowercase": lambda: answer_stripped == answer_stripped.lower(),
        "bullet_list": lambda: any(
            line.strip().startswith(("- ", "* ", "• "))
            for line in lines if line.strip()
        ),
        "numbered_list": lambda: any(
            re.match(r"^\d+[\.\)]\s", line.strip())
            for line in lines if line.strip()
        ),
        "paragraph_count": lambda: len([
            p for p in re.split(r"\n\s*\n", answer_stripped) if p.strip()
        ]) == (count or threshold),
        "sentence_count_min": lambda: len(re.findall(r"[.!?]+", answer_stripped)) >= (count or threshold),
        "comma_separated": lambda: "," in answer_stripped and "\n" not in answer_stripped.strip(),
        # IFEval adapter verifiers (dataset_adapters.py emits these names)
        "no_comma": lambda: "," not in answer_stripped,
        "has_title": lambda: bool(
            lines[0].strip() and len(lines[0].strip().split()) <= 10
            and lines[0].strip().istitle()
        ) if lines else False,
        "placeholder_count": lambda: len(re.findall(r'\[.*?\]', answer_stripped)) >= (count or 1),
        "bullet_count": lambda: sum(
            1 for line in lines
            if line.strip().startswith(("- ", "* ", "• "))
        ) >= (count or 1),
        "contains_keywords": lambda: all(
            kw.lower() in answer_stripped.lower() for kw in keywords
        ) if keywords else True,
        "no_forbidden_words": lambda: not any(
            fw.lower() in answer_stripped.lower() for fw in forbidden
        ) if forbidden else True,
        "language": lambda: True,  # Cannot verify without langdetect; pass through
        "non_empty": lambda: len(answer_stripped) > 0,
        "highlighted_sections": lambda: bool(
            re.search(r'\*\*[^*]+\*\*', answer_stripped)
            or re.search(r'^##\s+', answer_stripped, re.MULTILINE)
        ),
        # IFEval relation-based verifiers (word_count with at_least/at_most/exactly)
        "word_count": _word_count_by_relation,
        "sentence_count": _sentence_count_by_relation,
        "title_case": lambda: all(
            w[0].isupper() for w in words if w and w[0].isalpha()
        ) if words else False,
    }

    fn = verifiers.get(verifier)
    if fn is None:
        # Mirror score_answer's "Unknown scoring method" convention: an
        # unrecognized verifier is a config defect, not a silent substring
        # match that would score arbitrary answers as correct.
        raise ValueError(f"Unknown programmatic verifier: {verifier!r}")

    return fn()


def _score_substring(
    answer: str, expected: str, config: dict[str, Any]
) -> bool:
    """Check if expected text appears in output.

    Used for: Needle-in-haystack, simple factoid QA.

    Config:
        case_sensitive: Whether comparison is case-sensitive (default: False).

    Digit-group separators (commas/underscores/spaces sitting *between two
    digits*) are stripped from both sides before matching, so a correctly
    computed numeric answer formatted as "479,001,600" still matches the
    expected substring "479001600". Non-numeric text is untouched because the
    separator must be flanked by digits on both sides (e.g. "Hello, world" is
    left as-is). 2026-06-02: fixes the agentic factorial sentinel, which began
    failing on 06-01 once the compute-first prompt made the model emit
    comma-grouped results.
    """
    case_sensitive = config.get("case_sensitive", False)

    def _strip_digit_separators(s: str) -> str:
        return re.sub(r"(?<=\d)[,_ ](?=\d)", "", s)

    answer = _strip_digit_separators(answer)
    expected = _strip_digit_separators(expected)

    needle = expected.strip()
    if not needle:
        return False
    return _contains_text_unit(answer, needle, case_sensitive=case_sensitive)


def _score_f1(
    answer: str, expected: str, config: dict[str, Any]
) -> bool:
    """Token-level F1 scoring for QA tasks.

    Used for: HotpotQA, SQuAD-style reading comprehension.

    Computes precision/recall/F1 at the token level after normalization.
    A prediction is considered correct if F1 >= threshold.

    Config:
        extract_pattern: Regex to extract answer (default: <answer> tag).
        threshold: Minimum F1 to count as correct (default: 0.5).
        normalize: Whether to normalize text (default: True).
    """
    pattern = config.get("extract_pattern", r"<answer>(.*?)</answer>")
    threshold = config.get("threshold", 0.5)
    normalize = config.get("normalize", True)

    # Extract answer: find the LAST occurrence — models may emit
    # the tag multiple times before settling on a final answer.
    compiled_pattern = _compile_single_group_pattern(pattern)
    matches = compiled_pattern.findall(answer)
    if matches:
        extracted = matches[-1].strip()
    else:
        # Legacy fallback: try #### pattern for backward compatibility
        legacy_matches = re.findall(r"####[ \t]*\n?(.+)", answer, re.IGNORECASE)
        if legacy_matches:
            extracted = legacy_matches[-1].strip()
        else:
            extracted = _extract_answer(answer, pattern)
    if extracted is None:
        # Fallback: use last non-empty line
        lines = [ln.strip() for ln in answer.strip().split("\n") if ln.strip()]
        extracted = lines[-1] if lines else ""

    if normalize:
        extracted = _normalize_text(extracted)
        expected = _normalize_text(expected)

    # Tokenize
    pred_tokens = extracted.split()
    gold_tokens = expected.split()

    if not gold_tokens:
        return len(pred_tokens) == 0

    if not pred_tokens:
        return False

    # Compute multiset token overlap so repeated entities are counted honestly.
    from collections import Counter

    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    common = sum((pred_counts & gold_counts).values())

    if not common:
        return False

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return f1 >= threshold


def _normalize_text(text: str) -> str:
    """Normalize text for F1 scoring (SQuAD-style)."""
    import string
    import unicodedata

    # Fold diacritics before punctuation stripping so answer variants like
    # "Dusan Lajovic" and "Dušan Lajović" score as the same tokens.
    text = "".join(
        ch for ch in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(ch)
    )

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)

    # Collapse whitespace
    text = " ".join(text.split())

    return text


def _score_llm_judge(
    answer: str, expected: str, config: dict[str, Any]
) -> bool:
    """Score using a local LLM as semantic equivalence judge.

    Used for: PhysReason — symbolic physics/math answers where substring
    matching misses equivalent forms (e.g. mg/2 vs \\frac{mg}{2}).

    Calls an OpenAI-compatible endpoint to judge whether the model's answer is
    semantically equivalent to the expected answer.

    Endpoint resolution (realized-first, mirrors commits 5aa29f35/e97d4ed9):
    a hardcoded llama-server port (the old default 8082 = a worker_general
    *quarter* port) is dead on a quarters-only / eval-batch stack, so the judge
    now defaults to the ORCHESTRATOR API, which resolves the role to a LIVE
    backend itself. No new hardcoded llama port is introduced. An explicit
    ``judge_host``+``judge_port`` in ``scoring_config`` still wins (targeted
    override); otherwise the endpoint comes from ``ORCHESTRATOR_API_URL``
    (default ``http://localhost:8000``). A down/malformed judge remains an
    honest ScoringUnavailableError — we never launch anything and never
    silently fall back to substring.

    Config:
        judge_host + judge_port: Explicit judge server (both required to
            override; targets ``http://{host}:{port}/v1/chat/completions``).
        judge_url: Explicit full base URL override (wins over host/port).
        timeout: HTTP timeout in seconds (default: 30).
    """
    timeout = config.get("timeout", 30)
    judge_url = _resolve_llm_judge_base_url(config)

    # First try a boundary-aware substring fast path; contained words such as
    # "cat" in "concatenate" must still go to the judge.
    if _contains_text_unit(answer, expected.strip()):
        return True

    # Extract answer from \boxed{} if present
    boxed = re.search(r'\\boxed\{(.+?)\}', answer, re.DOTALL)
    candidate = boxed.group(1).strip() if boxed else answer.strip().split("\n")[-1].strip()

    judge_prompt = (
        "You are a physics answer equivalence judge. Determine whether two "
        "mathematical/physics answers are semantically equivalent.\n\n"
        "Consider:\n"
        "- Different but equivalent LaTeX forms (e.g. \\frac{mg}{2} vs mg/2)\n"
        "- Equivalent symbolic rearrangements\n"
        "- Same numerical value with different units notation\n"
        "- Simplified vs expanded forms\n\n"
        f"Expected answer: {expected}\n\n"
        f"Student answer: {candidate}\n\n"
        "Are these answers semantically equivalent? Reply with ONLY "
        "\"true\" or \"false\", nothing else."
    )

    import httpx

    try:
        resp = httpx.post(
            f"{judge_url}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": judge_prompt}],
                "max_tokens": 8,
                "temperature": 0.0,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        verdict = resp.json()["choices"][0]["message"]["content"].strip().lower()
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
        # HTTP/transport failure (httpx.*), non-2xx (HTTPStatusError from
        # raise_for_status), bad JSON (json.JSONDecodeError <: ValueError), or
        # an unexpected response shape (KeyError/IndexError/TypeError). A judge
        # that is down or malformed is scorer-unavailability — surface it as an
        # ERROR; do NOT silently fall back to substring (audit item B7 owns the
        # substring fast-path extraction semantics).
        raise ScoringUnavailableError(
            f"llm_judge unreachable or returned a malformed response at "
            f"{judge_url}; refusing to silently fall back to substring"
        ) from exc

    return verdict.startswith("true")


def _resolve_llm_judge_base_url(config: dict[str, Any]) -> str:
    """Resolve the llm_judge base URL (scheme://host:port), realized-first.

    Precedence: explicit ``judge_url`` > explicit ``judge_host``+``judge_port``
    > ``ORCHESTRATOR_API_URL`` env > ``http://localhost:8000``. Routing through
    the orchestrator API (rather than a hardcoded llama-server port) lets the
    orchestrator resolve the judge role to a LIVE backend on a quarters-only
    fleet — no new hardcoded ports, per commits 5aa29f35/e97d4ed9.
    """
    import os

    explicit_url = str(config.get("judge_url") or "").strip()
    if explicit_url:
        return explicit_url.rstrip("/")
    # Both host AND port must be supplied to take the legacy direct-port path;
    # a lone hardcoded port default is exactly the dead-endpoint trap we fix.
    host = config.get("judge_host")
    port = config.get("judge_port")
    if host and port:
        return f"http://{host}:{port}".rstrip("/")
    return os.environ.get("ORCHESTRATOR_API_URL", "http://localhost:8000").rstrip("/")


def _score_math_verify(
    answer: str, expected: str, config: dict[str, Any]
) -> bool:
    """Score using Math-Verify library for symbolic mathematical comparison.

    Used for: MATH-500 — where equivalent expressions should match
    (e.g. \\frac{mg}{2} ≡ mg/2, x^2+1 ≡ 1+x^2, {1,2,3} ≡ {3,1,2}).

    Requires: pip install math-verify (Apache-2.0, HuggingFace).

    Error semantics — NO silent scorer fallback:
        - math-verify not installed → ScoringUnavailableError. We refuse to
          quietly score math with exact_match; that silent swap is what
          mis-scored every threaded math eval.
        - The GOLD (``expected``) answer raising on parse, or parsing to an
          empty extraction, is a dataset/gold defect → ScoringUnavailableError.
        - The MODEL's answer failing to parse is a *task* failure → False.
        - ``verify`` itself raising is a scorer defect → ScoringUnavailableError.

    Thread safety:
        math-verify guards BOTH ``parse`` and ``verify`` with ``signal.alarm``,
        which raises ``ValueError("... signal only works in main thread ...")``
        on any non-main thread. The library's documented remedy is to disable
        those timeouts (``parse(..., parsing_timeout=None)`` and
        ``verify(..., timeout_seconds=None)``); we pass both whenever we are
        off the main thread — the eval-level watchdog still bounds pathological
        wall time — so threaded scoring runs the real math_verify path instead
        of silently degrading (or, if only parse were fixed, ERRORing on every
        threaded verify() call).

    Config:
        extraction_mode: "latex" (default), "expr", or "string"
    """
    try:
        from math_verify import parse, verify
    except ImportError as exc:
        raise ScoringUnavailableError(
            "math-verify not installed but scoring_method=math_verify "
            "requested; refusing to silently fall back to exact_match"
        ) from exc

    parse_kwargs: dict[str, Any] = {}
    verify_kwargs: dict[str, Any] = {}
    if threading.current_thread() is not threading.main_thread():
        # math-verify's own documented remedy for threaded use; the
        # eval-level watchdog bounds pathological wall time.
        parse_kwargs["parsing_timeout"] = None
        verify_kwargs["timeout_seconds"] = None

    try:
        gold = parse(expected, **parse_kwargs)
    except Exception as exc:
        raise ScoringUnavailableError(
            "math_verify could not parse the GOLD/expected answer "
            f"{expected!r} (dataset/gold defect)"
        ) from exc
    if not gold:
        raise ScoringUnavailableError(
            "math_verify extracted nothing from the GOLD/expected answer "
            f"{expected!r} (dataset/gold defect)"
        )

    try:
        pred = parse(answer.strip(), **parse_kwargs)
    except Exception:
        # The model's own answer failing to parse is a task failure, not a
        # scorer-unavailability condition — score it wrong, don't raise.
        return False

    try:
        # gold-first argument order — verify() is asymmetric.
        return bool(verify(gold, pred, **verify_kwargs))
    except Exception as exc:
        raise ScoringUnavailableError(
            "math_verify.verify() raised while comparing parsed answers "
            "(scorer defect)"
        ) from exc


# ── Helpers ────────────────────────────────────────────────────────────


def _extract_answer(text: str, pattern: str) -> str | None:
    """Extract answer from text using regex pattern."""
    compiled = _compile_single_group_pattern(pattern)
    match = compiled.search(text)
    if match and match.group(1):
        return match.group(1).strip()
    return None


def _compile_single_group_pattern(pattern: str) -> re.Pattern[str]:
    compiled = re.compile(pattern, re.IGNORECASE | re.DOTALL)
    if compiled.groups != 1:
        raise ValueError(
            f"extract_pattern must contain exactly one capture group, got {compiled.groups}"
        )
    return compiled


def _extract_boxed_answer(text: str) -> str | None:
    """Extract the final LaTeX \\boxed{...} payload, including nested braces."""
    last_start = text.rfind(r"\boxed{")
    if last_start < 0:
        return None
    i = last_start + len(r"\boxed{")
    depth = 1
    out: list[str] = []
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out).strip()
            out.append(ch)
        else:
            out.append(ch)
        i += 1
    return None


def _final_answer_region(text: str) -> str:
    """Return the final answer-bearing line/region, not earlier explanation."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return ""
    marker = re.compile(r"\b(final\s+answer|answer|result)\b", re.IGNORECASE)
    for line in reversed(lines):
        if marker.search(line):
            return line
    return lines[-1]


def _contains_text_unit(
    haystack: str,
    needle: str,
    *,
    case_sensitive: bool = False,
) -> bool:
    flags = 0 if case_sensitive else re.IGNORECASE
    needle = needle.strip()
    if not needle:
        return False
    left = r"(?<!\w)" if needle[0].isalnum() else ""
    right = r"(?!\w)" if needle[-1].isalnum() else ""
    return re.search(f"{left}{re.escape(needle)}{right}", haystack, flags) is not None


def _is_safe_entry_point(entry_point: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", str(entry_point)))


def _extract_code_block(text: str, language: str = "python") -> str | None:
    """Extract code from markdown code block or raw code."""
    # Try markdown code block first
    patterns = [
        rf"```{language}\s*\n(.*?)```",
        r"```\w*\s*\n(.*?)```",
    ]
    for pat in patterns:
        match = re.search(pat, text, re.DOTALL)
        if match:
            return match.group(1).strip()

    # Try to find a def/class statement (Python-specific)
    if language == "python":
        match = re.search(r"((?:def|class)\s+\w+.*?)(?:\n\n|\Z)", text, re.DOTALL)
        if match:
            return match.group(1).strip()

    # Last resort: if text looks like executable Python code, return it
    # Covers USACO-style stdin solutions (n = int(input()), sys.stdin, etc.)
    stripped = text.strip()
    if stripped and any(
        stripped.startswith(prefix)
        for prefix in ("def ", "class ", "import ", "from ", "n ", "t ", "for ", "while ", "if ", "#")
    ):
        return stripped

    # Also accept if it contains input() — likely a competitive programming solution
    if "input()" in stripped or "sys.stdin" in stripped:
        return stripped

    return None


def _is_valid_json(text: str) -> bool:
    """Check if text contains valid JSON."""
    # Try the whole text
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to find JSON in the text
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start >= 0 and end > start:
            try:
                json.loads(text[start : end + 1])
                return True
            except (json.JSONDecodeError, ValueError):
                pass

    return False


def score_batch(
    questions: list[dict[str, Any]],
    answers: list[str],
) -> list[dict[str, Any]]:
    """Score a batch of answers against their questions.

    Args:
        questions: List of question dicts with id, expected, scoring_method,
            scoring_config.
        answers: List of model answers (same order as questions).

    Returns:
        List of result dicts with id, passed, expected, actual_answer.
    """
    results = []
    for q, ans in zip(questions, answers):
        passed = score_answer(
            answer=ans,
            expected=q.get("expected", ""),
            scoring_method=q.get("scoring_method", "exact_match"),
            scoring_config=q.get("scoring_config"),
        )
        results.append({
            "id": q.get("id", "unknown"),
            "suite": q.get("suite", "unknown"),
            "passed": passed,
            "expected": q.get("expected", ""),
            "answer_preview": ans[:200] if ans else "",
        })
    return results
