"""Pin the B7 ratified scorer semantics against the golden delta corpus.

The B7 golden-delta report
(``orchestration/reports/b7_scorer_golden_delta_20260721/``) recorded, for
every corpus case, the outcome the POST-package scorer produced (the ``new``
column of ``results.jsonl``). That column is the *ratified* semantics of the
07a20a7c + 8f24679a scorer package.

This test re-runs the CURRENT ``scripts/benchmark/debug_scorer.py`` over the
same corpus and asserts it still produces exactly those recorded outcomes. If a
future scorer edit changes any outcome, this test fails — forcing whoever makes
the change to regenerate and re-review the golden file *deliberately* (via
``run_delta.py``), rather than silently drifting the eval instrument.

The fast subset (``pin_fast`` — pure in-process scoring, no subprocess/network)
is the hard, always-run pin. A second test covers the subprocess/network rows
(code_execution, llm_judge) and is skipped where their host infra is absent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "orchestration" / "reports" / "b7_scorer_golden_delta_20260721"
CORPUS_PATH = REPORT_DIR / "golden_corpus.jsonl"
RESULTS_PATH = REPORT_DIR / "results.jsonl"
CODE_EXEC_TMP = Path("/mnt/raid0/llm/tmp")

sys.path.insert(0, str(REPO_ROOT / "scripts" / "benchmark"))

from debug_scorer import score_answer  # noqa: E402


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _outcome(row: dict) -> Any:
    """Mirror run_delta.py: bool outcome, or 'ERROR:<ExcType>'."""
    try:
        return bool(
            score_answer(
                answer=row["answer"],
                expected=row["expected"],
                scoring_method=row["scoring_method"],
                scoring_config=row.get("scoring_config"),
            )
        )
    except Exception as exc:  # noqa: BLE001 — classify by exception type name
        return f"ERROR:{type(exc).__name__}"


def _recorded_new(value: Any) -> Any:
    # results.jsonl stores booleans as JSON true/false and errors as strings.
    return value


CORPUS = {r["case_id"]: r for r in _load_jsonl(CORPUS_PATH)}
RESULTS = _load_jsonl(RESULTS_PATH)
FAST_RESULTS = [r for r in RESULTS if r["pin_fast"]]
SLOW_RESULTS = [r for r in RESULTS if not r["pin_fast"]]


def test_corpus_and_results_are_aligned() -> None:
    corpus_ids = set(CORPUS)
    result_ids = {r["case_id"] for r in RESULTS}
    assert corpus_ids == result_ids
    assert len(RESULTS) == len(CORPUS)
    # internal consistency: recorded `changed` == (old != new)
    for r in RESULTS:
        assert r["changed"] == (r["old"] != r["new"]), r["case_id"]


def test_headline_delta_counts_are_pinned() -> None:
    # Pins the operator-facing numbers so an accidental corpus/scorer drift is
    # caught even before per-row diffing.
    assert len(RESULTS) == 146
    assert sum(1 for r in RESULTS if r["changed"]) == 50
    sentinel_changed = {
        r["provenance"].split(":", 1)[1].split("/", 1)[0]
        for r in RESULTS
        if r["changed"] and r["provenance"].startswith("sentinel:")
    }
    assert len(sentinel_changed) == 21


@pytest.mark.parametrize("res", FAST_RESULTS, ids=[r["case_id"] for r in FAST_RESULTS])
def test_fast_subset_reproduces_ratified_new(res: dict) -> None:
    """Hard pin: current scorer must reproduce every ratified POST outcome
    (pure in-process scoring only — runs in well under 5s)."""
    row = CORPUS[res["case_id"]]
    assert _outcome(row) == _recorded_new(res["new"])


@pytest.mark.skipif(
    not CODE_EXEC_TMP.is_dir(),
    reason="code_execution rows require the host sandbox dir /mnt/raid0/llm/tmp",
)
@pytest.mark.parametrize("res", SLOW_RESULTS, ids=[r["case_id"] for r in SLOW_RESULTS])
def test_subprocess_and_network_rows_reproduce_ratified_new(res: dict) -> None:
    """code_execution (subprocess) + llm_judge (unreachable-judge) rows. Still
    fast here (~a few subprocess spawns / refused TCP connects), but gated on
    host infra and split out from the always-run fast pin."""
    row = CORPUS[res["case_id"]]
    assert _outcome(row) == _recorded_new(res["new"])
