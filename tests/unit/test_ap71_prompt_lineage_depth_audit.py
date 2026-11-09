"""AP-71: hand-checked fixtures for the lineage-depth side-effect audit."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "autopilot"))

import prompt_lineage_depth_audit as pa  # noqa: E402

LEAK = "the quick brown fox jumps over the lazy dog today"  # 9 tokens, contains 8-grams
CORPUS = ["What is " + LEAK + "?", "unrelated question about tides"]


def _row(tid, file="frontdoor.md", q=1.0, decision="keep", species="prompt_forge"):
    return {
        "trial_id": tid, "species": species, "keep_revert_decision": decision,
        "quality": q, "git_tag": f"autopilot/trial-{tid}",
        "config_snapshot": {"file": file},
    }


def _suite(text):
    return {"gpqa"} if "gpqa" in text.lower() else set()


def test_added_text_only_new_lines():
    assert pa.added_text("a\nb\n", "a\nb\nc\n  d  \n") == "c\nd"


def test_overlap_requires_verbatim_8gram():
    corpus = pa.ngram_set(CORPUS)
    assert pa.overlap_line_count("tip: the quick brown fox jumps over the lazy dog", corpus) == 1
    # 7 shared tokens only: no hit
    assert pa.overlap_line_count("the quick brown fox jumps over the", corpus) == 0
    assert pa.overlap_line_count("quick brown fox jumps over THE lazy dog today", corpus) == 1


def test_spearman_known_values_and_undefined():
    assert pa.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert pa.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert pa.spearman([1, 2, 3, 4], [5, 5, 5, 5]) is None
    assert pa.spearman([1, 2], [1, 2]) is None


def test_kept_rows_filter_and_order():
    rows = [_row(5), _row(2, decision="revert"), _row(3, species="other"), _row(1)]
    assert [r["trial_id"] for r in pa.kept_prompt_rows(rows)] == [1, 5]


def test_icrh_signature_detected_on_rising_quality_and_leak():
    texts = {1: "base line", 2: "base line\nl2", 3: "base line\nl2\nl3 " + LEAK,
             4: "base line\nl2\nl3 " + LEAK + "\nl4 gpqa " + LEAK + " more",
             5: "base line\nl2\nl3 " + LEAK + "\nl4 gpqa " + LEAK + " more\nl5 gpqa " + LEAK}
    rows = [_row(i, q=float(i)) for i in texts]
    rep = pa.audit(rows, lambda r: texts[r["trial_id"]], CORPUS, _suite)
    f = rep["files"]["frontdoor.md"]
    table = f["rows"]
    assert [t["depth"] for t in table] == [1, 2, 3, 4, 5]
    assert table[0]["added_ngram_overlap"] is None  # no previous keep: not guessed
    assert [t["added_ngram_overlap"] for t in table[1:]] == [0, 1, 1, 1]
    assert [t["added_suite_mentions"] for t in table[1:]] == [0, 0, 1, 1]
    assert f["rho_vs_depth"]["quality"] == 1.0
    assert f["icrh_signature"] is True and f["verdict"] == "icrh_signature"


def test_no_signature_when_quality_flat_and_short_chain_is_insufficient():
    texts = {i: "x\n" + "\n".join(f"l{j}" for j in range(i)) for i in range(1, 6)}
    rows = [_row(i, q=1.0) for i in texts]
    rep = pa.audit(rows, lambda r: texts[r["trial_id"]], CORPUS, _suite)
    assert rep["files"]["frontdoor.md"]["verdict"] == "no_signature"
    short = pa.audit(rows[:3], lambda r: texts[r["trial_id"]], CORPUS, _suite)
    assert short["files"]["frontdoor.md"]["verdict"] == "insufficient_depth"


def test_unrecoverable_text_breaks_chain_not_guessed():
    texts = {1: "a", 2: None, 3: "a\nb", 4: "a\nb\nc"}
    rows = [_row(i) for i in texts]
    rep = pa.audit(rows, lambda r: texts[r["trial_id"]], CORPUS, _suite)
    assert rep["unrecovered"] == [{"trial_id": 2, "file": "frontdoor.md"}]
    t = rep["files"]["frontdoor.md"]["rows"]
    assert [x["trial_id"] for x in t] == [1, 3, 4]
    assert t[1]["added_ngram_overlap"] is None  # after the gap: no valid diff base
    assert t[2]["added_ngram_overlap"] == 0


def test_files_are_separate_lineages():
    rows = [_row(1, "a.md"), _row(2, "b.md"), _row(3, "a.md")]
    rep = pa.audit(rows, lambda r: f"t{r['trial_id']}", CORPUS, _suite)
    assert rep["files"]["a.md"]["keeps"] == 2 and rep["files"]["b.md"]["keeps"] == 1
