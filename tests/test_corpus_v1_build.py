#!/usr/bin/env python3
"""Schema + row-validation tests for the near-miss decision corpus v1 (H4 RC-3).

Hermetic: exercises the row schema, the rule-based mutation primitives, the
qid<->question-pool join, bug-report reconstruction, and the seeded cap over
small in-memory fixtures. Touches no large dataset and performs NO inference.

Run just this file:
    .venv/bin/python -m pytest tests/test_corpus_v1_build.py -q
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))

from corpus_v1 import common  # noqa: E402
from corpus_v1 import seed_mutations as sm  # noqa: E402
from corpus_v1 import mine_bugreports as mb  # noqa: E402
from corpus_v1 import assemble as asm  # noqa: E402


# --------------------------------------------------------------------------- #
# stable_qid mirrors eval_tower._stable_question_qid
# --------------------------------------------------------------------------- #
def test_stable_qid_matches_eval_tower_formula():
    suite, prompt = "general", "What is the capital of Australia?"
    expect = hashlib.sha1(f"{suite}\x00{prompt}".encode("utf-8")).hexdigest()[:16]
    assert common.stable_qid(suite, prompt) == expect
    assert len(common.stable_qid(suite, prompt)) == 16
    # stable across calls
    assert common.stable_qid(suite, prompt) == common.stable_qid(suite, prompt)


# --------------------------------------------------------------------------- #
# make_row / validate_row
# --------------------------------------------------------------------------- #
def _good_natural_row():
    return common.make_row(
        source_benchmark="c-crab",
        source_suite="python",
        domain="code",
        task="review this patch",
        candidate="diff --git a b\n+bad line",
        gold_label="reject",
        gold_source="human_review_comment+testgen_oracle",
        gold_confidence="multi_oracle",
        defect_origin="natural",
        row_key="unit-1",
        executable_oracle={"verdict": "fail", "oracle_type": "testgen_fail_then_pass"},
        reasoning_module_labels={"source": "human_review_comment", "labels": ["off by one"]},
        rationale_gold_cause="off by one",
        ambiguous_tail=False,
        natural_defect_control=True,
        decontamination={"repo": "r", "base_commit": "abc", "pull_number": 1, "created_at": "t"},
        provenance={"instance_id": "x"},
    )


def test_make_row_has_all_required_keys_and_deterministic_id():
    r = _good_natural_row()
    assert not common.validate_row(r)
    assert r["row_id"].startswith("nearmiss-v1:c-crab:")
    assert r["corpus_id"] == common.CORPUS_ID
    # deterministic id from row_key
    r2 = _good_natural_row()
    assert r["row_id"] == r2["row_id"]


def test_validate_row_catches_bad_domain():
    r = _good_natural_row()
    r["domain"] = "banana"
    errs = common.validate_row(r)
    assert any("bad domain" in e for e in errs)


def test_validate_row_requires_missing_key():
    r = _good_natural_row()
    del r["gold_source"]
    errs = common.validate_row(r)
    assert any("missing key: gold_source" in e for e in errs)


def test_single_oracle_must_route_to_arbitration():
    r = _good_natural_row()
    r["gold_confidence"] = "single_oracle"
    r["ambiguous_tail"] = False
    errs = common.validate_row(r)
    assert any("single_oracle" in e for e in errs)
    r["ambiguous_tail"] = True
    assert not common.validate_row(r)


def test_labeled_row_needs_a_gold_field():
    r = _good_natural_row()
    r["executable_oracle"] = None
    r["reasoning_module_labels"] = None
    errs = common.validate_row(r)
    assert any("neither executable_oracle nor reasoning_module_labels" in e for e in errs)


def test_null_gold_label_row_is_valid_without_gold_fields():
    # journal-outcome shape: candidate + labels pending recovery
    r = common.make_row(
        source_benchmark="autopilot-journal",
        source_suite="hotpotqa",
        domain="hotpotqa",
        task="q",
        candidate=None,
        gold_label=None,
        gold_source="programmatic_scorer:f1",
        gold_confidence="observation",
        defect_origin="natural",
        row_key="j1",
        provenance={"candidate_recovery_needed": True},
    )
    assert not common.validate_row(r)


# --------------------------------------------------------------------------- #
# mutation primitives (deterministic, rule-based)
# --------------------------------------------------------------------------- #
def test_mutate_multiple_choice_picks_different_letter():
    out = sm.mutate_multiple_choice("B", salt="q1")
    assert out is not None
    new, cause = out
    assert new in {"A", "C", "D", "E"} and new != "B"
    assert cause == "wrong_option_selected"
    assert sm.mutate_multiple_choice("B", salt="q1") == out  # deterministic
    assert sm.mutate_multiple_choice("not-a-letter") is None


def test_mutate_numeric_off_by_one():
    assert sm.mutate_numeric("18") == ("19", "numeric_off_by_one")
    assert sm.mutate_numeric("70,000") == ("70001", "numeric_off_by_one")
    new, cause = sm.mutate_numeric("3.14")
    assert new == "3.15" and cause == "numeric_off_by_one"
    assert sm.mutate_numeric("hello") is None


def test_mutate_boolean_negation_flip():
    assert sm.mutate_boolean("yes") == ("no", "negation_flip")
    assert sm.mutate_boolean("No") == ("Yes", "negation_flip")
    assert sm.mutate_boolean("Chief of Protocol") is None


def test_mutate_entity_substitutes_distinct():
    pool = ["Michio Sugeno", "Annick Bricaud", "Radcliffe College"]
    out = sm.mutate_entity("Michio Sugeno", pool, salt="q")
    assert out is not None
    new, cause = out
    assert new != "Michio Sugeno" and cause == "entity_substitution"
    assert out == sm.mutate_entity("Michio Sugeno", pool, salt="q")  # deterministic
    assert sm.mutate_entity("Solo", ["Solo"], salt="q") is None  # no alternate


def test_seed_answer_dispatch():
    assert sm.seed_answer("multiple_choice", "C", [], "s")[1] == "wrong_option_selected"
    assert sm.seed_answer("exact_match", "42", [], "s")[1] == "numeric_off_by_one"
    assert sm.seed_answer("f1", "yes", [], "s")[1] == "negation_flip"
    assert sm.seed_answer("f1", "Animorphs", ["Twilight", "Goosebumps"], "s")[1] == \
        "entity_substitution"
    assert sm.seed_answer("substring", "", [], "s") is None  # empty gold -> skip


def test_mutate_code_token_preserves_line_count_and_flips():
    diff = "@@ -1,2 +1,2 @@\n context\n+    if x == 1:\n-    if x != 1:"
    out = sm.mutate_code_token(diff)
    assert out is not None
    new, cause = out
    assert new.count("\n") == diff.count("\n")  # line count preserved -> valid diff
    assert "!=" in new and cause.startswith("code_operator_flip")
    # no eligible added line -> None
    assert sm.mutate_code_token(" context only\n-removed") is None


# --------------------------------------------------------------------------- #
# question-pool loader + join
# --------------------------------------------------------------------------- #
def test_load_question_pool_keys_by_qid(tmp_path):
    p = tmp_path / "pool.jsonl"
    rows = [
        {"__pool_metadata__": True, "total_questions": 1},
        {"id": "x/0", "suite": "general", "prompt": "What is 2+2?",
         "expected": "4", "scoring_method": "substring", "tier": 1,
         "dataset_source": "hf"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    pool = common.load_question_pool(p)
    qid = common.stable_qid("general", "What is 2+2?")
    assert qid in pool
    assert pool[qid]["expected"] == "4"
    assert pool[qid]["scoring_method"] == "substring"


# --------------------------------------------------------------------------- #
# bug-report reconstruction
# --------------------------------------------------------------------------- #
def test_reconstruct_prepatch_keeps_buggy_lines():
    patch = (
        "diff --git a/f.c b/f.c\n"
        "--- a/f.c\n+++ b/f.c\n"
        "@@ -1,3 +1,3 @@\n"
        " int main() {\n"
        "-    return 1;\n"
        "+    return 0;\n"
        " }\n"
    )
    buggy = mb._reconstruct_prepatch(patch)
    assert "return 1;" in buggy      # removed (buggy) line kept
    assert "return 0;" not in buggy  # fix line excluded
    assert "int main()" in buggy      # context kept


# --------------------------------------------------------------------------- #
# seeded cap
# --------------------------------------------------------------------------- #
def _mkrow(origin, i):
    r = _good_natural_row()
    r["defect_origin"] = origin
    r["row_id"] = f"nearmiss-v1:{origin}:{i:04d}"
    return r


def test_enforce_seeded_cap_downsamples_when_over_half():
    rows = [_mkrow("seeded", i) for i in range(80)] + [_mkrow("natural", i) for i in range(20)]
    capped, info = asm.enforce_seeded_cap(rows, cap=0.5)
    seeded = [r for r in capped if r["defect_origin"] == "seeded"]
    assert info["applied"] is True
    assert len(seeded) == 20  # seeded <= natural
    assert len(capped) == 40
    assert len(seeded) / len(capped) <= 0.5


def test_enforce_seeded_cap_noop_when_under_half():
    rows = [_mkrow("seeded", i) for i in range(10)] + [_mkrow("natural", i) for i in range(90)]
    capped, info = asm.enforce_seeded_cap(rows, cap=0.5)
    assert info["applied"] is False
    assert len(capped) == 100


def test_build_config_hash_is_deterministic():
    assert asm.build_config_hash() == asm.build_config_hash()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
