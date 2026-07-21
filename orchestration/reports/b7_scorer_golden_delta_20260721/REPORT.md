# B7 Scorer Semantics — Golden Before/After Delta

_Generated: 2026-07-21T09:26:30Z · op-bundle ESC-6 option B (operator-granted 2026-07-21)_

## What this is

The before/after behavioral delta the operator is owed for the **scorer-semantics package** — commits `07a20a7c + 8f24679a`'s changes to `scripts/benchmark/debug_scorer.py`. A golden input corpus is scored by BOTH the pre-package and the post-package scorer; every changed outcome is enumerated and tied to the audit finding that explains it.

- **PRE scorer** = `debug_scorer_pre.py`, extracted from `2a41c0bc` (post hard-fail A1, pre the semantics package).
- **POST scorer** = live `scripts/benchmark/debug_scorer.py` at `fe1da2f1 (HEAD, spec-dec-mtp-refresh-2026-06-22)`.
- **Corpus**: `golden_corpus.jsonl` — 146 rows (115 derived from every live sentinel, 31 synthetic per-finding).
- **Outcomes**: `50` of `146` rows changed outcome under the package.

## Summary — changed rows by scoring method

| scoring_method | rows | changed | unchanged |
|---|---:|---:|---:|
| code_execution | 5 | 3 | 2 |
| exact_match | 20 | 16 | 4 |
| f1 | 19 | 4 | 15 |
| llm_judge | 3 | 1 | 2 |
| multiple_choice | 35 | 5 | 30 |
| substring | 64 | 21 | 43 |
| **TOTAL** | **146** | **50** | **96** |

## Summary — changed rows by direction

| direction | count | meaning |
|---|---:|---|
| True->False | 27 | PRE credited a non-answer / POST rejects it (boundary, final-region) |
| False->True | 15 | PRE scored wrong-form as miss / POST now credits it (boxed, multiset, textual MC) |
| ERROR->X | 3 | PRE raised where POST now returns a bool (non-string expected coerced) |
| X->ERROR | 3 | POST refuses to score (ScoringUnavailableError / ValueError) where PRE returned a bool |
| ERROR->ERROR | 2 | both raise, but the ExcType changed (guarded ValueError vs incidental Attribute/Type error) |

## Direction × scoring method

| scoring_method | ERROR->ERROR | ERROR->X | False->True | True->False | X->ERROR |
|---|---|---|---|---|---|
| code_execution | 0 | 0 | 1 | 1 | 1 |
| exact_match | 1 | 1 | 7 | 6 | 1 |
| f1 | 1 | 1 | 2 | 0 | 0 |
| llm_judge | 0 | 0 | 0 | 0 | 1 |
| multiple_choice | 0 | 0 | 5 | 0 | 0 |
| substring | 0 | 1 | 0 | 20 | 0 |

## Live sentinel scoring changes

**25** sentinel-derived rows changed, spanning **21** distinct live sentinels (out of 39 in `scripts/autopilot/sentinel_questions.yaml`, the file eval_tower.py loads via `SENTINEL_PATH`).

Sentinels with at least one changed synthetic-answer row:

- `sentinel_agentic_01` (substring): nearmiss_embedded
- `sentinel_agentic_02` (exact_match): boxed_score16, cot_wrongfinal_score03
- `sentinel_agentic_03` (substring): nearmiss_embedded
- `sentinel_code_01` (substring): nearmiss_embedded
- `sentinel_code_02` (substring): nearmiss_embedded
- `sentinel_code_03` (substring): nearmiss_embedded
- `sentinel_escalation_01` (substring): nearmiss_embedded
- `sentinel_escalation_02` (substring): nearmiss_embedded
- `sentinel_escalation_03` (exact_match): boxed_score16, cot_wrongfinal_score03
- `sentinel_factual_01` (substring): nearmiss_embedded
- `sentinel_general_01` (substring): nearmiss_embedded
- `sentinel_instruction_01` (substring): nearmiss_embedded
- `sentinel_instruction_03` (substring): nearmiss_embedded
- `sentinel_instruction_04` (substring): nearmiss_embedded
- `sentinel_longctx_01` (substring): nearmiss_embedded
- `sentinel_math_01` (substring): nearmiss_embedded
- `sentinel_math_02` (exact_match): boxed_score16, cot_wrongfinal_score03
- `sentinel_math_03` (exact_match): boxed_score16, cot_wrongfinal_score03
- `sentinel_math_06` (substring): nearmiss_embedded
- `sentinel_multihop_01` (substring): nearmiss_embedded
- `sentinel_thinking_01` (substring): nearmiss_embedded

> These are changes on **synthetic probe answers** constructed to exercise the changed behaviors, not on real model transcripts. They show *which* sentinels' scoring is sensitive to the package (e.g. a `\boxed{}`-only answer to a numeric sentinel, or an embedded-digit near-miss). The sentinels' own gold values are unchanged.

## Every changed row

| case_id | method | PRE | POST | dir | audit finding |
|---|---|---|---|---|---|
| `audit_score04_entrypoint_noargs` | code_execution | True | ERROR:ScoringUnavailableError | X->ERROR | SCORE-04: entry_point + string expected without cases now ERRORs (no zero-arg synth) |
| `audit_score05_entrypoint_cases_pass` | code_execution | False | True | False->True | SCORE-05: entry_point_cases oracle is a post-package capability |
| `audit_score21_vacuous_assert` | code_execution | True | False | True->False | SCORE-21: vacuous `assert True` no longer counts as an executable oracle |
| `audit_score03_colon_cotwrong` | exact_match | True | False | True->False | SCORE-03: colon/quote fallback now confined to the final-answer region (not CoT) |
| `audit_score03_quote_cotwrong` | exact_match | True | False | True->False | SCORE-03: colon/quote fallback now confined to the final-answer region (not CoT) |
| `audit_score03_quote_final` | exact_match | False | True | False->True | SCORE-03: colon/quote fallback now confined to the final-answer region (not CoT) |
| `audit_score16_nested_frac` | exact_match | False | True | False->True | SCORE-16: exact_match now extracts \boxed{...} incl. nested braces |
| `audit_score16_simple` | exact_match | False | True | False->True | SCORE-16: exact_match now extracts \boxed{...} incl. nested braces |
| `audit_score23_int_exact` | exact_match | ERROR:AttributeError | True | ERROR->X | SCORE-23: non-string expected coerced to str instead of raising |
| `audit_score24_multigroup_em` | exact_match | True | ERROR:ValueError | X->ERROR | SCORE-24: F1 multiset overlap + single-capture-group enforcement |
| `audit_score24_zerogroup_em` | exact_match | ERROR:IndexError | ERROR:ValueError | ERROR->ERROR | SCORE-24: F1 multiset overlap + single-capture-group enforcement |
| `sentinel_agentic_02__em_boxed` | exact_match | False | True | False->True | SCORE-16: exact_match now extracts \boxed{...} incl. nested braces |
| `sentinel_agentic_02__em_cot_wrongfinal` | exact_match | True | False | True->False | SCORE-03: colon/quote fallback now confined to the final-answer region (not CoT) |
| `sentinel_escalation_03__em_boxed` | exact_match | False | True | False->True | SCORE-16: exact_match now extracts \boxed{...} incl. nested braces |
| `sentinel_escalation_03__em_cot_wrongfinal` | exact_match | True | False | True->False | SCORE-03: colon/quote fallback now confined to the final-answer region (not CoT) |
| `sentinel_math_02__em_boxed` | exact_match | False | True | False->True | SCORE-16: exact_match now extracts \boxed{...} incl. nested braces |
| `sentinel_math_02__em_cot_wrongfinal` | exact_match | True | False | True->False | SCORE-03: colon/quote fallback now confined to the final-answer region (not CoT) |
| `sentinel_math_03__em_boxed` | exact_match | False | True | False->True | SCORE-16: exact_match now extracts \boxed{...} incl. nested braces |
| `sentinel_math_03__em_cot_wrongfinal` | exact_match | True | False | True->False | SCORE-03: colon/quote fallback now confined to the final-answer region (not CoT) |
| `audit_score23_list_f1` | f1 | ERROR:TypeError | True | ERROR->X | SCORE-23: non-string expected coerced to str instead of raising |
| `audit_score24_multigroup_f1` | f1 | ERROR:AttributeError | ERROR:ValueError | ERROR->ERROR | SCORE-24: F1 multiset overlap + single-capture-group enforcement |
| `audit_score24_multiset_a` | f1 | False | True | False->True | SCORE-24: F1 multiset overlap + single-capture-group enforcement |
| `audit_score24_multiset_b` | f1 | False | True | False->True | SCORE-24: F1 multiset overlap + single-capture-group enforcement |
| `audit_judge_fastpath_concat` | llm_judge | True | ERROR:ScoringUnavailableError | X->ERROR | B7: llm_judge substring fast-path is now boundary-aware |
| `audit_mc_none_of_above` | multiple_choice | False | True | False->True | MC: multiple_choice resolves textual/overlapping choice labels |
| `audit_mc_overlap_blackcat` | multiple_choice | False | True | False->True | MC: multiple_choice resolves textual/overlapping choice labels |
| `audit_mc_overlap_cat` | multiple_choice | False | True | False->True | MC: multiple_choice resolves textual/overlapping choice labels |
| `audit_mc_paren_label` | multiple_choice | False | True | False->True | MC: multiple_choice resolves textual/overlapping choice labels |
| `audit_mc_textual_label` | multiple_choice | False | True | False->True | MC: multiple_choice resolves textual/overlapping choice labels |
| `audit_score06_2630` | substring | True | False | True->False | SCORE-06: substring now boundary-aware (digit/word units, not raw containment) |
| `audit_score06_630` | substring | True | False | True->False | SCORE-06: substring now boundary-aware (digit/word units, not raw containment) |
| `audit_score06_concat` | substring | True | False | True->False | SCORE-06: substring now boundary-aware (digit/word units, not raw containment) |
| `audit_score23_none_substr` | substring | ERROR:TypeError | False | ERROR->X | SCORE-23: non-string expected coerced to str instead of raising |
| `sentinel_agentic_01__substr_nearmiss` | substring | True | False | True->False | SCORE-06: substring now boundary-aware (digit/word units, not raw containment) |
| `sentinel_agentic_03__substr_nearmiss` | substring | True | False | True->False | SCORE-06: substring now boundary-aware (digit/word units, not raw containment) |
| `sentinel_code_01__substr_nearmiss` | substring | True | False | True->False | SCORE-06: substring now boundary-aware (digit/word units, not raw containment) |
| `sentinel_code_02__substr_nearmiss` | substring | True | False | True->False | SCORE-06: substring now boundary-aware (digit/word units, not raw containment) |
| `sentinel_code_03__substr_nearmiss` | substring | True | False | True->False | SCORE-06: substring now boundary-aware (digit/word units, not raw containment) |
| `sentinel_escalation_01__substr_nearmiss` | substring | True | False | True->False | SCORE-06: substring now boundary-aware (digit/word units, not raw containment) |
| `sentinel_escalation_02__substr_nearmiss` | substring | True | False | True->False | SCORE-06: substring now boundary-aware (digit/word units, not raw containment) |
| `sentinel_factual_01__substr_nearmiss` | substring | True | False | True->False | SCORE-06: substring now boundary-aware (digit/word units, not raw containment) |
| `sentinel_general_01__substr_nearmiss` | substring | True | False | True->False | SCORE-06: substring now boundary-aware (digit/word units, not raw containment) |
| `sentinel_instruction_01__substr_nearmiss` | substring | True | False | True->False | SCORE-06: substring now boundary-aware (digit/word units, not raw containment) |
| `sentinel_instruction_03__substr_nearmiss` | substring | True | False | True->False | SCORE-06: substring now boundary-aware (digit/word units, not raw containment) |
| `sentinel_instruction_04__substr_nearmiss` | substring | True | False | True->False | SCORE-06: substring now boundary-aware (digit/word units, not raw containment) |
| `sentinel_longctx_01__substr_nearmiss` | substring | True | False | True->False | SCORE-06: substring now boundary-aware (digit/word units, not raw containment) |
| `sentinel_math_01__substr_nearmiss` | substring | True | False | True->False | SCORE-06: substring now boundary-aware (digit/word units, not raw containment) |
| `sentinel_math_06__substr_nearmiss` | substring | True | False | True->False | SCORE-06: substring now boundary-aware (digit/word units, not raw containment) |
| `sentinel_multihop_01__substr_nearmiss` | substring | True | False | True->False | SCORE-06: substring now boundary-aware (digit/word units, not raw containment) |
| `sentinel_thinking_01__substr_nearmiss` | substring | True | False | True->False | SCORE-06: substring now boundary-aware (digit/word units, not raw containment) |

## Era note

This delta is part of the **E7 eval-instrument boundary** (the measurement trust boundary: MEASUREMENT.md, eval tower, scoring, safety gates, era registry). The scorer-semantics package tightens how the deterministic eval-tower scorer credits/refuses answers; per the instrument constitution these are human-amendment-only changes, and this golden file is the ratified before/after record. **op-bundle ESC-6 option B is satisfied by this document.**

## Reproduce / pin

```
# regenerate corpus (deterministic)
python orchestration/reports/b7_scorer_golden_delta_20260721/build_corpus.py
# re-run the delta
python orchestration/reports/b7_scorer_golden_delta_20260721/run_delta.py
# pin: current scorer must still reproduce results.jsonl 'new' column
python -m pytest tests/unit/test_b7_golden_corpus_pin.py -q
```

`results.jsonl` records the ratified POST (`new`) outcome per case. `tests/unit/test_b7_golden_corpus_pin.py` re-runs the live scorer over the corpus and asserts it still produces exactly those outcomes — so a future scorer change must update this golden file **deliberately**.

---

## B7b addendum — SCORE-07/08/09/12 scorer-semantics remainder (prepared for sign-off)

These four findings live in `scripts/autopilot/eval_tower.py` + `scripts/autopilot/rubric_scoring.py` (the rubric / confidence path), NOT in `debug_scorer.py`, so their corpus rows carry a `scorer` discriminator (`rubric_parse` / `rubric_aggregate` / `code_exec_confidence`) and are pinned against those surfaces by `_b7b_outcome`, segregated from the 146 debug_scorer rows. Regenerate with `build_b7b_rows.py` (idempotent — strips/re-appends `b7b_*`).

**13 new rows, 8 behavior changes** (corpus/results now 159 total; 58 changed; the 146/50/21 B7 sub-counts are unchanged and still pinned).

| # (SCORE) | Old behavior | New behavior |
|---|---|---|
| **07** rubric judge range validation | out-of-[0,1] judge score CLAMPED (`7` → perfect `1.0` on that dimension) | out-of-range REJECTED as scale drift (dimension omitted → deterministic heuristic fallback); `log.warning` counts rejections; exact `0.0`/`1.0` boundaries unchanged |
| **08** judge-vs-heuristic provenance | rubric scores stamped identically whether from a model judge or the structural heuristic fallback | `QuestionResult.rubric_source` ∈ {`judge`,`heuristic_fallback`}; aggregate `details["rubric_source_counts"]`; journal row carries `rubric_source`. Scores unchanged. |
| **09** empty rubric ≠ perfect | `aggregate_rubric_score({})` → **1.0** (zero positive weight defaulted `positive_score` to 1.0) | empty / all-missing → **0.0** (absence of evidence ≠ perfection). Production caller always merges the 8-key fallback, so live behavior only changes for genuinely-empty inputs. |
| **12** phantom pass_rate confidence | code_execution confidence = `scoring_config.get("pass_rate", correct)` — a static dataset field no scorer writes; `scoring_config: null` → AttributeError → error result | confidence = `float(correct)`, source `code_execution_binary_proxy` (never real, since n_probs suppressed per EV-CONF); non-dict `scoring_config` coerced to `{}` (no more error result) |

Row-level breakdown (`changed` in **bold**):

| case_id | finding | old → new |
|---|---|---|
| b7b_score07_out_of_range_high | 07 | **{factual_accuracy:1.0} → {}** |
| b7b_score07_out_of_range_negative | 07 | **{citation:0.0} → {}** |
| b7b_score07_in_range_kept | 07 | {outline:0.7} → {outline:0.7} |
| b7b_score07_boundary_one_kept | 07 | {presentation:1.0} → {presentation:1.0} |
| b7b_score07_boundary_zero_kept | 07 | {tool_calls:0.0} → {tool_calls:0.0} |
| b7b_score07_mixed_reject_and_keep | 07 | **{reasoning:0.25,tool_calls:1.0,content_stage:1.0} → {reasoning:0.25}** |
| b7b_score09_empty | 09 | **1.0 → 0.0** |
| b7b_score09_all_missing | 09 | **1.0 → 0.0** |
| b7b_score09_all_present_half | 09 | 0.5 → 0.5 |
| b7b_score09_partial_positive | 09 | 0.8 → 0.8 |
| b7b_score12_pass_rate_correct | 12 | **[0.9,"code_execution_pass_rate"] → [1.0,"code_execution_binary_proxy"]** |
| b7b_score12_pass_rate_wrong | 12 | **[0.9,"code_execution_pass_rate"] → [0.0,"code_execution_binary_proxy"]** |
| b7b_score12_no_pass_rate_source_renamed | 12 | **[1.0,"code_execution_pass_rate"] → [1.0,"code_execution_binary_proxy"]** |

(SCORE-08 is a metadata stamp with no verdict delta, so it has no corpus row; it is pinned by `test_eval_question_stamps_rubric_source_heuristic_fallback` / the judge-path assertion in `test_eval_question_uses_configured_local_rubric_judge`.)

**Status: PREPARED, not yet ratified.** This package awaits the operator's one-line sign-off (same discipline as B7) before it gates any decision; the E7 era boundary already covers the scorer change.
