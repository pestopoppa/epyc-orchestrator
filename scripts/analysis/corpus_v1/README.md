# Near-miss decision corpus v1 (`nearmiss-v1`) — builder

Builds the **calibration instrument for the Architect→Reviewer control plane**
(H4 task RC-3). Authoritative spec:
`epyc-root/handoffs/active/reviewer-calibration-accounting.md` (RC-2/RC-3).

This is a *versioned instrument*: reviewer FA/FR/Brier/ECE numbers are only
meaningful relative to a pinned `(corpus_id, content_sha256, gold-label source,
build_config_hash)`. All pre-P-REV-1 numbers are **observations** — nothing here
gates a decision. Layer A: reviewer metrics are **not** a T0–T3 model-quality
axis.

## Output (outside git)

`/mnt/raid0/llm/datasets/nearmiss-corpus-v1/`
- `rows.jsonl` — one row per candidate decision (schema below)
- `manifest.json` — corpus id, content sha256, build-config hash, per-source /
  per-domain / per-defect_origin / per-gold_label / per-gold_confidence counts,
  seeded ratio, arbitration count, journal↔pool join fraction, input-file
  provenance hashes, gap notes
- `_staging/*.jsonl` — per-miner intermediate output (regenerable)

## Build

```bash
# one-shot (runs every miner, then assembles):
.venv/bin/python scripts/analysis/corpus_v1/assemble.py --run-miners

# or run miners individually then assemble from staging:
.venv/bin/python scripts/analysis/corpus_v1/mine_ccrab.py
.venv/bin/python scripts/analysis/corpus_v1/mine_swecare.py      # self-relocates for pyarrow
.venv/bin/python scripts/analysis/corpus_v1/mine_journals.py
.venv/bin/python scripts/analysis/corpus_v1/seed_mutations.py    # depends on ccrab/swecare staging
.venv/bin/python scripts/analysis/corpus_v1/mine_bugreports.py
.venv/bin/python scripts/analysis/corpus_v1/assemble.py
```

**No inference anywhere.** Journals/logs are read-only. `seed_mutations` is
purely rule-based. The orchestrator venv deliberately lacks `pyarrow` (no new pip
deps); `mine_swecare` self-relocates to a pyarrow-capable interpreter (the
inference-research venv) to read the SWE-CARE parquet.

## Sources (one miner each)

| # | Miner | Source | Contribution | Defect origin |
|---|-------|--------|--------------|---------------|
| 1 | `mine_ccrab.py` | c-CRAB (SWE-CARE-derived, executable oracles) | 595 per-comment reject rows + 410 merged-patch accept controls | natural (code) |
| 2 | `mine_swecare.py` | SWE-CARE test split, deduped vs c-CRAB | ~361 residual reject rows + ~261 accept controls | natural (code) |
| 3 | `mine_journals.py` | autopilot journals (all shards) | 3.7k domain-labeled outcome rows, task+gold-answer recovered via qid↔pool join | natural (mixed) |
| 4 | `seed_mutations.py` | rule-based mutations of §3 gold answers + §1/2 merged patches | seeded reject + paired accept controls | seeded / natural |
| 5 | `mine_bugreports.py` | `epyc-root/bug-reports/` (orchestrator has none) | known-bad pre-patch code candidates | natural (code) |

### The journal↔question-pool join (key mechanism)

Journal `eval_details.question_results` entries store `(qid, suite, correct,
scoring_method)` but **not** the question text nor the model's answer text. The
qid is `sha1(f"{suite}\x00{prompt}")[:16]`
(`eval_tower._stable_question_qid`), so we reconstruct the **task prompt** and
**reference gold answer** for ~99% of distinct qids by joining against
`epyc-inference-research/benchmarks/prompts/question_pool.jsonl`.

Still missing: the model's *candidate* answer text (never persisted). Journal
rows therefore carry `candidate=null` + `provenance.candidate_recovery_needed`.
A later **non-inference** join to a captured-answer store (or an eval re-run)
completes them; `reasoning_module_labels` for those rows needs a later inference
labeling pass for WHY-diagnosis scoring.

## Row schema (`nearmiss_corpus_row.v1`)

Defined + validated in `common.py` (`make_row`, `validate_row`). Dual gold
labels — either may be null per source.

| field | meaning |
|-------|---------|
| `row_id`, `corpus_id`, `schema_version` | identity (`row_id` deterministic from a source-stable key) |
| `source_benchmark`, `source_suite`, `domain` | provenance + RC-3 6-domain enum (`source_suite` preserves the exact suite/language) |
| `task`, `candidate` | the task statement and the thing a reviewer judges (answer/diff/plan) |
| `gold_label` | consolidated verdict: `accept`/`reject`/`pass`/`fail`/`null` |
| `gold_source`, `gold_instrument_version`, `gold_confidence` | how gold was derived + instrument version + `multi_oracle`/`single_oracle`/`observation` |
| `executable_oracle` | executable-oracle verdict object, or null |
| `reasoning_module_labels` | reasoning-module / human-reviewer labels, or null |
| `rationale_gold_cause` | the WHY (right-for-wrong-reason discipline) |
| `defect_origin` | `natural` \| `seeded` |
| `ambiguous_tail` | route-to-human-arbitration flag |
| `natural_defect_control` | tags the natural-defect control slice |
| `decontamination` | `{repo, base_commit, pull_number, created_at}` (SWE-Bench-Illusion) |
| `provenance` | source-specific detail |

### Gold-confidence semantics (the ≥2-oracle gate)

- `multi_oracle` — gate-worthy: ≥2 independent oracles (c-CRAB human comment +
  testgen fail→pass) **or** synthetic ground truth (seeded: reference gold +
  deterministic rule).
- `single_oracle` — one human oracle only (SWE-CARE residual, c-CRAB stage1-only
  comments, bug-reports). **Forced** to `ambiguous_tail=true` (arbitration) by
  the validator.
- `observation` — non-gating (journal outcomes, merged-patch accept controls).

### Domain map

`code` ← coder/bigcodebench/cruxeval/debugbench/livecodebench + c-CRAB/SWE-CARE;
`thinking` ← thinking/math/gpqa; `hotpotqa`/`simpleqa`/`instruction_precision`
verbatim; everything else → `general`. The exact suite is always in
`source_suite`, so re-bucketing is lossless.

## Seeded mutation rules (`seed_mutations.py`, rule version `nearmiss-seed-rules-v1`)

Deterministic (content-hash-seeded), rule-based, no inference:
`numeric_off_by_one`, `negation_flip` (yes↔no / True↔False), `wrong_option_selected`
(multiple-choice), `entity_substitution` (plausible-but-wrong from same-suite pool),
`code_operator_flip` / `code_bool_flip` (single line-count-preserving flip inside
an accepted diff). Seeded rows are capped at ≤50% of the corpus by the assembler
(`enforce_seeded_cap`).

## Tests

`epyc-orchestrator/tests/test_corpus_v1_build.py` — hermetic schema/row-validation
+ mutation-rule + join + cap tests (no heavy datasets, no inference):

```bash
.venv/bin/python -m pytest tests/test_corpus_v1_build.py -q
```
