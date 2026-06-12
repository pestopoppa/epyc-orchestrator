# Autopilot Item Analytics

Generated: `2026-06-12T20:50:31Z`
Source rows: `664`

## last_100_trials

Trials: `100` (`685` to `784`)

Per-qid analytics: **unavailable** — journal rows do not persist per-question results yet; N2 ledger must add them.

### Pinned-Zero Watchlist Verdicts

| suite | p_correct | verdict | basis |
|---|---:|---|---|
| mode_advantage_hard | 0.000 | genuinely_hard_candidate | Pinned near zero with no structural artifact documented in the Fable5 audit; requires per-qid vectors/core_v2 calibration before treating it as a useful discriminator. |
| usaco | 0.035 | artifact | Fable5 audit found seed-42 T1 rows with expected='' behind a text-scorer gate; W4 replaced empty-expected sampled rows for future evals. |
| vl | 0.035 | artifact | Fable5 follow-up traced historical VL zeros to routing/OCR plumbing; W4 repaired image bypass and OCR MTMD fallback, so historical pinned-zero is artifact-contaminated. |
| instruction_precision | 0.069 | artifact | Fable5 audit found fixed-pool expected='' instruction-precision rows; historical nonzero values are tier/sentinel mixing, not proof of item health. |
| bigcodebench | 0.083 | artifact | Fable5 audit found the fixed BigCodeBench rows require pandas, which was absent in the orchestrator environment; W4 added the dependency. |

### Flagged Suites

| suite | obs | q_evals | p_correct | mean_q | std | verdict | flags |
|---|---:|---:|---:|---:|---:|---|---|
| mode_advantage_hard | 94 | 144 | 0.000 | 0.000 | 0.000 | genuinely_hard_candidate | pinned_zero_or_broken, errors_present |
| usaco | 94 | 144 | 0.035 | 0.080 | 0.337 | artifact | pinned_zero_or_broken, errors_present |
| vl | 94 | 144 | 0.035 | 0.080 | 0.337 | artifact | pinned_zero_or_broken, errors_present |
| instruction_precision | 94 | 144 | 0.069 | 0.160 | 0.673 | artifact | errors_present |
| bigcodebench | 94 | 144 | 0.083 | 0.191 | 0.733 | artifact | errors_present |
| gpqa | 94 | 144 | 0.472 | 1.404 | 0.367 | not_pinned | errors_present |
| simpleqa | 94 | 144 | 0.493 | 1.468 | 0.216 | not_pinned | errors_present |
| agentic | 94 | 144 | 0.500 | 1.500 | 0.000 | not_pinned | errors_present |
| mode_advantage | 94 | 144 | 0.500 | 1.500 | 0.000 | not_pinned | errors_present |
| skill_transfer | 94 | 144 | 0.500 | 1.500 | 0.000 | not_pinned | errors_present |
| long_context | 94 | 144 | 0.576 | 1.723 | 0.534 | not_pinned | errors_present |
| hotpotqa | 94 | 144 | 0.653 | 2.138 | 0.742 | not_pinned | errors_present |
| general | 94 | 144 | 0.938 | 2.840 | 0.462 | not_pinned | errors_present |
| tool_use | 87 | 325 | 0.966 | 2.372 | 0.992 | saturated_low_discrimination | saturated, errors_present |
| livecodebench | 94 | 144 | 0.972 | 2.936 | 0.303 | saturated_low_discrimination | saturated, errors_present |
| cruxeval | 94 | 144 | 0.993 | 2.984 | 0.154 | saturated_low_discrimination | saturated, errors_present |
| math | 94 | 144 | 0.993 | 2.984 | 0.154 | saturated_low_discrimination | saturated, errors_present |
| coder | 94 | 144 | 1.000 | 3.000 | 0.000 | saturated_low_discrimination | saturated, errors_present |
| debugbench | 94 | 144 | 1.000 | 3.000 | 0.000 | saturated_low_discrimination | saturated, errors_present |
| thinking | 94 | 144 | 1.000 | 3.000 | 0.000 | saturated_low_discrimination | saturated, errors_present |

## last_7_days

Trials: `112` (`673` to `784`)

Per-qid analytics: **unavailable** — journal rows do not persist per-question results yet; N2 ledger must add them.

### Pinned-Zero Watchlist Verdicts

| suite | p_correct | verdict | basis |
|---|---:|---|---|
| mode_advantage_hard | 0.000 | genuinely_hard_candidate | Pinned near zero with no structural artifact documented in the Fable5 audit; requires per-qid vectors/core_v2 calibration before treating it as a useful discriminator. |
| usaco | 0.035 | artifact | Fable5 audit found seed-42 T1 rows with expected='' behind a text-scorer gate; W4 replaced empty-expected sampled rows for future evals. |
| vl | 0.035 | artifact | Fable5 follow-up traced historical VL zeros to routing/OCR plumbing; W4 repaired image bypass and OCR MTMD fallback, so historical pinned-zero is artifact-contaminated. |
| instruction_precision | 0.069 | artifact | Fable5 audit found fixed-pool expected='' instruction-precision rows; historical nonzero values are tier/sentinel mixing, not proof of item health. |
| bigcodebench | 0.083 | artifact | Fable5 audit found the fixed BigCodeBench rows require pandas, which was absent in the orchestrator environment; W4 added the dependency. |

### Flagged Suites

| suite | obs | q_evals | p_correct | mean_q | std | verdict | flags |
|---|---:|---:|---:|---:|---:|---|---|
| mode_advantage_hard | 106 | 144 | 0.000 | 0.028 | 0.290 | genuinely_hard_candidate | pinned_zero_or_broken, errors_present |
| usaco | 105 | 144 | 0.035 | 0.071 | 0.319 | artifact | pinned_zero_or_broken, errors_present |
| vl | 105 | 144 | 0.035 | 0.071 | 0.319 | artifact | pinned_zero_or_broken, errors_present |
| instruction_precision | 106 | 144 | 0.069 | 0.170 | 0.693 | artifact | errors_present |
| bigcodebench | 105 | 144 | 0.083 | 0.171 | 0.696 | artifact | errors_present |
| gpqa | 105 | 144 | 0.472 | 1.414 | 0.348 | not_pinned | errors_present |
| simpleqa | 106 | 144 | 0.493 | 1.486 | 0.252 | not_pinned | errors_present |
| agentic | 106 | 144 | 0.500 | 1.571 | 0.318 | not_pinned | errors_present |
| mode_advantage | 105 | 144 | 0.500 | 1.500 | 0.000 | not_pinned | errors_present |
| skill_transfer | 105 | 144 | 0.500 | 1.500 | 0.000 | not_pinned | errors_present |
| long_context | 106 | 144 | 0.576 | 1.840 | 0.628 | not_pinned | errors_present |
| hotpotqa | 106 | 144 | 0.653 | 2.236 | 0.750 | not_pinned | errors_present |
| general | 106 | 144 | 0.938 | 2.858 | 0.438 | not_pinned | errors_present |
| tool_use | 98 | 325 | 0.966 | 2.198 | 1.087 | saturated_low_discrimination | saturated, errors_present |
| livecodebench | 105 | 144 | 0.972 | 2.943 | 0.287 | saturated_low_discrimination | saturated, errors_present |
| cruxeval | 105 | 144 | 0.993 | 2.986 | 0.146 | saturated_low_discrimination | saturated, errors_present |
| math | 106 | 144 | 0.993 | 2.986 | 0.145 | saturated_low_discrimination | saturated, errors_present |
| coder | 106 | 144 | 1.000 | 3.000 | 0.000 | saturated_low_discrimination | saturated, errors_present |
| debugbench | 105 | 144 | 1.000 | 3.000 | 0.000 | saturated_low_discrimination | saturated, errors_present |
| thinking | 106 | 144 | 1.000 | 3.000 | 0.000 | saturated_low_discrimination | saturated, errors_present |
