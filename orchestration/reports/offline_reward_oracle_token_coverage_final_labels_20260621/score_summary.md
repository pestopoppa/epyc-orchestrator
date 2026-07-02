# Offline Reward Oracle Token Coverage Scores

- Schema: `offline_reward_oracle_token_coverage_scores.v1`
- Model: `deterministic/reference-token-coverage-v1`
- Score source: `reference_token_coverage`
- Rows: `322`
- Score min / max / mean: `0.0` / `1.0` / `0.6165638222011433`

- Score definition: unique lowercase alphanumeric/underscore reference tokens
  present in response divided by unique reference tokens

## Stats

| Key | Value |
|---|---:|
| `rows` | `322` |
| `target_source:answer_equivalence_final_label` | `173` |
| `target_source:heldout_stress_binary_reward` | `144` |
| `target_source:original_binary_reward` | `5` |
| `variant_type:base` | `48` |
| `variant_type:confound` | `48` |
| `variant_type:paraphrase` | `48` |
