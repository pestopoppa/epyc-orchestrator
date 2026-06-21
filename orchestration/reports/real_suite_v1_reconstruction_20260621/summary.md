# Real Suite v1 Reconstruction

Generated: `2026-06-21T01:20:01+00:00`

This prompt-free report checks whether the F1 W3 50-row selection can be materialized from local inference tap events and existing question-pool ground truth.

## Coverage

- Selected rows: `50`
- Prompt recovered: `46`
- Response recovered: `37`
- Existing question-pool expected matches: `39`

## Materialization Status

- `expected_backed_ready`: `39`
- `missing_prompt`: `4`
- `needs_reference_or_rubric`: `7`

## Prompt Sources

- ``: `4`
- `architect_question_block`: `1`
- `raw_task_prompt`: `45`

## Privacy

- Committed outputs omit prompt text, response text, and expected text.
- Use `--private-output-jsonl` for local-only curation of prompts and rubrics.

## Next Step

Use the private output JSONL to fill deterministic expected answers or EV-9-style llm_judge rubrics for rows marked needs_reference_or_rubric, then emit benchmarks/prompts/debug/real_suite_v1.yaml and add real_suite_v1 to YAML_ONLY_SUITES.
