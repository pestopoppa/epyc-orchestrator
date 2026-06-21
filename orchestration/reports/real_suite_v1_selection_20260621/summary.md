# Real Suite v1 Selection

Generated: `2026-06-21T01:09:22+00:00`

This artifact selects the 50 prompt-free task records for F1 W3 real-suite v1 curation. It is not the final EvalTower YAML suite; prompt and rubric materialization remains gated on approved local-private task text.

## Counts

- Candidate rows: `372`
- Selected rows: `50`
- Status: `selection_manifest_ready_yaml_materialization_pending`

## Selected By Class

- `benchmark_eval_measurement`: `8` selected / `8` quota
- `ops_deploy_process`: `7` selected / `7` quota
- `code_change_implementation`: `7` selected / `7` quota
- `debug_root_cause`: `7` selected / `7` quota
- `governance_docs_handoff`: `7` selected / `7` quota
- `research_intake_deep_dive`: `7` selected / `7` quota
- `planning_architecture_review`: `7` selected / `7` quota

## Selected By Outcome

- `failure`: `22`
- `success`: `28`

## Privacy

- Prompt text present: `False`
- Prompt refs present: `False`
- Selected prompt-key paths: `0`

## Next Step

Materialize benchmarks/prompts/debug/real_suite_v1.yaml from approved private prompts and deterministic or llm_judge rubrics, then add the suite to YAML_ONLY_SUITES.
