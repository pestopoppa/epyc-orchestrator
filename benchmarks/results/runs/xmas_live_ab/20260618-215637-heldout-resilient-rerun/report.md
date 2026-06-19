# X-MAS held-out replay report

- source results: `benchmarks/results/runs/xmas_live_ab/20260618-215637-heldout-resilient-rerun/results.jsonl`
- replay mode: `replay`
- decision: `hold`
- prompt manifest: `benchmarks/results/runs/xmas_live_ab/20260618-heldout-resilient/prompts.jsonl`
- arm sequence: `baseline, xmas`
- prompt count: `25`
- score delta (xmas - baseline): `-0.350`
- latency ratio (xmas / baseline): `16.181`

## Validation
- status: pass

## Decision
- blockers:
  - overall score delta -0.350 < required 0.050
  - latency ratio 16.181 > allowed 1.100
  - no domain improved by >= 0.050
  - domain regressions: code, math, reasoning
- lift domains: none
- regression domains: code, math, reasoning

## Diagnostics
- score flips: baseline_only_better=7, both_correct=6, both_incorrect=7, unscored=5
- timeouts/errors: xmas=2
- top route transitions: coder_escalation->worker_general (16), coder_escalation->architect_general (3), coder_escalation-><none> (2), coder_escalation->ingest_long_context (2), coder_escalation->frontdoor (1)
- largest latency regressions:
  - knowledge_plan_simpleqa_general_00002 knowledge:plan: coder_escalation -> worker_general, 1.084s -> 201.54s (185.923x), score None -> None
  - reasoning_solve_gpqa_Molecular_Biology_0000 reasoning:solve: coder_escalation -> , 1.668s -> 240.104s (143.947x), score 0.0 -> 0.0
  - code_extract_cruxeval_output_0009 code:extract: coder_escalation -> worker_general, 1.502s -> 119.205s (79.364x), score 1.0 -> 0.0
  - knowledge_refine_simpleqa_general_00003 knowledge:refine: coder_escalation -> worker_general, 1.384s -> 36.553s (26.411x), score 1.0 -> 1.0
  - knowledge_extract_simpleqa_general_00004 knowledge:extract: coder_escalation -> worker_general, 1.463s -> 34.364s (23.489x), score 0.0 -> 0.0

## Next Clean-Window Run
- keep `xmas_routing.mode` off until this report is green and a new inference window is confirmed quiet
- reuse the exact held-out prompt manifest recorded above
- keep baseline restore enabled so the final arm leaves the orchestrator in `mode=off`
