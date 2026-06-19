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

## Next Clean-Window Run
- keep `xmas_routing.mode` off until this report is green and a new inference window is confirmed quiet
- reuse the exact held-out prompt manifest recorded above
- keep baseline restore enabled so the final arm leaves the orchestrator in `mode=off`
