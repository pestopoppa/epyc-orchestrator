# X-MAS held-out replay report

- source results: `benchmarks/results/runs/xmas_live_ab/20260703T213541Z-constrained-policy-v2/results.jsonl`
- replay mode: `replay`
- decision: `promote_candidate`
- prompt manifest: `benchmarks/results/runs/xmas_live_ab/20260618-heldout-resilient/prompts.jsonl`
- arm sequence: `baseline, xmas, xmas, baseline`
- prompt count: `25`
- score delta (xmas - baseline): `0.100`
- latency ratio (xmas / baseline): `0.938`

## Validation
- status: pass

## Decision
- blockers: none
- lift domains: reasoning
- regression domains: none

## Diagnostics
- score flips: both_correct=12, both_incorrect=6, unscored=5, xmas_only_better=2
- top route transitions: worker_general->worker_general (13), frontdoor->worker_general (4), worker_vision->worker_vision (3), worker_general->architect_general (2), worker_vision->worker_general (2)

## Next Clean-Window Run
- keep `xmas_routing.mode` off until this report is green and a new inference window is confirmed quiet
- reuse the exact held-out prompt manifest recorded above
- keep baseline restore enabled so the final arm leaves the orchestrator in `mode=off`
