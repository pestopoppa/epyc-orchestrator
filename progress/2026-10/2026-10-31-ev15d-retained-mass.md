# EV-15d retained score-token mass — 2026-10-31

Implemented the per-call score-token head parser and integrated its retained-mass floor into the offline paired readout consumer. Missing score-token IDs receive zero and are recorded; each call is checked before normalization, and any under-floor call produces an auditable typed refusal without a comparison. The raw-head v2 plan binds the ordered K=9 token IDs and floor into request IDs. The original v1 manifest, plan, request IDs, and math-only result shape remain compatible.

Focused APP validation passed **70 tests** plus Ruff and whitespace checks. The final R6 process receipt, full test transcript, source hashes, and first preflight-failure identity are in [`artifacts/audit/ni-ev15d-retained-mass-20261031/README.md`](../../artifacts/audit/ni-ev15d-retained-mass-20261031/README.md). No model/native run or measurement claim was made. [no-belief: source implementation and synthetic correctness controls; no measurement]
