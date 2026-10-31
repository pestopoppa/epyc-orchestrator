# EV-15d retained score-token mass — APP source proof (2026-10-31)

## Change

Added the per-call score-token head parser and connected it to the existing offline paired readout consumer. Each call records its raw retained mass and missing score-token IDs before normalization. A call below the explicit floor or with zero retained mass returns a typed refusal and cannot enter a paired judgment. Head IDs, score-token order, and the floor are bound into the v2 plan and request IDs; the consumer keeps each call separate and never pools mass.

The v1 manifest/plan contract, required keys, request-ID digest inputs, and output shape remain unchanged for the legacy normalized-vector math fixture API. V1 refuses raw heads; v2 refuses legacy vectors. This implementation is an offline software path and makes no native inference, production, or measurement claim.

## Verification

`focused-tests.log` is the complete bounded run transcript. It reports **70 passed**, one existing OpenTelemetry deprecation warning, Ruff check and format success, and `git diff --check` success. The tests cover parser mass accounting/refusals, the existing v1 contract and request IDs, v2 ordered-token/floor binding, raw-vs-legacy refusal, per-call pair refusal, and accepted/refused CLI output.

`R6-RECEIPT.json` records the final run: 2026-10-31 19:02:57.594853–19:03:01.491630 UTC, CPUs 82–83, mutex 34–35, OMP/BLAS/MKL/NumExpr threads set to one, boot ID and process birth ticks, active grants sampled during the run, 0.1-second `/proc/<pid>/statm` process-tree RSS sampling, 8 GiB ceiling, 900-second ceiling, and drained captured process tree. Peak sampled RSS was 213,286,912 bytes and elapsed time was 3.919 seconds.

`first-format-gate-failure.json` preserves the earlier attempt identity. It failed before pytest because Ruff requested formatting; the test file was formatted and the final bounded run above passed. An earlier exploratory 63-test run lacked a captured affinity/mutex/RSS receipt and is not accepted as validation evidence.

## Scope

No model was run and no production/native behavior was exercised. All fixtures are synthetic software controls. No measurement or quality claim is made. [no-belief: source implementation and synthetic correctness controls; no measurement]
