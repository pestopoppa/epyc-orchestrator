# Compression episode usage — source controls

The opt-in compression A/B path snapshots the request-owned native usage ledger
after REPL construction and before two-stage or graph execution, then emits a
post-repair `episode_usage` event. Prior routing calls are outside the boundary.
Failed two-stage attempts that fall through are included; a successful early
two-stage return has no episode event. Unbound, mismatched, estimated, pending,
and incomplete observations remain unknown or partial. Content-cache replays
are counted separately and contribute no new backend tokens. These fields do
not change the existing per-turn `rolling_input_tokens` metric.

The exporter reads the same native bytes it binds by per-task offset, length,
and SHA-256; it rechecks each source before consolidating captured bytes. Missing,
duplicate, or invalid episode events export as unknown. The pre-existing single
graph-final requirement remains: multiple graph finals refuse export.

`tests.log` records 24 passed and one deselected. The deselected TestClient
lifespan test had failed earlier because this bounded projection lacks a script;
its subsequent startup would touch shared host lock state, so it was not rerun.
`canary.log` records two passes of the real REPL executor entry with a model-free
simulated graph and native request ledger, through the sidecar and ROOT strict
reader to a private ledger append/read. The reviewed ROOT adapter snapshot SHA-256
is recorded in that log. Both controls used CPU 28–29 under the bounded R4
supervisor. These are software controls, not an inference or performance result.
