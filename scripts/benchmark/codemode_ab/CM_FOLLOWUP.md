# CM-1..CM-5 follow-up: what code now does, what still needs a live measurement

No inference was run to produce this. Every flag below is DEFAULT OFF.

Probe result that motivates it (CODEMODE-AB, 35B): per-request top-8 tool selection in the initial prompt raised success
0.10 -> 0.38; C1 (embedding) and C2 (decision model) tie on success, C2 has better recall (0.65 vs 0.45); direct native
calls beat REPL code mode (0.43 vs 0.10). 27B selector arms were floor-bound. Rescore note: `verdict-rescore.json` next to
each `verdict.json` uses the fixed score.py (tied success no longer yields X/Y_better); original verdicts stand under the
old sha.

## CM-1 (direct native path) - code done
`ORCHESTRATOR_TOOL_SELECTOR=off|embed|oneshot`, `..._K` (default 8). `select_native_tools` narrows the client `tools` list in
`_run_client_tool_completion`. No provider is registered at startup; with none, lexical top-k is used and recorded.
Measure: wire `make_native_provider("embed", embedder=<v10 production embedder>)` and `("oneshot", primitives=<worker>)` at
stack start in a measurement window; run the codemode_ab arm A task set through the live `x_tool_mode=client` endpoint with
the flag off / embed / oneshot (paired on task x seed). Question: does success rise with tools >> k at equal or lower tokens?
Expected effect: +0.2..0.3 success on novel-tier tasks at 123 tools; no effect when the list is already <= k. Record selector
recall (any-of needed tool in top-k) and selector tokens. Stop on futility: |d_success| < 0.05 and token gap < 10 percent at
the N=20 look. Open design question: oneshot uses one NOUL per tool, which grows with list size; check cost at 123 tools.

## CM-2 (search_tools / describe_tool) - code done, `ORCHESTRATOR_REPL_TOOL_SEARCH=1`
Measure 1 (no inference): prompt tokens vs tool count for the current prompt (`constants.py`) vs a search-only prompt.
Measure 2 (live): arm B vs B+search on the CODEMODE-AB tasks; acceptance = prompt-token reduction at equal accuracy.
Gate: S4 gate or an operator exception to "no new REPL tools before S4" before enabling anywhere real. Ranker is lexical;
an embedding ranker is the alternative to compare.

## CM-3 (typed read-only returns) - code done, `ORCHESTRATOR_STRUCTURED_READ_ONLY=1`
Run `check_shape_consistency` (probe 5 vs max batch) against every real read-only tool with realistic arguments; needs
real handlers and fixtures, not inference. Acceptance: no read-only tool returns a bare string on success. Check consumers
(REPL code, prompts) tolerate the envelope; note errors become envelopes rather than exceptions.

## CM-4 (classify in the REPL) - code done, `ORCHESTRATOR_REPL_CLASSIFY=1`
Code now also exposes the existing classifiers read-only, typed and capped (same flag): `route_classify(items)` (rule
routing + Trinity role), `risk_bands(items, role="")` (factual-risk and difficulty bands) and, when the REPL has a MemRL
retriever, `q_scores(items, k=3)` (the routing q-values QScorer writes, read via `retrieve_for_routing` only). Cap: 200
items per call, `ORCHESTRATOR_REPL_CLASSIFY_BUDGET` (default 2000) per session, 4000 chars per item. The QScorer reward
function is a write-path computation over progress logs and is deliberately not exposed. No code remains for CM-4.
Measure (live): label N items
(N in 20, 100, 500) by script `classify` vs the model labelling in-prompt; compare accuracy against a gold labelling,
prompt tokens, and wall time. Gate as CM-2.

## CM-5 (handles, branch-replayed store) - code done, `ORCHESTRATOR_REPL_HANDLES=1`
Code now records for replay by default while handles are on (`environment.py`, `codemode_tools.replay_default_on`; any
explicit `REPLAY_RECORDING` setting wins in both directions). The recorded seed carries the handle store the run started
from (`repl.handle_store_cas`) and `GraphHarness` restores it into the rebuilt REPL; a checkpoint taken without handles
clears the store on restore. No code remains for CM-5. Measure (live): tasks with large
intermediate results, handle reference vs `peek` re-read; tokens and success. Confirm handles survive a real branch
checkpoint/restore in the live session store (unit-tested at the REPL level only).
