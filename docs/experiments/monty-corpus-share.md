# D-RI-MONTY-T1 results: share of real REPL cells that run unchanged under the Monty adapter

Date 2026-11-18. Orchestrator branch `exp/monty-corpus` (base `658724c5`). CPU only, offline, no inference servers.
Raw per-cell data: [`monty-corpus-share.json`](monty-corpus-share.json) (sha, flags, verdict, failure class, 100-char head;
no full code bodies). Harness: `scripts/benchmark/monty_corpus_share.py`, run with the private venv
`/mnt/raid0/llm/tmp/monty-slice-venv` (pydantic-monty 1.6.0). Method and adapter: `monty-slice-results.md`.

## Headline

**This host has no persisted, role-labelled log of executed worker REPL cells, so the worker-role share cannot be
measured directly.** The orchestrator's code log (`REPLEnvironment._code_log`) lives only inside checkpoints and none
is stored; the session log (`src/graph/session_log.py`) keeps a hash and first/last line, not the code; the trace
capture (`EPYC_REPL_TRACE_CAPTURE`) stores hashes and spans only and its directory does not exist. The one source with
full executed code is the graph REPL tap, which has no role field. What was measured instead:

| Stratum | unique cells | runs unchanged (strict) | + runs but state not JSON | fail | indeterminate |
|---|---|---|---|---|---|
| A tap, all (2592 executions) | 98 | 52 (62.7% of determinate) | 0 | 31 | 15 |
| A tap, **not found verbatim in tests/ or scripts/** | 53 | **16 (37.2%)**; 58.1% weighted by repeat count | 0 | 27 | 10 |
| A tap, found verbatim in tests/ or scripts/ (fixtures) | 45 | 36 (90.0%) | 0 | 4 | 5 |
| B eval, role-labelled, all | 112 | 4 (4.2%) | 41 (46.9% incl.) | 51 | 16 |
| B `architect_coding` (3way FINAL answers) | 42 | 0 | 27 (77.1% incl.) | 8 | 7 |
| B `architect_general` (3way FINAL answers) | 26 | 0 | 14 (77.8% incl.) | 4 | 8 |
| B `qwen36_q8_0` (tool_compliance) | 8 | 2 (25%) | 0 | 6 | 0 |
| B `supergemma4_26b_mm_q8` | 5 | 1 (20%) | 0 | 4 | 0 |
| B `supergemma4_31b_q4km` | 11 | 1 (10%) | 0 | 9 | 1 |
| B `coder_qwen3_coder_30b_a3b` | 7 | 0 | 0 | 7 | 0 |
| B `gemma4_26b_a4b_q4km_mtp` | 9 | 0 | 0 | 9 | 0 |
| B `minimax_m27_q8` | 4 | 0 | 0 | 4 | 0 |

No stratum is a worker role (`worker_*` does not appear in any labelled source). Shares are "ran to completion in the
confined Monty worker with the AST gate on, no denied callback, JSON-compatible state", over determinate cells
(indeterminate = the cell is not valid Python even for CPython, or it errored in the original run, or it followed a
timeout in its session). Treat every percentage as indicative: the strata hold 4 to 53 unique cells.

## Cell sources and counts

* **A: `/mnt/raid0/llm/tmp/repl_tap.log`**: the graph REPL tap (`src/graph/repl_tap.py`), 2026-10-20 to 2026-11-18,
  2592 executed cells with the original outcome (`ERROR:` or `FINAL`), no role. Only 98 are distinct (Oct 20-Nov 1
  smoke runs repeat the same probes). 45 of the 98 appear verbatim in `tests/` or `scripts/` and are flagged as
  fixtures; 53 are not, and are the closest thing here to real model-emitted cells, though a probe script outside the
  repo could also be among them. Sessions were replayed in turn order through one adapter so cross-cell state resolves.
* **B: eval outputs with a role label**: 68 `3way_*.jsonl` role answers containing a REPL `FINAL(` call
  (`architect_coding` 42, `architect_general` 26) plus 47 `tool_compliance` responses to the REPL-tool prompt in
  `runs/**/*.json` (seven model roles). These are model-emitted code, not executed cells; 3way answers are mostly a
  function definition plus `FINAL`, so they are the least representative of multi-turn exploration.
* Considered and rejected: `logs/dispatch/dispatch.jsonl` (290 rows, no code), `events.sqlite`, `skills.db` (empty),
  autopilot journals, `tool_call_audit.jsonl` (`run_shell` text only), `repl_output/`, seeding JSON.
* Sampling: nothing needed capping (210 unique cells total, cap 2000), so there was no stratified draw.

## Pipeline per cell

(a) REPL AST gate (`_validate_code`); (b) a static Monty-subset screen (own AST scan); (c) real Monty execution through
`MontyREPLAdapter` with the worker confined under the SEC-EXEC shim (fail closed). `peek`/`grep`/`context_len` ran
against a synthetic 10 kB context. Every other REPL callable is denied by the adapter with a logged `PermissionError`,
so no cell reached a tool, a file or a server. 3 s per-cell deadline.

## Failure classes (unique cells that did not run unchanged)

| Class | cells | notes |
|---|---|---|
| Uses a REPL callable beyond peek/grep/context_len/FINAL | 57 | `file_write_safe` 19, `list_dir` 15, `ocr_document` 5, `llm_batch` 4, `TOOL` 4, `run_shell` 3, `file_info` 3, `list_tools` 1, `llm_call` 1, `peek`/`grep` with a `file_path` argument 2; this is the dominant class and is a design boundary of the lane, not a Monty gap |
| Non-JSON state (cell ran; a function it defined cannot cross the boundary) | 41 | all from B 3way answers (`def` plus `FINAL`); harmless for a one-cell answer, a failure for a multi-cell session |
| Runtime error | 17 | assertion failures in model code, deliberate `ZeroDivisionError`, names defined in earlier cells that are not in the replayed session; the original also errored where known (A), unknown for B |
| Plain `NameError` for names the cell never defined | 4 | |
| AST gate refusal | 3 | `import os`, `input` |
| `NotImplementedError` (Monty feature) | 1 | a `class` definition with `__init__` ("monty syntax parser does not yet support ...") |
| `statistics`/`numpy`/`fractions` (absent), generators, user exception subclasses, unsupported syntax | **0 observed** | the failure classes the task text anticipates did not occur once in this corpus |
| Not valid Python even in CPython (excluded as indeterminate) | 16 | truncated or prose-wrapped model output |

## Does a static screen work?

Screen: "in subset" means no generator, no class inheritance, no absent-stdlib/package name, no REPL callable outside
the bridged four, valid CPython syntax. Against Monty's actual outcome (determinate cells, "runs" includes the
state-loss ones):

| | Monty runs | Monty fails |
|---|---|---|
| static screen says in subset | 88 | 24 |
| static screen says out | 9 | 58 |

Precision of "in subset" is 79% (88 of 112); the 24 misses are mostly runtime errors, undefined cross-cell names and gate refusals
that a name screen cannot see, plus model bugs. All 58 failing cells that the screen flagged for an outside-the-bridge callable failed under Monty (59 flagged, the 59th
was indeterminate), and no callable-boundary failure was missed. A first pass missed 12 (`TOOL`, `llm_batch`, `list_tools`,
`llm_call`, and `peek`/`grep` with `file_path`); the screen now knows those names and checks the `file_path` keyword, and the
table above is from the rerun. The Monty-language-feature checks
(generators, absent modules) never fired on a cell that mattered.

## Recommendation

**The precondition for T2 is not met on this evidence, but not refuted either.** Three points:

1. The corpus cannot answer the question asked. There are 53 unique non-fixture cells, none role-labelled, none
   from a worker role; the 62.7% / 37.2% / 58.1% figures bound nothing about served worker cells.
2. What the data does say: the lane's coverage limit on this corpus is the **bridged callable set**, not Monty's
   language subset. Cells that stay in peek/grep/context_len/FINAL plus plain Python mostly run unchanged; cells that call
   `list_dir`, `file_write_safe`, `run_shell`, `llm_call`, `TOOL` are refused by design. The statistics/numpy/generator/
   exception-subclass gaps measured in the slice are real but rare here (zero cells).
3. A static screen is therefore cheap: a gate check plus a callable-name check against the full REPL tool-name list and
   the `file_path` kwarg on peek/grep (implemented in `static_screen`), about 20 lines of `ast`, with no need for a Monty feature model. If T2 goes ahead, wire exactly that screen, route in-screen cells to
   Monty and everything else to CPython, and log the screen verdict with each cell. Do not build a larger feature screen.

**Before T2 commits effort, close the measurement gap:** persist the REPL code with its role. Done in this branch: the tap now records `role`, `request_id` and `task_id` on its header line (`monty_corpus_share.py`
reads them when present). Once the API is reloaded, run for a few days of real traffic, then re-run `monty_corpus_share.py` unchanged on that log. Until then
the share for worker roles is unmeasured, and the only defensible claim is the screen's design: it has no false positive on the
callable boundary in this corpus, and the Monty-language gaps it would also test for never occurred.

## Limits

Small unique-cell counts; the A tap mixes live runs with smoke probes (fixture matching is exact-text only); B cells
were not executed in the original REPL, so B errors cannot be split into model bugs and Monty gaps; each tap session
was replayed against a synthetic context, so cells whose output depends on the real context are scored on whether they run,
not on whether they return the same text; the 3 s deadline was not tripped (no timeouts). No CPython comparison run
was made (it would execute the original tool calls), so "unchanged" means "runs to completion with no denied call",
not byte-identical output.
