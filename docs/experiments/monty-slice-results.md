# D-RI-MONTY-SLICE results: a restricted Monty adapter on twelve existing REPL cases

Date 2026-11-18. Orchestrator branch `feat/monty-slice` (base `f0cf8838`). CPU only, no inference servers.
Raw numbers: [`monty-slice-results.json`](monty-slice-results.json). Harness:
`scripts/benchmark/monty_slice_compare.py`. Unit tests: `tests/unit/test_monty_adapter.py` (33 tests).

## Recommendation

**GO to retain the adapter as an opt-in, default-OFF lane. NO-GO for enabling it by default, and this slice
does not show whole-REPL fitness.**

On the twelve cases the lane preserved the admitted semantics (12/12 original-test assertions hold; 11/12
byte-identical output, error text, FINAL result and parent state; the twelfth differs only in SyntaxError wording)
and it adds a real containment property CPython-in-process cannot offer: the cell runs in a separate, killable,
memory-limited, Landlock+seccomp-confined worker, and a worker loss resumes from a JSON checkpoint. Overhead on a
warm cell is about 0.1 to 0.2 ms. Against that: Monty cannot run a measurable part of what the REPL admits today
(no `statistics`, `fractions`, numpy/scipy, generators, exception subclasses), so it can only serve cells that stay
inside its subset, and the host callbacks it needs still run in the API process.

What would justify the next step (a wider slice, then a model-bearing paired probe in its own window): a real
corpus of REPL cells from served worker roles to measure the share that stays inside Monty's subset. This slice
cannot say that share. Do not infer it from twelve unit-test snippets.

## What was run

| Arm | What it is | Supports |
|---|---|---|
| `cpython_warm` | production `REPLEnvironment.execute`, one env per case, reused for warm timing | everything |
| `cpython_contained` | the EXISTING SEC-EXEC child (`run_python_code`, Landlock + seccomp + scrubbed env), one cold child per cell | stateless self-contained cells only. D-RI-CONTAIN-T2 (a warm, REPL-aware contained lane) is not built, so there is no peek/grep/context/artifacts/FINAL and no state across cells |
| `monty` | `MontyREPLAdapter`, REPL AST gate ON, worker under the SEC-EXEC shim (default) | the twelve cases |
| `monty_native` | same adapter, AST gate OFF, used only to see what Monty itself refuses | security probes |

Runtime: `pydantic-monty` **1.6.0** (PyPI wheels, `monty-runtime 1.6.0`), Python 3.11.17, in a PRIVATE venv
`/mnt/raid0/llm/tmp/monty-slice-venv` (the shared `.venv` was not touched; it does not contain the package).
Provenance caveat: the intake pinned the clone SHA `86fb685c` whose workspace version is 1.6.0. The installed wheel
matches that VERSION; this host has no `cargo`, so the wheel's content is not proven to be that commit. Packaging:
optional extra `monty = ["pydantic-monty==1.6.0"]` in `pyproject.toml` plus `uv.lock` (`uv lock`: three added
packages plus the extra's metadata, nothing else changed), so a default install does not pull it. Flag: `monty_repl`
(`ORCHESTRATOR_MONTY_REPL`), default OFF in test and prod; the adapter refuses to construct without it and nothing in
the default REPL path imports the module.

### Adapter design (src/repl_environment/monty_adapter.py)

* The adapter wraps an existing `REPLEnvironment` (the parent boundary) and returns the existing `ExecutionResult`.
* **State stays in the parent.** After every cell the JSON-compatible user names and `artifacts` are copied out of the
  sandbox into `parent._globals` / `parent.artifacts`, so `get_state()`, `checkpoint()` and `restore()` work
  unchanged and the checkpoint is plain JSON. Monty's opaque `dump()` bytes are never used. After `restore()` or a
  worker loss, a fresh worker is rebuilt from that JSON.
* **Callbacks are bounded value messages.** Only `peek`, `grep`, `context_len` (context only; `file_path` refused,
  `n` clamped to 100k chars, pattern at most 512 chars) and `FINAL` are bridged, plus a sandbox-side `FINAL_VAR`.
  Every other callable the REPL exposes is present in the REPL and DENIED here with a `PermissionError` that is
  logged. The REPL's per-snippet tool-call budget and wall-clock deadline are enforced on every callback.
* **Same refusals.** The REPL's own AST gate (`parent._validate_code`) runs before Monty sees the code.
* **Limits.** Monty `max_feed_duration_secs` = the REPL timeout, `max_memory` (default 256 MiB), recursion depth,
  suspension count, a pool `request_timeout` backstop, and the REPL's SIGALRM deadline around the whole cell (this
  is what interrupts a slow host callback, because Monty's clock stops while the sandbox waits on the host).
* **Terminal semantics.** A timeout is terminal for the parent exactly as in the REPL; a memory trip or crash
  discards the worker and the next cell resumes from the last synced checkpoint.
* **Worker confinement (defense in depth).** Monty documents itself as a language-level sandbox with no seccomp or VM.
  The adapter launches the worker through a private launcher that execs the existing SEC-EXEC shim
  (`exec_sandbox.wrap`), and FAILS CLOSED (`MontyUnavailable`) if that cannot be applied. Verified on the worker:
  `Seccomp: 2`, `NoNewPrivs: 1`, empty environment (no API key reached it).

## The twelve cases

Copied verbatim (code, context, config) from `tests/unit/test_repl_environment.py`; each carries the original test's
assertion as an oracle. Selected to cover the slice's operation classes: peek/grep persistent feeds (C01-C03),
list/dict/comprehension and try/except (C04-C05), error capture (C06-C07), artifact, FINAL and state continuity
(C08-C11), and one security refusal (C12).

| Case | Source test | Cells | out / err / final / state identical to CPython | CPython warm | Monty warm (median of 50) | Monty p95 | contained CPython child |
|---|---|---|---|---|---|---|---|
| C01 | `TestPeekFunction::test_peek_default` | 1 | yes | 6.293 ms | 6.481 ms | 9.553 ms | n/s (needs peek) |
| C02 | `TestGrepFunction::test_grep_regex_pattern` | 1 | yes | 5.463 ms | 5.685 ms | 8.925 ms | n/s (needs grep) |
| C03 | `TestGrepFunction::test_grep_truncation` | 1 | yes | 5.547 ms | 5.771 ms | 9.007 ms | n/s (needs grep) |
| C04 | `TestAllowedBuiltins::test_dict_comprehension_works` | 1 | yes | 0.119 ms | 0.194 ms | 0.218 ms | 33 ms/cell |
| C05 | `TestAllowedBuiltins::test_try_except_works` | 1 | yes | 0.115 ms | 0.216 ms | 0.287 ms | 31 ms/cell |
| C06 | `TestREPLBasicExecution::test_runtime_error_captured` | 1 | yes | 0.031 ms | 0.173 ms | 0.360 ms | 32 ms/cell |
| C07 | `TestREPLBasicExecution::test_syntax_error_captured` | 1 | error class only (see below) | 0.012 ms | 0.086 ms | 0.105 ms | 31 ms/cell |
| C08 | `TestREPLBasicExecution::test_artifacts_can_be_modified` | 2 | yes | 0.170 ms | 0.348 ms | 0.437 ms | n/s (state) |
| C09 | `TestFinalFunction::test_output_before_final_captured` | 1 | yes | 0.039 ms | 0.132 ms | 0.213 ms | n/s (FINAL) |
| C10 | `TestFinalFunction::test_final_var_returns_artifact` | 2 | yes | 0.114 ms | 0.310 ms | 0.400 ms | n/s (FINAL_VAR) |
| C11 | `TestREPLState::test_state_persists_across_executions` | 2 | yes | 0.153 ms | 0.332 ms | 0.368 ms | n/s (state) |
| C12 | `TestSecuritySandbox::test_import_os_blocked` | 1 | yes (byte-identical refusal text) | 0.005 ms | 0.005 ms | 0.006 ms | 31 ms/cell, no refusal |

Reading the table:

* C07 is the only difference: CPython says `SyntaxError: '(' was never closed (<string>, line 1)`, Monty says
  `SyntaxError: unexpected EOF while parsing`. Same class, different wording, and the model sees the text.
* "Warm" is per pass over the case's cells, in-process, perf_counter, one host, 50 repetitions. The adapter costs
  about +0.1 ms per cell (a feed plus a state read-back round trip to the worker); it is not significant next to the
  parent's own work. C01-C03 are about 6 ms in BOTH arms because the parent's `peek`/`grep` record a research-context
  embedding (`_track_research`); that cost is shared, not Monty's. C12 is a gate refusal that never reaches Monty.
* The contained CPython child costs 31-35 ms per cell because every cell is a cold child; it keeps no state and has no
  REPL surface, so it is marked not supported (n/s) for 7 of the 12 cases, not scored as a failure.
* Cold start: parent env 0.2 ms; adapter first cell 20 ms with the confined worker (the shim's Python start-up),
  1.8 ms with an unconfined worker, 1.0 ms on a pre-warmed shared pool. Worker RSS about 11 MB. These are first
  measurements on one machine and are not vendor-comparable.

## Security refusals (10 probes)

| Probe | CPython warm (REPL) | Monty adapter (gate on) | Monty native (gate off) | existing contained CPython child |
|---|---|---|---|---|
| `import os` | refused (AST gate) | refused, same text | ran, no error (stub module) | ran |
| `from os import system` | refused | refused, same text | ImportError | ran |
| `__import__('os')` | refused | refused | NameError | ran |
| `eval('1+1')` | refused | refused | **ran, printed 2** (sandboxed) | **ran, printed 2** |
| `exec('x=1')` | refused | refused | ran (sandboxed) | ran |
| `open('/etc/hostname')` | refused | refused | PermissionError (no FS by default) | **read the file** (Landlock only fences credentials) |
| `''.__class__` | refused | refused | AttributeError | ran |
| `getattr(object,'__class__')` | refused | refused | AttributeError | ran |
| `object.__subclasses__()` | refused | refused | AttributeError | **listed subclasses incl. `os._wrap_close`** |
| `globals()` | refused | refused | NameError | ran |

Reading: with the gate on, the refusals are byte-identical to the REPL's because they ARE the REPL's. Monty alone
refuses LESS syntactically (eval, exec and `import os` are accepted and run inside Monty) but denies by capability
(no filesystem, no `os.system`, no dunder attribute access, no `__import__`). The existing contained CPython child
refuses nothing at the language level; it relies entirely on kernel containment (no sockets, credential read-deny).
Neither is an escape suite: Monty's only published escape is intake-1870 (a v0.0.16 use-after-free) and its repo
documents a bounty and fuzzers, not an escape suite. The Landlock+seccomp confinement on the worker is therefore the
reason a Monty escape would not be a free credential or network path.

## Denied and delayed callbacks, terminal resource errors, disposal

| Scenario | CPython warm | Monty adapter |
|---|---|---|
| `peek(.., file_path=..)` / `grep(.., file_path=..)` | returns `[ERROR: Path not in allowed locations]` text | `PermissionError`, logged (context-only lane) |
| `run_shell('echo hi')` | runs (contained child) | `PermissionError: not available in the Monty lane`, logged |
| `web_fetch`, `list_dir` | run (fail on path or network) | denied, logged |
| unknown name | `NameError` | `NameError` (identical) |
| tool-call budget 5, loop of 20 `peek` | `RuntimeError: Tool-call budget exceeded: more than 5 tool calls in one snippet (at peek); ...` | identical (the parent's own `_charge_tool_call`) |
| callback delayed 0.2 s inside a 5 s deadline | ok, 207 ms | ok, 208 ms |
| callback delayed 1.6 s, deadline 1 s | `REPLTimeout`, 1000 ms, env terminal, follow-up refused | `REPLTimeout`, 1001 ms, env terminal, follow-up refused, worker dead |
| catastrophic regex in bridged `grep`, deadline 1 s | `REPLTimeout`, 1000 ms | `REPLTimeout`, 1001 ms |
| `while True: pass`, deadline 1 s | `REPLTimeout`, 1000 ms, terminal | `REPLTimeout`, 1001 ms, terminal, worker dead |
| dynamic memory bomb | NOT RUN (in-process; the static guard sees only literal sizes) | `MemoryError: memory limit exceeded` at 64 MiB, parent NOT terminal, new worker, old worker dead, state resumed (`[1, 2, 3] {'k': 'v'}`) |
| same bomb in the contained child | `MemoryError` at its 16 GiB RLIMIT_AS | n/a |
| runaway recursion | `RecursionError`, REPL alive | `RecursionError`, worker alive |
| 10 MB output flood (200k lines) | whole 10.2 MB accumulated in memory, then spilled (38 ms) | bounded to about 1 MB before accumulation, then spilled (116 ms) |
| SIGKILL of the captured worker | n/a | `[ERROR: Monty worker crashed ...]`, next cell resumes, state intact |
| disposal | n/a | `close()` proves every captured pid dead (process state, not `kill -0`); verified in tests and harness |

Notes: the slow-callback parity needed the REPL's SIGALRM around the whole cell. Without it the adapter ran a 1.6 s
callback to completion (1608 ms) because Monty's clocks pause while the sandbox waits on the host (also confirmed
directly: a 1.5 s host callback under a 1 s pool `request_timeout` returned normally). A time limit inside Monty leaves
no guarantee about the heap, so a timeout kills the captured worker and the cell is terminal for the parent, as in the
REPL. Memory limits are allocator bytes, not RSS, and are opt-in in Monty; the adapter always sets one, and an OS or
cgroup limit would still be required (the confined worker has the shim's 16 GiB RLIMIT_AS only).

## JSON checkpoint export and import through the parent boundary

Cells defining list, dict, str, int and an `artifacts` entry ran in worker A. `adapter.checkpoint()` (the parent's own
checkpoint) matched CPython's `user_globals` and `artifacts` exactly, serialised as 1,493 bytes of JSON; a new parent
and adapter `restore()`d it into a NEW worker, continued (`cnt += 10`, new artifact) and the state was correct. No
Monty dump bytes were used. State that is not JSON-compatible (a user-defined function, a set) is REPORTED in
`adapter.unavailable` with a reason and not carried; it lives only in the worker that made it.

## Coverage limits and refusals (recorded, not failures)

| Cell | CPython REPL | Monty adapter |
|---|---|---|
| `statistics.mean(...)` (preloaded in the REPL) | works | `NameError` (no `statistics`) |
| `numpy` / `scipy` (preloaded when installed) | works | `NameError` (no third-party packages) |
| `import fractions` | works | `ModuleNotFoundError` |
| generator function | works | `NotImplementedError` (no generators) |
| user exception subclass | works | `NotImplementedError` (no class inheritance) |
| set, sorted, f-string; `json`; `re`; `collections` | works | identical output |

Other adaptations and divergences: (1) `FINAL` is recorded host-side, so `try: FINAL(x) except BaseException: ...`
still finishes (in the REPL a swallowed `FINAL` is lost); (2) structured mode, context bundles and the CM-5 handle
store are refused at construction; (3) a worker is always a subprocess (Monty has no in-process mode), so the first
cell pays a spawn; (4) the lane returns nothing for a trailing expression (the REPL does not echo either); (5) state
crossing the boundary is JSON only; (6) `execute()` raises `MontyUnavailable` rather than returning an error result
when worker confinement cannot be applied (fail closed, loud).

## Disclosure

The first harness run used a contained-child memory bomb of 2e9 elements (16 GB), just under that child's 16 GiB cap,
and it actually allocated for a moment (host has 1.1 TB, no effect observed). The harness now uses 3e9 (24 GB), which
fails at mmap, and says why in a comment. No in-process CPython memory bomb was ever run.

## What this does not show

Twelve unit-test snippets, one host, no model in the loop, microbenchmarks. It says nothing about the share of real
worker-role cells that fit Monty's subset, nothing about model behaviour on code-mode tasks (the codemode intake's
sources are mixed on that and none measures a 7B-35B quantized worker), and nothing about whole-REPL fitness. Intake
figures for Monty start-up (0.8 ms) are single-vendor and were not reproduced as such.

## Remaining

1. Wiring: no call site constructs the adapter (`maybe_monty_adapter` is the entry); choosing where it plugs into
   `create_repl_environment`/the API per-request path needs the D-f lease and request lifecycle (not done here).
2. A corpus measurement: replay recorded REPL cells (code log) through the gate and Monty to get the in-subset share.
3. Whether to prefer Monty only for cells that pass a static "in-subset" screen, with CPython for the rest.
4. A real cgroup memory/process limit for the worker (RLIMIT_AS is the only OS cap now); D-RI-CONTAIN-T2 overlap.
5. Belief wiring for the findings above (committed-conclusion register); not done in this slice.
