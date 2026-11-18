# D-RI-MONTY-LANE results: the Monty REPL execution lane

Date 2026-11-18. Orchestrator branch `feat/monty-lane` (rebased on `072d1583`). CPU only, offline, no inference
servers. Lane: `src/repl_environment/monty_adapter.py`. Tests: `tests/unit/test_monty_adapter.py` (77) and the
suite-parity plugin `tests/monty_lane_parity.py`. Corpus replay: `scripts/benchmark/monty_lane_corpus.py`, raw
per-cell data in [`monty-lane-corpus.json`](monty-lane-corpus.json). Runtime: `pydantic-monty` 1.6.0 in the private
venv `/mnt/raid0/llm/tmp/monty-slice-venv` (the shared venv was not touched). Predecessors:
[`monty-slice-results.md`](monty-slice-results.md) and [`monty-corpus-share.md`](monty-corpus-share.md).

## Contract

With `monty_repl` ON, a cell that CPython would run must not fail. Every cell is routed:

1. **Static screen** (AST, before anything runs). A cell goes to CPython if any of these holds: the REPL AST gate
   refuses it; CPython cannot parse it; it uses something Monty lacks or renders differently (generators, `del`,
   class inheritance, method decorators, set displays and `set()`/`frozenset()` (iteration order), `print(file=)`,
   `except*`, async, imports outside json/math/re/collections/itertools/functools/datetime/copy/decimal/dataclasses/
   typing, `OrderedDict`/`ChainMap`, `statistics`/`fractions`/numpy/scipy); it reads a name nothing defines (CPython
   resolves a callee before its arguments, Monty does not, so the error text would differ); it reads parent state
   that cannot cross; it calls `remember`/`SCRIPT`/`checkpoint_create`/`checkpoint_restore`, which read the cell's
   namespace mid-cell; the parent is in structured mode, has a context bundle, or the child lane
   (`ORCHESTRATOR_REPL_CHILD_LANE`) owns execution; the flag was switched off; or the lane is unavailable.
2. **Monty attempt** in a warm, confined worker. **Every** REPL callable is bridged. The host callback calls
   `parent._globals[name]`, the same object CPython code calls. That object is the counted wrapper (per-snippet
   budget), plus per-tool argument validation when it is on, plus each tool's own path checks, write dedupe and
   untrusted-text handling. The lane does not reimplement any of it. Tool prints are captured into the cell output as
   under CPython. `FINAL_VAR` goes through the parent's validated global against the cell's live artifacts.
3. **Fallback** on a Monty gap: `NotImplementedError`, `AttributeError`, `TypeError`, `ImportError`, `SyntaxError`,
   `RecursionError`, `MemoryError`, a `NameError` on a name CPython has, Monty's compute limit (half the REPL
   deadline), a worker crash, a checkout failure, an unconvertible callback argument or result, or state that cannot
   cross back. The cell re-runs on CPython from the pre-cell state. Monty writes nothing to the parent until the cell
   has finished and its read-back has validated, so the pre-cell state is intact. Callbacks the attempt already made
   are **replayed** from a journal (same name and arguments, in order, give the recorded result or exception and
   output) and charged to the budget. A side-effecting tool therefore runs once. User errors (assertions,
   `ZeroDivisionError`, `KeyError`, `ValueError`, a tool's own exception, ...) are not retried; they surface with
   CPython's text, including CPython's `AssertionError` form (Monty appends the assert source; the lane strips it).

**State.** Values cross both ways as plain trees (None/bool/int/float/str/bytes, list/tuple/set/frozenset/dict).
Definitions cross **by source**: top-level `def`, `class`, `name = lambda` and `import` statements whose evaluation
calls nothing (decorators other than lru_cache/cache/dataclass/total_ordering/wraps, defaults or class bodies that
call something are excluded). They are re-executed in the parent after a Monty cell and re-fed to every new worker.
Definitions made by CPython-routed cells are learned the same way. Names a function declares `global` are tracked.
The lane mirrors what the worker holds and pushes only the differences before each Monty cell. Those differences
come from CPython cells, `restore()`, tools that mutate artifacts, and in-place mutation. A name deleted on the
parent side forces a fresh worker, because Monty has no `del`.

**What cannot cross**, and stays on CPython: closures and other functions that are not top-level definitions,
instances of user classes, `Counter`/`defaultdict`/`deque` objects, datetime/Decimal values, module objects bound by
a non-import statement, and definitions Monty cannot execute (recorded in `uncarriable`). A Monty cell that creates
one falls back. A later cell that reads one is screened to CPython, so the value lives only in the parent.

**Worker limits.** Each worker runs under the SEC-EXEC shim (Landlock + seccomp, empty environment, `cd /`) with
`RLIMIT_AS` (default 1 GiB), plus EITHER an owned cgroup v2 leaf with BOTH `memory.max` and `pids.max` (when a
writable, delegated cgroup with both controllers exists; it reuses `child_lane._Cgroup`) OR `RLIMIT_NPROC`. On this
host `/sys/fs/cgroup` is writable but delegates no controllers, so the lane uses rlimits. The caps are **verified on
the live worker** after its first reply: `/proc/<pid>/limits`, cgroup membership, and `Seccomp: 2`. If verification
fails, the lane is disabled for that environment and every cell runs on CPython (`ORCHESTRATOR_MONTY_CGROUP=require`
makes the cgroup mandatory). `RLIMIT_NPROC` is per-uid, so the cap is "host task count at spawn + 1024". An earlier
version counted once at pool start. The count drifted on this shared host, the runtime's own thread spawn failed
(`Resource temporarily unavailable`, exit 101), and the fallback absorbed it. The count is now taken at each spawn.

**Pool.** There is one warm pool per process (`ORCHESTRATOR_MONTY_POOL_WARM`=2, `_MAX`=8) with
`max_checkouts_per_worker=1`, so a worker serves exactly one environment (request) and then retires. A cell reuses
its environment's worker. When every worker is held, the checkout times out after 1 s and the cell runs on CPython.
Environments are reference cycles, so before that the lane runs one `gc.collect()` and retries. A weakref finalizer
returns a worker when its adapter is collected, and `shutdown_pools()` runs at exit and also removes the cgroup and
the launcher directory.

**Wiring.** `create_repl_environment` and the per-request path (`chat_pipeline/repl_executor.py`) call
`install_monty_lane(env)`. With the flag OFF it returns the environment untouched and never imports the lane (tested
in a subprocess). With the flag ON it installs per-instance `execute`/`restore`, so callers keep their
`REPLEnvironment`. It fails open (CPython kept, a `lane_unavailable` telemetry row written) when the package is
missing, containment cannot apply, the pool will not start, or construction raises. It is not installed on a
RestrictedPython environment. The flag is re-read for every cell.

**Telemetry.** One JSONL row per cell goes to `ORCHESTRATOR_MONTY_LANE_LOG` (default `logs/monty_lane.jsonl`):
route, screen reason, fallback reason and detail, monty/cpython/total ms, worker reuse, sessions, refusals, callbacks,
replayed/diverged counts, a code sha (no code), task id and role.

## Route and fallback shares

### T1 corpus, replayed through both arms (210 unique cells)

The corpus is the same one T1 used: tap sessions in turn order, plus role-labelled eval cells. Every REPL tool except
the pure context helpers (peek, grep, context_len, chunk_context, FINAL, FINAL_VAR) is an inert recording stub, the
SAME stub in both arms. Nothing was written, executed or sent to a model, and the comparison is byte parity.

| Stratum | cells | Monty route | screened out | screened in | fell back (of screened in) | output/error/FINAL identical | state identical | CPython ok, lane failed |
|---|---|---|---|---|---|---|---|---|
| all | 210 | 157 (74.8%) | 49 | 161 | 4 (2.5%) | 209 | 210 | **0** |
| A tap | 98 | 77 (78.6%) | 19 | 79 | 2 (2.5%) | 97 | 98 | 0 |
| A tap, non-fixture | 53 | 40 (75.5%) | 12 | 41 | 1 (2.4%) | 53 | 53 | 0 |
| B eval (role-labelled) | 112 | 80 (71.4%) | 30 | 82 | 2 (2.4%) | 112 | 112 | 0 |

Screen-out reasons (all): `name:undefined` 23 (cells calling TOOL/llm_* etc. in an environment without a registry or
primitives; CPython raises NameError for them too), `syntax:cpython_rejects` 17, `feature:set_order` 5, `ast_gate` 3,
`import:time` 1. Fallbacks: `runtime:TypeError` 2, `runtime:AttributeError` 2. Every one re-ran on CPython with the
same result CPython gives. The single non-identical cell prints 12 kB and spills. The spill notice names a
per-environment random directory (`repl_output/<id>/turn_1.txt`), which differs between ANY two environments,
CPython included. It is not a lane divergence.

Against T1 (37.2% of non-fixture tap cells ran unchanged, with the bridge limited to peek/grep/context_len/FINAL):
75.5% of the same cells now run on Monty, and **every** cell gives CPython's result. The bridged-callable failure
class (57 cells in T1) is gone. Before the undefined-name screen was added, one eval cell
(`llm_batch([... peek(1000, 1000)])` with no `llm_batch` defined) failed with a different error than CPython: Monty
raised a `TypeError` from the argument, CPython a `NameError` from the callee. Both failed. The screen now routes
such cells to CPython.

**Latency (corpus, this host).** A warm Monty cell (the environment's worker reused) costs about 0.5 ms against
about 0.2 ms on CPython. The FIRST Monty cell of an environment costs a median 36 ms (p95 68 ms): checkout, prelude,
cap verification and the state push, plus a respawn under the shim when the replay outruns the 2 warm workers. Most
corpus groups are single cells, so the corpus median is cold. A fallback cell costs the attempt plus the CPython run
(31 to 48 ms here, mostly the cold start). Raise `ORCHESTRATOR_MONTY_POOL_WARM` for burstier traffic.

### Slice harness rerun (`monty_slice_compare.py`, the twelve cases)

12/12 oracles hold and 12/12 are byte-identical to CPython. C07 is now identical too: CPython parses the cell and
Monty does not, so the cell falls back. Warm per-pass ms on CPython vs the lane: C01 8.09 vs 8.48, C04 0.18 vs 0.64,
C08 0.24 vs 1.09, C11 0.22 vs 1.05, C12 0.05 vs 0.19. The lane adds about 0.4 to 0.9 ms per pass. That is more than
the slice's 0.1 to 0.2 ms, because each cell now pushes differences, reads state back for the diff and builds the full
callback table.

### Suite parity: the existing suites through the flag-on path

`pytest -p tests.monty_lane_parity` installs the lane on EVERY `REPLEnvironment` the suites build, with the flag
forced on, and the suites' own assertions are the oracle. The suites were test_repl_environment, all eight codemode
suites, tests/security, test_repl_child_lane and test_repl_tool_arg_validation: **458 passed, 6 xfailed, 0 failed**.
The lane served 355 cells: 206 on Monty, 149 on CPython (screen: `parent:child_lane` 109, `ast_gate` 19,
`parent:timed_out` 5, `parent:structured_mode` 3, tool-reads-state 3, syntax 2, other 4). Of the 211 cells screened
in, 5 fell back (2.4%): `runtime:NameError` 4 (`__builtins__` probes) and `state:not_transferable` 1 (a `Counter`).
One white-box test (`test_alarm_timeout_marks_terminal_and_refuses_later_execute`) injects its timeout by hooking
`builtins.exec` on `repl._globals`, which tests the CPython executor itself, so its environments stay on CPython and
the plugin counts it.

Flag OFF, shared venv, same suites plus test_monty_adapter: **462 passed, 62 skipped (the Monty-only tests), 6
xfailed**. test_monty_adapter flag on: **77 passed**.

## Bugs found and fixed while building it

* A tool exception subclass (`ToolArgumentError`) reaches Monty as its builtin base (`ValueError`). Host errors
  were matched by exact type and so got the wrong text. They now match by MRO.
* The sandbox-side `FINAL_VAR` skipped the parent's validated `FINAL_VAR` and dropped its return value. It is now
  bridged.
* Host-callback prints went to the process's stdout instead of the cell output. They are now captured and replayed.
* The worker pid was verified before the launcher had exec'd (the pid was still bash's). Verification now runs after
  the first reply.
* `RLIMIT_NPROC` counted once (see above) caused rare worker panics.
* Pool exhaustion blocked forever. There is now a checkout timeout, GC and retry, then fallback.

## Known residuals (not failures)

* A cell that never finishes costs 1.5x the deadline: Monty gets half, then CPython gets the full deadline before
  the terminal timeout. A slow HOST callback is still terminal at the deadline, as under CPython. If a cell spends
  more than 0.5x the deadline computing on Monty and also waits on host callbacks, the sum can reach the REPL's
  SIGALRM while CPython would have finished. This is unmeasured and would need a cell near the deadline.
* Replay is positional. A cell whose callback sequence depends on nondeterminism diverges (`replay_diverged` is
  logged) and then makes the remaining calls live.
* A `MemoryError` falls back to CPython in-process. That is required by the contract, and it gives up containment
  for that cell (the bomb runs where it would have run with the flag off).
* The corpus has no worker-role cells (see T1). These shares bound nothing about served worker roles.

## Dependency decision

`pydantic-monty` **stays an optional extra** (`monty`). The flag is default off and the lane fails open without the
package, so a default install loses nothing. The wheel ships a native binary plus a Rust worker whose content is not
proven to be the reviewed commit (no cargo on this host; see the slice provenance caveat), and adding that to every
default install of the API venv is a supply-chain decision the measurement does not need. Enabling the lane for an
autopilot trial requires installing the extra into the API venv first; without it the toggle is a logged no-op
(`lane_unavailable`). `pyproject.toml` and `uv.lock` are unchanged.
