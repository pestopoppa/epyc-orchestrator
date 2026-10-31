# MHS-12 eval-identity screen

Source-only implementation of the original MHS-12 task in the root PromptForge
mutation safety handoff. No inference, live activation, reload, promotion gate,
measurement threshold, claim grading rule or journal mutation is part of this work.

The existing `screen_static_safety()` seam in
`src/autopilot_core/mutation_guards.py` now rejects added executable accesses to
`workload_class`, `scoring_method`, `batch_id`, request/metadata `suite` and `tier`
keys, and the literal `eval_batch`, with `MutationEffect.UNSAFE` and a reason.
All five native allowlisted targets are Python. The guard uses their parsed ASTs;
comments, documentation strings and write-only names do not become identity reads.
Explicit name/attribute/key reads, `get`/`pop`/`setdefault` and literal-key
`getattr`/`hasattr` are screened. Expression multiplicity, lexical scope and
enclosing control context preserve existing equal accesses while refusing a new
copy, modified use or branch involving a grandfathered field. Existing MHS-2
import/capability grandfathering and its other restrictions are unchanged.

For scopes with `force_role` handling, the guard preserves force-bearing
statements, assignments to selected role receivers (names, attributes and keys,
including explicit local propagation), abrupt exits and their enclosing control
context. Removing forced selection, resetting `force_role`, overwriting the
selected role, adding an early return/raise, changing a condition or wrapping the
existing forced path in `if False` is refused. Unrelated straight-line edits
remain possible; the whole function is not frozen.

This is a static explicit-access guard, not a general sandbox or a proof against
arbitrarily obfuscated dynamic keys, interprocedural aliases, infinite loops,
indirect side effects or whole-program behavioral changes. It intentionally
refuses changes to existing identity expressions and uncertain forced-routing
control flow; a benign semantic rewrite of those expressions can be refused.
Request/metadata `suite`/`tier` recognition is syntactic (`request`, `req`,
`metadata`, `request_metadata`); it does not infer object types or arbitrary aliases.

The existing native caller path is unchanged:
`PromptForge._screen_code_mutation` → `screen_static_safety` →
`propose_code_mutation` sets the unsafe effect/reason, invalidates the candidate
and restores its original content → `_action_code_mutation` returns before eval.
The isolated apply path rejects the invalid mutation before `ctx.apply_file`.
The new test drives that real proposal/action/apply path with the author and
metadata capture replaced by model-free fixtures.

Validation source base: APP `0c4833bc3b0a38275e0b41433f9be68d6cf9b11f`.
The exact original guard accepted both a native chat no-op and a chat-file branch
on `request.workload_class == 'eval_batch'`, reproducing the gap before the fix.
Native allowlist no-ops, added identity reads, context/multiplicity grandfathering,
direct forced-routing bypasses and the actual native caller path are covered by
`tests/unit/test_prompt_forge_eval_identity.py`.

The broad controls include mutation safety, guard re-export/import safety,
transfer safety/risk, and mutation extraction. One inherited S2 source-scanning
test fails against both original and candidate guards: it flags the existing
`prompt_forge_mod.CODE_MUTATION_ALLOWLIST` patch at line 560 of the unchanged
`test_prompt_forge_mutation_safety.py`. The final passing broad run explicitly
deselects that one test; it is not a claim that the entire repository suite passes.
The retained baseline log records the independent original-guard failure.
Committed log text removes trailing horizontal whitespace for Git hygiene; the
unmodified supervisor stdout remains in the retained task scratch.

Execution uses the original standing pair-30 mutex and R6 admission, relocated to
SMT CPUs 126–127 only after all four physical quarters were covered by correctness
grants. OMP/OpenBLAS/MKL threads are 1; the sampled supervisor enforces a bounded
owned process tree, at most 8 GiB and 900 seconds, and refuses timing grants.
No models are loaded. The private source fixture/admin do not edit the shared
canonical HEAD/index or the protected reusable orchestrator lane.

Broad controls before the narrow force-write extension: **272 passed, 1 deselected** in 64.85 seconds. Final supervisor:
66.230 seconds, sampled owned-tree peak 1,549,709,312 bytes, 653 samples,
children drained, no monitoring/admission failure. The final focused native
mutation-safety/S1 run passed 154 tests. At that pre-force-write-delta revision the new module contained 48 cases.
`ruff check` on both changed Python files and the new test module's format check
passed; `git diff --check` passed. Raw command results are retained beside this file.

`make gates` was also invoked in the bounded source fixture. Its result is recorded
in `make-gates.log.txt`; that fixture omits repository shell/docs trees, so the
invocation does not establish complete-repository gate coverage. The native Python
tests and lint above are the relevant code-change controls. No full-repository
lint/test pass is claimed.

Final review extension: explicit literal-key `setattr`/`delattr` writes to
`force_role` are tracked separately from identity reads. The original MHS-2 screen
already rejects these calls in current native chat (neither capability exists
there). Exact original-guard controls also show that a legitimate grandfathered
unrelated `setattr`/`delattr` call licenses a later forced-role reset; the added
MHS-12 write screen closes that case. A grandfathered ordinary
`setattr(request, 'workload_class', None)` remains a write rather than a new read.
Before the ordering fix, that source passed **159 focused controls** (53 new cases plus existing native
mutation safety and S1 seam), in 5.08 seconds; final supervisor 6.512 seconds,
242,499,584 sampled peak owned-tree RSS bytes, 65 samples, children drained.
Touched Python lint and the new test format check passed. The broad suite was not
rerun after this minimal extension; its exact earlier source hashes are retained
in `source-sha256-pre-force-write.txt`, and final hashes in `source-sha256.txt`.

The scoped `make gates` invocation exited 2 on fixture Markdown checks. The fixture
stores the tracked AGENTS symlink blob as text and omits the repository lint
configuration and other trees. Its Markdown output is not evidence of a failure
of the complete original repository. This limitation does not affect the native
Python code/read set, which is materialized byte-for-byte from the stated base
except for the two changed files.

Final native validation ordering fix: deleting the real `_try_cheap_first` function
previously returned `REPLACE`/`missing public names` before reaching MHS-12. The
underlying guard already classified that deletion `UNSAFE`; the native pre-fix
counterexample is retained in `early-gate.stdout.log.txt`. PromptForge now computes
its existing static screen once after successful syntax parsing and gives only
MHS-12 violation reasons precedence over generic shrink/public-name rejections.
When no MHS-12 violation exists, the original generic reports keep their precedence;
the cached static report is reused at the original return seam. No new helper
interface, dispatch change or duplicate static analysis was added.

Final complete source: **163 focused tests passed**, including 57 new cases,
native entire-force-function deletion, catastrophic force removal, and unchanged
non-MHS12 generic `REPLACE` reports, each asserting one static-screen invocation.
Pytest 5.26 seconds; supervisor 6.476 seconds, 212,828,160 sampled peak owned-tree
RSS bytes, 65 samples, children drained. Lint passed on all three changed Python
files. The prior 272-test broad result predates the two narrowly reviewed changes;
the retained exact source hashes distinguish those results from this final source.
GitNexus reported LOW upstream impact for the native validation seam (three nodes
across Species/Autopilot); index currency is UNKNOWN because the index is stale.

Publication parent refresh: APP main advanced from the tested base to
`ec6f81769020d695b046895d654e36a39595ac81` while this change was reviewed. Its
machine-frame dashboard/proof changes are preserved. The exact guard, proposer,
new test and all five native allowlisted files have no peer delta; the tested
code/read set is unchanged. The private index merged the disjoint trees and the
complete MHS-12 diff was committed atomically atop that fresh peer, with the
normal canonical pre-commit wrapper and APP extras over the full staged diff.
