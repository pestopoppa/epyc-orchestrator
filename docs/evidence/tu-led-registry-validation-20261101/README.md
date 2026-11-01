# Request-local registry validation observation

This prospective, opt-in observation records the native registry's argument
validation result for individual registry invocations. It does not cover direct
REPL functions, parser failures, unknown tools, or permission denials as validated
calls. Invocation IDs carry no arguments or tool names. No tool-call repair link is
inferred from order or names. The query-memory registry and other instance-overridden
or subclassed registries are left untouched and recorded as unknown.
At most 4,096 calls are retained per episode; the observer counts omitted registry
invocations and marks the observation censored. Collection errors leave counts
unknown and also mark censoring. Censored episodes do not enter the first-registry
invocation fraction.

The TOC sidecar's optional diagnostic binds each observation to original trajectory
bytes, strict boolean supplied harness outcomes, and graph-episode usage. A requested
arm comparison matches explicit task IDs and keeps gains, givebacks, missing and
unresolved pairs separate. The first-registry-invocation rate is scoped to that
instrument, not all tool attempts. Token sums are graph-episode scope; unresolved
grade or incomplete token coverage is preserved. Verifier failure, no-valid-tool,
task-level first-try-valid, tool repair recovery and timeout buckets remain unknown.
This is partial instrumentation, not completion of TU-LED-1 or a quality score.

`tests-bounded-final.log.txt` records the model-free regression: 110 passed,
one legacy-default skip and one deselected API startup test. That startup test
failed in this projection because a tracked script is absent and would also touch
host CPU-lock files during lifespan startup. Scoped follow-up checks are recorded
separately; no inference or live report was run.

The shared-carrier canary failed against `/workspace` because its current TOC
adapter did not expose episode-token fields. The same canary passed against the
previously reviewed TOC adapter snapshot at
`/mnt/raid0/llm/tmp/codex3-code-app-20261101/root-canary`; its `toc_ab.py`
SHA-256 is `6817b910d9e7887b20c16ff887dc93b64d4c61f2072400d45c49b21cd2f16a38`.
Both original outputs are retained. This does not establish that the current
workspace adapter carries these fields.
