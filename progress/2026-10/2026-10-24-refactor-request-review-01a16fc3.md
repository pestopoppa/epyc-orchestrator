# Request-path review closeout — 2026-10-24

Owning main accepted the pre-first-chunk cancellation repair after reviewing the source diff and endpoint regression. Checkpoint before this closeout: `fb9362b4399bb578e742b9b2fe78d717529c3bad`; accepted source fix: `a1c695930`.

- Created isolated worktree `/mnt/raid0/llm/worktrees/refactor-01a16fc3-request-review` on `refactor/request-reviewed-01a16fc3`, based on `origin/refactor/request-path`.
- Merged accepted test isolation `origin/refactor/test-isolation-reviewed-01a16fc3` before testing. The merge kept request-branch content and isolation protections; no request source pins or NI63 pinned-closure files changed.
- Reviewed the nine O1 stage-one changes against `impl_O1.1.md` and the multirepo handoff. GitNexus on the worktree reported LOW impact for `openai_chat_completions` (0 upstream dependants) and `_iter_thread_call_chunks` (3 impacted symbols within the route).
- Found `/v1` streaming awaited its first generated event before returning `StreamingResponse`; disconnect polling existed only on the nonstreaming path. Added a regression with a blocked initial client-tool generation and disconnecting request. It failed before the repair because backend cancellation was not observed.
- Repair starts a disconnect watcher before the initial `anext`, joins it on cancellation, typed 503, and stream cleanup, and combines request-level cancellation with per-call cancellation so an in-flight backend callback sees the disconnect.
- The expanded targeted suite passed 1,011 tests with 3 skips. `ruff check`, regression formatting, and `git diff --check` passed. No full suite, model inference, service operation, or production change was performed.
- Proposed feature commits accepted by main: `6ac1e53d` regression and `a1c69593` fix. Small follow-on test formatting commits: `24223136`, `fb9362b4`. Isolation merge: `de12bf45`.
- Pushing only `HEAD:refs/heads/refactor/request-reviewed-01a16fc3` is authorized by owning main. No other branch or deployment is in scope. Keep the worktree for main cleanup.
