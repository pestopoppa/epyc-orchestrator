# Session artifact query controls — 2026-11-01

The opt-in registry tool uses the existing request-local artifact registry and
inode-checked reader. Its private registry clone retains the original permission
and policy objects. The scoped-tool exception binds both the active private
registry and its exact tool; a nested or forged same-name registry cannot use it.
The shared graph spill writer and structured REPL dispatcher are unchanged.

`related-v2.log` records 79 passed and one legacy-default skip. One unrelated
chat-route test was deselected after its missing fake-request `headers` failure
was reproduced against the unmodified base in `base-chat-test.log`. The query
controls cover structured `TOOL` dispatch, one model call, exact-quote rejection,
session separation, nested and forged tools, role denial, replaced artifacts,
default-off behavior, oversized refusal, and pre-existing tool-name collisions.
`ruff.log` records the scoped lint check. Tests ran without models under the
28–29 prep mutex with the bounded supervisor shown in the logs.

The API's session lifetime, 12,000-character cap and exact-quote limits are
documented in `docs/chapters/03-repl-environment.md`. These are source and
synthetic regression controls; no live efficacy or model-quality claim is made.
