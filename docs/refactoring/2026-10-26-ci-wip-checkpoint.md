# CI package WIP checkpoint — 2026-10-26

This branch is a backup of an unvalidated O8 draft, not acceptance or production promotion. Base: `3ee8eea2ab514e8fac4c15c839d4ad1644e0ba38`. The draft adds PR/manual CI with pinned native carrier, gate comparators, dependency metadata parity, centralized pytest/server markers, NI63 current-source provenance, and no-op async-mark cleanup.

Private-cache online UV lock and isolated sync passed under Python 3.11. Pytest/Ruff version checks, workflow YAML parse and make-test dry-run passed. Formatting of the new helpers/tests completed. Preparation then stopped on one unused Path import; that import is removed, but renewed lint and focused pytest remain unverified. The replacement preparation wrapper was cancelled before region admission to freeze this backup; no test result is inferred from cancellation.

Full gates, accepted-integration baseline datasets, full-suite results, native capture/shared grade and coverage remain pending. No empty baseline or historical native attestation is supplied. The accepted integration SHA must be established before baseline generation. No P5 source changes or application behavior changes are authorized by this checkpoint.

Durable private evidence and exact snapshot paths: `/mnt/raid0/llm/tmp/refactor-01a16fc3/ci/checkpoint-20261026/` and `run4/`. The latter records the one-line source delta, verified source hashes, cancellation and isolated focused runner. `current-run.json` is the live resume pointer. Publication of this branch preserves work only.
