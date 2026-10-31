# Active stack template launch gate — 2026-10-31

`cmd_start` now validates the explicit or active stack template before its capacity gate, registry prelude, stack-change gate, or launch actions. Invalid, missing, and malformed profiles refuse before those effects. Migration still validates its named target, and `--validate-only` exits without startup effects. The active `default` template currently validates cleanly (0 errors, 0 warnings); the ROOT handoff's old nine-error inventory is stale, and no template YAML was edited.

Source implementing commit: `c01e72bfe8d69a79c955f7d4b2bef5ea459bce96`. Verification details and commands: [`artifacts/audit/stack-template-launch-gate-20261031/verification.md`](../../artifacts/audit/stack-template-launch-gate-20261031/verification.md).

115 focused tests passed; Ruff and `git diff --check` passed. ROOT task checkbox/index closeout belongs to the owning session. Result marker: `[no-belief: software source publication and regression verification]`.
