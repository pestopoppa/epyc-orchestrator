# Active stack template launch gate — 2026-10-31

## Source

- Implementing commit: `c01e72bfe8d69a79c955f7d4b2bef5ea459bce96`
- Base: `3e6b7b2ec02eb30abbc0f6449cc61035032e8035`
- Changed source: `scripts/server/stack_commands.py`
- Regression coverage: `tests/unit/test_stack_template_launch_gate.py`,
  `tests/unit/test_orchestrator_stack_reload.py`

## Behavior

`cmd_start` resolves the explicit profile or the active profile from
`ORCHESTRATOR_STACK_PROFILE` (whose native default is `default`) and calls
`load_template` plus `validate_template` before the capacity gate, registry
compilation, stack-change launch gate, or launch actions. Invalid, missing, and
malformed active templates fail with status 1 before those effects. Migration
continues to validate its explicitly selected target through
`migrate_to_template`; it does not validate an unrelated active profile.
`--validate-only` returns after validation without startup effects.

## Current default template audit

The task in ROOT
`handoffs/active/numa-topology-cutover-resume-20260730.md:1305-1308` retains a
historical claim of nine errors. On the implementing checkout, the native active
profile resolution selected `default`; `validate_template(load_template("default"))`
returned `valid=True`, 0 errors, 0 warnings. No `stack_templates/default.yaml`
edit was made.

## Verification

Using `/mnt/raid0/llm/epyc-orchestrator/.venv/bin/python` with the implementing
checkout on `PYTHONPATH`:

- `pytest -q tests/unit/test_stack_template_launch_gate.py tests/unit/test_orchestrator_stack_reload.py tests/unit/test_orchestrator_stack_validate_only.py tests/unit/test_stack_templates_v2.py` — **115 passed**.
- `python -m ruff check scripts/server/stack_commands.py tests/unit/test_stack_template_launch_gate.py tests/unit/test_orchestrator_stack_reload.py` — passed.
- `git diff --check` — passed.
- The gate tests cover default/env/explicit profile resolution, valid continuation,
  invalid/missing/malformed no-effect refusal, and migration behavior.

No live service, model, or shared runtime configuration was changed.

## Task closure routing

The owning session applies the semantic checkbox to the original task at
`handoffs/active/numa-topology-cutover-resume-20260730.md:1305` and handles the
ROOT index/source-table closeout. Result marker: `[no-belief: software source
publication and regression verification]`.
