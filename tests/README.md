# Orchestrator tests

Run the complete test suite with `make test` (or `just test`). Pytest's sole
test root is `tests/`; this includes top-level `tests/test_*.py` files as well
as unit, integration, classifier, retrieval, vision, and security tests. Use
`make test PYTEST_ARGS="-n 1 -k name"` to narrow a local run.

## Isolation and opt-in tests

The root `tests/conftest.py` installs the hermetic kernel-store and runtime-store
fixtures before application imports. Unit tests also isolate inference locks,
contention reads, KV-slot I/O, and inference-tap files. Tests in a `real_*`
fixture opt-out marker deliberately leave one of those seams real so that the
test can control and verify it directly; those markers are not suite exclusions.
`asyncio_mode = "auto"` handles async test functions, so a separate
`@pytest.mark.asyncio` decorator is unnecessary.

The only markers excluded from pull-request CI are:

| Marker | Reason |
| --- | --- |
| `requires_server` | Requires a live llama-server process, which a hermetic hosted runner does not provide; opt in locally with `--run-server`. |
| `requires_live_models` | Requires the operator's live orchestrator models and model files. |
| `requires_materialized_trace` | Reads the host's materialized trace database rather than a test-owned fixture. |

These tests remain collected and are deselected explicitly by the CI command.
Other failures are run and compared by test ID against the checked-in baseline;
they are reported as known debt, while a new failing or erroring ID fails CI.
The baseline includes its integrated source HEAD and a minimum collection count.

## Test rules

- Use `tmp_path` or `tmp_path_factory` for files and child processes. Do not
  write test data into the checkout, shared runtime stores, or live governance
  paths.
- Use pytest's configured `pythonpath`; don't mutate `sys.path` in test modules.
- Don't use a bare `MagicMock` as a config object when code reads path fields.
- Prefer behavioral assertions over source-text checks. Keep meaningful
  independent expectations when a source clone is built from the active tree.
- Use `real_inference_lock_state`, `real_kv_slot_io`, or `real_inference_tap`
  only when the test itself needs that real host boundary; document the resource
  it controls.

The canonical commands and pytest configuration live in `Makefile`, `Justfile`,
and `pyproject.toml`. CI runs `make gates` and the complete `tests/` suite in
mock mode on Python 3.11.
