# Binding router wiring verification

## Source and integration commits

- Implementation commit: `053c4858466fff785b7d745e2dc752210904915b` (`git rev-parse 053c4858`).
- Fresh `origin/main` commit integrated without conflict: `33693ca71d8a6c0a06a5aef121ac6df62475eccb`.
- Local integration commit: `580132abd7b67dcf13dfd7092bbb4936a873d6dd` (`git rev-parse HEAD`).
- Branch: `codex/td29-m4-cache-20261030`.

## Native routing tests

Command executed under the required prep mutex:

```bash
flock -w 30 /mnt/raid0/llm/tmp/compute-flags/prep-locks/34-35.lock \
  python3 /mnt/raid0/llm/tmp/codex-ni-binding-router-20261031/run-native-tests.py
```

The runner set `OMP_NUM_THREADS=1`, set `PYTHONPATH` to the APP worktree, and invoked the shared interpreter:

```bash
/mnt/raid0/llm/epyc-orchestrator/.venv/bin/python -m pytest -q \
  tests/unit/test_binding_router_routing.py \
  tests/unit/test_routing_bindings.py \
  tests/unit/test_chat_routing.py \
  tests/unit/test_pipeline_routing.py
```

Result: **135 passed**, 1 dependency deprecation warning, 35.36 s wall time. The runner measured 38.62 s child user CPU and 0.21 s child system CPU; sampled peak RSS was 158,396 KiB. The full output is in [`native-tests.log`](native-tests.log).

The new test exercises the real API lifespan: flag-off leaves the state router unset and keeps the heuristic route; flag-on creates `state.binding_router`, the test registers a `RoutingBinding` through the existing API, and the real `select_initial_route` classifier path returns the override. Forced and explicit role requests preserve their existing precedence. The prior `test_binding_router_tripwire.py`, which asserted the feature stayed disconnected, was removed.

## Repository gates

`PATH=/mnt/raid0/llm/epyc-orchestrator/.venv/bin:$PATH make gates` passed schema, shellcheck, formatting, and lint gates. `git diff --check` passed.

## Scope

The enabled lifespan creates an empty process-owned registry. Rules are consumed only when explicitly registered with the existing `RoutingBinding` API. No automatic configuration loading, default rules, per-session isolation, or runtime flag activation was added. The feature flag remains default-off.
