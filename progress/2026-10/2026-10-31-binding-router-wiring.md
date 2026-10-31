# Binding router wiring — 2026-10-31

The `binding_routing` flag was default-off and the routing override guard already existed, but API startup never constructed a `BindingRouter` and the heuristic route caller did not pass one. This left the flag unable to apply the existing `RoutingBinding` resolver.

The API lifespan now constructs an empty process-owned `BindingRouter` when `binding_routing` is enabled, stores it on `AppState`, and clears it during shutdown. The existing `select_initial_route` heuristic-classifier path passes that same instance only while the flag is enabled. Force-role, explicit-role, image, and long-context requests still return before that branch. The former AST tripwire was replaced with native routing tests that run the real API lifespan, add a declared override via the existing `RoutingBinding` API, and verify the real route result and force/explicit precedence.

The registry starts empty. Enabling the flag alone does not load or create routing rules; rules must be explicitly registered through the existing API. No default override, automatic configuration source, per-session isolation, or live flag activation is claimed. The feature remains default-off.

Source implementation commit: `053c4858466fff785b7d745e2dc752210904915b` (`git rev-parse 053c4858`). The branch also merged fresh `origin/main` commit `33693ca71d8a6c0a06a5aef121ac6df62475eccb` without conflicts; current integration commit is `580132abd7b67dcf13dfd7092bbb4936a873d6dd`.

Verification used the shared APP interpreter and the 34-35 compute mutex with `OMP_NUM_THREADS=1`: 135 targeted binding/routing tests passed in 35.36 s; child CPU was 38.62 s user + 0.21 s system; sampled peak RSS was 158,396 KiB. `make gates` passed, and `git diff --check` passed. Full command and raw test output are in [`artifacts/audit/binding-router-wiring-20261031/verification.md`](../../artifacts/audit/binding-router-wiring-20261031/verification.md) and [`native-tests.log`](../../artifacts/audit/binding-router-wiring-20261031/native-tests.log).

ROOT handoff checkbox application is prepared for main in [`ROOT-completion-prep.md`](../../artifacts/audit/binding-router-wiring-20261031/ROOT-completion-prep.md); this session did not write the ROOT checkout. Result marker: `[no-belief: software source publication and regression verification; no measurement]`.
