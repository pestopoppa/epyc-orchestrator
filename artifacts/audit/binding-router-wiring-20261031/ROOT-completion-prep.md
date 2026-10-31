# ROOT completion preparation — binding router

Prepared for MAIN to apply to the original ROOT handoff. Preserve the original task prose; only the checkbox state and completion evidence markers are supplied here.

```markdown
- [x] **`binding_router` is a parameter nobody passes, gating a whole feature.** `src/api/state.py:97` ✅ 2026-10-31 [implementer: epyc-orchestrator@053c4858466fff785b7d745e2dc752210904915b] [test-artifact: `repos/epyc-orchestrator/artifacts/audit/binding-router-wiring-20261031/verification.md`] [no-belief: software source publication and regression verification; no measurement]
```

Source implementation commit: `053c4858466fff785b7d745e2dc752210904915b`.

Proof: 135 native routing tests passed; raw output and command are in the linked test artifact. The feature remains default-off. When enabled at API lifespan, it constructs an empty process registry; the existing explicit `RoutingBinding` API supplies rules. No automatic configuration source or default override was added.
