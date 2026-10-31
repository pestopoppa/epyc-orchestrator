# ROOT handoff draft — UFH14-B6d

MAIN applies this draft in ROOT after APP publication is verified. The original active handoff and its checkbox remain untouched by this APP change.

Replace the original unchecked item with this checked item, retaining its acceptance wording:

> - [x] **UFH14-B6d — route escalations around a parked target.** (filed 2026-10-03, workspace-ec, the part of B6c that was not built) Today a request whose role is parked gets a fast 503 `role_parked`; nothing re-routes it. An escalation chain whose next hop is parked (for example `coder_escalation` or `architect_critic` on :8083) fails instead of trying the next eligible role. In the routing/escalation layer, treat a parked target like an unavailable one: skip to the next role in the chain when one exists, record the skip on the serving record, and return `role_parked` only when no alternative remains. Direct role calls and the passthrough keep the 503: a named-role caller asked for that role. Done when a test with :8083's roles parked shows an escalation served by the next eligible role, with the skip recorded. **Done 2026-10-31:** APP `d0059590fc1eb8e42ef290c4bfc9fb81783dfc91`; typed parked-only chain advance, serialized serving trace and v1 step record; force-pinned named target retains its typed refusal. Focused verification: 68 parked-role/v1 tests and 65 stage/primitive caller tests passed; `make gates` passed. [Evidence capsule](../../repos/epyc-orchestrator/docs/evidence/ufh14-b6d-20261031/CAPSULE.md). [no-belief: code-and-test verification only]

The source task asks MAIN to apply the original checkbox, so this file is a draft only. It creates no second handoff row and makes no measurement or model-quality claim.
