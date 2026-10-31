# USAGE-PROV-1 declaration carrier — 2026-10-31

Published the optional source-authored measurement declaration carrier in APP commit `0d01dc94c294`,
integrated onto APP main at `3d35934f7f29`. Omitted declarations remain `None`; begin copies the
bounded declaration and settled events preserve the same self-hashed dependency. Existing callers,
token counting, and cache physical accounting are unchanged.

The focused usage suites passed **41 tests** using the shared orchestrator environment. The companion
ROOT suite exercises the actual writer, strict reader, source ingest, and planning-context canary
with a synthetic-only declaration; it verifies no leakage to another model target.

No production caller currently authors the declaration, and no live inference/native usage capture
was run. The open `USAGE-PROV-1` enrollment still requires source-authored native capture and its
runtime canary. No wired/READY or measured-cost claim is made. [no-belief: software implementation and
synthetic correctness controls; no measurement]
