# Machine-frame transport projection — source proof

Source base: `0c4833bc3b0a38275e0b41433f9be68d6cf9b11f`. Approved source patch SHA-256: `9ad8cb47beced1bcb2bb23812223432d2cf003bdb517bc2a6336e8a51c68986e`.

The existing snapshot and SSE routes accept optional `view=machine`; the default full response is preserved. The machine page opts in from ROOT. Projection omits only an equal `region_locks.live_frame` alias; the complete canonical `live_frame`, lifecycle history, identities, timestamps, UNKNOWN/degradation and freshness fields remain intact. Missing or divergent aliases remain visible, including JSON scalar type and negative-zero differences. Comparison is iterative so a native 650-level frame can serve through both full and machine routes without extra Python recursion. No native producer, service setting, kernel or health fold changes.

Verification: `deep-after-suite.stdout` has 143 passing native projection/route, telemetry, region-lock, freshness, CORS and panel controls; `deep-before.stdout` retains the reproduced recursive-comparison failure. Native project imports were confined to the bounded fixture. Lint passes for all three changed Python files and both new files pass formatting. `make gates` exits 2 at inherited ShellCheck warnings: pristine-base and candidate outputs are byte-identical. Whole-dashboard formatting debt also matches the base. These are explicit inherited failures, not a full green gate claim.

`fixture-size-controls.json` contains deterministic source-fixture UTF-8 sizes only. No live payload, bandwidth, performance or deployment result is asserted. The native producer still builds and serializes the full frame before projection; its production build cost is unchanged by this work. R6 receipts retain the two-CPU, OMP=1, sampled owned-tree memory and deadline checks. ROOT's 15 existing/new page controls are retained with its companion source commit.

The source was committed through the normal canonical pre-commit wrapper and APP extras. Exact staged-tree PII custody remains private; final publication receipts bind its verified digest to the source tree. Shared canonical HEAD/index were not used for staging. The isolated Git admin is `/mnt/raid0/llm/tmp/codex3-machineframe-20261031/app.git` with its own index; no additional full checkout was created. Artifact-local whitespace attributes preserve significant unified-diff context spaces in the original patch/format logs.

[no-belief: software transport projection and deterministic regression controls; no live measurement or deployment]
