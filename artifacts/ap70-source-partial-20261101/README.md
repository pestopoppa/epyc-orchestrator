# AP-70 source partial: rotation-boundary observation contracts

Published orchestrator source: `a464a03eca67d78b55f40d2a49eae8c09188226e` on `main`.
Source was published by the serialized push tool from a clean candidate branch after fetching `origin/main`; fetched `origin/main` was the candidate's ancestor. This artifact commit packages the reviewed source evidence only and is separate from the source commit.

## Source scope

The source adds opt-in, default-off AP-70 observation instrumentation at the actual core-rotation boundary, an EvalTower diagnostic scoring route, and served-runtime/config identity attestation. It predeclares positive integer K and planned boundaries before pair reads; retains the exact outgoing/incoming indices and incumbent/runtime pins; computes descriptive `q(outgoing) - q(incoming)` from the existing quality extractor; and leaves failed, incomplete, unavailable, or nonfinite pairs explicit without a finalized mean. K=1 has an undefined SE; for K≥2 the summary uses completed matched pairs. The diagnostic is not connected to fitness, archive, baseline, promotion, candidate selection, or the normal rotation RNG. Default-off callers do not make an extra attestation request.

Runtime identity refuses missing, ambiguous, or drifting identity. Prompt-incumbent evidence is currently supported; nonprompt incumbent identity is explicitly unsupported. An API/load-balanced request reaching a different worker can conservatively make identity unavailable. A model path or prefix digest is not proof of model-content identity.

## Validation evidence

The final hosted workflow run `41292163512` and job `124805446809` completed successfully on Python 3.13.15: 50 tests passed; zero failed, errored, or skipped. The original GitHub artifact ZIP is retained byte-for-byte at `final-41292163512/ap70-observation-41292163512-1.zip` (artifact id `13077757826`, 14,312 bytes; SHA-256 `0001d8ba90d9afbd5e0bcc96bb07698d4d46d344e5cd40f2739c465961a50dc6`). Raw run, job, and artifact API JSON plus extracted JUnit, source manifest, stdout, exit code, dependencies and Python package freeze are in that directory. JUnit SHA-256 is `d60ca93aff005c1d33fa49d06b9f711b582683a57f762688a7ff66dca38066a8`.

Controls are preserved independently; their test counts are not additive:

- `intermediate-41289827708/`: initial green workflow, 34 passed, source commit `f0bbbad714809f70282e1e768e41e0acaf945d7f`, original run/job/artifact API metadata and essential result files.
- `failed-41291417354/`: failed correction control at source commit `fb0345763ff86c14c1def9363a48791fcebff82f`, 47 passed / 1 failed. The positive fixture used the wrong checkpoint root. Raw run/job/artifact API JSON, original workflow artifact ZIP (artifact id `13076658995`, SHA-256 `494998f3d59449c9660c6fff4c70166a822f8564ebeb530c7e7785d03351e572`) and essential result files are retained.
- `intermediate-41291640201/`: corrected fixture green, 48 passed at `3969ecf338c662cf765a22e548c2232c373d3b33`; original run/job/artifact metadata and ZIP (artifact id `13077468049`, SHA-256 `45bdafd9d6fb7852b3f2c61921141ff2b2be87d555bed3e29441fa4164d64cb7`) and essential results are retained.

## Status and limits

This is source-only partial work. No local pytest, model call, AutoPilot runtime, API reload, native K-boundary evaluation, ingestion, or empirical drop/efficacy/generalization claim was made. AP-70 remains open in the owning ROOT handoff until its full acceptance and prospective belief-wiring work are completed. The source instrumentation remains opt-in and future evidence must be captured at write time; no historical journal rows are retrofilled.
