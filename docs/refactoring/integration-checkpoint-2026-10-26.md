# Isolated orchestrator integration checkpoint — 2026-10-26

Owning main reviewed and accepted this isolated candidate relative to immutable
base `84f6f6f15f713d21098dbad2b153d32f8cc13624`. The pre-checkpoint candidate HEAD
was `3ee8eea2ab514e8fac4c15c839d4ad1644e0ba38`; the checkpoint includes its
reviewed 17-file working diff. This acceptance does not establish compatibility
with subsequent main changes or authorize main promotion, deployment, runtime
activation, production-kernel changes, or P5 evaluator-era changes.

The expanded 111-file union collected 2,905 tests: 2,897 passed and 8 skipped.
All four selected source coverage floors were 100%. Final focused validation
passed two runtime-path fixture regressions and five NI63 controls. Registry
loader AST equivalence, all 32 actual-byte Git blob/SHA-256 source pins (31
unchanged against candidate HEAD), four exact P5 file comparisons, scoped Ruff,
formatting, compile and diff checks passed. Existing baseline debt was 18 Ruff
findings and 133 formatter paths; these were not broadly reformatted.

The final runner's historical result remains FAIL (exit 1, completed
2026-10-25T21:46:10Z): only its formatter baseline comparison failed. It compared
against mutable `origin/main`, which had advanced to `d825274d7078501f7dcc03fd253ba3b422dda0b4`,
while its stored baseline was generated at 07:59 UTC against accepted base
`84f6f6f1`. Exact reconstructed scope increased from 182 to 232 files, with 50
added paths and none removed. All 44 additional formatter failures were outside
the original scope, and every one is byte-identical to the accepted base. The
133 stored failing baseline snapshots also match accepted-base blobs. Intersecting
the existing run's failures with the original 182-file scope produces exactly
the stored 133-path baseline. Therefore introduced formatter failure paths within
the accepted scope are zero. This is a correction from existing evidence, not
a fresh formatter run or a replacement of the historical FAIL record.

Durable correction JSON, exact path lists, and blob identity proofs are in
[integration-01a16fc3-evidence](integration-01a16fc3-evidence/).
The source diff and new runtime-path test matched their run-start snapshots at
checkpoint review. No measurement trust grade or native attestation is claimed.
GitNexus coverage remains UNKNOWN after index corruption; review used manual scope.

Full local run artifacts remain under
`/mnt/raid0/llm/tmp/refactor-01a16fc3/orchestrator-integration/`:
`current-run.json`, `final-union-manifest.json`, `final-union-pytest.log`,
`final-union-coverage.json`, `final-proof-run2.log`, `final-proof-run2.exit`,
`final-proof-status.txt`, `registry-loader-ast-proof.txt`, `final-focused-tests.log`,
`final-ni63-tests.log`, `final-integrity.txt`, and `final-format-compare.txt`.
The manifest's acceptance field predates owning-main acceptance described here.
