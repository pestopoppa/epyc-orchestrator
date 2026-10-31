# HIL-15b v2 frozen source publication

## Scope

Closed the source-publication task codex-ni-ufh08-hil15b-v2-publish-20261025 by publishing the reviewed 13-path payload on orchestrator origin/main. This is a source and packaging result only. No model was loaded, no live task was run, no empirical-difficulty or quality claim was made, and the owner’s HIL-15b 60-attempt cohort and parent acceptance remain open.

## Published implementation

- Source commit: fe7a0b7bb1cb7423905f90e1aa72eb152d3b43d8 (feat(trace): publish frozen HIL-15b v2 assets). It is verified as an ancestor of fresh origin/main; the serialized push used agent codex-ni-01a16d6b and the APP push lock.
- The frozen taskset SHA-256 is 0089c2a840bc40367433de754424b17ad9a5edca954b312513e6ecaa0abd4312.
- The three ignored historical .log receipts were published as .log.txt copies. Their bytes match the original receipts:
  - PREFLIGHT.log.txt: 4aba478a5900408781df84c0f492c0d36cd593b0386e4ee3e7c9debc3e940099
  - PREFLIGHT-TOOLPROTOCOL.log.txt: dbc4475a02a819218f2f7621779e2a6dcf83d456ac4c8b40537b44a7784aaa25
  - PREFLIGHT-MOCK40.log.txt: 51e6856b2e521379f8ee456e72e3b3616643ff9d3a42be9b4ed65c1003c7edb9
- All 13 published paths and their SHA-256 values:
  - 93c615ffd325760ed98b8ebe87a785bc02b2b172eca46c4ded0c165a1c66a94a — artifacts/hil/hil15b-v2-20261025/ASSET-CONTRACT.txt
  - 4ace073096af4a6976086b3ce9674a6da7f4522c33b65cfaa6645509c61e0c64 — artifacts/hil/hil15b-v2-20261025/PREFLIGHT-RECEIPTS.txt
  - 28ca4ff22f76bd3388d51a13fafea7556f7d12bd51b5f1c0496cc5476e8a42d6 — artifacts/hil/hil15b-v2-20261025/calibration_tasks.json
  - a3f25ec7514fbd0d851bf5cf0818c885b6dcdff13fe8e5e76b1e66277bfc9849 — artifacts/hil/hil15b-v2-20261025/heldout_tasks.json
  - 0089c2a840bc40367433de754424b17ad9a5edca954b312513e6ecaa0abd4312 — artifacts/hil/hil15b-v2-20261025/hil15b_v2_tasks.json
  - 67ae0d3e55a4746b56e7d6af733a30eb37b1aced930e63f78b598eef80b66dd0 — artifacts/hil/hil15b-v2-20261025/hil15b_v2_warrants.json
  - 51e6856b2e521379f8ee456e72e3b3616643ff9d3a42be9b4ed65c1003c7edb9 — artifacts/hil/hil15b-v2-20261025/receipts/PREFLIGHT-MOCK40.log.txt
  - dbc4475a02a819218f2f7621779e2a6dcf83d456ac4c8b40537b44a7784aaa25 — artifacts/hil/hil15b-v2-20261025/receipts/PREFLIGHT-TOOLPROTOCOL.log.txt
  - 4aba478a5900408781df84c0f492c0d36cd593b0386e4ee3e7c9debc3e940099 — artifacts/hil/hil15b-v2-20261025/receipts/PREFLIGHT.log.txt
  - d9cca2fdebb81ce2eb0fec161a0567dc0198e946ce520bda3c0bc1a08685819d — artifacts/hil/hil15b-v2-20261025/receipts/mock40-manifest.json
  - be0907a29cf26b7b66b0553c07e5eb8a9bf5788dd25cedb487cd911b4a158970 — scripts/trace/hil15b_v2_make_tasks.py
  - 1b6cc6968fa29b562f5461077f81d6a7fd1fa416256804f2ec19f2b1cca1bd4e — scripts/trace/validate_hil15b_v2.py
  - 7fb565571f36ae0450903f8147165215ff8201f2e23e50f3a0f09ef15fa27aa2 — tests/unit/test_hil15b_v2_assets.py

The native driver was not changed; scripts/trace/hil15a_flagon_run.py retains the frozen SHA-256 75abff0842a550051b3a25f3bf079a505044120a9ee57e80cc08ef2661bb4c63. Current-tree inspection and the passing integration test confirm load_tasks and grade still accept these assets. The generator, independent validator, and test remain byte-identical to the reviewed payload.

## Verification

Ran model-free checks on the current APP source under the R6 32-33 prep lock, taskset to CPUs 32-33, OMP=1, sampled process-group RSS cap 8 GiB, and a 300-second command timeout:

- scripts/trace/validate_hil15b_v2.py artifacts/hil/hil15b-v2-20261025: 20 independent answer/input/checker comparisons and 5 semantic fixtures passed.
- Shared APP venv pytest target tests/unit/test_hil15b_v2_assets.py: 3 passed, 1 dependency deprecation warning, 0.34 s. This covers frozen asset bytes and anchor, current native loader/grade integration and calibration/heldout partition, and unchanged native driver SHA.
- make gates: passed (schema, shellcheck/shfmt check, markdownlint).
- Ruff check on the two frozen scripts and test: passed. Ruff format check reports all three pinned Python files would be reformatted; they were left unchanged to preserve accepted payload hashes. No Python source or task semantics were changed to satisfy formatting.

Attempt provenance retained: an initial system-Python unittest invocation failed to import the test because that interpreter lacks requests; a later shared-venv unittest discovery collected 0 tests because the test uses pytest functions. Neither is counted as a pass. The correct shared-venv pytest invocation then collected and passed all 3 tests.

## Limits

The historical receipts describe their original 2026-10-25 model-free preflight. The 2026-10-31 checks above verify current source/package compatibility; they do not repeat or revise that preflight, run a model, or establish empirical task difficulty. No inference READY was created for a source-only publication.
