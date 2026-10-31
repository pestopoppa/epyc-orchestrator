# NI per-instance `-np` declaration proof

## Audit and implementation

The ROOT M task remains needed at APP `HEAD` `16dbdaaa8ac744098d1b275a4258278acaf535d8`. The master schema only declared `serving_shape.slots_by_shape`, keyed by shape class. The compiler already emits `runtime.cache.slots_by_port`, and the launcher already resolves that artifact by physical port. No declaration-level per-instance field existed.

The optional declaration is `server_mode.<role>.serving_shape.slots_by_instance`, keyed by declared instance port. Precedence is per-instance; existing shape-class declaration; role scalar; mode fallback. Compiler output remains in the existing `runtime.cache.slots_by_port` field. The native `resolve_instance_slots` wrapper leaves CRITICAL-impact `resolve_slots` unchanged. The parser validates known physical ports and positive integer counts consistently across compiler, native resolver, and batch certification.

The master registry parser preserves explicit caller-provided `port_shape_classes`; ambient topology classes only fill missing ports. Batch certification reads the owning registry's declaration, not ambient master data. It only applies instance widths to local declared physical endpoints. For a URL pool, the shared width is the minimum of declared endpoint widths and the admission cap is evaluated per normalized endpoint (stripping `full:`); remote/unknown endpoints do not inherit local port declarations. Invalid declarations fail closed at width 1. The physical `eval_batch_frontdoor` lane at port 18070 retains its independent fallback of 2 and does not inherit frontdoor declarations.

## GitNexus and caller trace

`gitnexus status` reported `/workspace` indexed at `97268ec` (2026-10-24) while current ROOT was `85b5aa4` (2026-10-31); APP checkout itself reported not indexed. Treat graph output as supplemental and manually trace actual current callers. Explicit APP impact was LOW for `_resolve_parallel_slots` (8 dependants), HIGH for `_slots_by_port` (6), and CRITICAL for `resolve_slots` (67 overall, 9 direct). The implementation does not edit `resolve_slots` and uses the existing per-port cache contract through compiler, launcher, admission, and runtime attestation.

## Verification

Test commands used the R6 lock `/mnt/raid0/llm/tmp/compute-flags/prep-locks/34-35.lock`, affinity CPUs `34,35,130,131`, OMP/OpenBLAS/MKL set to 1, and a 900-second timeout for the broad run. Final broad run included all changed module suites plus admission and stack-change/attestation controls: **502 passed**, peak RSS **178580 KiB**. Log: `full-changed-tests-final-rss.log`.

The four output-usage failures from the preceding broad run were checked against exact HEAD source using a bounded import overlay in scratch (no new checkout/worktree). Base source commit was `16dbdaaa8ac744098d1b275a4258278acaf535d8`; original source file SHA-256 values: backend `cceeaa03e7f7e03371f634f049ca3e86b711fa171cc653f6e261af50cad2d743`, stack manifest `bc49258dca10b16bc83dc45c10a13117fc8410f58bfb070dcb40811e09651893`, stack priors `304f652cba847e157e8ac9e3098b5e9817e0c07071b921d0a3b359f81add1f5d`. Candidate source SHA-256 values after implementation commit `eda654775ddac4a7bec41b36563d7e4860cd895d`: backend `6717415c645b0837a71444c1f12c82c29b736d0ab3b0fae07e5b2e384b312afa`, stack manifest `8329d81f8a7435f0e1cfce1e3a32e4a8635d0e80bc92fcf21100fd2e813b9143`, stack priors `213042a3aa5fe678a5dd70515ce1e2c8949faa8ae5d359f232e9ee7afcfee8ce`. All changed-source hashes are in `source-sha256.txt`. With corrected current-protocol test fixtures, the exact HEAD source overlay passed **4/4**, and the candidate passed **4/4** (`head-corrected-fixture-control.log`). The fixtures now use `llama_server:/completion.tokens_predicted` and `llama_server:/v1/chat/completions.usage.completion_tokens`; expected reported counts and source contributors remain explicit. The total-token assertion uses the reported count 9, matching the result's `output_tokens_reported=9` instead of the legacy estimate 10.

A full default `compile_stack_priors()` comparison against the shipped derived artifact checked all 16 roles: **0 runtime cache differences** (`default-cache-compatibility.log`). Synthetic declaration tests verify unchanged cache keys and provenance hashes over the handed registry bytes. `ruff check` passed for all changed source/test files; `git diff --check` passed.

APP implementation commit: `eda654775ddac4a7bec41b36563d7e4860cd895d` (`git rev-parse HEAD` at implementation checkpoint). It passed the active pre-commit hook, including the topology/contention freshness check. `make gates` passed after fresh-main integration. The documentation/proof commit and remote publication are recorded in the final progress update.

## Review files

- `source-sha256.txt` — implementation commit and per-file source hashes
- `full-changed-tests-final-rss.log` — final broad test run and peak RSS
- `head-corrected-fixture-control.log` — exact HEAD source overlay control for four corrected tests
- `ROOT-completion-prep.md` — unchanged task wording and MAIN-only checkbox application note
- `focused-tests.log` — earlier focused selection
- `default-cache-compatibility.log` — 16-role default cache comparison
