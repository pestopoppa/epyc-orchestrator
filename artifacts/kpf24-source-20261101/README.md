# KPF-24 source contracts — 2026-11-01

PROPOSED software verification; no deployment or inference claim.

Implementation commit: `45952718bc270ccf93e3dd431b18d812eaeed1bf`.
Tested final workflow/source tree: `2293e16ec4428996bcf8ae04329e6786d70527fa`.
The application and test blobs are identical between these commits.

The native gate reads the selected server's `llamacpp:kv_cells_used` and
`llamacpp:kv_cells_size` gauges, in attention cells, and the same native pool
snapshot exposed in `/slots`. Shared cells count once. Missing, invalid and
fork-off counters stay unknown. Valid zero usage requires positive capacity.
The higher consecutive same-capacity snapshot is retained; capacity drift
explicitly refuses admission until a consistent refresh.

The gate adds own outstanding reservations and the candidate to native usage
plus ratio-weighted pending decode. Own tickets have no exact server-task
binding before dispatch, so their overlap cannot safely be subtracted: this is
a conservative bound that may reserve overlapping own cells twice. It does
not establish a global reservation ledger across API workers or predict future
external-client allocations. Legacy/index-off behavior is preserved. Fork
credit is capped at prompt minus one, retains the generation allocation, and
is labeled `cache_credit_source: "fork"` after the source-liveness check.

## Original hosted runs

Each directory contains the original run/jobs/artifact API JSON, artifact ZIP,
GitHub log ZIP, extracted JUnit/stdout/dependencies/source hashes, and a custody
inspection. ZIP digests were checked against the original GitHub artifact API;
all source hashes were checked against the run's immutable Git blobs.

| Run | Source | Focused contracts | Broad gates / disposition | Original artifact SHA-256 |
| --- | --- | --- | --- | --- |
| [41297457980](https://github.com/pestopoppa/epyc-orchestrator/actions/runs/41297457980) | `e45a6ac1` | 275 pass, 3 fail, 6 skip | Three stale flag-off fixture expectations; hosted shfmt absent. Failure retained. | `93f2cfbcc800f59d11cefe8e825170af8899fc56688a50bdc84fa0799a8b860b` |
| [41297851772](https://github.com/pestopoppa/epyc-orchestrator/actions/runs/41297851772) | `45952718` | 281 pass, 0 fail/error, 6 skip | Immutable main `29fdfc5a` reproduces the original three flag-off failures. Broad gate lacked the explicit async plugin. | `28d225b6a5eb4ddc693a368500f7e880430e658b385f0be6bef21041cafed813` |
| [41298276218](https://github.com/pestopoppa/epyc-orchestrator/actions/runs/41298276218) | `fca6dd61` | 281 pass, 0 fail/error, 6 skip | Exit capture repaired. Candidate and baseline broad gates each report 101 fail, 267 pass, 17 skip, 1 error with plugin autoload disabled. | `7ceda05e665387b159765901dfeb5261ce1675f1378ee31bd383e97f19257c5f` |
| [41298363381](https://github.com/pestopoppa/epyc-orchestrator/actions/runs/41298363381) | `2293e16e` | **281 pass, 0 fail/error, 6 skip**; Ruff exit **0** | Explicit async plugin enabled. Candidate and baseline `make gates` both exit **2**, with byte-identical configuration failures before integration collection. | `1ac896cbc6d99a5439e88970809d50bd11cf3f385639922a2ba3514798a6aa8a` |

All four overall GitHub conclusions are **failure**, not green full-suite
receipts. The final broad-gate stdout files share SHA-256
`e2f993bb85b62f279417d1dc83766504fb3b1a5adca994727cccca4d75dc041c`.
The error is `pytest.PytestDeprecationWarning` for unset
`asyncio_default_fixture_loop_scope`, raised by the declared pytest-asyncio
dependency under the repository's warning policy. It occurs on immutable main
`29fdfc5a` and the candidate identically. No gate JUnit is created when pytest
fails during configuration; its absence is preserved.

The first two workflows inherited GitHub's bash `-e`: failed diagnostic
pipelines exited before some exit-code files or the broad baseline comparison
could be written. Original logs retain their exits; missing files were not
invented. Later workflows explicitly capture these failures before comparing
the baseline. The records above are separate runs, never aggregated into a
passing denominator.

The six focused skips are five checks requiring the frozen production kernel
source tree on the hosted runner and one check requiring a historical study
artifact. Their native reasons remain in JUnit and `inspection.json`. No native
kernel test, model call, performance, quality, adoption or KPF-26 result is
claimed. No production tree, build, serving process or runtime flag was changed.

The adjacent flag-off fixture repair allows exactly the four lease-observation
keys already added by `7ecfbe3f`, while keeping synthetic legacy serving records
valid and asserting the actual admission ticket/lease values on real records.
It does not invent lease observations in fixtures.

`inspect-ci.py <ci-directory>` reproduces the custody and focused-JUnit summary
from these original files and a repository containing the referenced commits.
It performs no inference or grading. [no-belief: software verification]
