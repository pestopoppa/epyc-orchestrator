# Archived KVQ reconsideration: source and synthetic control record

APP source commit: `6eb94e9597a0243bbd20516e1b8dcb350f8d1c9b` (parent
`2c4350972645856fc20c356d0d4659da20edd57f`). This record binds the focused
controls to that source; it does not assert a live deployment or a production
measurement. The report is an explicit snapshot reader for human review and
does not reverse decisions, write a ledger, or promote a candidate.

## Source pins

SHA-256 of the tested on-disk files:

| File | SHA-256 |
| --- | --- |
| `scripts/autopilot/kvq_reconsider.py` | `f1b3539c0ec8f0ea5390b1b114dcb484078e4cba5781b44d2c189379b737a16c` |
| `tests/unit/test_kvq_reconsider.py` | `1e637b7613842151289b4e9e26ac4475552da84fa728014e3c657c9301161422` |
| ROOT `scripts/vidya/fold.py` | `42e1ccb832c59415501254aca427fb8e0d0cb2f7bc9e259ec3a8b10e7a369ef6` |
| ROOT `scripts/vidya/ledger.py` | `50d4872ab822b4912ba83ab8d05febf12b28e273b41237f3655bb13eb67a2e3e` |

The Vidya files used by the tests were a private projection of host ROOT
`/mnt/micron/codex3-lanes/root`, whose recorded HEAD was
`84fc682bb0a94cb88d3f4a2019e1579d096cb0f3`. The two projected files matched
the on-disk host files by the hashes above. These are file hashes, not a
loaded-byte attestation; they do not attest other dependencies in that tree.

## Controls and results

The captured synthetic suite used the native ROOT fold and ledger projection,
a temporary native ledger and archive copy, and the actual report CLI. It
covered receipt replay and the declared/exposed intersection; direct and
dependency correction and withdrawal, including a withdrawn claim and
multiple paths with one evidence label; already-reviewed corrections;
unrelated changes; original-prefix and later-frame time visibility; invalid
or missing boundaries and timestamps; malformed JSON; and unchanged input
copies. The CLI test also checked the report's input and source provenance.
The recorded run passed **12/12**. [Captured pytest output](native-synthetic-tests.log)
SHA-256: `6403128e726f2cef51a7e113eb11f50b08448f7bf6f7f9d693f8e735304e96bb`.
Trailing whitespace was removed for the repository whitespace check; the raw
capture SHA-256 was `86e672e2d71f8c3e5c9e85357267684448dd0b73ec5d88c5e05ff06c43c80488`.

Command, executed from the isolated APP checkout:

```sh
VIDYA_DIR=/mnt/raid0/llm/tmp/codex3-kvq-reconsider-20261101/vidya \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest --noconftest -q tests/unit/test_kvq_reconsider.py
```

`--noconftest` was necessary because the local Python environment lacked the
APP dependencies imported by the repository's ordinary conftest; pytest also
reported its unknown `asyncio_mode` configuration warning. This is a focused
synthetic control result, not a full APP regression suite. The test's native
Vidya projection was available for this run; environments without it skip the
module, so a skip does not constitute this evidence.

`ruff check scripts/autopilot/kvq_reconsider.py tests/unit/test_kvq_reconsider.py`
passed. [Captured Ruff output](ruff.log) SHA-256:
`82b3e6a6c090a57601d22943bd23fca9218d1031dbe5a7b754092f9a156b4f18`.
The source commit also passed the installed native commit hooks.

No live archive, AutoPilot state, or canonical ledger was read or modified for
these controls; there was no model inference, activation, production promotion,
or claim of operational wiring. The native KVQ receipt producer folds the full
original prefix before its source-selection time filter, so the report replays
that original prefix faithfully and time-filters only later frames.
