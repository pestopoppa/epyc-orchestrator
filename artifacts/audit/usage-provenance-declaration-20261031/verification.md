# Request usage declaration carrier: source proof

Date: 2026-10-31

APP source commit: `0d01dc94c294`

Published APP main: `3d35934f7f29`

## Change

`UsageLedger.begin` now accepts an optional source-authored measurement declaration. The default is
`None`, so existing callers remain unannotated. The writer validates a bounded JSON declaration,
copies it at begin, and includes the same declaration in the self-hashed begin and settled events.
The declaration names the exact usage counter and carries its metric, unit, category, direction,
protocol, claim, and optional existing ClaimTuple context. The writer invents none of those values.

## Verification

Command, run with the shared orchestrator environment and the existing 36–37 CPU prep lock:

```text
flock /mnt/raid0/llm/tmp/compute-flags/prep-locks/36-37.lock taskset -c 132,133 env OMP_NUM_THREADS=1 PYTHONPATH=/mnt/raid0/llm/worktrees/mains/codex-ni-orchestrator /mnt/raid0/llm/epyc-orchestrator/.venv/bin/python -m pytest -q tests/unit/test_usage_provenance.py tests/unit/test_usage_provenance_log.py
```

Result: **41 passed**, one upstream OpenTelemetry deprecation warning. Ruff passed for the changed
writer and focused log tests. The tests cover absent-by-default declarations, detached input data,
self-hashed begin/settled equality, exact zero, and invalid direction refusal.

The companion ROOT test invokes the actual writer modules from the published APP source checkout,
then runs the strict reader, generic source ingest, and public planning-context reader. Its test-only
declaration is explicitly labeled synthetic. The canary confirms that the scoped synthetic target
sees the fixture and a different model target does not. It does not claim runtime measurement.

## Scope status

No production caller authors a declaration yet, and no native inference capture or live runtime
observation was made. Cache replays that lack the original declaration remain unjoined and cannot
create another physical observation. `USAGE-PROV-1` remains open for actual source-authored native
capture and its runtime canary; this change does not establish that the source is wired or READY.

Source SHA-256:

```text
src/llm_primitives/usage.py  0d9a0a83aad471ff2a5b902951709ab3d7d4c392fde31ba5c7f74946c81d1e75
tests/unit/test_usage_provenance_log.py  43337ec3cd18a9950caaceb02cf75457d0270cde5d7d512c872f379d1f5053cd
```

`[no-belief: source implementation and synthetic correctness controls; no measurement]`
