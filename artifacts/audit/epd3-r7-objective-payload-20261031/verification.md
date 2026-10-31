# EPD-3-R7 full-objective payload preservation

## Change

Both eval reward payload writers in `scripts/benchmark/seeding_injection.py` now carry the complete source `prompt` in `task_description` and pass that same complete prompt to `_precompute_embedding`. The embedder input remains governed by the existing canonical `embedding_text_for` builder and its shared 2,000-character cap. No shared embedding budget or normalization policy changed, and no historical records were rewritten.

The native request model, `/chat/reward` handler, `store_external_reward`, `QScorer.score_external_result`, and private SQLite `EpisodicStore` round trip are exercised in `tests/unit/test_seeding_injection_objective_storage.py`. It covers the 3-way eval route, the per-role eval route, two 2,100-character shared prefixes with different suffixes, Unicode/short text, the canonical embedder request, and native empty/missing request semantics.

## Verification

Command (R6 34-35 mutex, OMP/OpenBLAS/MKL=1, shared APP interpreter):

```bash
flock /mnt/raid0/llm/tmp/compute-flags/prep-locks/34-35.lock timeout 900s /usr/bin/time -v env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/mnt/raid0/llm/worktrees/mains/codex-ni-orchestrator /mnt/raid0/llm/epyc-orchestrator/.venv/bin/python -m pytest -q tests/unit/test_seeding_injection_objective_storage.py tests/unit/test_seeding_injection.py tests/unit/test_seeding_injection_additional.py tests/unit/test_one_embedding_convention_writers.py tests/unit/test_epd3_r5_r6_r8.py
```

Observed: **68 passed, 4 warnings, 4.78 seconds**, maximum RSS **172548 kB**, exit status **0**. The warnings were dependency deprecations from OpenTelemetry and SWIG bindings.

`ruff check scripts/benchmark/seeding_injection.py tests/unit/test_seeding_injection_objective_storage.py` and `git diff --check` passed. The main session reviewed and approved the corrected source and native-path test before publication.

## Published source pins

- Implementing source commit after clean rebase onto current `origin/main`: `9c388042475d835f88e62355443e7036986b9a34`
- Upstream base: `b85b44f12ec857fc0249bc956e239feadfe92ca4`
- `git rev-parse HEAD:scripts/benchmark/seeding_injection.py`: `1dbdc9f525ba1dee221b4a202878418c8071ec48`
- `git rev-parse HEAD:tests/unit/test_seeding_injection_objective_storage.py`: `e8b8a255b7d662b56843ff189a0aff8ff9600e08`
- `scripts/benchmark/seeding_injection.py` SHA-256: `b21ddc3690b4bc9c80c7b80ecffa2172df3bfca7315909f5bc7ee8bd06d33165`
- `tests/unit/test_seeding_injection_objective_storage.py` SHA-256: `bef88ff2fd330bed1da90b68b48bb87245a034a4075b9626d3bc386e6fbc5705`

The superseded first design and its observed 12-test pass are retained in [`superseded-design-history.md`](superseded-design-history.md); review correctly rejected that design because a 200-character embedder slice diverged from the stored objective under EPD-3-R3. The corrected 68-test run above verifies the canonical builder input.
