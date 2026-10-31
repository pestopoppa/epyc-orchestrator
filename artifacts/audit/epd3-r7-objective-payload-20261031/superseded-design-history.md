# EPD-3-R7 initial test proof — superseded embedding slice

This records the first implementation proof before review caught its embedding-contract mismatch. The run was real and passed, but the design was rejected: both reward payloads carried full objective text while `_precompute_embedding` still received only `prompt[:200]`, which violates EPD-3-R3's canonical-vector correspondence to the stored objective.

Command (from `/mnt/raid0/llm/worktrees/mains/codex-ni-orchestrator`):

```bash
flock /mnt/raid0/llm/tmp/compute-flags/prep-locks/34-35.lock timeout 900s /usr/bin/time -v env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/mnt/raid0/llm/worktrees/mains/codex-ni-orchestrator /mnt/raid0/llm/epyc-orchestrator/.venv/bin/python -m pytest -q tests/unit/test_seeding_injection_objective_storage.py tests/unit/test_seeding_injection.py tests/unit/test_seeding_injection_additional.py
```

Observed result: `12 passed, 4 warnings in 4.33s`; max RSS `174100 kB`; exit status 0. A second run after formatting the new test file observed the same 12 passing tests (`4.46s`, max RSS `173960 kB`, exit 0). Those runs validate the storage-path test mechanics only. The corrected follow-up removes the 200-character embedding slice, verifies the actual canonical builder input at the 2,000-character shared limit, and must be re-run before acceptance.

# Corrected design verification

After removing the 200-character slice from both `_precompute_embedding` calls, the round-trip test was strengthened to observe actual `json["content"]` sent by `_precompute_embedding` and compare it to `embedding_text_for(full_objective, task_type)`. Both long objectives are 2,100 shared characters plus distinct suffixes, so they collide under the existing 2,000-character canonical builder while remaining distinct in SQLite.

Command:

```bash
flock /mnt/raid0/llm/tmp/compute-flags/prep-locks/34-35.lock timeout 900s /usr/bin/time -v env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/mnt/raid0/llm/worktrees/mains/codex-ni-orchestrator /mnt/raid0/llm/epyc-orchestrator/.venv/bin/python -m pytest -q tests/unit/test_seeding_injection_objective_storage.py tests/unit/test_seeding_injection.py tests/unit/test_seeding_injection_additional.py tests/unit/test_one_embedding_convention_writers.py tests/unit/test_epd3_r5_r6_r8.py
```

Observed result: `68 passed, 4 warnings in 4.78s`; max RSS `172548 kB`; exit status 0.
