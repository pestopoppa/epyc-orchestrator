# Replacement evidence for historical N-way and AP-69 closures

Evidence packet assembled 2026-10-31 12:35:37 UTC. The earlier `/tmp` test logs cited during the audit were absent. The runs below are new, honestly dated replacement runs against selected APP `origin/main` source at `eb4561b0247f0a94b34da2851bc7f5a2be184efd`; they do not recreate or claim the missing historical logs. Both audit items are prior work and remain excluded from 2026-10-31's new-work quota.

## N-way metric and tooltip

Implementing commit: APP `b387c5dded6619bfbcad4c86edab5f5759deabf8` (2026-10-30). Pinned source blobs: `src/api/routes/dashboard.html` `c4c62425a993adae928da50d4f28d17eb0af8c54`; `src/scheduling/contention_gate.py` `e313372ac6a2f810cd5fc5548fe8cc16eb813699`; renderer tests `0d14be086df4b2dbf0f8db905f29d0b4c7a6962f`; gate tests `030d8175981825c27b7cc98c886ab4c2a597a3dd`.

- Command: `python -m pytest -q tests/unit/test_dashboard_contention_nway_restricted.py tests/unit/test_scheduling_contention_gate.py`
- R6 mutex: `/mnt/raid0/llm/tmp/compute-flags/prep-locks/34-35.lock`; affinity CPUs 34, 35, 130, 131; `OMP_NUM_THREADS=1`; 900-second supervisor timeout.
- PID 967793; `/proc` start ticks 157254926; initial sampled RSS 588 KiB; max RSS 146276 KiB; process absent after wait.
- Completed 2026-10-31 UTC: **33 passed**, one existing deprecation warning.
- Captured log SHA-256: `79cc618f99e218c936c89792a1a133db41ee33c91529c91c91cbc7794299ddca`.

```text
============================= test session starts ==============================
platform linux -- Python 3.11.17, pytest-9.0.3, pluggy-1.6.0
rootdir: /mnt/raid0/llm/tmp/ni-xsource-snapshot-full-20261031
configfile: pyproject.toml
plugins: asyncio-1.3.0, timeout-2.4.0, langsmith-0.7.30, cov-7.1.0, xdist-3.8.0, hypothesis-6.151.13, anyio-4.13.0
asyncio: mode=Mode.AUTO, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collected 33 items

tests/unit/test_dashboard_contention_nway_restricted.py ......           [ 18%]
tests/unit/test_scheduling_contention_gate.py .......................... [ 96%]
.                                                                        [100%]

=============================== warnings summary ===============================
../../epyc-orchestrator/.venv/lib/python3.11/site-packages/opentelemetry/util/_importlib_metadata.py:32
  /mnt/raid0/llm/epyc-orchestrator/.venv/lib/python3.11/site-packages/opentelemetry/util/_importlib_metadata.py:32: DeprecationWarning: SelectableGroups dict interface is deprecated. Use select.
    return EntryPoints(ep for group_eps in eps.values() for ep in group_eps)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================== 33 passed, 1 warning in 1.42s =========================
```

## AP-69 proposal-filter label

Implementing commits: APP `dff7dabbd4cc95ee2e310349cc0930e6eaa4eee4` and `e822e3788ec016be1f7ee06d13225179a6dc816e` (2026-10-30). Pinned test blobs: actions `de26dfea92ab8f1ab45b67f84e464b2f949b97a0`, coordinator `081f888b1f55ea517e40a6fdab8ecc4dfb045a1c`, providers `d8d9a356929e7dee5603915ff1a58d5c76431460`, health report `b9f9f733060bcc5e9b0342c1f4e3a8622917f545`.

- Command: seven named regression cases across `test_autopilot_actions.py`, `test_autopilot_planner_coordinator.py`, `test_autopilot_planner_providers.py`, and `test_planner_provider_health_report.py`:

  - `test_autopilot_actions.py::test_prior_planner_decision_digest_labels_legacy_kind_unknown`
  - `test_autopilot_planner_coordinator.py::test_planner_archive_records_provider_trace`
  - `test_autopilot_planner_coordinator.py::test_planner_archive_marks_nonverdict_unknown_not_as_filter_or_validation`
  - `test_autopilot_planner_providers.py::test_planner_tap_start_marks_critic_as_advisory_proposal_filter`
  - `test_autopilot_planner_providers.py::test_critic_provider_archive_records_advisory_kind_only_for_critic_role`
  - `test_planner_provider_health_report.py::test_build_report_counts_local_draft_and_critique_success`
  - `test_planner_provider_health_report.py::test_report_labels_explicit_critic_kind_and_preserves_legacy_unknown`
- R6 mutex: `/mnt/raid0/llm/tmp/compute-flags/prep-locks/34-35.lock`; affinity CPUs 34, 35, 130, 131; `OMP_NUM_THREADS=1`; 900-second supervisor timeout.
- PID 971060; `/proc` start ticks 157259735; initial sampled RSS 444 KiB; max RSS 176428 KiB; process absent after wait.
- Completed 2026-10-31 UTC: **7 passed**, one existing deprecation warning.
- Captured log SHA-256: `0754bf58477968e6b954bc28908c8c6d59e8b21588928f7f7b2e8878deffab83`.

```text
============================= test session starts ==============================
platform linux -- Python 3.11.17, pytest-9.0.3, pluggy-1.6.0
rootdir: /mnt/raid0/llm/tmp/ni-xsource-snapshot-full-20261031
configfile: pyproject.toml
plugins: asyncio-1.3.0, timeout-2.4.0, langsmith-0.7.30, cov-7.1.0, xdist-3.8.0, hypothesis-6.151.13, anyio-4.13.0
asyncio: mode=Mode.AUTO, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collected 7 items

tests/unit/test_autopilot_actions.py .                                   [ 14%]
tests/unit/test_autopilot_planner_coordinator.py ..                      [ 42%]
tests/unit/test_autopilot_planner_providers.py ..                        [ 71%]
tests/unit/test_planner_provider_health_report.py ..                     [100%]

=============================== warnings summary ===============================
../../epyc-orchestrator/.venv/lib/python3.11/site-packages/opentelemetry/util/_importlib_metadata.py:32
  /mnt/raid0/llm/epyc-orchestrator/.venv/lib/python3.11/site-packages/opentelemetry/util/_importlib_metadata.py:32: DeprecationWarning: SelectableGroups dict interface is deprecated. Use select.
    return EntryPoints(ep for group_eps in eps.values() for ep in group_eps)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
========================= 7 passed, 1 warning in 1.14s =========================
```

The APP snapshot used for these current-source checks also contained the separately proposed, test-only `x_request_source` assertion correction in the provider test file. The seven AP-69 node IDs are disjoint from those three local-chat cases. The exact lost historical run time/counts are not claimed here.
