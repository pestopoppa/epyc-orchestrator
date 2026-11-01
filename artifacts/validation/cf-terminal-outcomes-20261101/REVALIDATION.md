# CF terminal outcome telemetry revalidation

- Tested source commit: `df5645e96` (`Track compaction give-up and non-termination outcomes`)
- Base: `origin/main` at `a1f4942edff28dd99c108a1299395a3ec00be8f6`
- Environment: isolated worktree `/mnt/raid0/llm/worktrees/cf-terminal-outcomes-20261101`
- Compute: calendar keeper reported `IDLE-ALARM free for >=2 checks: q1` at
  `2026-11-01T03:57:37Z`; `region-lock status` showed q1 free immediately before execution.
  Both command groups acquired q1 as `role=build` in 0.0 seconds using physical CPUs 24-47.
  No inference or serving process was run.

## Focused tests

Exact command:

```bash
scripts/region-lock run --cpu-list 24-47 --role build --tag cf-terminal-outcomes-validation-20261101 --timeout-s 10 -- /bin/bash -c 'set -o pipefail; TMPDIR=/mnt/raid0/llm/tmp/cf-terminal-outcomes-20261101 .venv/bin/python -m pytest tests/unit/test_session_log.py tests/unit/test_langgraph_phase2.py tests/test_langgraph_durable_resume.py tests/integration/test_observability.py -q 2>&1 | tee artifacts/validation/cf-terminal-outcomes-20261101/pytest.log'
```

Exit status: `0`. Output: `109 passed in 5.94s` in [pytest.log](pytest.log).

## Ruff and committed-diff check

Exact command and region-lock acquisition output are in [quality.log](quality.log).
Strict Ruff reports three existing `F401` findings in `src/graph/session_log.py`
(`datetime`, `timezone`, and a local `time` import); filtered Ruff (`--ignore F401`)
passes. These findings are present in the `origin/main` version of the file. The
committed diff check `git diff --check origin/main...HEAD` exits `0`; its output is
in [diff-check.log](diff-check.log).

Individual Ruff outputs: [ruff-strict.log](ruff-strict.log) and
[ruff-ignore-f401.log](ruff-ignore-f401.log).
