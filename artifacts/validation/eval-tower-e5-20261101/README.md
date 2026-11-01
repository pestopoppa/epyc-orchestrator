# Eval Tower E5 partial source repair — 2026-11-01

The candidate fixes the report-discovery stat race and serializes the AutoPilot TUI prompt tap write. It does not set a suite-health freshness limit; ET-8 and the EvalTower callback contract were not changed after GitNexus reported HIGH impact. E5 remains open.

The native run was made on the following source-relative Git blobs:

- `scripts/autopilot/eval_tower.py`: `cdebe3cade6ddf7160497d78486cb2edd2b92bc1`
- `scripts/autopilot/autopilot_tui.py`: `272ec77b9f3d37b8fd3cb16c629d5e8f359f6588`
- `tests/unit/test_eval_tower_e5_low_repairs.py`: `ff1d8a3ca4659f38c1e334b80560230d7152c149`

[Fixed candidate log](fixed-control.log): 29 passed, including four new tests, the existing seeding TUI suite, the existing promotion eval test and the existing callback test. The R4 physical guard used pair 78–79 under flock, two CPUs, OMP/BLAS threads set to one, 120-second bound, and reported child rc 0 and children drained. RSS is sampled, not a kernel-enforced peak.

[Negative control log](old-control.log): the lock test failed at its nonblocking lock assertion, as expected, after the [control script](old_control.py) extracted the original `set_prompt` AST from `scripts/autopilot/autopilot_tui.py` at base `fb2036d93203eb4e5a87918ecf0572c29ba73fd3` (blob `cb9cf5e16ea5f1f258873e18f81a2abe4343759f`) and installed it in the test module. Guard child rc 1, children drained. The script's absolute scratch input is the unmodified `git show` of that blob; the log records the actual run. No inference or stack reload was performed.
