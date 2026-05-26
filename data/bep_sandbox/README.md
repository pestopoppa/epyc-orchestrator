# BEP-2 / DCP-6 scratch workload (Phase 2)

5 deterministic synthetic code-edit tasks in `tasks.jsonl`, chosen as batch-edit *sweet spots*
(multi-file + cross-file dependency), each with an independent deterministic verifier. Per
`handoffs/active/bep-dcp-falsification-harness.md`.

| id | kind | exercises |
|----|------|-----------|
| t1_create_util | create | single-file create |
| t2_add_and_use | multi-file modify + dependency | add fn in calc.py, use in main.py |
| t3_method_and_caller | multi-file modify + dependency | add method, pre-written caller must run |
| t4_rename_module | rename + update import | exercises the BEP-1b rename promotion fix |
| t5_bugfix | single-file modify | fix a wrong constant |

Schema per line: `{id, kind, prompt, files:{path:content}, verifier_cmd}`. The driver writes
`files` into a fresh scratch git repo, points `ORCHESTRATOR_EDIT_ROOT` at it, runs the arm, then
runs `verifier_cmd` in the scratch repo (exit 0 + "PASS" in output = quality pass). The verifier
is an independent oracle (pytest/assert/grep), never the applying model.

**Validated (no inference):** each verifier FAILS on the initial files and PASSES on a known-good
solution (fresh dir per check). **Driver requirements learned here:** (1) fresh scratch dir per
task×arm×rep (reset to pristine); (2) run verifiers with `PYTHONDONTWRITEBYTECODE=1` (else a prior
rep's `.pyc` can shadow an edited module within the same dir).
