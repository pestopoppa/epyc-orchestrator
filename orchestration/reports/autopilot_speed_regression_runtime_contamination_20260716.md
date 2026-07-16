# AutoPilot Speed Regression / Runtime Contamination Audit - 2026-07-16

## Scope

This report records the quiet-window investigation of the post-frontier AutoPilot throughput drop and the runtime flag contamination found during that investigation.

Decision status:

- AutoPilot is intentionally stopped. The process table, not stale phase JSON, is authoritative.
- No kernel promotion decision is made here.
- The frontier replay numbers are current-floor evidence, not clean model-quality evidence.
- The runtime flag contamination fix is implemented in code and covered by focused tests.

## Evidence Summary

### Frontier parameter replay

Existing report: `orchestration/reports/frontier_numeric_replay_20260716.md`.

Quiet-window exact numeric parameter replays, using current `legacy_pool_seed_42_n50` with `AUTOPILOT_W6_AUDIT_BLOCK=0` and `AUTOPILOT_TOOL_SENTINELS=0`, confirmed a current throughput floor around half the June 28 frontier rows:

| Source trial | Surface | Historical speed | Replay speed | Replay q | Replay r | Errors | Artifact |
|---:|---|---:|---:|---:|---:|---:|---|
| `996` | `think_harder` | `69.21354521045092` | `34.75560694205509` | `1.7399999999999998` | `0.86` | `7` | `orchestration/reports/frontier_replay_996_20260716T001046Z.json` |
| `1003` | `memrl_retrieval` | `68.35713081738538` | `35.309929890347824` | `1.7999999999999998` | `0.86` | `7` | `orchestration/reports/frontier_replay_1003_20260715T235917Z.json` |

Quality caveat: these were exact parameter replays, not exact historical qid-vector replays. The current vector overlaps historical rows by only `8/50` qids for trial `996` and `7/50` qids for trial `1003`. The raw quality/reliability drop is also polluted by pre-fix runtime errors, including the plan-review reroute-target bug fixed separately in `bc8d3303`.

### Runtime contamination: trial 1403

Trial `1403` is the confirmed runtime contamination event:

- `timestamp`: `2026-07-15T23:14:01.953749+00:00`
- `action_type`: `structural_experiment`
- `reasoning`: `{"type": "structural_experiment", "flags": {"plan_review": true}}`
- `active_flags`: `["plan_review=True"]`
- `quality`: `0.5454545454545454`
- `speed`: `32.5932099310475`
- `reliability`: `0.2727272727272727`
- `errors`: `40`
- nested `eval_details.details.flag_attestation`: `status=ok`, `workers_seen=6`, `expected={"plan_review": true}`, `diffs=[]`
- nested `eval_details.details.flag_revert_result`: `{"status": "error", "error": "[Errno 111] Connection refused"}`
- journal row still had `outcome_status="ok"` and no `bug_corrupted_by`.

Impact:

- The flag apply was real and attested across workers.
- The safety gate rejected the trial.
- The revert failed.
- Later rows showed `active_flags=[]` while live runtime state still had `plan_review=true`.
- That created a misleading journal/runtime split and allowed subsequent batches to include plan-review overhead while being recorded as unflagged.

This explains part of the observed routing/throughput shift, especially the unexpected architect review traffic in later numeric batches. It does not by itself prove every t/s regression source; EvalTower routing mix, W6/tool-sentinel overlays, question-vector drift, and concurrency bottleneck-role selection remain separate contributors to measure.

## Kernel Lineage Check

Read-only v7 audit found no evidence that the current speed regression was caused by a production kernel source edit:

- Running production server binary reports the old June 26 build line (`version: 9774 (91745611f)`) and mtime `2026-06-26 21:12:22`.
- Production tree: `/mnt/raid0/llm/llama.cpp`, branch `production-consolidated-v6`, HEAD `91a8424e`.
- Experimental tree: `/mnt/raid0/llm/llama.cpp-experimental`, branch `experimental-v7-candidate`, HEAD `46f876c1`.
- Production is not formally an ancestor of experimental, but the production-only ROCm fp8 guard is patch-equivalent to the experimental line and the remaining production-only commit is docs-only.
- IQK references are present in both trees.

Conclusion: v7 is feature-complete enough for a candidate audit, but it is not promoted. Promotion still requires a clean candidate state, fresh build, server/model smokes, canonical v6-vs-v7 performance A/B, and v6/v7 MMLU-Pro + GPQA-Diamond quality gates.

## Code Fix

Implementation files:

- `scripts/autopilot/actions.py`
- `src/autopilot_core/learning_exclusions.py`
- `tests/unit/test_autopilot_actions.py`
- `tests/unit/test_classify_learning_exclusion.py`

Behavior now:

- `structural_experiment` snapshots current live flag values before applying a candidate.
- It refuses to run without an exact restore snapshot for every changed flag.
- It requires successful apply status and successful worker attestation before eval.
- It restores exact prior values after a failed gate instead of blindly inverting candidate flags.
- It requires successful revert status and worker attestation.
- If revert fails after an eval, the eval is marked `bug_corrupted_by="structural_flag_revert_failure"`.
- If apply/attestation fails before eval, the skip outcome is marked `bug_corrupted_by="structural_flag_apply_failure"`.
- `classify_learning_exclusion()` now honors `eval_result.bug_corrupted_by` before exogenous/MAD learning categories, so contaminated evals do not feed learning, baselines, or blacklists as normal evidence.

Focused validation:

- `.venv/bin/python -m pytest tests/unit/test_autopilot_actions.py tests/unit/test_classify_learning_exclusion.py -q`
- Result: `151 passed in 1.00s`

GitNexus impact before edits:

- `_action_structural_experiment`: LOW risk, no upstream impacted nodes reported.
- `classify_learning_exclusion`: LOW risk, upstream path through `scripts/autopilot/autopilot.py:_run_loop_inner`.

## Live Remediation

Runtime flag state was corrected without restarting AutoPilot:

- `POST /config` set `plan_review=false` at `2026-07-16T09:14:52Z`.
- `orchestration/runtime_flags.json` now records `plan_review.value=false`.
- `scripts/validate/attest_flags.py --expect plan_review=false --expect specialist_routing=true --expect skillbank=true --min-workers 6` passed with `workers_seen=6`, no expected diffs, and no heterogeneity.

AutoPilot has not been restarted after this remediation.

## Next Measurements

Before restart, the next clean measurement should isolate overhead sources:

1. Re-run the current `trial 1003` frontier replay protocol after the `plan_review=false` remediation.
2. Compare with a forced `plan_review=false` control and record route distribution, aggregate t/s, median request t/s, wall seconds, request count, errors, and tap-side auxiliary request count.
3. If still slow, run a bottleneck-role/concurrency A/B for EvalTower routing: current default versus worker-oriented bottleneck selection.
4. Only after the orchestrator/runtime controls are clean, run v6-vs-v7 kernel A/B.

Restart boundary: restart AutoPilot only after deciding whether the next action is controlled replay measurement or authority-daemon resume.
