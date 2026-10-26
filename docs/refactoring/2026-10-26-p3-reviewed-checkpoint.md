# P3 actions and journal package — reviewed candidate checkpoint

Candidate parents: P3 WIP `bbccaaefed8f0f482b0b712f5d7ed06e14ef719b` and accepted integration `851955112b8ba8c2911a7684469551afa1541623`. This reviewed two-parent merge is published as the private candidate checkpoint described below. Planned publication ref: `codex/refactor-autopilot-actions-01a16fc3`; no main promotion or deployment.

Final targeted functional proof: **402 passed, 1 skipped** in 39.49s, 12 files. Compile, protected P5/W3e equality, diff and unchanged source seal passed. Prior expanded90 run: **1917 passed, 2 skipped, 11 failed** under earlier source; those failures were diagnosed and corrected, then exercised by the final targeted selection. This is not a claim that all1930 cases passed on current source. The first focused34 attempt had 817 passed/1 skipped/3 failed, later corrected. Graph coverage remains UNKNOWN; manual direct-caller evidence was used.

| Finding | Behavioral warrant in package selections |
|---|---|
| B3-01 | Recorded PID/exact API argv/cwd env identity; fail closed on mismatch; missing json import fixed after actual positive-control failure. |
| B3-02 | Superseded corrupt trials leave planner information/trend views. |
| B3-03 | NaN/±Inf rejection in controller/coordinator/applicator validators. |
| B3-05 | Formerly omitted action types reach mocked loop dispatch. |
| B1-17 | Ordered action tuple agrees with dispatcher registry. |
| B3-06 | Failed provider call schema and bounded sanitized critique-trace digest diagnostics. |
| B3-07 | Archive write failure observability, trace warning-once, unknown utilization failure. |
| B3-08 | Synthetic 200KB stderr child drains; timeout fake models kill/reap, no real Claude invocation in failed fixture. |
| B3-12 | Refusal precedes prewarm/pause/registry side effects. |
| B3-M1 | Journal/trial context forwards through apply_params and mocked successful restart boundary. |
| B3-04 | Loaded journal injection retains separate LEGACY all-era ceiling; generation caches, readonly nested consumers, reset/teardown and legacy all_entries adapter. |
| B3-23 | Bounded archive tail matches full scan behavior. |
| B4-01 | Gapped/numeric shards and explicit malformed-row policy shared by report readers. |
| B3-15 | Dead resume plumbing removed; provider/controller compatibility retained. |
| B3-16 | Dead diagnostic APIs removed; neutral pin reader preserves legacy quality and unknown revision/era in report consumers. |
| B3-M2 | Digest exposes action diversity; unused pareto query removed. |

Raw scoped Ruff exits1 with **16 inherited E701** findings; normalized immutable851 comparison finds **zero introduced**. Raw formatter failures include legacy debt; final review identified exactly three newly failing files (eval_task_coverage_report.py, journal_shards.py, test_journal_shards.py). Their owner-granted formatting closure passed: all three ASTs are identical before/after, non-target source bytes are unchanged, format-check3/compile/no-introduced-Ruff/protected/diff proofs passed. Package accepted for private branch publication. No broad legacy formatting is authorized.

Evidence root: `/mnt/raid0/llm/tmp/refactor-01a16fc3/autopilot-actions/`. `final402-immutable/` retains final functional receipts; `attempt2-receipts/` retains prior90 receipts; `format-closure-20261026/` owns the final formatting proof. AP54 isolation draft was never applied; observed real pool reads were readonly and the exact consumer was not proven. No production evaluator/P5 edits, inference, service reload, thresholds/objectives/history changes or canonical dataset writes.

Final formatting artifact SHA-256 custody:

```text
516adcb60c83400838610c9d1227ab8ce5a516a9975cf33e88fae8001eb0ff2b  AST-custody.json
8fc0a98089bd7ba038ac1ac0563008101b98d0803d6c032283dd62b5c1a71de3  source-before.sha256
f1ac49abed0c3cc11b2fceb2d3cb2eedbaed6b972c61c615bbfc5e57b15d8f4c  source-after.sha256
4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945  introduced-ruff.json
eee76f865de5a9f2456950f710ae5d5f9b5bd0ea02aa54f614c5eed6642a0bfd  format-check.log
```
