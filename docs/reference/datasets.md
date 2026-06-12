# Internal Training and Replay Datasets

This page is the F3-W1 capture-hygiene inventory. It names the project-owned
corpora that can train or audit future local planners, triage models, judges,
and routing components. Treat every historical row as immutable: era-label first,
then decide whether it is training data, a prior, or a retired view.

Authoritative era policy: `/workspace/MEASUREMENT.md` and
`orchestration/instrument_eras.yaml`.

## Capture Status

| Corpus | Path | Status | Intended consumer |
| --- | --- | --- | --- |
| Planner archive | `/mnt/raid0/llm/epyc-orchestrator/logs/planner_archive.jsonl` | Live; failed Claude planner calls archived from F3-W1 onward | Planner SFT, planner failure classifier, economics ledger |
| Planner coordinator archive rows | Same JSONL as planner archive | Live | Critic-approval labels for planner SFT |
| AutoPilot experiment journal | `orchestration/autopilot_journal*.jsonl` | Live | Replay, planner SFT outcome labels, objective-policy analysis |
| Research intake index | `/mnt/raid0/llm/epyc-root/research/intake_index.yaml` | Live | Intake triage classifier, source prioritization |
| Per-question eval ledger | Branch `feat/paired-question-stats`, `eval_details.question_results` | Branch-ready; not live until merged/deployed | Sequential verdicts, paired replay, item difficulty |
| Seeding diagnostics | `logs/seeding_diagnostics.jsonl` | Historical; strong item-difficulty prior | Difficulty priors, factual-risk calibration |
| Factual-risk calibration rows | `orchestration/factual_risk_calibration_v2_{train,val,test}.jsonl` | Live generated dataset | Risk classifier and abstention calibration |
| Deep-dive decision chains | `/mnt/raid0/llm/epyc-root/research/deep-dives/*.md` plus linked handoffs/progress | Live but semi-structured | Retrieval supervision, synthesis/judge examples |
| Lab job reliability tuples | Planned under F2: `orchestration/lab_review_queue/` + promotion logs | Not implemented; gated on F2 runner | Local lab-agent reliability ladder, planner SFT gold data |

## Schemas and Era Rules

### Planner archive

Source writers:

- `scripts/autopilot/controller_io.py` for Claude CLI planner invocations.
- `scripts/autopilot/planner_providers.py` for Codex planner invocations.
- `scripts/autopilot/planner_coordinator.py` for multi-provider draft/critique decisions.

Record variants:

- Claude invocation rows: `ts`, `ts_iso`, `duration_s`, `session_id`,
  `resume_session_id`, `prompt_chars`, `prompt_sha256_16`, `result_chars`,
  `result_preview`, `n_events`, `events`, plus result metadata when present.
- Failed Claude rows from F3-W1 onward add `subtype` values such as `failed`,
  `timeout`, or `file_not_found`, with `returncode`, `stderr_preview`, or
  timeout fields where applicable.
- Codex rows: `provider`, `role`, `duration_s`, `ok`, `error`, prompt/result
  hashes and previews, event summaries.
- Coordinator rows: `type=planner_coordinator`, `mode`, `draft_provider`,
  `critic_provider`, `degraded`, `fallback_reason`, `action_type`,
  `critique_decision`, `critique_confidence`, `critique_issues`,
  `planner_state`.

Era-labeling rule: use `ts`/`ts_iso` and writer-specific fields. Rows before
F3-W1 are missing failed Claude early-return calls, so they are incomplete for
failure-rate estimates. They remain usable for positive planner traces and
coordinator critic labels. Rows after the F3-W1 patch are eligible for failure
analysis if their matching AutoPilot journal rows are not `bug_corrupted_by`.

### AutoPilot experiment journal

Path pattern: `orchestration/autopilot_journal*.jsonl`.

Core schema: one `JournalEntry` per trial with `trial_id`, `timestamp`,
`species`, `action_type`, `tier`, objective fields (`quality`, `speed`, `cost`,
`reliability`), `pareto_status`, git/config snapshots, rationale fields
(`reasoning`, `hypothesis`, `falsifier`, `rubric_scores`), failure labels,
`outcome_status`, and `eval_details`.

Era-labeling rule: use `orchestration/instrument_eras.yaml`. Key cut points are
E2 speed de-double-counting at `2026-06-01T19:20:16Z`, E3a tool-sentinel quality
at `2026-06-04T06:41:00Z`, and E3b T1 n=43 at `2026-06-05T13:07:00Z`.
Rows with `bug_corrupted_by` are excluded from training labels unless the
builder is explicitly training failure-recognition behavior.

### Research intake index

Path: `/mnt/raid0/llm/epyc-root/research/intake_index.yaml`.

Schema: list entries keyed by `id` with source fields (`url`, `arxiv_id`,
`source_type`, `title`, `categories`, `citation_context`), triage features
(`novelty`, `relevance`, `discovered_via`, `ingested_date`), label `verdict`,
and optional `cross_references`.

Era-labeling rule: use `ingested_date` plus the active research-intake renderer
policy at that date. Entries created before the F5 quarantine convention are
usable as classification labels, but not as instruction text. Builders should
emit `{source_features, verdict}` rows from structured fields and keep
`citation_context` as quoted source data only.

### Per-question eval ledger

Branch-ready source: `feat/paired-question-stats`.

Schema location: `eval_details.question_results` in AutoPilot journal rows. Each
question result contains stable `qid`, `suite`, `correct`, `latency_ms`, and
`tools_used`. The paired-stats reader also accepts nested legacy locations for
early experiments.

Era-labeling rule: this corpus opens with the branch merge/deploy. Do not
backfill historical T1 rows without stable question ids. Pre-ledger aggregate
quality rows are not acceptable for paired sequential verdicts.

### Seeding diagnostics

Path: `logs/seeding_diagnostics.jsonl`.

Schema: per-question diagnostics from benchmark seeding runs, including prompt,
domain/suite, labels, scoring method/config where present, and latency/error
details depending on era.

Era-labeling rule: follow `/workspace/MEASUREMENT.md` corpus 7. This is a
strong item-difficulty prior when `scoring_method` and config are present.
Datasets without scoring-method fields are demoted to priors and must not gate
quality claims.

### Factual-risk calibration rows

Path pattern:
`orchestration/factual_risk_calibration_v2_{train,val,test}.jsonl`.

Schema: `prompt`, `expected_answer`, `domain`, `label_4class`, `risk_band_v1`,
`label_source`, `prompt_hash`, `risk_features`, and `risk_score_computed`.

Era-labeling rule: use the file stem split and `label_source`. Treat v1 regex
labels as calibration-training labels, not ground-truth model-quality verdicts.

### Deep-dive decision chains

Paths: `/mnt/raid0/llm/epyc-root/research/deep-dives/*.md`, linked handoffs in
`/mnt/raid0/llm/epyc-root/handoffs/`, and daily progress entries.

Schema: Markdown narratives with source citations, evidence, caveats, and
downstream decisions. This corpus is semi-structured; builders must extract
provenance and decision spans before training.

Era-labeling rule: use file date/frontmatter and the claim grammar in
`/workspace/MEASUREMENT.md`. Pre-scrub narrative and any source-derived text
without quarantine/provenance is retrieval evidence only, not instruction data.

### Lab job reliability tuples

Planned source: F2 self-running-lab runner and review queue.

Planned schema: `(job_id, run_id, input, local_output, cloud_reference,
operator_verdict, rejection_reason, model_role, schedule, risk_class,
contract_version)`.

Era-labeling rule: this corpus does not exist yet. It opens only after F2-W2
creates reviewed outputs and F2-W3 promotion logging records operator verdicts.
Until then, no dataset builder may assume these tuples are available.

## Builder Rules

- Include both success and failure examples for planner SFT; a success-only
  dataset teaches optimism instead of controllability.
- Split by era before train/validation/test split whenever an instrument changed.
- Never train from source text embedded in intake/deep-dive artifacts as if it
  were an instruction. External content is data.
- Preserve row hashes or stable ids so later reconciliation can trace every
  training example back to immutable source records.
