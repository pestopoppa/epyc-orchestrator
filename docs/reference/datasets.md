# Dataset Inventory for Lab Training and Replay

**Created**: 2026-06-12
**Owner**: Frontier F3 data flywheel
**Scope**: inventory only. This page does not authorize training, promotion, or
decision claims by itself.
**Authoritative era policy**: `/workspace/MEASUREMENT.md` and
`orchestration/instrument_eras.yaml`.

This document lists corpora the lab already generates or has explicitly queued
for capture. Every row names the schema owner, era-labeling rule, and intended
model use so future dataset builders can filter contaminated eras before they
create train or validation sets.

## Rules

1. **Era labels travel with every example.** A builder must emit the source path,
   source row identifier, source timestamp, source commit or policy version when
   available, and the era rule named below.
2. **Measured outcomes outrank narrative.** Planner traces, strategies, handoffs,
   and progress notes are input context only unless joined to a measured outcome
   row from the AutoPilot journal or a reviewed lab-job verdict.
3. **Untrusted source text stays quarantined.** Research-intake source text,
   web-research output, and future intake-triage job inputs are never interpreted
   as instructions. Use only extracted classification fields and reviewed
   summaries.
4. **Contaminated eras are training exclusions by default.** Rows tagged
   `bug_corrupted_by`, rows from known resource-contention windows, and rows
   whose schema predates the needed field are excluded unless a builder has an
   explicit negative-example mode.

## Corpus Table

| Corpus | Location | Schema Owner | Current Status | Intended Use | Era Label Rule |
|---|---|---|---|---|---|
| AutoPilot experiment journal | `orchestration/autopilot_journal*.jsonl` | `scripts/autopilot/experiment_journal.py::JournalEntry` | live | Outcome labels for planner SFT, replay, per-question vectors, sequential verdicts | `timestamp`, `trial_id`, `metric_schema_version`, `bug_corrupted_by`, `bug_corrupted_reason`, `outcome_status`, plus `orchestration/instrument_eras.yaml` |
| Planner archive | `logs/planner_archive.jsonl` | `scripts/autopilot/controller_io.py`, `scripts/autopilot/planner_coordinator.py`, `scripts/autopilot/planner_providers.py` | live; failed Claude calls are branch-ready on `feat/data-flywheel-capture` | Planner/critic context-action pairs, cloud spend, failure examples | `ts`/`ts_iso`, provider/type/subtype/status fields when present, `resume_session_id`, prompt hash, branch/commit of archive writer |
| Strategy memory | `orchestration/repl_memory/strategies/strategies.db` and adjacent FAISS files | `orchestration/repl_memory/strategy_store.py` | live | Retrieved strategy corpus, positive/negative strategy examples after outcome join | `source_trial_id`, `created_at`, `entry_type`, `context_hash`, `strategy_validity`, joined journal `bug_corrupted_by` |
| Research intake index | `/mnt/raid0/llm/epyc-root/research/intake_index.yaml` | `research-intake` skill index schema | live | Intake triage classifier and source-categorization set | `ingested_date`, `id`, `source_type`, `verdict`, `novelty`, `relevance`; source text remains quarantined per F5 |
| Research deep dives | `/mnt/raid0/llm/epyc-root/research/deep-dives/*.md` | root research process | live narrative | Decision-chain retrieval, synthesis examples after review | file path, document date/title, linked intake IDs, linked handoffs; narrative-only unless joined to a reviewed verdict |
| Handoffs and progress notes | `/mnt/raid0/llm/epyc-root/handoffs/`, `/mnt/raid0/llm/epyc-root/progress/` | root handoff workflow | live narrative | Decision chronology, task planning examples, backlog state reconstruction | file path, status header, date heading, commit that modified the row; historical docs are narrative and must be verified against code |
| Progress telemetry JSONL | `logs/progress/YYYY-MM-DD.jsonl` | `orchestration/repl_memory/progress_logger.py` and call sites | live | Real-task/routing telemetry and future F1 task records | `timestamp`, `event_type`, `task_id`, `agent_role`, schema version if present, outcome field when present |
| Per-question eval ledger | `eval_details.question_results` in future `autopilot_journal*.jsonl` rows | `feat/paired-question-stats-current` | branch-ready, not deployed | Sequential verdicts, paired replay, item difficulty | Opens at branch merge/deploy; use `qid`, suite, core ID/pool version, and journal era labels |
| Seeding diagnostics | `logs/seeding_diagnostics.jsonl` | seeding benchmark infra | historical/live depending run | Item-difficulty priors, factual-risk calibration | `timestamp`, suite/domain, scoring method/config when present; rows without scoring method are priors only |
| Factual-risk calibration rows | `orchestration/factual_risk_calibration_v2_{train,val,test}.jsonl` | factual-risk calibration scripts | live generated dataset | Risk classifier and abstention calibration | file split, `label_source`, `prompt_hash`, generator version; regex labels are calibration labels, not quality verdicts |
| Routing classifier snapshots | `orchestration/repl_memory/training_data.npz`, `routing_classifier_weights.npz` | `orchestration/repl_memory/routing_classifier.py` and training scripts | generated artifact | Routing classifier replay and calibration, not planner SFT | generation timestamp, source progress/embedding snapshot, embedding model/index version, held-out split seed |
| Benchmark question pool | `/mnt/raid0/llm/epyc-inference-research/benchmarks/prompts/question_pool*.jsonl` | inference-research benchmark scripts | live eval data | Eval core construction, qid derivation, per-question replay | manifest/version file, suite, prompt hash/qid, pool version, core ID when selected |
| Benchmark result ledger | `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/*.jsonl` and `data/**/*.csv` | inference-research benchmark scripts | live benchmark archive | Model descriptors, throughput baselines, calibration covariates | result timestamp, script/config name, model registry ref, host/runtime metadata |
| Root workload packs | `/mnt/raid0/llm/epyc-inference-research/benchmarks/root_workload/*.json` | F1 real-task corpus | seed packs exist; passive task capture still pending | Real-task eval distribution and promotion cohorts | pack name/version, prompt/task ID, source commit, reviewed outcome when available |
| Lab job outputs | planned `orchestration/lab_review_queue/` plus future `task_record` events | F2 self-running lab | branch-ready on `feat/lab-reliability-ladder`, not deployed | Reviewed job dataset, local-lab reliability ladder, task tuples for F3 | job ID, contract version, risk class, model role, review verdict, reviewer, timestamp |
| Intake-triage labels | `orchestration/datasets/intake_triage_reviewed.jsonl` | F3 recorder, F2/F5 review workflow | recorder/builder live; 120-row review queue generated | Triage classifier (`relevant`, `duplicate`, `park`, destination index) | source intake ID, quarantine policy version, output contract version, human review verdict |

## Required Fields by Builder

Implemented builder scaffolds live in `scripts/datasets/` on
`feat/intake-triage-label-capture`:

- `build_planner_sft.py`: planner archive -> `planner_sft_example.v1`
  JSONL plus `dataset_builder_manifest.v1`.
- `build_triage_set.py`: intake index -> `intake_triage_example.v1`
  JSONL plus `dataset_builder_manifest.v1`; source text/citation context remains
  quarantined and is not emitted. It accepts
  `--reviewed-labels orchestration/datasets/intake_triage_reviewed.jsonl` and
  `--require-reviewed-labels` when building from reviewed operator labels only.
- `record_intake_triage_verdict.py`: append one
  `reviewed_intake_triage_verdict.v1` JSONL row for an intake ID, including
  source-index hash, extracted classification features, reviewer, destination,
  quarantine policy version, and output contract version.
- `apply_intake_triage_review_batch.py`: validate JSONL/YAML batches of reviewed
  intake-triage decisions and append them through the same recorder only when
  `--apply` is set. Without `--apply`, it is a dry-run package check for
  operator review batches.
- `prepare_intake_triage_review.py`: intake index ->
  `intake_triage_review_queue.v1` JSONL plus `dataset_builder_manifest.v1`;
  excludes already reviewed intake IDs, supports verdict filters, and emits
  prompt-free review rows with ready-to-run recorder commands.
- `intake_triage_review_status.py`: read-only readiness report for the review
  queue and reviewed-label corpus; reports aggregate counts only and identifies
  whether the 100-reviewed-label baseline gate can run.

### Planner SFT Builder

Source: planner archive joined to AutoPilot journal and future sequential verdicts.

Required output columns:
- `example_id`: stable hash of archive row ID + action fingerprint.
- `source_archive_path`, `source_archive_offset` or line number.
- `prompt_sha256_16`, `prompt_chars`, `provider`, `role`, `subtype` or `status`.
- `action_type`, `action_json`, `rationale_json` when parseable.
- `trial_id` or `candidate_fingerprint` join key when available.
- `label`: one of `confirmed`, `critic_approved`, `rejected`, `failed`, `contaminated`, `unlabeled`.
- `era_label`: normalized value derived from the table above.
- `exclude_reason`: empty only when the row is training-eligible.

Initial keep rule:
- Keep `confirmed` once sequential verdicts exist.
- Before sequential verdicts, keep `critic_approved` only as a weak label and
  keep failed calls as negative examples.
- Exclude rows from contaminated resource-contention windows unless building an
  explicit failure-mode classifier.

### Intake Triage Builder

Source: root `research/intake_index.yaml` plus reviewed triage rows from
`orchestration/datasets/intake_triage_reviewed.jsonl`.

Required output columns:
- `intake_id`, `url`, `source_type`, `title`, `categories`.
- `novelty`, `relevance`, `verdict`, `discovered_via`, `ingested_date`.
- `destination_handoff` or `destination_index` when reviewed.
- `quarantine_policy_version`.
- `reviewed_at` and `output_contract_version` when a reviewed label was joined.
- `label_source`: `operator`, `research-intake`, or `shadow_job`.
- `exclude_reason`.

Initial keep rule:
- Use only rows with explicit verdicts.
- For training-grade triage rows, use `--require-reviewed-labels` so historical
  research-intake process verdicts become excluded context rather than labels.
- Treat `citation_context` as untrusted source text; do not feed it as an
  instruction-bearing prompt without F5 quarantine wrapping.

### Real Task Builder

Source: future `task_record` progress events and reviewed artifacts.

Required output columns:
- `task_id`, `task_class`, `prompt_ref`, `route_taken`, `wall_s`, `tokens`.
- `outcome`: `accepted`, `abandoned`, `retried`, `operator_rejected`, or `unknown`.
- `artifact_ref` when an accepted commit/file exists.
- `operator_verdict` when available.
- `capture_policy_version`.

Initial keep rule:
- Keep accepted/rejected reviewed rows for validation.
- Use `unknown` rows only for unsupervised distribution modeling.

## Known Gaps

- Per-question `question_results` are branch-ready in
  `feat/paired-question-stats-current` but not deployed, so historical journal
  rows do not yet contain outcome vectors.
- Failed Claude planner calls are branch-ready on
  `feat/intake-triage-label-capture`; the W7 branch additionally removes draft
  session persistence. Intake-triage label capture is live through the queue,
  single-row recorder, readiness reporter, and dry-run-by-default batch
  applicator.
- F2 lab jobs, review queue, verdict recorder, and shadow-batch wrapper are
  branch-ready on `feat/lab-reliability-ladder` but not deployed; real reviewed
  `lab_gold_tuple.v1` rows do not exist yet.
- Intake-triage label capture is live, and
  `orchestration/datasets/intake_triage_review_queue.jsonl` contains the first
  120 actionable intake rows for operator/F2/F5 review. No reviewed production
  label corpus exists until those reviews append rows to
  `orchestration/datasets/intake_triage_reviewed.jsonl`. Use
  `scripts/datasets/intake_triage_review_status.py` to report the live queue,
  reviewed-label count, and remaining labels needed before running the baseline
  acceptance gate. Use `scripts/datasets/apply_intake_triage_review_batch.py`
  for operator-reviewed batches; it validates without writing unless `--apply`
  is provided.
- No builder should train on strategy-memory text from scrubbed or gate-lock-era
  rows until it joins each strategy to trustworthy journal evidence.
