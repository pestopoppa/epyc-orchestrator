# Fable5 Gate Report

Ready: false

## Blockers

- ds_e1_dynamic_stack: ri10_canary: RI-10 raw high-risk sample-count coverage exists, but enforce/shadow arm-attributed decision telemetry is not sufficient.
- ds_e1_dynamic_stack: kv_size_measurements: No direct DS-E1 production KV-size measurement series was found.
- xmas_production_path: xmas_routing.mode is off; enforce remains default-off
- xmas_production_path: latest X-MAS held-out A/B policy is incumbent_constrained_v1; required incumbent_constrained_cheapfirst_v2
- xmas_production_path: latest X-MAS held-out A/B decision is hold

## Next Actions

### collect_w8_promotion_eval_evidence

- Priority: `P0`
- Status: `active`
- Reason: W4/W6 authority is restart-ready; W8 still needs live promotion-eval finalization evidence before closing the tail.
- Evidence:
  - `w8_promotion_status`: "none"
  - `open_requirements`: ["combined_E_below_required", "fresh_promotion_eval_required", "seq_confirmation_required"]
  - `pending_candidate`: null
  - `pending_source_trial_id`: null
  - `pending_attempts`: null
  - `last_blocked_trial_id`: null
  - `last_blocked_candidate`: null
  - `last_blocked_reason`: null
  - `latest_seq_trial_id`: 1083
  - `latest_candidate`: "ea4334a6c3ddbadc"
  - `latest_combined_E`: 0.935508
  - `latest_required_E`: 100.0
  - `latest_confirmed`: false
  - `latest_fresh_eval`: false
  - `latest_seq_state`: "accumulating"
  - `latest_baseline_reference_state`: "fresh"
  - `baseline_reference_last_forced_trial_id`: 1082
  - `baseline_reference_last_forced_reason`: "10 trusted profile trials since baseline reference draw"
  - `baseline_reference_blocked_reason`: null
- Command: `cd /mnt/raid0/llm/epyc-orchestrator && uv run python scripts/autopilot/restart_readiness_report.py --json --strict --require-seq-cutover --require-w6-audit`
- Follow-up: `uv run python scripts/autopilot/fable5_gate_report.py --json --strict`

### run_ds_e1_kv_measurements

- Priority: `P0`
- Status: `blocked`
- Reason: DS-E1 cannot decide DS-7/DS-6 profiles until direct production KV-size rows exist.
- Requires: attested clean window with AutoPilot and live llama-server processes stopped/coordinated
- Blocked by:
  - active AutoPilot process(es): 1671930 .venv/bin/python3 scripts/autopilot/autopilot.py start --max-trials 2000
  - live llama-server process(es): 1116996 /mnt/raid0/llm/llama.cpp/build/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf -md /mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2 --draft-p-min 0.0 --threads-draft 16 -ub 512 --no-mmap --reasoning off --jinja --host 127.0.0.1 --port 8072 -np 1 -c 16384 -t 96 -ctk q8_0 -ctv q8_0 --flash-attn on; 1117328 /mnt/raid0/llm/llama.cpp/build/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf -md /mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2 --draft-p-min 0.0 --threads-draft 16 -ub 512 --no-mmap --reasoning off --jinja --host 127.0.0.1 --port 8082 -np 1 -c 16384 -t 48 -ctk q8_0 -ctv q8_0 --flash-attn on; 1117611 /mnt/raid0/llm/llama.cpp/build/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf -md /mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2 --draft-p-min 0.0 --threads-draft 16 -ub 512 --no-mmap --reasoning off --jinja --host 127.0.0.1 --port 8182 -np 1 -c 16384 -t 48 -ctk q8_0 -ctv q8_0 --flash-attn on; 1117883 /mnt/raid0/llm/llama.cpp/build/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf -md /mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2 --draft-p-min 0.0 --threads-draft 16 -ub 512 --no-mmap --reasoning off --jinja --host 127.0.0.1 --port 8282 -np 1 -c 16384 -t 48 -ctk q8_0 -ctv q8_0 --flash-attn on; ... +24 more
- Command: `cd /mnt/raid0/llm/epyc-inference-research && scripts/benchmark/ds_e1_kv_measurements.sh --execute`
- Follow-up: `cd /mnt/raid0/llm/epyc-orchestrator && uv run python scripts/server/dynamic_stack_evidence_packet.py --output orchestration/reports/ds_e1_evidence_packet_$(date -u +%Y%m%dT%H%M%SZ).md --strict`

### collect_ri10_canary_arm_telemetry

- Priority: `P0`
- Status: `active`
- Reason: RI-10 has raw high-risk samples, but arm-attributed canary telemetry is not yet decision-grade.
- Command: `uv run python scripts/analysis/ri10_canary_sample_report.py`

### run_xmas_constrained_policy_ab

- Priority: `P0`
- Status: `blocked`
- Reason: X-MAS enforce needs a fresh held-out A/B carrying incumbent_constrained_cheapfirst_v2 and a promote_candidate verdict.
- Requires: attested quiet window; runner preflight refuses AutoPilot and competing benchmark coordinators
- Blocked by:
  - active AutoPilot process(es): 1671930 .venv/bin/python3 scripts/autopilot/autopilot.py start --max-trials 2000
- Command: `cd /mnt/raid0/llm/epyc-orchestrator && uv run python scripts/benchmark/xmas_live_ab.py --prompts benchmarks/results/runs/xmas_live_ab/20260618-heldout-resilient/prompts.jsonl --reps 2 --host-quiet-confirmed --output benchmarks/results/runs/xmas_live_ab/$(date -u +%Y%m%dT%H%M%SZ)-constrained-policy`

## Sections

### phase_health

- Status: `ready`
- Summary: AutoPilot phase heartbeat is active at trial 1085 / dispatch_action (T1 50/60).
- Details:
  - `ok`: true
  - `status`: "active"
  - `trial_id`: 1085
  - `phase`: "dispatch_action"
  - `action_type`: "structural_experiment"
  - `heartbeat_age_s`: 36.40364336967468
  - `pid`: 1671930
  - `pid_alive`: true
  - `process_started_at_s`: 1783088602.07
  - `require_current_code`: false
  - `code_stale`: false
  - `code_stale_paths`: []
  - `eval_label`: "T1"
  - `eval_completed_questions`: 50
  - `eval_total_questions`: 60
  - `eval_correct_questions`: null
  - `eval_correct_pct`: null
  - `eval_concurrency`: null
  - `planner_hints_enabled`: true
  - `seq_verdict_enabled`: true
  - `w6_audit_accrual_enabled`: true
  - `w6_audit_shadow_only`: true
  - `w6_audit_n`: "10"
  - `w6_audit_every_n_trials`: "1"
  - `autopilot_planner_timeout`: "600"

### w4_w6_restart_cutover

- Status: `ready`
- Summary: W4/W6 strict restart cutover is ready.
- Details:
  - `restart_ready`: true
  - `archive_source_surface_ok`: true
  - `archive_source_surface_count`: 6
  - `archive_source_surface_failed_count`: 0
  - `seq_cutover_ready`: true
  - `seq_trusted_vector_trials`: 219
  - `seq_min_trusted_vector_trials`: 120
  - `seq_trusted_vector_trials_remaining`: 0
  - `seq_shadow_rows`: 142
  - `seq_min_shadow_rows`: 30
  - `seq_shadow_rows_remaining`: 0
  - `w8_promotion_status`: "none"
  - `w8_open_requirements`: ["combined_E_below_required", "fresh_promotion_eval_required", "seq_confirmation_required"]
  - `w8_pending_candidate`: null
  - `w8_pending_source_trial_id`: null
  - `w8_pending_attempts`: null
  - `w8_last_finalized_trial_id`: null
  - `w8_last_finalized_candidate`: null
  - `w8_last_finalized_combined_E`: null
  - `w8_last_finalized_delta_excludes_regression`: null
  - `w8_last_blocked_trial_id`: null
  - `w8_last_blocked_candidate`: null
  - `w8_last_blocked_reason`: null
  - `w8_latest_seq_trial_id`: 1083
  - `w8_latest_candidate`: "ea4334a6c3ddbadc"
  - `w8_latest_combined_E`: 0.935508
  - `w8_latest_required_E`: 100.0
  - `w8_latest_confirmed`: false
  - `w8_latest_seq_state`: "accumulating"
  - `w8_latest_baseline_reference_state`: "fresh"
  - `w8_latest_fresh_eval`: false
  - `w8_baseline_reference_last_forced_trial_id`: 1082
  - `w8_baseline_reference_last_forced_reason`: "10 trusted profile trials since baseline reference draw"
  - `w8_baseline_reference_last_forced_stale`: false
  - `w8_baseline_reference_blocked_trial_id`: null
  - `w8_baseline_reference_blocked_reason`: null
  - `w6_audit_cutover_ready`: true
  - `w6_audited_trial_count`: 56
  - `w6_min_audited_trials`: 30
  - `w6_audited_trial_count_remaining`: 0
  - `w6_alarm_clearance_clean_trials_required`: 0
  - `w6_raw_audited_trial_count`: 56
  - `w6_trusted_audited_trial_count`: 56
  - `w6_untrusted_audited_trial_count`: 0
  - `w6_untrusted_audited_trial_ids`: []
  - `w6_gaming_alarm`: false
  - `w6_potential_overfit_divergences`: 0
  - `cutover_horizon_clean_trials_remaining`: 0
  - `cutover_horizon_blocker`: null
  - `cutover_horizon_components`: {"seq_shadow_rows": 0, "seq_trusted_vectors": 0, "w6_alarm_clearance": 0, "w6_audited_trials": 0}
  - `baseline_seed_append_ready`: false
  - `baseline_seed_append_required`: false
  - `baseline_seed_append_expect_trial_counter`: null
  - `baseline_seed_append_expect_journal_max_trial_id`: null
  - `durable_journal_max_trial_id`: 1084
  - `state_trial_counter`: 1085
  - `snapshot_restart_readiness`: "tail_fold_ready"
  - `snapshot_payload_journal_max_trial_id`: 1084

### ds_e1_dynamic_stack

- Status: `blocked`
- Summary: DS-E1 dynamic-stack packet is not decision-ready.
- Blockers:
  - ri10_canary: RI-10 raw high-risk sample-count coverage exists, but enforce/shadow arm-attributed decision telemetry is not sufficient.
  - kv_size_measurements: No direct DS-E1 production KV-size measurement series was found.
- Details:
  - `ready_for_profile_decision`: false
  - `generated_at`: "2026-07-03T14:45:56Z"
  - `section_statuses`: {"contention_matrix": "ready", "ds5_roster_manifest": "ready", "kv_size_measurements": "missing", "ri10_canary": "insufficient_data", "stack_roster": "ready"}
  - `kv_required_measurements`: {"architect_general": [2048, 8192], "frontdoor": [2048, 8192, 32768], "ingest_long_context": [2048, 8192, 32768], "worker_general": [2048, 8192]}
  - `kv_observed_measurements`: null
  - `kv_missing_measurements`: null
  - `kv_expected_csv_columns`: ["role", "model_id", "model_path", "context_length", "max_context", "status", "rss_load_mb", "rss_after_prefill_mb", "server_kv_size_mb", "prompt_tokens", "prompt_tps", "log_file", "notes"]
  - `kv_candidate_paths`: null
  - `kv_searched_globs`: ["orchestration/reports/ds_e1*kv*", "orchestration/reports/dynamic_stack*kv*", "../epyc-inference-research/data/dynamic_stack/**/kv*", "../epyc-inference-research/data/kv_measurements/**"]
  - `clean_window_ready`: false
  - `clean_window_blockers`: ["active AutoPilot process(es): 1671930 .venv/bin/python3 scripts/autopilot/autopilot.py start --max-trials 2000", "live llama-server process(es): 1116996 /mnt/raid0/llm/llama.cpp/build/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf -md /mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2 --draft-p-min 0.0 --threads-draft 16 -ub 512 --no-mmap --reasoning off --jinja --host 127.0.0.1 --port 8072 -np 1 -c 16384 -t 96 -ctk q8_0 -ctv q8_0 --flash-attn on; 1117328 /mnt/raid0/llm/llama.cpp/build/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf -md /mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2 --draft-p-min 0.0 --threads-draft 16 -ub 512 --no-mmap --reasoning off --jinja --host 127.0.0.1 --port 8082 -np 1 -c 16384 -t 48 -ctk q8_0 -ctv q8_0 --flash-attn on; 1117611 /mnt/raid0/llm/llama.cpp/build/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf -md /mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2 --draft-p-min 0.0 --threads-draft 16 -ub 512 --no-mmap --reasoning off --jinja --host 127.0.0.1 --port 8182 -np 1 -c 16384 -t 48 -ctk q8_0 -ctv q8_0 --flash-attn on; 1117883 /mnt/raid0/llm/llama.cpp/build/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf -md /mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2 --draft-p-min 0.0 --threads-draft 16 -ub 512 --no-mmap --reasoning off --jinja --host 127.0.0.1 --port 8282 -np 1 -c 16384 -t 48 -ctk q8_0 -ctv q8_0 --flash-attn on; ... +24 more"]

### xmas_production_path

- Status: `blocked`
- Summary: X-MAS production routing remains gated.
- Blockers:
  - xmas_routing.mode is off; enforce remains default-off
  - latest X-MAS held-out A/B policy is incumbent_constrained_v1; required incumbent_constrained_cheapfirst_v2
  - latest X-MAS held-out A/B decision is hold
- Details:
  - `config_path`: "/mnt/raid0/llm/epyc-orchestrator/orchestration/classifier_config.yaml"
  - `candidate_table_path`: "/mnt/raid0/llm/epyc-orchestrator/orchestration/xmas_winner_table.yaml"
  - `quiet_window_ready`: false
  - `quiet_window_blockers`: ["active AutoPilot process(es): 1671930 .venv/bin/python3 scripts/autopilot/autopilot.py start --max-trials 2000"]
  - `mode`: "off"
  - `winner_table_path`: "orchestration/xmas_winner_table.yaml"
  - `require_complete_table`: true
  - `config_validation_errors`: []
  - `candidate_table_errors`: []
  - `candidate_table_ready`: true
  - `latest_ab_summary_path`: "/mnt/raid0/llm/epyc-orchestrator/benchmarks/results/runs/xmas_live_ab/20260621T112005Z-constrained-policy/summary.json"
  - `latest_ab_results_path`: "/mnt/raid0/llm/epyc-orchestrator/benchmarks/results/runs/xmas_live_ab/20260621T112005Z-constrained-policy/results.jsonl"
  - `latest_ab_decision_status`: "hold"
  - `latest_ab_score_delta`: -0.25
  - `latest_ab_latency_ratio`: 0.7142973457091678
  - `latest_ab_blockers`: ["overall score delta -0.250 < required 0.050", "no domain improved by >= 0.050", "domain regressions: code, math, reasoning"]
  - `latest_ab_policy`: "incumbent_constrained_v1"
  - `required_ab_policy`: "incumbent_constrained_cheapfirst_v2"
